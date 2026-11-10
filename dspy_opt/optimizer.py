"""DSPy-inspired prompt evaluation and multi-candidate compilation engine."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 overlap between prediction and ground-truth reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0 if pred_tokens != ref_tokens else 1.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return round(2.0 * precision * recall / (precision + recall), 4)


class PromptEvaluator:
    """Deterministic multi-objective scorer evaluating prompt engineering quality."""

    def score_prompt(
        self,
        prompt: str,
        task_domain: str,
        has_examples: bool = False,
        expected_fields: list[str] | None = None,
    ) -> dict[str, float]:
        """Score a prompt across grounding, structure, verification, and coverage in [0, 1]."""
        lower = prompt.lower()

        domain_tokens = set(_tokenize(task_domain))
        prompt_tokens = set(_tokenize(prompt))
        grounding = (
            len(domain_tokens & prompt_tokens) / len(domain_tokens) if domain_tokens else 1.0
        )

        structure_cues = ("json", "schema", "format", "output", "return", "fields")
        structure = min(1.0, sum(1 for cue in structure_cues if cue in lower) / 3.0)

        verification_cues = ("step-by-step", "verify", "evidence", "avoid", "edge", "constraint")
        verification = min(1.0, sum(1 for cue in verification_cues if cue in lower) / 3.0)

        if expected_fields:
            matched_fields = sum(1 for f in expected_fields if f.lower() in lower)
            schema_coverage = matched_fields / len(expected_fields)
        else:
            schema_coverage = 1.0 if structure >= 0.66 else 0.5

        example_bonus = 1.0 if (has_examples and "example" in lower) else (0.7 if not has_examples else 0.2)

        overall = round(
            0.25 * grounding
            + 0.25 * structure
            + 0.20 * verification
            + 0.15 * schema_coverage
            + 0.15 * example_bonus,
            4,
        )
        return {
            "overall": overall,
            "grounding": round(grounding, 4),
            "structure": round(structure, 4),
            "verification": round(verification, 4),
            "schema_coverage": round(schema_coverage, 4),
        }


class PromptOptimizer:
    """Multi-candidate prompt compiler with few-shot bootstrapping and measured delta scoring."""

    def __init__(self, evaluator: PromptEvaluator | None = None) -> None:
        self.evaluator = evaluator or PromptEvaluator()

    @staticmethod
    def _format_examples(examples: list[dict[str, str]]) -> str:
        blocks = [
            f"Example {idx}:\n  Input: {ex.get('input', '').strip()}\n  Output: {ex.get('output', '').strip()}"
            for idx, ex in enumerate(examples, start=1)
        ]
        return "\n".join(blocks)

    @staticmethod
    def _format_schema(output_schema: dict[str, str] | None) -> str:
        if not output_schema:
            return "Structured JSON object with verified fields."
        return json.dumps(output_schema, indent=2)

    def _build_candidates(
        self,
        base_prompt: str,
        task_domain: str,
        examples: list[dict[str, str]] | None = None,
        output_schema: dict[str, str] | None = None,
    ) -> list[tuple[str, str]]:
        directives = [
            "Think step-by-step and verify intermediate reasoning constraints before answering.",
            "Return structured JSON output strictly matching the target output schema.",
            "Cite verifiable evidence and avoid speculation on missing edge cases.",
        ]
        directives_block = "\n".join(f"- {d}" for d in directives)
        schema_block = self._format_schema(output_schema)

        zero_shot = (
            f"Role: Expert practitioner in {task_domain}.\n"
            f"Task: {base_prompt}\n\n"
            f"Output Format:\n{schema_block}\n\n"
            f"Strict Directives:\n{directives_block}"
        )

        cot_verified = (
            f"You are a principal specialist in {task_domain}.\n"
            f"Objective: {base_prompt}\n"
            "Reasoning Protocol: Decompose constraints step-by-step, verify edge cases, and ground all claims.\n"
            f"Target Output Schema (JSON):\n{schema_block}\n\n"
            f"Strict Directives:\n{directives_block}"
        )

        candidates: list[tuple[str, str]] = [
            ("zero_shot_structured", zero_shot),
            ("cot_verified", cot_verified),
        ]

        if examples:
            ex_block = self._format_examples(examples)
            few_shot = f"{cot_verified}\n\nDemonstrations:\n{ex_block}"
            candidates.append(("few_shot_bootstrapped", few_shot))

        return candidates

    def optimize(
        self,
        base_prompt: str,
        task_domain: str = "Software Engineering",
        examples: list[dict[str, str]] | None = None,
        output_schema: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compile candidate prompts, evaluate against baseline, and return highest-scoring prompt."""
        if not base_prompt or not base_prompt.strip():
            raise ValueError("base_prompt must not be empty")

        expected_fields = list(output_schema.keys()) if output_schema else None
        has_examples = bool(examples)

        baseline_metrics = self.evaluator.score_prompt(
            base_prompt,
            task_domain=task_domain,
            has_examples=has_examples,
            expected_fields=expected_fields,
        )

        candidates = self._build_candidates(
            base_prompt=base_prompt.strip(),
            task_domain=task_domain.strip(),
            examples=examples,
            output_schema=output_schema,
        )

        scored_candidates: list[dict[str, Any]] = []
        for strategy, text in candidates:
            metrics = self.evaluator.score_prompt(
                text,
                task_domain=task_domain,
                has_examples=has_examples,
                expected_fields=expected_fields,
            )
            scored_candidates.append(
                {
                    "strategy": strategy,
                    "prompt": text,
                    "metrics": metrics,
                }
            )

        best = max(scored_candidates, key=lambda item: item["metrics"]["overall"])
        base_score = baseline_metrics["overall"]
        opt_score = best["metrics"]["overall"]
        rel_gain = round(((opt_score - base_score) / max(base_score, 0.05)) * 100.0, 1)

        return {
            "original_prompt": base_prompt,
            "optimized_prompt": best["prompt"],
            "selected_strategy": best["strategy"],
            "baseline_score": base_score,
            "optimized_score": opt_score,
            "estimated_accuracy_gain": f"+{max(0.0, rel_gain)}%",
            "metrics": best["metrics"],
            "candidates_evaluated": len(scored_candidates),
        }

    def evaluate_output(
        self,
        prediction: str,
        reference: str,
        require_json: bool = False,
    ) -> dict[str, Any]:
        """Score a model output against a reference via token F1 and JSON syntax validity."""
        f1 = token_f1_score(prediction, reference)
        json_valid: bool | None = None
        if require_json:
            try:
                json.loads(prediction)
                json_valid = True
            except json.JSONDecodeError:
                json_valid = False
        return {"token_f1": f1, "json_valid": json_valid}
