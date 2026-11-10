# DSPy Prompt Optimizer

Multi-candidate prompt compilation, bootstrapping, and deterministic evaluation engine with a FastAPI REST interface.

## Features

- **Multi-Candidate Compilation (`PromptOptimizer`)**: Compiles candidate strategies across domain grounding, structured output constraints, verification rubrics, and bootstrapped few-shot examples, selecting the highest-scoring prompt with measured before/after score deltas.
- **Deterministic Scorer (`PromptEvaluator`)**: Evaluates grounding, formatting cues, verification directives, and expected schema field coverage in `[0, 1]`.
- **Output Verification (`token_f1_score` & JSON checks)**: Computes normalized token-level F1 overlap and optional JSON validity against ground-truth references.
- **REST API**: FastAPI endpoints `GET /health`, `POST /api/v1/optimize`, and `POST /api/v1/evaluate`.

## Local Development

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt ruff
ruff check .
pytest -v tests/
```
