"""FastAPI endpoints for DSPy-style prompt compilation and metric evaluation."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dspy_opt.optimizer import PromptOptimizer

app = FastAPI(title="DSPy Prompt Optimizer", version="1.0.0")
opt = PromptOptimizer()


class OptReq(BaseModel):
    """Prompt optimization request payload."""

    prompt: str = Field(..., min_length=3, max_length=8000)
    domain: str = Field(default="Software Engineering", min_length=2, max_length=200)
    examples: list[dict[str, str]] | None = Field(default=None)
    output_schema: dict[str, str] | None = Field(default=None)


class EvalReq(BaseModel):
    """Output metric evaluation request."""

    prediction: str = Field(..., min_length=1)
    reference: str = Field(..., min_length=1)
    require_json: bool = Field(default=False)


@app.get("/health", tags=["Monitoring"])
def health_check() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/optimize", tags=["Optimization"])
def optimize_prompt(req: OptReq) -> dict[str, Any]:
    """Compile and select the highest-scoring structured prompt."""
    try:
        return opt.optimize(
            req.prompt,
            req.domain,
            examples=req.examples,
            output_schema=req.output_schema,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/evaluate", tags=["Evaluation"])
def evaluate_prediction(req: EvalReq) -> dict[str, Any]:
    """Evaluate prediction fidelity against a reference answer."""
    return opt.evaluate_output(
        req.prediction,
        req.reference,
        require_json=req.require_json,
    )
