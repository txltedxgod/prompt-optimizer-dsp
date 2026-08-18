from fastapi import FastAPI
from pydantic import BaseModel
from dspy_opt.optimizer import PromptOptimizer

app = FastAPI(title="DSPy Prompt Optimizer", version="0.1.0")
opt = PromptOptimizer()

class OptReq(BaseModel):
    prompt: str
    domain: str = "Software Engineering"

@app.post("/api/v1/optimize")
def optimize_prompt(req: OptReq):
    return opt.optimize(req.prompt, req.domain)
