from fastapi.testclient import TestClient

from dspy_opt.api import app
from dspy_opt.optimizer import PromptOptimizer, token_f1_score


def test_prompt_optimization() -> None:
    opt = PromptOptimizer()
    res = opt.optimize("Write a SQL query for top users", "Database Optimization")
    assert "Strict Directives" in res["optimized_prompt"]
    assert res["optimized_score"] > res["baseline_score"]
    assert res["candidates_evaluated"] >= 2


def test_few_shot_and_schema_prompt_compilation() -> None:
    opt = PromptOptimizer()
    res = opt.optimize(
        "Extract invoice totals and vendor tax identifiers",
        task_domain="Financial Document Extraction",
        examples=[
            {
                "input": "Acme Corp Invoice #42 Total: $199.50",
                "output": '{"vendor": "Acme Corp", "total": 199.50}',
            }
        ],
        output_schema={"vendor": "string", "total": "float"},
    )
    assert res["selected_strategy"] == "few_shot_bootstrapped"
    assert "Acme Corp" in res["optimized_prompt"]
    assert "vendor" in res["optimized_prompt"]
    assert res["metrics"]["schema_coverage"] == 1.0


def test_token_f1_and_http_endpoints() -> None:
    assert token_f1_score("SELECT id FROM users", "SELECT id, name FROM users") > 0.7

    client = TestClient(app)
    assert client.get("/health").status_code == 200

    opt_resp = client.post(
        "/api/v1/optimize",
        json={
            "prompt": "Classify customer support priority",
            "domain": "Customer Support Triage",
            "output_schema": {"priority": "string"},
        },
    )
    assert opt_resp.status_code == 200
    assert opt_resp.json()["optimized_score"] > opt_resp.json()["baseline_score"]

    eval_resp = client.post(
        "/api/v1/evaluate",
        json={
            "prediction": '{"priority": "high"}',
            "reference": '{"priority": "high"}',
            "require_json": True,
        },
    )
    assert eval_resp.status_code == 200
    assert eval_resp.json()["json_valid"] is True
    assert eval_resp.json()["token_f1"] == 1.0
