from dspy_opt.optimizer import PromptOptimizer

def test_prompt_optimization():
    p = PromptOptimizer()
    res = p.optimize("Write a SQL query for top users", "Database Optimization")
    assert "Strict Directives" in res["optimized_prompt"]
