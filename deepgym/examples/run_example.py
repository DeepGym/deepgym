"""End-to-end example: define env, run solution, get score."""
from deepgym import DeepGym, Environment
from pathlib import Path

dg = DeepGym()

env = Environment(
    task=Path("examples/python_sorting/task.md").read_text(),
    verifier_code=Path("examples/python_sorting/verifier.py").read_text(),
    language="python",
    timeout=30,
    difficulty="easy",
    domain="coding",
    tags=["sorting", "algorithms"],
)

solution = Path("examples/python_sorting/reference_solution.py").read_text()
result = dg.run(env, model_output=solution)

print(f"Score: {result.score}")
print(f"Passed: {result.passed}")
print(f"Time: {result.execution_time_ms:.0f}ms")
print(f"Output: {result.output}")
