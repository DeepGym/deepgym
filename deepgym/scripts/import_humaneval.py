#!/usr/bin/env python3
"""Import HumanEval benchmark into DeepGym environment format.

Downloads the 164 HumanEval problems from HuggingFace and generates
a DeepGym-compatible environment directory for each one, including:
  - task.md (the prompt)
  - reference_solution.py (prompt + canonical solution)
  - verifier.py (wraps the HumanEval test code)
  - metadata.json

Also generates environments/humaneval/registry.json indexing all problems.
"""

import json
from pathlib import Path


def generate_verifier(task_id: str, entry_point: str, prompt: str, test_code: str) -> str:
    """Generate a DeepGym-compatible verifier from HumanEval test code.

    HumanEval tests define a ``check(candidate)`` function that runs assertions.
    The verifier loads the solution module, extracts the entry-point function,
    and passes it as ``candidate``.

    A few tests call helpers the prompt defines (e.g. ``poly``, ``encode_cyclic``)
    or the entry point by name. The tests run in a namespace built from the
    dataset's own prompt plus the solution's entry point, and nothing else the
    solution defines: its own ``range`` or ``encode_cyclic`` could rig the asserts.
    """
    return f'''#!/usr/bin/env python3
"""DeepGym verifier for {task_id}."""
import sys
import json
from deepgym_solution import load_solution


# HumanEval prompt (trusted helpers and imports) and test code, run in the judge.
_PROMPT = {prompt!r}
_TEST_CODE = {test_code!r}


def verify(solution_path):
    """Load the solution and run HumanEval tests against it."""
    # Load solution module
    # The solution runs in its own isolated process; calls go through a proxy.
    try:
        mod = load_solution(solution_path)
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Import error: {{type(e).__name__}}: {{e}}"}}

    if {entry_point!r} not in dir(mod):
        return {{"score": 0.0, "passed": False, "details": "Missing function: {entry_point}"}}

    # Helpers come from the prompt; only the entry point comes from the solution.
    ns = {{}}
    exec(_PROMPT, ns)

    try:
        # Fetching the entry point can run solution code, so a failure there fails the run.
        ns[{entry_point!r}] = candidate = getattr(mod, {entry_point!r})
        # Execute the test code (defines check(candidate)) in the solution namespace
        exec(_TEST_CODE, ns)
        # Call check with the candidate function
        ns["check"](candidate)
        return {{"score": 1.0, "passed": True, "details": "All tests passed"}}
    except AssertionError as e:
        return {{"score": 0.0, "passed": False, "details": f"Test failed: {{e}}"}}
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Error: {{type(e).__name__}}: {{e}}"}}


if __name__ == "__main__":
    try:
        result = verify(sys.argv[1])
    except Exception as e:
        # The judge failed, not the solution: report a verifier error, never a score.
        print(json.dumps({{"score": 0.0, "passed": False, "details": f"Verifier error: {{type(e).__name__}}: {{e}}", "error_type": "runtime_error"}}))
        sys.exit(2)
    print(json.dumps(result))
'''


def main():
    from datasets import load_dataset

    output_dir = Path.home() / '.deepgym' / 'environments' / 'humaneval'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading HumanEval dataset from HuggingFace...')
    ds = load_dataset('openai/openai_humaneval', split='test')
    print(f'Loaded {len(ds)} problems.')

    registry = []

    for row in ds:
        task_id = row['task_id'].replace('/', '_')
        entry_point = row['entry_point']
        prompt = row['prompt']
        canonical = row['canonical_solution']
        test_code = row['test']

        env_dir = output_dir / task_id
        env_dir.mkdir(exist_ok=True)

        # task.md — the problem prompt
        (env_dir / 'task.md').write_text(prompt, encoding='utf-8')

        # reference_solution.py — prompt + canonical solution (complete runnable file)
        (env_dir / 'reference_solution.py').write_text(
            prompt + canonical, encoding='utf-8'
        )

        # verifier.py
        verifier_src = generate_verifier(task_id, entry_point, prompt, test_code)
        (env_dir / 'verifier.py').write_text(verifier_src, encoding='utf-8')

        # metadata.json
        metadata = {
            'id': task_id,
            'name': entry_point,
            'difficulty': 'medium',
            'domain': 'coding',
            'benchmark': 'humaneval',
            'tags': ['humaneval', 'function-completion'],
        }
        (env_dir / 'metadata.json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
        )

        registry.append({**metadata, 'path': f'environments/humaneval/{task_id}'})

    # registry.json
    (output_dir / 'registry.json').write_text(
        json.dumps(registry, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Generated {len(registry)} HumanEval environments in {output_dir}')


if __name__ == '__main__':
    main()
