#!/usr/bin/env python3
"""Import MBPP benchmark into DeepGym environment format.

Downloads the 500 MBPP test problems from HuggingFace and generates
a DeepGym-compatible environment directory for each one, including:
  - task.md (the problem description)
  - reference_solution.py (the canonical code)
  - verifier.py (runs the assert-based test cases)
  - metadata.json

Also generates environments/mbpp/registry.json indexing all problems.
"""

import ast
import json
import re
from pathlib import Path


def extract_entry_point(code: str) -> str:
    """Extract the first function name from the reference code."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else 'solution'


def trusted_names(reference_code: str, test_code: str) -> tuple[str, list[str]]:
    """Return (the reference's import statements, the names the tests take from a solution).

    Tests may call any function the reference defines, not only its first one, and a
    few test a function named like a builtin (``sum``). So the names a solution must
    supply are those the tests load that the reference binds at top level; every other
    name, builtins included, resolves to trusted code.
    """
    reference = ast.parse(reference_code)
    imports = '\n'.join(
        ast.get_source_segment(reference_code, node)
        for node in reference.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    defined = set()
    for node in reference.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
    loaded = {
        node.id
        for node in ast.walk(ast.parse(test_code))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return imports, sorted(defined & loaded)


def task_prompt(text: str, reference_code: str, test_code: str) -> str:
    """Return the problem text plus the signatures the tests call, which MBPP's text omits.

    Only names and parameters are shown, never a test or an expected output.
    """
    _, required = trusted_names(reference_code, test_code)
    defs = {
        node.name: node
        for node in ast.parse(reference_code).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    shown = [
        f'class {name}' if isinstance(defs.get(name), ast.ClassDef)
        else f'def {name}({ast.unparse(defs[name].args)})' if name in defs
        else name
        for name in required
    ]
    return f'{text}\n\nYour code must define: ' + ', '.join(f'`{item}`' for item in shown) + '\n'


def generate_verifier(
    task_id: str,
    test_list: list[str],
    challenge_test_list: list[str],
    test_setup_code: str,
    reference_code: str,
) -> str:
    """Generate a DeepGym-compatible verifier from MBPP test assertions.

    MBPP tests are simple assert strings (e.g. ``assert foo(1) == 2``). The
    verifier runs them against the solution's proxy, in a namespace holding only
    the reference's imports and the solution names the tests call: a solution
    that exported its own ``set`` or ``math`` could otherwise rig the asserts.
    """
    # Combine regular and challenge tests
    all_tests = list(test_list)
    if challenge_test_list:
        all_tests.extend(challenge_test_list)

    imports, required = trusted_names(reference_code, '\n'.join([*all_tests, test_setup_code]))

    return f'''#!/usr/bin/env python3
"""DeepGym verifier for MBPP_{task_id}."""
import sys
import json
from deepgym_solution import load_solution


# MBPP test assertions
_TESTS = {all_tests!r}
_TEST_SETUP = {test_setup_code!r}
# The reference solution's imports (trusted) and the names the tests take from the solution.
_IMPORTS = {imports!r}
_REQUIRED = {required!r}


def verify(solution_path):
    """Load the solution and run MBPP test assertions against it."""
    # Load solution module
    # The solution runs in its own isolated process; calls go through a proxy.
    try:
        mod = load_solution(solution_path)
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Import error: {{type(e).__name__}}: {{e}}"}}

    missing = [name for name in _REQUIRED if name not in dir(mod)]
    if missing:
        return {{"score": 0.0, "passed": False, "details": f"Missing function: {{', '.join(missing)}}"}}

    ns = {{}}
    exec(_IMPORTS, ns)

    # Fetching the solution's names can run its code, so a failure there fails the run,
    # as does the optional test setup code.
    try:
        ns.update({{name: getattr(mod, name) for name in _REQUIRED}})
        exec(_TEST_SETUP, ns)
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Setup error: {{type(e).__name__}}: {{e}}"}}

    passed = 0
    total = len(_TESTS)
    details = []

    for i, test_code in enumerate(_TESTS):
        try:
            exec(test_code, ns)
            passed += 1
        except AssertionError as e:
            details.append(f"Test {{i}}: assertion failed: {{e}}")
        except Exception as e:
            details.append(f"Test {{i}}: {{type(e).__name__}}: {{e}}")

    score = passed / total if total > 0 else 0.0
    summary = f"{{passed}}/{{total}} passed"
    if details:
        summary += ". " + "; ".join(details)

    return {{"score": score, "passed": passed == total, "details": summary}}


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

    output_dir = Path.home() / '.deepgym' / 'environments' / 'mbpp'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading MBPP dataset from HuggingFace...')
    ds = load_dataset('google-research-datasets/mbpp', 'full', split='test')
    print(f'Loaded {len(ds)} problems.')

    registry = []

    for row in ds:
        task_id = row['task_id']
        text = row['text']
        code = row['code']
        test_list = row['test_list']
        test_setup_code = row.get('test_setup_code', '')
        challenge_test_list = row.get('challenge_test_list', [])

        entry_point = extract_entry_point(code)
        dir_name = f'MBPP_{task_id}'

        env_dir = output_dir / dir_name
        env_dir.mkdir(exist_ok=True)

        # task.md -- the problem description and the signatures the tests call
        tests = '\n'.join([*test_list, *challenge_test_list, test_setup_code])
        (env_dir / 'task.md').write_text(task_prompt(text, code, tests), encoding='utf-8')

        # reference_solution.py -- the canonical code
        (env_dir / 'reference_solution.py').write_text(code, encoding='utf-8')

        # verifier.py
        verifier_src = generate_verifier(
            task_id, test_list, challenge_test_list, test_setup_code, code
        )
        (env_dir / 'verifier.py').write_text(verifier_src, encoding='utf-8')

        # metadata.json
        metadata = {
            'id': dir_name,
            'name': entry_point,
            'difficulty': 'medium',
            'domain': 'coding',
            'benchmark': 'mbpp',
            'tags': ['mbpp', 'function-completion'],
        }
        (env_dir / 'metadata.json').write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
        )

        registry.append({**metadata, 'path': f'environments/mbpp/{dir_name}'})

    # registry.json
    (output_dir / 'registry.json').write_text(
        json.dumps(registry, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Generated {len(registry)} MBPP environments in {output_dir}')


if __name__ == '__main__':
    main()
