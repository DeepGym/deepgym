"""Verifier for the CLI Task (Git Operations)."""

import json
import os
import shutil
import subprocess
import sys


def verify(solution_path, test_cases_path=None):
    """Verify the solution creates a git repo with the correct state."""
    repo_dir = '/tmp/my_repo'

    # Clean up any previous run.
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    try:
        proc = subprocess.run(
            [sys.executable, solution_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Solution timed out after 15s',
        }

    checks = {}

    # Check repo directory exists.
    checks['repo_exists'] = os.path.isdir(repo_dir)

    # Check it is a git repo.
    git_dir = os.path.join(repo_dir, '.git')
    checks['is_git_repo'] = os.path.isdir(git_dir)

    # Check 'feature' branch exists.
    if checks['is_git_repo']:
        branch_result = subprocess.run(
            ['git', 'branch', '--list', 'feature'],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        checks['feature_branch_exists'] = 'feature' in branch_result.stdout
    else:
        checks['feature_branch_exists'] = False

    # Check hello.txt exists with correct content.
    hello_path = os.path.join(repo_dir, 'hello.txt')
    if os.path.isfile(hello_path):
        with open(hello_path) as f:
            content = f.read().strip()
        checks['hello_file_exists'] = True
        checks['hello_content_correct'] = content == 'Hello, DeepGym!'
    else:
        checks['hello_file_exists'] = False
        checks['hello_content_correct'] = False

    # Check git log has a commit with expected message.
    if checks['is_git_repo']:
        log_result = subprocess.run(
            ['git', 'log', '--oneline', '--all'],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
        checks['has_commit'] = 'Add hello.txt' in log_result.stdout
    else:
        checks['has_commit'] = False

    # Check solution prints 'done'.
    checks['prints_done'] = 'done' in proc.stdout.strip().lower()

    passed_count = sum(checks.values())
    total = len(checks)
    score = passed_count / total

    details_parts = [f'{k}={v}' for k, v in checks.items()]

    cases = []
    for idx, (check_name, check_passed) in enumerate(checks.items()):
        cases.append(
            {
                'id': f'check_{idx}',
                'input_summary': check_name,
                'passed': bool(check_passed),
                'schema_version': '1.0',
                'score': 1.0 if check_passed else 0.0,
                'cases': [],
                'expected_summary': 'True',
                'actual_summary': str(check_passed),
            }
        )

    return {
        'schema_version': '1.0',
        'score': score,
        # Threshold < 1.0: CLI tasks use discrete filesystem/git checks where
        # minor output differences (e.g. missing "done" print) should not fail
        # an otherwise correct solution.
        'passed': score >= 0.8,
        'details': f'{passed_count}/{total} checks passed. ' + ', '.join(details_parts),
        'cases': cases,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'cases': [],
                    'details': 'Usage: verifier.py <solution_path>',
                }
            )
        )
        sys.exit(1)

    try:
        result = verify(sys.argv[1])
    except Exception as e:
        result = {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Verifier error: {type(e).__name__}: {e}',
        }

    print(json.dumps(result))
