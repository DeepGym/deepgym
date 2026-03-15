"""Verifier for the File Organizer task."""

import json
import os
import subprocess
import sys
import tempfile


def verify(solution_path, test_cases_path=None):
    """Verify the solution creates the correct directory structure."""
    work_dir = tempfile.mkdtemp(prefix='deepgym_file_org_')
    project_dir = os.path.join(work_dir, 'project')

    env = os.environ.copy()
    env['PROJECT_DIR'] = project_dir

    try:
        proc = subprocess.run(
            [sys.executable, solution_path],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
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

    # Check directories exist.
    for dirname in ['src', 'tests', 'docs']:
        dirpath = os.path.join(project_dir, dirname)
        checks[f'{dirname}_dir_exists'] = os.path.isdir(dirpath)

    # Check README files exist and have correct content.
    expected_content = {
        'src/README.md': '# Source Code',
        'tests/README.md': '# Tests',
        'docs/README.md': '# Documentation',
    }

    for filepath, expected_heading in expected_content.items():
        full_path = os.path.join(project_dir, filepath)
        if os.path.isfile(full_path):
            with open(full_path) as f:
                content = f.read()
            checks[f'{filepath}_exists'] = True
            checks[f'{filepath}_content'] = expected_heading in content
        else:
            checks[f'{filepath}_exists'] = False
            checks[f'{filepath}_content'] = False

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
        # Threshold < 1.0: filesystem tasks use discrete checks where minor
        # omissions (e.g. missing "done" print) should not fail an otherwise
        # correct directory-structure solution.
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
