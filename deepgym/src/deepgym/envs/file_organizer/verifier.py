"""Verifier for the File Organizer task."""

import json
import os
import subprocess
import sys
import tempfile

from deepgym_solution import read_output, run_solution


def verify(solution_path, test_cases_path=None):
    """Verify the solution creates the correct directory structure."""
    work_dir = tempfile.mkdtemp(prefix='deepgym_file_org_')
    project_dir = os.path.join(work_dir, 'project')

    try:
        proc = run_solution(
            solution_path, cwd=work_dir, env={'PROJECT_DIR': project_dir}, timeout=15
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

    # Check README files exist and hold the requested lines (blank lines and
    # surrounding whitespace aside).
    expected_content = {
        'src/README.md': ['# Source Code', 'Main application source files.'],
        'tests/README.md': ['# Tests', 'Unit and integration tests.'],
        'docs/README.md': ['# Documentation', 'Project documentation and guides.'],
    }

    for filepath, expected_lines in expected_content.items():
        try:
            content = read_output(os.path.join(project_dir, filepath), work_dir)
            checks[f'{filepath}_exists'] = True
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            checks[f'{filepath}_content'] = lines == expected_lines
        except (OSError, ValueError):  # missing, a symlink, or not UTF-8 text
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
    try:
        result = verify(sys.argv[1])
    except Exception as e:
        # The judge failed, not the solution: report a verifier error, never a score.
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'details': f'Verifier error: {type(e).__name__}: {e}',
                    'error_type': 'runtime_error',
                }
            )
        )
        sys.exit(2)
    print(json.dumps(result))
