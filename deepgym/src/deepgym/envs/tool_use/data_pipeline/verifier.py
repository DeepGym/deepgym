"""Verifier for the Data Pipeline task."""

import csv
import io
import json
import os
import subprocess
import sys
import tempfile

from deepgym_solution import read_output, run_solution

INPUT_CSV = """name,age,city
Alice,25,New York
Bob,35,San Francisco
Charlie,40,Chicago
Diana,28,Boston
Eve,45,Seattle
Frank,22,Denver
Grace,33,Austin
Henry,50,Portland
Ivy,29,Miami
Jack,38,Dallas
"""
FIELDS = ('name', 'age', 'city')


def _records(rows):
    """Return CSV rows as sorted (name, age, city) tuples, whitespace-trimmed."""
    return sorted(tuple((row.get(field) or '').strip() for field in FIELDS) for row in rows)


def verify(solution_path, test_cases_path=None):
    """Verify the solution filters CSV rows correctly."""
    reader = csv.DictReader(io.StringIO(INPUT_CSV))
    expected = _records(row for row in reader if int(row['age']) > 30)

    # Create a temp directory with input.csv.
    work_dir = tempfile.mkdtemp(prefix='deepgym_data_pipeline_')
    input_path = os.path.join(work_dir, 'input.csv')
    output_path = os.path.join(work_dir, 'output.csv')

    with open(input_path, 'w') as f:
        f.write(INPUT_CSV.strip() + '\n')

    try:
        proc = run_solution(solution_path, cwd=work_dir, timeout=15)
    except subprocess.TimeoutExpired:
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Solution timed out after 15s',
        }

    checks = {}

    # Check script ran without error.
    checks['no_error'] = proc.returncode == 0

    # Check output.csv exists.
    try:
        output_text = read_output(output_path, work_dir)
        checks['output_exists'] = True
    except (OSError, ValueError):  # missing, a symlink, or not UTF-8 text
        checks['output_exists'] = False

    try:
        rows = list(csv.DictReader(io.StringIO(output_text))) if checks['output_exists'] else []
    except csv.Error:
        rows = []
    records = _records(rows)
    checks['correct_count'] = len(records) == len(expected)
    checks['correct_names'] = {name for name, _, _ in records} == {name for name, _, _ in expected}
    # Every field of every row: altered ages or cities are not the filtered data.
    checks['correct_records'] = records == expected
    checks['has_headers'] = bool(rows) and set(rows[0]) >= set(FIELDS)

    # Check stdout reports the count.
    checks['prints_count'] = '6' in proc.stdout.strip()

    passed_count = sum(checks.values())
    total = len(checks)
    score = passed_count / total

    details_parts = [f'{k}={v}' for k, v in checks.items()]
    if proc.stderr.strip():
        details_parts.append(f'stderr={proc.stderr.strip()[:200]}')

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
        # Threshold < 1.0: a minor miss (e.g. the count print) doesn't fail an
        # otherwise correct solution, but the written rows must be exactly right.
        'passed': checks['correct_records'] and score >= 0.8,
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
