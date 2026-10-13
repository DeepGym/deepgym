"""Verifier for the CLI Task (Git Operations)."""

import json
import os
import re
import subprocess
import sys
import tempfile
import zlib

from deepgym_solution import read_output, read_output_bytes, run_solution


def _read_repo(repo_dir, root):
    """Return (branch names, commit messages) read straight from .git.

    Never runs git: a repo the solution built can set .git/config options that
    make git execute commands. Reads only loose objects, through read_output_bytes
    (no symlinks, nothing outside *root*).
    """
    git_dir = os.path.join(repo_dir, '.git')

    def read(rel):
        try:
            return read_output_bytes(os.path.join(git_dir, rel), root, limit=10_000_000)
        except OSError:
            return None

    branches = {}
    packed = read('packed-refs') or b''
    for line in packed.decode('utf-8', 'replace').splitlines():
        sha, _, name = line.partition(' ')
        if name.startswith('refs/heads/'):
            branches[name[len('refs/heads/') :]] = sha.strip()
    heads = os.path.join(git_dir, 'refs', 'heads')
    try:
        names = [] if os.path.islink(heads) else os.listdir(heads)
    except OSError:  # missing, or made unreadable by the solution
        names = []
    for name in names:
        sha = read(os.path.join('refs', 'heads', name))
        if sha is not None:
            branches[name] = sha.decode('ascii', 'replace').strip()

    messages, seen, todo = [], set(), list(branches.values())
    while todo and len(seen) < 1000:
        sha = todo.pop()
        if sha in seen or not re.fullmatch(r'[0-9a-f]{40}', sha):
            continue
        seen.add(sha)
        compressed = read(os.path.join('objects', sha[:2], sha[2:]))
        if compressed is None:
            continue
        try:
            raw = zlib.decompressobj().decompress(compressed, 1_000_000)
        except zlib.error:
            continue
        header, _, body = raw.partition(b'\0')
        if not header.startswith(b'commit '):
            continue
        meta, _, message = body.decode('utf-8', 'replace').partition('\n\n')
        messages.append(message.strip())
        todo += [line[len('parent ') :] for line in meta.splitlines() if line.startswith('parent ')]
    return set(branches), messages


def verify(solution_path, test_cases_path=None):
    """Verify the solution creates a git repo with the correct state."""
    work_dir = tempfile.mkdtemp(prefix='deepgym_cli_task_')
    repo_dir = os.path.join(work_dir, 'my_repo')

    try:
        proc = run_solution(solution_path, cwd=work_dir, timeout=15, allow_subprocess=True)
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

    # Check it is a git repo, and read its branches and commits without running git.
    git_dir = os.path.join(repo_dir, '.git')
    checks['is_git_repo'] = os.path.isdir(git_dir) and not os.path.islink(git_dir)
    branches, messages = _read_repo(repo_dir, work_dir) if checks['is_git_repo'] else (set(), [])

    # Check 'feature' branch exists.
    checks['feature_branch_exists'] = 'feature' in branches

    # Check hello.txt exists with correct content.
    try:
        content = read_output(os.path.join(repo_dir, 'hello.txt'), work_dir).strip()
        checks['hello_file_exists'] = True
        checks['hello_content_correct'] = content == 'Hello, DeepGym!'
    except (OSError, ValueError):  # missing, a symlink, or not UTF-8 text
        checks['hello_file_exists'] = False
        checks['hello_content_correct'] = False

    # Check a commit has the expected message.
    checks['has_commit'] = any(m.startswith('Add hello.txt') for m in messages)

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
