"""Verifier for the CLI Task (Git Operations)."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zlib

from deepgym_solution import read_output, read_output_bytes, run_solution

_SHA = re.compile(r'[0-9a-f]{40}')
# The most of one file the judge holds in memory, as deepgym_solution's read_output_bytes:
# an object that inflates past it isn't read at all (a zip bomb would exhaust the judge).
_JUDGE_HOLDS = 64 * 1024 * 1024


class _Repo:
    """A git repository read straight from .git, trusting only what checks out.

    Never runs git: a repo the solution built can set .git/config options that make git
    execute commands. Reads only loose objects, through read_output_bytes (no symlinks,
    nothing outside *root*), and takes an object only if its content hashes to its name
    with the type and length its header declares, so nothing can be forged by writing
    files that merely look like git's.
    """

    def __init__(self, repo_dir, root):
        self.git_dir = os.path.join(repo_dir, '.git')
        self.root = root
        packed = (self._read('packed-refs') or b'').decode('utf-8', 'replace')
        self.packed = {}
        for line in packed.splitlines():
            sha, _, name = line.partition(' ')
            if not line.startswith(('#', '^')):
                self.packed[name.strip()] = sha.strip()

    def _read(self, rel):
        try:
            return read_output_bytes(os.path.join(self.git_dir, rel), self.root)
        except OSError:  # missing, a link, outside root, or more than the judge holds
            return None

    def ref(self, name):
        """The commit id *name* (refs/heads/...) points to, or None."""
        loose = self._read(name)
        sha = (
            loose.decode('ascii', 'replace').strip() if loose is not None else self.packed.get(name)
        )
        return sha if sha and _SHA.fullmatch(sha) else None

    def head(self):
        """The commit id HEAD resolves to, through its branch or detached, or None."""
        head = (self._read('HEAD') or b'').decode('utf-8', 'replace').strip()
        if head.startswith('ref: refs/heads/'):
            return self.ref(head[len('ref: ') :])
        return head if _SHA.fullmatch(head) else None

    def branches(self):
        """Branch name to commit id, loose refs over packed ones."""
        names = {
            name[len('refs/heads/') :] for name in self.packed if name.startswith('refs/heads/')
        }
        heads = os.path.join(self.git_dir, 'refs', 'heads')
        try:
            names |= set() if os.path.islink(heads) else set(os.listdir(heads))
        except OSError:  # missing, or made unreadable by the solution
            pass
        refs = {name: self.ref(f'refs/heads/{name}') for name in names}
        return {name: sha for name, sha in refs.items() if sha}

    def object(self, sha, kind):
        """The body of loose object *sha* if it really is that *kind* of object, else None."""
        if not sha or not _SHA.fullmatch(sha):
            return None
        compressed = self._read(os.path.join('objects', sha[:2], sha[2:]))
        if compressed is None:
            return None
        inflater = zlib.decompressobj()
        try:
            raw = inflater.decompress(compressed, _JUDGE_HOLDS)
        except zlib.error:
            return None
        if inflater.unconsumed_tail or not inflater.eof or inflater.unused_data:
            return None  # more than the judge holds, cut short, or trailing bytes
        if hashlib.sha1(raw).hexdigest() != sha:
            return None
        header, _, body = raw.partition(b'\0')
        return body if header == f'{kind} {len(body)}'.encode() else None

    def commit(self, sha):
        """(tree id, parent ids, message) of commit *sha*, or None unless it is one whose
        tree is a real, well-formed tree."""
        body = self.object(sha, 'commit')
        if body is None:
            return None
        meta, _, message = body.partition(b'\n\n')
        lines = meta.decode('utf-8', 'replace').split('\n')
        tree = lines[0][len('tree ') :]
        if not lines[0].startswith('tree ') or self.entries(tree) is None:
            return None
        parents = [line[len('parent ') :] for line in lines[1:] if line.startswith('parent ')]
        return tree, parents, message.decode('utf-8', 'replace')

    def entries(self, tree):
        """{name: (mode, object id)} of *tree*, or None unless every entry is well formed:
        an octal mode, a name, a NUL and a 20-byte id, as git itself parses it."""
        body = self.object(tree, 'tree')
        if body is None:
            return None
        found = {}
        while body:
            entry, sep, rest = body.partition(b'\0')
            mode, space, name = entry.partition(b' ')
            if not sep or not space or not name or len(rest) < 20 or b'/' in name:
                return None
            if not mode or any(c not in b'01234567' for c in mode) or name in found:
                return None
            found[name] = (mode, rest[:20].hex())
            body = rest[20:]
        return found

    def file_in(self, tree, name):
        """The content of regular file *name* at the top of well-formed *tree*, or None."""
        mode, sha = (self.entries(tree) or {}).get(name.encode(), (None, None))
        return self.object(sha, 'blob') if mode in (b'100644', b'100755') else None

    def commits(self, starts):
        """(message, tree id) of every valid commit reachable from *starts*."""
        found, seen, todo = [], set(), list(starts)
        while todo:
            sha = todo.pop()
            if sha in seen:
                continue
            seen.add(sha)
            commit = self.commit(sha)
            if commit is not None:
                tree, parents, message = commit
                found.append((message, tree))
                todo += parents
        return found


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

    # Check it is a git repo whose HEAD resolves to a real commit, read without running git.
    git_dir = os.path.join(repo_dir, '.git')
    repo = _Repo(repo_dir, work_dir) if os.path.isdir(git_dir) else None
    if repo is not None and os.path.islink(git_dir):
        repo = None
    checks['is_git_repo'] = repo is not None and repo.commit(repo.head()) is not None
    branches = repo.branches() if repo is not None else {}

    # Check 'feature' branch exists and points to a real commit.
    checks['feature_branch_exists'] = (
        repo is not None and repo.commit(branches.get('feature')) is not None
    )

    # Check hello.txt exists with correct content.
    try:
        content = read_output(os.path.join(repo_dir, 'hello.txt'), work_dir).strip()
        checks['hello_file_exists'] = True
        checks['hello_content_correct'] = content == 'Hello, DeepGym!'
    except (OSError, ValueError):  # missing, a symlink, or not UTF-8 text
        checks['hello_file_exists'] = False
        checks['hello_content_correct'] = False

    # Check a real commit has the expected message and commits the file itself.
    checks['has_commit'] = repo is not None and any(
        message.startswith('Add hello.txt')
        and (repo.file_in(tree, 'hello.txt') or b'').strip() == b'Hello, DeepGym!'
        for message, tree in repo.commits(branches.values())
    )

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
