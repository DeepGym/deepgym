"""apply_patch and load_files: repo-patch verifiers patch fixture files and import the result."""

import pytest

from deepgym.core import DeepGym
from deepgym.judge import local_isolation_kind
from deepgym.models import Environment
from deepgym.solution_runtime import apply_patch, load_files

needs_sandbox = pytest.mark.skipif(
    local_isolation_kind() == 'none', reason='no sandbox-exec or bwrap on this machine'
)

_CALC = 'def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n'
_FIX = (
    '--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n'
    ' def add(a, b):\n-    return a - b\n+    return a + b\n'
)


def test_apply_patch_fixes_a_file_and_leaves_the_input_alone() -> None:
    files = {'calc.py': _CALC, 'other.py': 'x = 1\n'}

    patched = apply_patch(_FIX, files)

    assert patched == {'calc.py': _CALC.replace('a - b', 'a + b', 1), 'other.py': 'x = 1\n'}
    assert files['calc.py'] == _CALC


def test_apply_patch_ignores_prose_and_fences_around_the_diff() -> None:
    answer = f'Here is the fix:\n```diff\ndiff --git a/calc.py b/calc.py\n{_FIX}```\nDone.\n'

    patched = apply_patch(answer, {'calc.py': _CALC})['calc.py']

    assert patched.startswith('def add(a, b):\n    return a + b')


def test_apply_patch_handles_two_hunks_insertions_and_stripped_blank_context() -> None:
    patch = (
        '--- a/calc.py\n+++ b/calc.py\n'
        '@@ -2,3 +2,3 @@\n-    return a - b\n+    return a + b\n\n \n'
        '@@ -5,0 +6,1 @@\n+    """Subtract."""\n'
    )

    patched = apply_patch(patch, {'calc.py': _CALC})['calc.py']

    assert patched == (
        'def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    """Subtract."""\n'
        '    return a - b\n'
    )


_NO_EOL = '\\ No newline at end of file\n'


@pytest.mark.parametrize('ending', ['\r\n', '\n'])
def test_apply_patch_keeps_crlf_in_a_crlf_file(ending: str) -> None:
    # Git writes a CRLF file's lines with their \r; a model may drop it. Both keep CRLF.
    patch = '--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n' + f'-old{ending}+new{ending} same{ending}'

    assert apply_patch(patch, {'f.txt': 'old\r\nsame\r\n'})['f.txt'] == 'new\r\nsame\r\n'


@pytest.mark.parametrize(
    ('before', 'hunk', 'after'),
    [
        ('old', f'-old\n{_NO_EOL}+new\n{_NO_EOL}', 'new'),
        ('old', f'-old\n{_NO_EOL}+old\n', 'old\n'),
        ('old\n', f'-old\n+new\n{_NO_EOL}', 'new'),
    ],
)
def test_apply_patch_honors_no_newline_at_end_of_file(before: str, hunk: str, after: str) -> None:
    patch = f'--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n{hunk}'

    assert apply_patch(patch, {'f.txt': before})['f.txt'] == after


@pytest.mark.parametrize(
    ('quoted', 'name'),
    [
        ('caf\\303\\251.py', 'caf\u00e9.py'),  # core.quotePath=true escapes the bytes
        ('caf\u00e9\\".py', 'caf\u00e9".py'),  # quotePath=false leaves them, escapes the quote
    ],
)
def test_apply_patch_reads_git_quoted_file_names(quoted: str, name: str) -> None:
    patch = _FIX.replace('a/calc.py', f'"a/{quoted}"').replace('b/calc.py', f'"b/{quoted}"')

    patched = apply_patch(patch, {name: _CALC})

    assert patched[name].startswith('def add(a, b):\n    return a + b')


def test_apply_patch_keeps_unicode_line_separators_inside_lines() -> None:
    patch = _FIX.replace('return a + b', "return a + b  # '\u2028'")

    patched = apply_patch(patch, {'calc.py': _CALC})['calc.py']

    assert "return a + b  # '\u2028'\n" in patched
    assert apply_patch(patch.replace('\n', '\r\n'), {'calc.py': _CALC})['calc.py'] == patched


@pytest.mark.parametrize(
    ('patch', 'message'),
    [
        ('just prose', 'no unified diff found'),
        (_FIX.replace('calc.py', 'secret.py'), "edits 'secret.py'"),
        (_FIX.replace('a/calc.py', '/dev/null', 1), 'not supported'),
        (_FIX.replace(' def add', ' def plus'), 'does not match line 1'),
        (_FIX.replace('@@ -1,2 +1,2 @@', '@@ -1,3 +1,3 @@'), 'patch ends inside a hunk'),
        (_FIX + '@@ -1,1 +1,1 @@\n-def add(a, b):\n+def add(a, c):\n', 'out of order'),
        ('--- a/calc.py\n+++ b/calc.py\n@@ -999,0 +999,1 @@\n+X = 1\n', 'past the end'),
        (_FIX.replace('a/calc.py', 'a/missing.py'), "renames 'missing.py' to 'calc.py'"),
        ('rename from missing.py\nrename to calc.py\n' + _FIX, 'renaming or copying'),
        ('deleted file mode 100644\n' + _FIX, 'creating, deleting'),
        ('--- a/calc.py\n+++ b/calc.py\nDone.\n', 'no hunk follows the file header'),
        (_FIX + '@@ -1 +1 @\n-old\n+new\n', 'malformed hunk header'),  # git: corrupt
    ],
)
def test_apply_patch_rejects_bad_patches(patch: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_patch(patch, {'calc.py': _CALC})


@pytest.mark.parametrize(
    ('files', 'entry'),
    [({'/abs.py': ''}, '/abs.py'), ({'../up.py': ''}, '../up.py'), ({'a.py': ''}, 'b.py')],
)
def test_load_files_rejects_unsafe_paths_and_a_missing_entry(files: dict, entry: str) -> None:
    with pytest.raises(ValueError):
        load_files(files, entry)


def _run(verify_body: str) -> float:
    """Score a verify() that uses load_files; the solution itself is unused."""
    source = (
        'import os\n'
        'from deepgym_solution import load_files\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        + ''.join(f'    {line}\n' for line in verify_body.strip().splitlines())
    )
    return DeepGym(mode='local').run(Environment(task='t', verifier_code=source), '').score


def test_load_files_imports_siblings_in_the_isolated_process() -> None:
    body = """
main = 'from helper import VALUE\\n\\ndef answer():\\n    return VALUE + 1\\n'
module = load_files({'helper.py': 'VALUE = 41\\n', 'main.py': main}, 'main.py')
return 1.0 if module.answer() == 42 else 0.0
"""
    assert _run(body) == 1.0


def test_load_files_keeps_the_package_of_a_nested_entry() -> None:
    body = """
files = {'pkg/__init__.py': '', 'pkg/helper.py': 'X = 41\\n'}
main = 'from . import helper\\nimport pkg.helper\\n\\n'
files['pkg/main.py'] = main + 'def answer():\\n    return helper.X + 1\\n'
module = load_files(files, 'pkg/main.py')
return 1.0 if module.answer() == 42 else 0.0
"""
    assert _run(body) == 1.0


def test_load_files_imports_the_entry_as_python_would() -> None:
    # pkg/__init__ runs first and the entry sits in sys.modules, as under a normal import.
    body = """
files = {'pkg/__init__.py': 'import builtins\\nbuiltins.ANSWER = 42\\n'}
main = 'import sys\\n\\ndef answer():\\n'
files['pkg/main.py'] = main + '    return ANSWER if sys.modules.get(__name__) else 0\\n'
module = load_files(files, 'pkg/main.py')
return 1.0 if module.answer() == 42 else 0.0
"""
    assert _run(body) == 1.0


def test_load_files_runs_a_package_entry_once() -> None:
    body = """
init = 'import builtins\\nbuiltins.RUNS = getattr(builtins, "RUNS", 0) + 1\\n\\n'
init += 'def runs():\\n    import builtins\\n    return builtins.RUNS\\n'
module = load_files({'pkg/__init__.py': init}, 'pkg/__init__.py')
return 1.0 if module.runs() == 1 else 0.0
"""
    assert _run(body) == 1.0


def test_load_files_refuses_files_named_like_loaded_modules() -> None:
    # main.py would get the runtime's json instead of the json.py next to it.
    body = """
files = {'main.py': 'import json\\n', 'json.py': 'def dumps(value):\\n    return 1\\n'}
try:
    load_files(files, 'main.py')
except ImportError as exc:
    return 1.0 if 'already loaded' in str(exc) else 0.0
return 0.0
"""
    assert _run(body) == 1.0


@needs_sandbox
def test_code_from_load_files_cannot_read_the_judge() -> None:
    body = """
code = 'def peek(path):\\n    try:\\n        return open(path).read()\\n'
code += '    except OSError:\\n        return None\\n'
module = load_files({'main.py': code}, 'main.py')
return 1.0 if module.peek(os.path.abspath(__file__)) is None else 0.0
"""
    assert _run(body) == 1.0
