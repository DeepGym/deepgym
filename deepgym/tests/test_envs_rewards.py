"""Built-in env rewards: independent correct solutions score 1.0, plausible wrong ones don't,
and a failing judge reports a verifier error instead of a score."""

import ast
import random
import runpy
import sys
import tempfile
from pathlib import Path

import pytest

from deepgym import solution_runtime
from deepgym.core import DeepGym
from deepgym.models import Environment
from deepgym.registry import load_environment
from deepgym.sandbox import build_run_result, parse_verifier_output

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


# Follows each present node's children; never reads a slot under a missing node.
_BFS = """
def level_order(tree):
    frontier = [0] if tree and tree[0] is not None else []
    levels = []
    while frontier:
        levels.append([tree[i] for i in frontier])
        frontier = [
            child
            for i in frontier
            for child in (2 * i + 1, 2 * i + 2)
            if child < len(tree) and tree[child] is not None
        ]
    return levels
"""

# Misreads the list as LeetCode's format, where a missing node has no child slots.
_LEETCODE_FORMAT = """
def level_order(tree):
    if not tree or tree[0] is None:
        return []
    levels, frontier, i = [], [tree[0]], 1
    while frontier:
        levels.append(frontier)
        nxt = []
        for _ in range(2 * len(frontier)):
            if i < len(tree) and tree[i] is not None:
                nxt.append(tree[i])
            i += 1
        frontier = nxt
    return levels
"""


def test_level_order_scores_a_parent_following_bfs(dg: DeepGym) -> None:
    env = load_environment('level_order_traversal')

    correct = dg.run(env, _BFS)
    wrong = dg.run(env, _LEETCODE_FORMAT)

    assert correct.score == 1.0, correct.output
    assert wrong.passed is False
    assert wrong.score < 1.0


def test_string_manipulation_scores_split_and_slice(dg: DeepGym, string_env: Environment) -> None:
    split_and_slice = "def transform(s):\n    return ' '.join(w[::-1] for w in s.split(' '))\n"
    collapses_spaces = "def transform(s):\n    return ' '.join(w[::-1] for w in s.split())\n"

    correct = dg.run(string_env, split_and_slice)
    wrong = dg.run(string_env, collapses_spaces)

    assert correct.score == 1.0, correct.output
    assert wrong.passed is False
    assert wrong.score < 1.0


_ORGANIZE = """
import os
from pathlib import Path

root = Path(os.environ['PROJECT_DIR'])
for name, readme in [
    ('src', '# Source Code\\nMain application source files.'),
    ('tests', '# Tests\\nUnit and integration tests.'),
    ('docs', '# Documentation\\nProject documentation and guides.'),
]:
    (root / name).mkdir(parents=True)
    (root / name / 'README.md').write_text(readme + '\\n')
print('done')
"""


def test_file_organizer_prompt_names_the_checked_directory(dg: DeepGym) -> None:
    env = load_environment('file_organizer')
    in_cwd = _ORGANIZE.replace("Path(os.environ['PROJECT_DIR'])", "Path('.')")

    correct = dg.run(env, _ORGANIZE)
    wrong = dg.run(env, in_cwd)

    assert 'PROJECT_DIR' in env.task
    assert '/tmp/project' not in env.task
    assert correct.score == 1.0, correct.output
    assert wrong.passed is False


def test_file_organizer_readmes_need_their_descriptions(dg: DeepGym) -> None:
    """Heading-only READMEs used to earn 1.0; the task asks for a description line too."""
    headings_only = _ORGANIZE.replace('write_text(readme', 'write_text(readme.splitlines()[0]')

    result = dg.run(load_environment('file_organizer'), headings_only)

    assert result.passed is False
    assert result.score < 1.0


_FILTER = """
import csv

with open('input.csv', newline='') as handle:
    header, *rows = list(csv.reader(handle))
kept = [row for row in rows if int(row[1]) > 30]
with open('output.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(kept)
print(len(kept))
"""


def test_data_pipeline_rejects_corrupted_records(dg: DeepGym) -> None:
    env = load_environment('data_pipeline')
    corrupted = _FILTER.replace(
        'writer.writerows(kept)', "writer.writerows([name, '999', 'Invented'] for name, *_ in kept)"
    )

    correct = dg.run(env, _FILTER)
    wrong = dg.run(env, corrupted)

    assert correct.score == 1.0, correct.output
    assert correct.passed is True
    assert wrong.passed is False
    assert wrong.score < 1.0


@pytest.mark.parametrize(
    ('env_name', 'target', 'solution'),
    [
        ('cli_task', 'tempfile.mkdtemp', ''),
        (
            'coin_change',
            'random.Random',
            (ENVS_DIR / 'coin_change' / 'reference_solution.py').read_text(encoding='utf-8'),
        ),
    ],
)
def test_judge_failure_is_a_verifier_error(
    env_name: str,
    target: str,
    solution: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A judge crash (here: a full disk, a broken library) exits 2 and never scores the solution."""
    solution_path = tmp_path / 'solution.py'
    solution_path.write_text(solution, encoding='utf-8')

    def judge_breaks(*args: object, **kwargs: object) -> None:
        raise OSError(28, 'No space left on device')

    module, name = target.split('.')
    monkeypatch.setattr({'tempfile': tempfile, 'random': random}[module], name, judge_breaks)
    monkeypatch.setitem(sys.modules, 'deepgym_solution', solution_runtime)
    monkeypatch.setattr(sys, 'argv', ['verifier.py', str(solution_path)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(load_environment(env_name).verifier_path), run_name='__main__')

    result = build_run_result(
        parse_verifier_output(capsys.readouterr().out), 0.0, 'local', exit_code=exit_info.value.code
    )
    assert exit_info.value.code == 2
    assert result.error_type == 'runtime_error'
    assert result.has_verdict is False


_FUNCTION_ENVS = sorted(
    path.parent
    for path in ENVS_DIR.rglob('verifier.py')
    if 'load_solution' in path.read_text(encoding='utf-8') and 'multi_turn' not in path.parts
)


@pytest.mark.parametrize('env_dir', _FUNCTION_ENVS, ids=lambda path: path.name)
def test_unsendable_export_is_a_scored_failure(dg: DeepGym, env_dir: Path) -> None:
    """Looking the entry point up runs solution code: its failure must stay the solution's."""
    reference = ast.parse((env_dir / 'reference_solution.py').read_text(encoding='utf-8'))
    name = next(node.name for node in reference.body if isinstance(node, ast.FunctionDef))
    env = Environment(task='t', verifier_path=env_dir / 'verifier.py')

    result = dg.run(env, f'{name} = []\n{name}.append({name})\n')  # too deep to send

    assert result.has_verdict is True, result.output
    assert result.score == 0.0


def test_solution_import_error_is_still_a_scored_failure(
    dg: DeepGym, sorting_env: Environment
) -> None:
    """The judge-error exit doesn't swallow the solution's own failures: those still score 0."""
    result = dg.run(sorting_env, 'raise ImportError("no such module")\n')

    assert result.score == 0.0
    assert result.has_verdict is True
    assert 'Import error' in result.output


@pytest.mark.parametrize('answer', ['["203.0.113.42"]', '"GET /download?file=x"'])
@pytest.mark.parametrize(
    'env_name', ['cyber/cve_2021_44228_log_triage', 'cyber/web_path_traversal_toy']
)
def test_non_object_json_answer_is_scored_not_a_crash(
    dg: DeepGym, env_name: str, answer: str
) -> None:
    """Valid JSON that isn't an object is the model's mistake: score it, don't crash."""
    result = dg.run(load_environment(env_name), answer)

    assert result.has_verdict is True
    assert result.passed is False


@pytest.mark.parametrize('env_name', ['data_pipeline', 'cyber/web_path_traversal_toy'])
def test_an_empty_answer_earns_nothing(dg: DeepGym, env_name: str) -> None:
    # Both used to pay for an empty answer: a clean exit, or a safe GET by default.
    result = dg.run(load_environment(env_name), '')

    assert result.has_verdict
    assert result.score == 0.0


@pytest.mark.parametrize(
    ('env_name', 'function'),
    [
        ('anagram_check', 'is_anagram'),
        ('palindrome_check', 'is_palindrome'),
        ('valid_parentheses', 'is_valid'),
    ],
)
@pytest.mark.parametrize('constant', ['True', 'False', '0'])
def test_a_constant_answer_to_a_yes_no_task_earns_nothing(
    dg: DeepGym, env_name: str, function: str, constant: str
) -> None:
    # Plain accuracy paid the majority class's share, and 0 == False passed as False.
    answer = f'def {function}(*args):\n    return {constant}\n'

    result = dg.run(load_environment(env_name), answer)

    assert result.has_verdict
    assert result.score == 0.0
