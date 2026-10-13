"""Computer-use env verifiers: correct end state scores 1, shortcuts score 0.

Runs each verifier locally against hand-built end states, so no Daytona needed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym import ComputerUseEnvironment, load_environment, load_suite

ENV_NAMES = ['signup_form', 'todo_list', 'settings_tabs', 'expense_sheet', 'file_manager']


def _items(*pairs: tuple[str, bool]) -> dict:
    return {'items': [{'text': text, 'done': done} for text, done in pairs]}


def _rows(*pairs: tuple[str, str]) -> dict:
    return {'saved': True, 'rows': [{'name': name, 'amount': amount} for name, amount in pairs]}


# (env, end state, expected score). None state means the app never saved.
STATE_CASES = [
    ('signup_form', None, 0.0),
    (
        'signup_form',
        {'submitted': True, 'name': '', 'email': '', 'plan': 'free', 'agreed': False},
        0.0,
    ),
    (
        'signup_form',
        {
            'submitted': True,
            'name': 'Ada Lovelace',
            'email': 'ada@example.com',
            'plan': 'pro',
            'agreed': True,
        },
        0.75,
    ),
    (
        'signup_form',
        {
            'submitted': True,
            'name': ' Ada Lovelace',
            'email': 'ADA@example.com',
            'plan': 'team',
            'agreed': True,
        },
        1.0,
    ),
    ('todo_list', None, 0.0),
    ('todo_list', _items(('Buy milk', False), ('Call mom', False), ('Write report', False)), 0.0),
    ('todo_list', _items(('Call mom', True), ('Book flights', False), ('Pay rent', False)), 0.0),
    (
        'todo_list',
        _items(
            ('Call mom', True),
            ('Write report', False),
            ('Book flights', False),
            ('Pay rent', False),
        ),
        1.0,
    ),
    ('settings_tabs', None, 0.0),
    (
        'settings_tabs',
        {
            'saved': True,
            'display_name': 'ghopper',
            'email_notifications': True,
            'push_notifications': True,
            'weekly_digest': False,
            'theme': 'light',
        },
        0.0,
    ),
    (
        'settings_tabs',
        {
            'saved': True,
            'display_name': 'Grace Hopper',
            'email_notifications': False,
            'push_notifications': False,
            'weekly_digest': False,
            'theme': 'dark',
        },
        0.0,
    ),
    (
        'settings_tabs',
        {
            'saved': True,
            'display_name': 'Grace Hopper',
            'email_notifications': False,
            'push_notifications': True,
            'weekly_digest': False,
            'theme': 'dark',
        },
        1.0,
    ),
    ('expense_sheet', None, 0.0),
    ('expense_sheet', _rows(('Rent', '1200'), ('Grocries', '300'), ('Internet', '60')), 0.0),
    (
        'expense_sheet',
        _rows(('Rent', '1200'), ('Groceries', '0'), ('Internet', '75'), ('Gym', '45')),
        0.0,
    ),
    (
        'expense_sheet',
        _rows(('Rent', '1200'), ('Groceries', '300'), ('Internet', '75.00'), ('Gym', '$45')),
        1.0,
    ),
]

_FILES = {
    'report.pdf': 'Q3 revenue summary\n',
    'notes.txt': 'milk, eggs, bread\n',
    'photo.jpg': 'not really a photo\n',
}

# (description, files under ~/Documents after the episode, expected score)
FILE_CASES = [
    ('untouched', dict(_FILES), 0.0),
    (
        'deleted instead of moved',
        {'photo.jpg': _FILES['photo.jpg'], 'todo.txt': _FILES['notes.txt']},
        0.0,
    ),
    (
        'everything archived',
        {
            'Archive/report.pdf': _FILES['report.pdf'],
            'Archive/photo.jpg': _FILES['photo.jpg'],
            'todo.txt': _FILES['notes.txt'],
        },
        0.0,
    ),
    (
        'copied, not moved',
        {**_FILES, 'Archive/report.pdf': _FILES['report.pdf']},
        1 / 3,
    ),
    (
        'done',
        {
            'Archive/report.pdf': _FILES['report.pdf'],
            'todo.txt': _FILES['notes.txt'],
            'photo.jpg': _FILES['photo.jpg'],
        },
        1.0,
    ),
]


def _run_verifier(env: ComputerUseEnvironment, work: Path, home: Path) -> dict:
    """Run *env*'s verifier the way Session.score does, from the work directory."""
    for rel, text in env.files.items():
        if rel.startswith('_shared/'):
            (work / rel).parent.mkdir(parents=True, exist_ok=True)
            (work / rel).write_text(text)
    (work / '_final_verifier.py').write_text(env.final_verifier_code)
    proc = subprocess.run(
        [sys.executable, '-I', '_final_verifier.py'],
        capture_output=True,
        text=True,
        cwd=work,
        # The verifier's HOME is its own; the agent's home comes by name, as in a session.
        env={'HOME': str(work), 'DEEPGYM_AGENT_HOME': str(home)},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('name', ENV_NAMES)
def test_env_loads_as_computer_use(name: str) -> None:
    env = load_environment(name)
    assert isinstance(env, ComputerUseEnvironment)
    assert env.setup_script
    assert '_shared/state_server.py' in env.files
    assert 'verifier.py' not in env.files
    assert not any('checks' in rel for rel in env.files)
    assert 'def report(' in env.final_verifier_code


@pytest.mark.parametrize('name', ENV_NAMES)
def test_setup_needs_no_root(name: str) -> None:
    """Setup runs as the unprivileged agent; the desktop is locked down before it runs."""
    env = load_environment(name)
    scripts = [env.setup_script, *(t for rel, t in env.files.items() if rel.endswith('.sh'))]
    assert not any('sudo' in script for script in scripts)


def test_computer_use_suite_is_separate() -> None:
    assert sorted(env.name for env in load_suite('computer-use')) == sorted(ENV_NAMES)
    assert not any(isinstance(env, ComputerUseEnvironment) for env in load_suite('all'))


@pytest.mark.parametrize(('name', 'state', 'expected'), STATE_CASES)
def test_browser_verifier(name: str, state: dict | None, expected: float, tmp_path: Path) -> None:
    if state is not None:
        (tmp_path / 'state.json').write_text(json.dumps(state))
    result = _run_verifier(load_environment(name), tmp_path, tmp_path)
    assert result['score'] == pytest.approx(expected)
    assert result['passed'] is (expected == 1.0)


@pytest.mark.parametrize(('label', 'files', 'expected'), FILE_CASES, ids=[c[0] for c in FILE_CASES])
def test_file_manager_verifier(label: str, files: dict, expected: float, tmp_path: Path) -> None:
    home = tmp_path / 'home'
    for rel, text in files.items():
        target = home / 'Documents' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    work = tmp_path / 'work'
    work.mkdir()
    result = _run_verifier(load_environment('file_manager'), work, home)
    assert result['score'] == pytest.approx(expected)


# --- Every variant: the solved state scores 1, saving untouched state scores 0. ---

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs' / 'computer_use'
INITIAL_TODOS = ['Buy milk', 'Call mom', 'Write report']
INITIAL_SETTINGS = {
    'saved': True,
    'display_name': 'ghopper',
    'email_notifications': True,
    'push_notifications': True,
    'weekly_digest': False,
    'theme': 'light',
}
UNTOUCHED = {
    'signup_form': {'submitted': True, 'name': '', 'email': '', 'plan': '', 'agreed': False},
    'todo_list': _items(*[(text, False) for text in INITIAL_TODOS]),
    'settings_tabs': INITIAL_SETTINGS,
    'expense_sheet': _rows(('Rent', '1200'), ('Grocries', '300'), ('Internet', '60')),
}


def _variants(name: str) -> list[dict]:
    return json.loads((ENVS_DIR / name / 'variants.json').read_text())


def _solved_state(name: str, p: dict) -> dict:
    if name == 'signup_form':
        return {
            'submitted': True,
            'name': p['name'],
            'email': p['email'],
            'plan': p['plan'].lower(),
            'agreed': True,
        }
    if name == 'todo_list':
        kept = [(text, text == p['complete']) for text in INITIAL_TODOS if text != p['delete']]
        return _items(*kept, (p['add1'], False), (p['add2'], False))
    if name == 'settings_tabs':
        state = {**INITIAL_SETTINGS, 'display_name': p['display_name'], 'theme': p['theme']}
        state[p['toggle']] = not state[p['toggle']]
        return state
    return _rows(
        ('Rent', '1200'),
        ('Groceries', '300'),
        ('Internet', p['internet']),
        (p['new_item'], p['new_amount']),
    )


BROWSER_VARIANTS = [(name, i) for name in UNTOUCHED for i in range(len(_variants(name)))]


@pytest.mark.parametrize(('name', 'variant'), BROWSER_VARIANTS)
def test_every_browser_variant(name: str, variant: int, tmp_path: Path) -> None:
    env = load_environment(name, variant=variant)
    assert '$' not in env.task
    for state, expected in (
        (_solved_state(name, _variants(name)[variant]), 1.0),
        (UNTOUCHED[name], 0.0),
    ):
        (tmp_path / 'state.json').write_text(json.dumps(state))
        assert _run_verifier(env, tmp_path, tmp_path)['score'] == expected


@pytest.mark.parametrize('variant', range(len(_variants('file_manager'))))
def test_every_file_manager_variant(variant: int, tmp_path: Path) -> None:
    p = _variants('file_manager')[variant]
    solved = {
        f'{p["folder"]}/{p["move"]}': _FILES[p['move']],
        p['rename_to']: _FILES[p['rename_from']],
        p['keep']: _FILES[p['keep']],
    }
    env = load_environment('file_manager', variant=variant)
    for label, files, expected in (('solved', solved, 1.0), ('untouched', _FILES, 0.0)):
        home = tmp_path / label
        for rel, text in files.items():
            (home / 'Documents' / rel).parent.mkdir(parents=True, exist_ok=True)
            (home / 'Documents' / rel).write_text(text)
        (home / 'work').mkdir()
        assert _run_verifier(env, home / 'work', home)['score'] == expected, label


def test_duplicate_rows_cannot_satisfy_guards(tmp_path: Path) -> None:
    rows = _rows(
        ('Rent', '999'), ('Rent', '1200'), ('Groceries', '300'), ('Internet', '75'), ('Gym', '45')
    )
    (tmp_path / 'state.json').write_text(json.dumps(rows))
    assert _run_verifier(load_environment('expense_sheet'), tmp_path, tmp_path)['score'] == 0.0
