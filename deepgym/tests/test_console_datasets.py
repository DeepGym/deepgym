"""Tests for the console's dataset API: build into the datasets directory, list, verify."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console import record_episode
from deepgym.console.app import create_app
from deepgym.console.datasets import DEFAULT_DATASETS_DIR
from deepgym.console.db import connect
from deepgym.console.label_store import LabelStore
from deepgym.console.schemas import LabelQuery
from deepgym.dataset import DatasetSpec, build_dataset, manifest_sha256
from deepgym.dataset import manifest as manifest_module
from deepgym.dataset.manifest import MAX_FOLDER_BYTES
from deepgym.labels import LabelCreate
from deepgym.traces import Trace
from tests.console_client import make_client
from tests.test_dataset import _trace, _unjudged

SPEC = {'test_ratio': 0.0, 'val_ratio': 0.0}
BUILD = {'spec': SPEC, 'name': 'notes', 'version': '1'}


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / 'data' / 'datasets'


@pytest.fixture()
def client(tmp_path: Path, root: Path) -> TestClient:
    return make_client(tmp_path, datasets_dir=root)


def _traces() -> list[Trace]:
    return [_trace('t1', 'a'), _trace('t2', 'b', passed=False), _unjudged('t3', 'c')]


def _record(db: Path, traces: list[Trace]) -> None:
    for trace in traces:
        record_episode(trace, db_path=db)


def _reversed(db: Path) -> None:
    """An expert passes t2, which the verifier failed."""
    LabelStore(db).add(
        LabelCreate.model_validate({'target': {'kind': 'trace', 'id': 't2'}, 'name': 'passed',
                                    'value': True, 'source': 'human', 'author': 'ana',
                                    'comment': 'the verifier missed it'})
    )  # fmt: skip


def _files(path: Path) -> set[Path]:
    return set(path.rglob('*'))


def test_build_list_and_verify(client: TestClient, db: Path, root: Path) -> None:
    traces = _traces()
    _record(db, traces)
    _reversed(db)

    response = client.post('/api/datasets/build', json=BUILD)

    assert response.status_code == 201
    built = response.json()
    folder = root / 'notes-1'
    assert built['dataset'] == 'notes-1'
    assert built['sha256'] == manifest_sha256(folder)
    labels = LabelStore(db).page(LabelQuery(), 0, None).labels
    expected = build_dataset(
        traces, DatasetSpec.model_validate(SPEC), labels, None,
        {'episode_without_trace': 0, 'merged_copy': 0, 'conflicting_copy': 0,
         'unreadable_trace': 0},
    )  # fmt: skip
    assert built['counts'] == expected.counts.model_dump(mode='json')
    assert (built['counts']['kept'], built['counts']['overridden']) == (2, 1)
    assert built['counts']['dropped'] == {'no_verdict': 1}

    listed = client.get('/api/datasets').json()
    assert listed['total'] == 1
    (entry,) = listed['datasets']
    assert (entry['dataset'], entry['sha256'], entry['error']) == ('notes-1', built['sha256'], None)
    assert (entry['manifest']['name'], entry['manifest']['version']) == ('notes', '1')

    verified = client.post(
        '/api/datasets/verify', json={'dataset': 'notes-1', 'sha256': built['sha256'].upper()}
    ).json()
    assert (verified['valid'], verified['problems'], verified['sha256']) == (
        True,
        [],
        built['sha256'],
    )
    assert verified['manifest'] == entry['manifest']


def test_build_counts_episode_labels_and_unread_traces(
    client: TestClient, db: Path, tmp_path: Path
) -> None:
    trace = _trace('t1', 'a')
    _record(db, [trace.model_copy(update={'result': None, 'finished_at': None}), trace])
    episode = client.get('/api/episodes').json()['episodes'][0]['id']
    LabelStore(db).add(
        LabelCreate.model_validate({'target': {'kind': 'episode', 'id': episode},
                                    'type': 'invalidate', 'source': 'human', 'author': 'ana',
                                    'comment': 'sandbox was broken'})
    )  # fmt: skip
    _record(db, [_trace('t2', 'b')])

    counts = client.post('/api/datasets/build', json=BUILD).json()['counts']

    assert counts['unread'] == {
        'episode_without_trace': 0,
        'merged_copy': 1,
        'conflicting_copy': 0,
        'unreadable_trace': 0,
    }
    assert counts['dropped'] == {'invalidated': 1}
    assert counts['kept'] == 1


def test_verify_finds_tampering(client: TestClient, db: Path, root: Path) -> None:
    _record(db, _traces())
    built = client.post('/api/datasets/build', json=BUILD).json()
    split = root / 'notes-1' / 'train.jsonl'
    split.write_bytes(split.read_bytes() + b'\n')

    tampered = client.post('/api/datasets/verify', json={'dataset': 'notes-1'}).json()
    unpublished = client.post(
        '/api/datasets/verify', json={'dataset': 'notes-1', 'sha256': 'ab' * 32}
    ).json()

    assert (tampered['valid'], tampered['manifest']) == (False, None)
    assert tampered['sha256'] == built['sha256']
    assert any('train.jsonl' in problem for problem in tampered['problems'])
    assert all('\n' not in problem for problem in tampered['problems'])
    assert unpublished['valid'] is False
    assert any(f'not the published {"ab" * 32}' in p for p in unpublished['problems'])


def test_verify_reports_a_missing_or_unreadable_manifest(client: TestClient, root: Path) -> None:
    (root / 'empty-1').mkdir(parents=True)
    (root / 'bad-1').mkdir()
    (root / 'bad-1' / 'manifest.json').write_text('{"name": "bad"}\n', encoding='utf-8')

    empty = client.post('/api/datasets/verify', json={'dataset': 'empty-1'}).json()
    bad = client.post('/api/datasets/verify', json={'dataset': 'bad-1'}).json()

    assert (empty['valid'], empty['sha256'], empty['problems']) == (
        False,
        None,
        ['empty-1/manifest.json not found'],
    )
    assert bad['valid'] is False
    assert bad['sha256'] == manifest_sha256(root / 'bad-1')
    assert 'manifest.json is not a dataset manifest: version: Field required' in bad['problems']
    listed = {entry['dataset']: entry for entry in client.get('/api/datasets').json()['datasets']}
    assert listed['bad-1']['error'] == (
        'manifest.json is not a dataset manifest (12 problems); verify the dataset to list them'
    )


def test_a_symlinked_manifest_is_never_read(client: TestClient, db: Path, root: Path) -> None:
    _record(db, _traces())
    client.post('/api/datasets/build', json=BUILD)
    (root / 'linked-2').mkdir()
    (root / 'linked-2' / 'manifest.json').symlink_to(root / 'notes-1' / 'manifest.json')

    verified = client.post('/api/datasets/verify', json={'dataset': 'linked-2'}).json()
    listed = {entry['dataset']: entry for entry in client.get('/api/datasets').json()['datasets']}

    expected = 'linked-2/manifest.json is not a regular file (a symlink is never followed)'
    assert (verified['valid'], verified['sha256'], verified['problems']) == (
        False,
        None,
        [expected],
    )
    assert (listed['linked-2']['manifest'], listed['linked-2']['error']) == (None, expected)


def test_unreadable_folders_are_reported_one_by_one(
    client: TestClient, db: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(db, _traces())
    client.post('/api/datasets/build', json=BUILD)
    client.post('/api/datasets/build', json={**BUILD, 'version': '2'})
    real = manifest_sha256

    def locked(folder: Path) -> str:
        if folder.name == 'notes-2':
            raise PermissionError(13, 'Permission denied', str(folder / 'manifest.json'))
        return real(folder)

    monkeypatch.setattr('deepgym.console.datasets.manifest_sha256', locked)

    listed = client.get('/api/datasets').json()['datasets']
    verified = client.post('/api/datasets/verify', json={'dataset': 'notes-2'}).json()

    assert [entry['error'] for entry in listed] == [
        None,
        "Can't read manifest.json: Permission denied",
    ]
    assert (verified['valid'], verified['problems']) == (
        False,
        ["Can't read the dataset: Permission denied"],
    )


@pytest.mark.parametrize(
    'changes',
    [
        {'name': '../escape'},
        {'name': 'a/b'},
        {'version': '1/../../escape'},
        {'version': '..\\escape'},
        {'name': '.hidden'},
        {'name': '/abs'},
        {'name': ' padded'},
        {'name': 'c:notes'},
        {'name': 'a' * (MAX_FOLDER_BYTES - 1)},
        {'name': '\u00e9' * 110},
        {'formats': []},
        {'formats': ['csv']},
        {'spec': {'test_ratio': 0.6, 'val_ratio': 0.6}},
        {'extra': 1},
    ],
)
def test_build_refuses_bad_input_and_writes_nothing(
    client: TestClient, db: Path, tmp_path: Path, changes: dict
) -> None:
    _record(db, _traces())
    before = _files(tmp_path)

    response = client.post('/api/datasets/build', json={**BUILD, **changes})

    assert response.status_code == 422
    assert _files(tmp_path) == before


def test_build_refuses_a_used_folder_and_no_traces(
    client: TestClient, db: Path, tmp_path: Path
) -> None:
    empty = client.post('/api/datasets/build', json=BUILD)
    _record(db, _traces())
    first = client.post('/api/datasets/build', json=BUILD)
    again = client.post('/api/datasets/build', json=BUILD)

    assert (empty.status_code, empty.json()['detail']) == (409, 'The console holds no traces')
    assert first.status_code == 201
    assert (again.status_code, again.json()['detail']) == (
        409,
        'notes-1 is not a new or empty directory',
    )
    assert str(tmp_path) not in again.text


def test_the_longest_folder_name_builds(client: TestClient, db: Path, root: Path) -> None:
    _record(db, _traces())
    name = 'a' * (MAX_FOLDER_BYTES - 2)

    response = client.post('/api/datasets/build', json={**BUILD, 'name': name})

    assert response.status_code == 201
    assert len(response.json()['dataset'].encode()) == MAX_FOLDER_BYTES
    assert (root / f'{name}-1' / 'manifest.json').is_file()


def test_build_skips_and_counts_traces_that_do_not_read(
    client: TestClient, db: Path, root: Path
) -> None:
    trace = _trace('t1')
    _record(db, [trace, trace.model_copy(update={'task': 'Another task'})])
    _record(db, [_trace('Ab', 'b'), _trace('t2', 'd')])
    broken = record_episode(_trace('ab', 'c'), db_path=db)
    with connect(db) as conn:
        conn.execute("UPDATE episodes SET trace = '{' WHERE id = ?", (broken,))

    response = client.post('/api/datasets/build', json=BUILD)

    assert response.status_code == 201
    built = response.json()
    assert built['counts']['unread'] == {
        'episode_without_trace': 0,
        'merged_copy': 0,
        'conflicting_copy': 2,
        'unreadable_trace': 1,
    }
    assert [conflict['trace_id'] for conflict in built['conflicts']] == ['t1']
    assert built['counts']['kept'] == 2  # Ab collides with no trace that reads
    assert built['unreadable'] == [broken]


def test_build_reports_an_only_conflicting_console(client: TestClient, db: Path) -> None:
    trace = _trace('t1')
    _record(db, [trace, trace.model_copy(update={'task': 'Another task'})])

    response = client.post('/api/datasets/build', json=BUILD)

    assert (response.status_code, response.json()['detail']) == (
        409,
        'The console holds no traces that read as one (1 left out)',
    )


def test_build_into_a_datasets_directory_that_cannot_be_written(tmp_path: Path, db: Path) -> None:
    blocked = tmp_path / 'blocked'
    blocked.write_text('a file, not a folder', encoding='utf-8')
    client = make_client(tmp_path, datasets_dir=blocked)
    _record(db, _traces())

    response = client.post('/api/datasets/build', json=BUILD)

    assert response.status_code == 409
    assert response.json()['detail'].startswith("Can't write notes-1 in the datasets directory: ")
    assert str(tmp_path) not in response.text


def test_a_build_that_loses_the_race_for_its_folder_is_a_conflict(
    client: TestClient, db: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record(db, _traces())
    real = manifest_module.verify_dataset

    def racing(folder: Path, *, sha256: str | None = None) -> object:
        (root / 'notes-1').mkdir(parents=True, exist_ok=True)
        (root / 'notes-1' / 'other.txt').write_text('another build', encoding='utf-8')
        return real(folder, sha256=sha256)

    monkeypatch.setattr(manifest_module, 'verify_dataset', racing)

    response = client.post('/api/datasets/build', json=BUILD)

    assert (response.status_code, response.json()['detail']) == (
        409,
        'notes-1 was created by another build while this one ran; nothing was written',
    )
    assert sorted(path.name for path in root.iterdir()) == ['notes-1']


def test_build_refuses_a_split_with_no_train(client: TestClient, db: Path, root: Path) -> None:
    _record(db, _traces())

    response = client.post(
        '/api/datasets/build', json={**BUILD, 'spec': {'test_ratio': 0.99, 'val_ratio': 0.0}}
    )

    assert response.status_code == 409
    assert 'No trace landed in train' in response.json()['detail']
    assert not root.exists() or not any(root.iterdir())


@pytest.mark.parametrize(
    ('body', 'code'),
    [
        ({'dataset': '../console.db'}, 422),
        ({'dataset': '..'}, 422),
        ({'dataset': '.'}, 422),
        ({'dataset': '.staging'}, 422),
        ({'dataset': 'a/b'}, 422),
        ({'dataset': '/etc'}, 422),
        ({'dataset': 'notes-1', 'sha256': 'xyz'}, 422),
        ({'dataset': 'notes-1', 'folder': '/tmp'}, 422),
        ({'dataset': 'missing-1'}, 404),
        ({'dataset': 'linked-1'}, 404),
        ({'dataset': 'file-1'}, 404),
    ],
)
def test_verify_takes_only_a_folder_of_the_datasets_directory(
    client: TestClient, db: Path, root: Path, tmp_path: Path, body: dict, code: int
) -> None:
    _record(db, _traces())
    client.post('/api/datasets/build', json=BUILD)
    (root / 'linked-1').symlink_to(root / 'notes-1')
    (root / 'file-1').write_text('x', encoding='utf-8')

    assert client.post('/api/datasets/verify', json=body).status_code == code


def test_list_pages_folders_by_name(client: TestClient, db: Path, root: Path) -> None:
    _record(db, _traces())
    for version in ('2', '1'):
        client.post('/api/datasets/build', json={**BUILD, 'version': version})
    (root / 'loose-1').mkdir()
    (root / '.notes-3-staging').mkdir()
    (root / 'readme.txt').write_text('x', encoding='utf-8')
    (root / 'linked-1').symlink_to(root / 'notes-1')

    every = client.get('/api/datasets').json()
    page = client.get('/api/datasets', params={'offset': 1, 'limit': 1}).json()

    assert [entry['dataset'] for entry in every['datasets']] == ['loose-1', 'notes-1', 'notes-2']
    assert every['total'] == 3
    assert every['datasets'][0] == {
        'dataset': 'loose-1',
        'sha256': None,
        'manifest': None,
        'error': 'loose-1/manifest.json not found',
    }
    assert ([entry['dataset'] for entry in page['datasets']], page['total']) == (['notes-1'], 3)


@pytest.mark.parametrize(
    'params', [{'name': 'notes'}, [('limit', '1'), ('limit', '2')], {'limit': 0}]
)
def test_dataset_queries_are_strict(client: TestClient, params: object) -> None:
    assert client.get('/api/datasets', params=params).status_code == 422  # type: ignore[arg-type]
    assert client.post('/api/datasets/build', params={'out': '/tmp'}, json=BUILD).status_code == 422


def test_list_without_a_datasets_directory_is_empty(client: TestClient, root: Path) -> None:
    assert client.get('/api/datasets').json() == {'datasets': [], 'total': 0}
    assert not root.exists()


def test_datasets_directory_defaults_to_the_home_folder(tmp_path: Path) -> None:
    app = create_app(tmp_path / 'console.db', tmp_path / 'qa', token='t')
    assert DEFAULT_DATASETS_DIR == Path.home() / '.deepgym' / 'datasets'
    assert app.state.datasets_dir == DEFAULT_DATASETS_DIR.resolve()


def test_datasets_directory_is_expanded_and_resolved_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'real').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    monkeypatch.chdir(tmp_path)

    home = create_app(tmp_path / 'a.db', tmp_path / 'qa', token='t', datasets_dir=Path('~/sets'))
    relative = create_app(tmp_path / 'b.db', tmp_path / 'qa', token='t', datasets_dir=Path('link'))

    assert home.state.datasets_dir == (tmp_path / 'sets').resolve()
    assert relative.state.datasets_dir == (tmp_path / 'real').resolve()


def test_verify_reports_a_manifest_nested_too_deep(client: TestClient, root: Path) -> None:
    (root / 'deep-1').mkdir(parents=True)
    (root / 'deep-1' / 'manifest.json').write_text('[' * 100_000, encoding='utf-8')

    verified = client.post('/api/datasets/verify', json={'dataset': 'deep-1'}).json()
    listed = client.get('/api/datasets').json()['datasets']

    assert verified['valid'] is False
    assert verified['problems'][0].startswith('deep-1/manifest.json is not JSON with unique keys')
    assert listed[0]['manifest'] is None
    assert listed[0]['error'].startswith('manifest.json is not a dataset manifest')
