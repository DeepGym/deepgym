"""OSWorld importer: the generated envs score the file the task saves, not the screen.

Runs offline: the task configs and the office files here are built by hand, and the
importer's own functions are the ones under test.
"""

import base64
import copy
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from deepgym import Action, ComputerUseEnvironment
from deepgym.registry import load_environment_dir
from tests.office_fixtures import Date, Duration, docx_content_types, make_docx, make_xlsx

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_osworld import (  # noqa: E402
    _REFERENCE_ACTIONS,
    _TASK_API_URL,
    _TASK_URL,
    SNAPSHOT,
    TASKS,
    Task,
    build_environment,
    build_registry_entry,
    build_setup,
    build_task_md,
    fetch_file,
    fetch_task,
    install_shared,
    load_checks_module,
    parse_task,
)

TASK_ID = '00000000-0000-4000-8000-000000000024'
CALC = 'libreoffice_calc'
WRITER = 'libreoffice_writer'
FILE = 'budget.xlsx'
DOC = 'report.docx'
DESKTOP = '/home/user/Desktop'
TITLE = 'A task title'

_START = {
    'A1': 'Item',
    'B1': 'Cost',
    'A2': 'Rent',
    'B2': 1200,
    'A3': 'Food',
    'B3': 300,
    'A4': 'Total',
}
_GOLD = {**_START, 'B4': 1500}
_MERGED = ['A5:B5']
"""The gold file's merge; is_merged() is openpyxl's MergedCell rule, so B5, not A5."""

_DOC_TEXT = ['Presentation Instruction', 'Describe some customs in China.']
_DOC_GOLD = ['presentation instruction', 'describe some customs in china.']


def _cell(sheet_idx: object, coordinate: str, prop: str, spec: dict) -> dict:
    return {
        'type': 'check_cell',
        'sheet_idx': sheet_idx,
        'coordinate': coordinate,
        'props': {prop: spec},
    }


def _rules() -> list[dict]:
    """Two rules the starting file fails and one it already holds."""
    return [
        _cell(0, 'A2', 'value', {'method': 'eq', 'ref': 'Rent'}),
        _cell('RNSheet1', 'B4', 'value', {'method': 'eq', 'ref': 1500}),
        _cell(0, 'B5', 'merge', {'method': 'eq', 'ref': True}),
    ]


def _task_config(
    file_name: str = FILE,
    *,
    func: str = 'compare_table',
    rules: list[dict] | None = None,
    instruction: str = 'Fill in the total and merge the closing row.',
    expected: bool = True,
    task_id: str = TASK_ID,
) -> dict:
    """An OSWorld task config shaped like the LibreOffice ones the importer reads."""
    remote = f'https://example.invalid/{file_name}'
    download = {
        'type': 'download',
        'parameters': {'files': [{'path': f'{DESKTOP}/{file_name}', 'url': remote}]},
    }
    evaluator: dict = {
        'func': func,
        'result': {'type': 'vm_file', 'path': f'{DESKTOP}/{file_name}'},
    }
    if expected:
        evaluator['expected'] = {
            'type': 'cloud_file',
            'path': f'https://example.invalid/gold/{file_name}',
        }
    if func == 'compare_table':
        evaluator['options'] = {'rules': _rules() if rules is None else rules}
    return {
        'id': task_id,
        'instruction': instruction,
        'config': [download, {'type': 'open', 'parameters': {'path': f'{DESKTOP}/{file_name}'}}],
        'evaluator': evaluator,
    }


def _parse(config: dict, *, domain: str = CALC) -> Task:
    return parse_task(config['id'], domain, TITLE, config)


def _import(
    tmp_path: Path, config: dict, source: Path, gold: Path | None, *, domain: str = CALC
) -> tuple[ComputerUseEnvironment, Path]:
    """Import one task from the local *source* and *gold* files, and load it back."""
    task = parse_task(config['id'], domain, TITLE, config)
    out = tmp_path / 'environments' / 'osworld'
    out.mkdir(parents=True, exist_ok=True)
    install_shared(out)
    build_environment(task, source, gold, out, load_checks_module())
    return load_environment_dir(out / task.task_id), out


def _calc_env(tmp_path: Path, *, rules: list[dict] | None = None):
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', _START, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', _GOLD, _MERGED)])
    return _import(tmp_path, _task_config(rules=rules), source, gold)


def _doc_env(tmp_path: Path, *, gold_text=None):
    source = make_docx(tmp_path / 'source.docx', _DOC_TEXT)
    gold = make_docx(tmp_path / 'gold.docx', _DOC_GOLD if gold_text is None else gold_text)
    config = _task_config(DOC, func='compare_docx_files')
    return _import(tmp_path, config, source, gold, domain=WRITER)


def _run_verifier(env: ComputerUseEnvironment, home: Path, work: Path) -> dict:
    """Run *env*'s verifier as Session.score does: its own HOME, the agent's by name."""
    for rel, text in env.files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text)
    (work / '_final_verifier.py').write_text(env.final_verifier_code)
    proc = subprocess.run(
        [sys.executable, '-I', '_final_verifier.py'],
        capture_output=True,
        text=True,
        cwd=work,
        env={'HOME': str(work), 'DEEPGYM_AGENT_HOME': str(home)},
        timeout=25,  # under the 30 s a session gives the verifier
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _save(tmp_path: Path, home: Path, cells: dict, merges: list[str], *, name: str = FILE) -> None:
    """Save a workbook to ~/Desktop/<name> of the agent's home, where the task saves."""
    saved = make_xlsx(tmp_path / 'state.xlsx', [('Sheet1', cells, merges)])
    (home / 'Desktop').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(saved, home / 'Desktop' / name)


@pytest.fixture()
def work(tmp_path: Path) -> Path:
    path = tmp_path / 'work'
    path.mkdir()
    return path


# --- The generated folder, as the loader and deepgym check see it. ---


def test_imported_env_loads_as_computer_use(tmp_path: Path) -> None:
    env, out = _calc_env(tmp_path)
    env_dir = out / TASK_ID

    assert isinstance(env, ComputerUseEnvironment)
    assert sorted(path.name for path in env_dir.iterdir()) == [
        'assets',
        'metadata.json',
        'setup.sh',
        'task.md',
        'variants.json',
        'verifier.py',
    ]
    # Spec files stay out of the session; the starting file is staged instead.
    assert set(env.files) == {f'assets/{FILE}.b64'}
    assert not any('checks' in rel for rel in env.files)
    assert 'def report(' in env.final_verifier_code
    asset = (env_dir / 'assets' / f'{FILE}.b64').read_text()
    assert base64.b64decode(asset) == (tmp_path / 'source.xlsx').read_bytes()
    assert f'assets/{FILE}.b64' in env.setup_script
    assert env.task.startswith('Fill in the total and merge the closing row.')
    assert '$' not in env.task


def test_metadata_and_registry_entry_describe_a_desktop_task(tmp_path: Path) -> None:
    _, out = _calc_env(tmp_path)
    metadata = json.loads((out / TASK_ID / 'metadata.json').read_text())
    assert metadata == {
        'id': TASK_ID,
        'name': TITLE,
        'type': 'computer-use',
        'difficulty': 'medium',
        'domain': 'libreoffice-calc',
        'tags': ['osworld', 'libreoffice', 'calc', 'desktop', 'gui'],
        'max_steps': 30,
        'snapshot': SNAPSHOT,
    }
    assert load_environment_dir(out / TASK_ID).snapshot == SNAPSHOT
    assert build_registry_entry(_parse(_task_config())) == {
        'id': TASK_ID,
        'name': TITLE,
        'path': f'environments/osworld/{TASK_ID}',
        'type': 'computer-use',
        'benchmark': 'osworld',
        'variants': 1,
        'difficulty': 'medium',
        'domain': 'libreoffice-calc',
        'family': 'computer-use',
        'tags': ['osworld', 'libreoffice', 'calc', 'desktop', 'gui'],
    }


def test_task_md_names_the_file_the_app_and_the_done_action() -> None:
    text = build_task_md(_parse(_task_config()))
    assert text.startswith('Fill in the total and merge the closing row.')
    assert f'~/Desktop/{FILE}' in text
    assert 'LibreOffice Calc' in text and 'Ctrl+S' in text and '`done`' in text

    writer = _parse(_task_config(DOC, func='compare_docx_files'), domain=WRITER)
    assert 'LibreOffice Writer' in build_task_md(writer)

    # task.md is substituted as a string.Template at load time.
    loud = _parse(_task_config(instruction='Show the $ total.'))
    with pytest.raises(ValueError, match='contains'):
        build_task_md(loud)


def test_setup_script_decodes_the_staged_file() -> None:
    script = build_setup(_parse(_task_config()))
    assert f'assets/{FILE}.b64' in script
    assert f'"$HOME/Desktop/{FILE}"' in script
    assert '--calc' in script and '--writer' not in script
    assert 'sudo' not in script
    # A document LibreOffice never opens fails setup (no verdict), not the episode.
    assert f'.~lock.{FILE}#' in script and 'exit 1' in script

    writer = _parse(_task_config(DOC, func='compare_docx_files'), domain=WRITER)
    assert '--writer' in build_setup(writer)


# --- Scoring: the saved file, never the screen. ---


def test_scores_untouched_half_done_and_finished_files(tmp_path: Path, work: Path) -> None:
    env, _ = _calc_env(tmp_path)
    home = tmp_path / 'home'
    for label, cells, merges, score in (
        ('untouched', _START, [], 0.0),
        ('total only', {**_START, 'B4': 1500}, [], 0.5),
        ('finished', _GOLD, _MERGED, 1.0),
    ):
        _save(tmp_path, home, cells, merges)
        result = _run_verifier(env, home, work)
        assert result['score'] == score, label
        assert result['passed'] is (score == 1.0), label
        assert {case['id'] for case in result['cases']} == {
            'workbook_saved',
            'rule0_check_cell',
            'rule1_check_cell',
            'rule2_check_cell',
        }, label

    # Nothing saved: every check fails and the guard says why.
    result = _run_verifier(env, tmp_path / 'empty-home', work)
    assert result['score'] == 0.0
    assert 'workbook_saved' in result['details']


def test_rules_the_starting_file_holds_become_guards(tmp_path: Path, work: Path) -> None:
    """Saving the starting file untouched must earn nothing, so a rule it already
    holds is a guard, and only the rules that need work are progress."""
    env, out = _calc_env(tmp_path)
    variant = json.loads((out / TASK_ID / 'variants.json').read_text())[0]

    assert variant['result'] == f'Desktop/{FILE}'
    assert [rule['coordinate'] for rule in variant['progress']] == ['B4', 'B5']
    assert [rule['coordinate'] for rule in variant['guards']] == ['A2']
    assert [rule['name'] for rule in variant['progress']] == [
        'rule1_check_cell',
        'rule2_check_cell',
    ]

    home = tmp_path / 'home'
    _save(tmp_path, home, _START, [])
    assert _run_verifier(env, home, work)['score'] == 0.0


def test_a_rule_an_empty_new_sheet_holds_is_a_guard(tmp_path: Path, work: Path) -> None:
    """'D1 is not merged' on a sheet the task adds holds as soon as an empty sheet is
    added, so it is a guard: adding a blank Sheet2 earns nothing."""
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', _START, [])])
    summary = {'A1': 'Summary'}
    gold = make_xlsx(
        tmp_path / 'gold.xlsx', [('Sheet1', _START, []), ('Sheet2', summary, ['A1:B1'])]
    )
    rules = [
        _cell('RNSheet2', 'A1', 'value', {'method': 'eq', 'ref': 'Summary'}),
        _cell('RNSheet2', 'B1', 'merge', {'method': 'eq', 'ref': True}),
        _cell('RNSheet2', 'C1', 'merge', {'method': 'eq', 'ref': False}),
    ]
    env, out = _import(tmp_path, _task_config(rules=rules), source, gold)
    variant = json.loads((out / TASK_ID / 'variants.json').read_text())[0]
    assert [rule['coordinate'] for rule in variant['progress']] == ['A1', 'B1']
    assert [rule['coordinate'] for rule in variant['guards']] == ['C1']

    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    for label, sheets, score in (
        ('blank Sheet2', [('Sheet1', _START, []), ('Sheet2', {}, [])], 0.0),
        ('finished', [('Sheet1', _START, []), ('Sheet2', summary, ['A1:B1'])], 1.0),
    ):
        make_xlsx(home / 'Desktop' / FILE, sheets)
        assert _run_verifier(env, home, work)['score'] == score, label


def test_a_date_only_matches_a_cell_shown_as_that_date(tmp_path: Path, work: Path) -> None:
    """pandas reads a date-formatted cell as a datetime, so the right serial typed as
    a plain number does not match the reference's date."""
    head = {'A1': 'Start', 'B1': 'Due'}
    start = {**head, 'A2': Date(40437)}
    done = {**start, 'B2': Date(40557)}
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', start, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', done, [])])
    rules = [{'type': 'sheet_data', 'sheet_idx0': 0, 'sheet_idx1': 'EI0'}]
    env, _ = _import(tmp_path, _task_config(rules=rules), source, gold)

    home = tmp_path / 'home'
    for label, cells, score in (
        ('the serial as a plain number', {**start, 'B2': 40557}, 0.0),
        ('the date', done, 1.0),
    ):
        _save(tmp_path, home, cells, [])
        assert _run_verifier(env, home, work)['score'] == score, label


def _sheet_data_env(tmp_path: Path, start: dict, done: dict):
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', start, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', done, [])])
    rules = [{'type': 'sheet_data', 'sheet_idx0': 0, 'sheet_idx1': 'EI0'}]
    return _import(tmp_path, _task_config(rules=rules), source, gold)[0]


def test_the_workbook_scored_is_the_one_the_package_names(tmp_path: Path, work: Path) -> None:
    """A stray workbook holding the right answer is never what gets scored: here the
    relationships name another workbook, the content types this one, and a package
    whose declarations disagree is unreadable."""
    start, done = {'A1': 'Total'}, {'A1': 'Total', 'A2': 123}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    right = make_xlsx(tmp_path / 'right.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(right)) as archive:
        book = archive.read('xl/workbook.xml').decode()
        book_rels = archive.read('xl/_rels/workbook.xml.rels').decode()
    wrong = make_xlsx(tmp_path / 'wrong.xlsx', [('Sheet1', {'A1': 'Total', 'A2': 999}, [])])
    with zipfile.ZipFile(wrong) as archive:
        wrong_sheet = archive.read('xl/worksheets/sheet1.xml').decode()
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="xl/declared.xml" Type="http://schemas.openxmlformats.org'
        '/officeDocument/2006/relationships/officeDocument"/></Relationships>'
    )
    _repacked(
        home / 'Desktop' / FILE,
        right,
        {
            '_rels/.rels': rels,
            'xl/declared.xml': book,
            'xl/_rels/declared.xml.rels': book_rels.replace('sheet1.xml', 'wrong.xml'),
            'xl/worksheets/wrong.xml': wrong_sheet,
        },
    )

    assert _run_verifier(env, home, work)['score'] == 0.0


def test_a_1904_workbook_compares_calendar_dates(tmp_path: Path, work: Path) -> None:
    """The same serial is a different date in a 1904-based workbook: it fails."""
    start = {'A1': 'Start', 'B1': 'Due', 'A2': Date(40437)}
    done = {**start, 'B2': Date(40557)}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        book = archive.read('xl/workbook.xml').decode()
    shifted = book.replace('<sheets>', '<workbookPr date1904="1"/><sheets>', 1)
    _repacked(home / 'Desktop' / FILE, good, {'xl/workbook.xml': shifted})

    assert _run_verifier(env, home, work)['score'] == 0.0


def test_shared_strings_the_declarations_disagree_on_are_unreadable(
    tmp_path: Path, work: Path
) -> None:
    start, done = {'A1': 'Total'}, {'A1': 'Total', 'A2': 'Correct'}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        types = archive.read('[Content_Types].xml').decode()
        strings = archive.read('xl/sharedStrings.xml').decode()
    _repacked(
        home / 'Desktop' / FILE,
        good,
        {
            '[Content_Types].xml': types.replace('/xl/sharedStrings.xml', '/xl/other.xml'),
            'xl/other.xml': strings.replace('Correct', 'Wrong'),
        },
    )

    result = _run_verifier(env, home, work)

    assert result['score'] == 0.0
    assert 'workbook_saved' in result['details']


def test_an_elapsed_time_does_not_match_a_date(tmp_path: Path, work: Path) -> None:
    """openpyxl reads [h]:mm:ss as a timedelta, so the date's serial shown as elapsed
    time is not the date."""
    start = {'A1': 'Start', 'B1': 'Due', 'A2': Date(40437)}
    done = {**start, 'B2': Date(40557)}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    for label, cells, score in (
        ('elapsed time', {**start, 'B2': Duration(40557)}, 0.0),
        ('the date', done, 1.0),
    ):
        _save(tmp_path, home, cells, [])
        assert _run_verifier(env, home, work)['score'] == score, label


def test_a_part_declared_twice_is_unreadable(tmp_path: Path, work: Path) -> None:
    """Readers disagree on which of two declarations counts, so the file is a 0."""
    start, done = {'A1': 'Total'}, {'A1': 'Total', 'A2': 123}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        types = archive.read('[Content_Types].xml').decode()
    twice = types.replace(
        '</Types>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.spreadsheetml.worksheet+xml"/></Types>',
    )
    _repacked(home / 'Desktop' / FILE, good, {'[Content_Types].xml': twice})

    assert _run_verifier(env, home, work)['score'] == 0.0


def test_a_number_in_a_format_openpyxl_reads_as_a_date_is_not_that_number(
    tmp_path: Path, work: Path
) -> None:
    """'*d0' pads with 'd', and openpyxl's classifier counts it as a date format."""
    start, done = {'A1': 'Total'}, {'A1': 'Total', 'A2': 123}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        sheet = archive.read('xl/worksheets/sheet1.xml').decode()
    styles = (
        f'<styleSheet xmlns="{_X}"><numFmts><numFmt numFmtId="164" formatCode="*d0"/>'
        '</numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/></cellXfs></styleSheet>'
    )
    padded = sheet.replace('<c r="A2">', '<c r="A2" s="1">')
    assert padded != sheet
    _repacked(
        home / 'Desktop' / FILE,
        good,
        {'xl/styles.xml': styles, 'xl/worksheets/sheet1.xml': padded},
    )

    assert _run_verifier(env, home, work)['score'] == 0.0


def test_workbooks_openpyxl_cannot_open_are_unreadable(tmp_path: Path, work: Path) -> None:
    """A style index past the stylesheet, or a workbook content type openpyxl doesn't
    know, makes openpyxl fail; here it makes the file a 0, not a pass."""
    start, done = {'A1': 'Total'}, {'A1': 'Total', 'A2': 123}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        sheet = archive.read('xl/worksheets/sheet1.xml').decode()
        types = archive.read('[Content_Types].xml').decode()
    for label, parts in (
        (
            'a missing style',
            {'xl/worksheets/sheet1.xml': sheet.replace('<c r="A2">', '<c r="A2" s="999">')},
        ),
        (
            'an unknown workbook type',
            {'[Content_Types].xml': types.replace('spreadsheetml.sheet.main', 'nonsense.main')},
        ),
    ):
        _repacked(home / 'Desktop' / FILE, good, parts)
        assert _run_verifier(env, home, work)['score'] == 0.0, label


def test_a_boolean_is_read_as_openpyxl_reads_it(tmp_path: Path, work: Path) -> None:
    """openpyxl reads <v>2</v> in a boolean cell as True, so it doesn't match False."""
    start, done = {'A1': 'Flag'}, {'A1': 'Flag', 'A2': False}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        sheet = archive.read('xl/worksheets/sheet1.xml').decode()
    two = sheet.replace('<c r="A2" t="b"><v>0</v></c>', '<c r="A2" t="b"><v>2</v></c>')
    assert two != sheet
    _repacked(home / 'Desktop' / FILE, good, {'xl/worksheets/sheet1.xml': two})

    assert _run_verifier(env, home, work)['score'] == 0.0


def test_an_int_column_does_not_match_a_float_column(tmp_path: Path, work: Path) -> None:
    """pandas' equals compares dtypes: int64 1 against float64 1.00001 is unequal, even
    though both round to 1 at the rule's precision."""
    start, done = {'A1': 'Amount'}, {'A1': 'Amount', 'A2': 1}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    for label, cells, score in (('a float', {**start, 'A2': 1.00001}, 0.0), ('the int', done, 1.0)):
        _save(tmp_path, home, cells, [])
        assert _run_verifier(env, home, work)['score'] == score, label


def test_a_cell_past_the_shared_strings_is_unreadable(tmp_path: Path, work: Path) -> None:
    """An extra cell naming a shared string that doesn't exist is not an empty cell:
    openpyxl can't open the file, so it scores 0 rather than the right answer's 1.0."""
    start, done = {'A1': 'Amount'}, {'A1': 'Amount', 'A2': 42}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', done, [])]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(good)) as archive:
        sheet = archive.read('xl/worksheets/sheet1.xml').decode()
    extra = sheet.replace(
        '</sheetData>', '<row r="3"><c r="A3" t="s"><v>999</v></c></row></sheetData>'
    )
    _repacked(home / 'Desktop' / FILE, good, {'xl/worksheets/sheet1.xml': extra})

    result = _run_verifier(env, home, work)

    assert result['score'] == 0.0
    assert 'workbook_saved' in result['details']


def test_a_boolean_does_not_match_a_number(tmp_path: Path, work: Path) -> None:
    start, done = {'A1': 'Flag'}, {'A1': 'Flag', 'A2': 1}
    env = _sheet_data_env(tmp_path, start, done)
    home = tmp_path / 'home'
    for label, cells, score in (('TRUE for 1', {**start, 'A2': True}, 0.0), ('1', done, 1.0)):
        _save(tmp_path, home, cells, [])
        assert _run_verifier(env, home, work)['score'] == score, label


def _patch_zip(data: bytes, member: str, edit) -> bytes:
    """*data* with *edit(header, payload)* applied to *member*'s local header and data."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.getinfo(member)
    start = info.header_offset
    name_len, extra_len = struct.unpack('<HH', data[start + 26 : start + 30])
    body = start + 30 + name_len + extra_len
    header, payload = bytearray(data[start:body]), bytearray(data[body:])
    edit(header, payload)
    return data[:start] + bytes(header) + bytes(payload)


def _headers(data: bytes, local: int, central: int, update) -> bytes:
    """*data* with the 2-byte field at *local* / *central* of every member's local and
    central directory header replaced by *update(old value)*."""
    out = bytearray(data)
    for signature, offset in ((b'PK\x03\x04', local), (b'PK\x01\x02', central)):
        at = out.find(signature)
        while at != -1:
            (value,) = struct.unpack('<H', out[at + offset : at + offset + 2])
            out[at + offset : at + offset + 2] = struct.pack('<H', update(value))
            at = out.find(signature, at + 4)
    return bytes(out)


def _flip(_: bytearray, payload: bytearray) -> None:
    for at in range(8, 40):
        payload[at] ^= 0xFF


_X = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
_STYLE_BOMB = (
    f'<styleSheet xmlns="{_X}"><numFmts><numFmt numFmtId="164" formatCode="{"[" * 1_000_000}"/>'
    '</numFmts><cellXfs><xf numFmtId="164"/></cellXfs></styleSheet>'
)
"""A 1 MB format code that a backtracking regex would take minutes over."""
_REF_BOMB = (
    f'<worksheet xmlns="{_X}"><sheetData><row r="1">'
    f'<c r="{"A" * 2_000_000}1"><v>1</v></c></row></sheetData></worksheet>'
)
"""A cell reference two million letters long."""
_DTD = (
    '<?xml version="1.0"?><!DOCTYPE worksheet [<!ENTITY a "aaaaaaaaaa">'
    '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>'
    f'<worksheet xmlns="{_X}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>&b;</t>'
    '</is></c></row></sheetData></worksheet>'
)
"""A sheet that declares entities: office parts never carry a DTD."""


def _repacked(path: Path, good: bytes, parts: dict[str, str], method: int = zipfile.ZIP_DEFLATED):
    """Write *good*'s parts to *path*, replaced or added from *parts*."""
    with zipfile.ZipFile(io.BytesIO(good)) as source, zipfile.ZipFile(path, 'w', method) as out:
        for name in source.namelist():
            out.writestr(name, parts.pop(name, None) or source.read(name))
        for name, text in parts.items():
            out.writestr(name, text)


def _understated(path: Path, good: bytes) -> None:
    """The sheet part padded to 20 MB unpacked while it declares only its real size."""
    sheet = 'xl/worksheets/sheet1.xml'
    with zipfile.ZipFile(io.BytesIO(good)) as source:
        real = source.read(sheet)
    _repacked(path, good, {sheet: real.decode() + ' ' * (20 << 20)})
    data = bytearray(path.read_bytes())
    for signature, size_at, name_at in ((b'PK\x03\x04', 22, 30), (b'PK\x01\x02', 24, 46)):
        at = data.find(signature)
        while at != -1:
            if data[at + name_at : at + name_at + len(sheet)] == sheet.encode():
                data[at + size_at : at + size_at + 4] = struct.pack('<I', len(real))
            at = data.find(signature, at + 4)
    path.write_bytes(bytes(data))


def _one_part_many_sheets(path: Path, tmp_path: Path) -> None:
    """1,001 sheets that all name one 1,000-cell part: past the cell budget together."""
    column = {f'A{row}': row for row in range(1, 1001)}
    base = make_xlsx(tmp_path / 'one.xlsx', [('Sheet1', column, [])])
    sheets = ''.join(f'<sheet name="S{i}" sheetId="{i + 1}" r:id="rId1"/>' for i in range(1001))
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(path, 'w') as out:
        for name in source.namelist():
            data = source.read(name).decode()
            if name == 'xl/workbook.xml':
                data = re.sub('<sheets>.*</sheets>', f'<sheets>{sheets}</sheets>', data)
            out.writestr(name, data)


def _plant(label: str, path: Path, tmp_path: Path) -> None:
    """Leave *path* the way an agent could, by the GUI or a macro, instead of saving."""
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', _GOLD, _MERGED)]).read_bytes()
    deflated = tmp_path / 'deflated.xlsx'
    with zipfile.ZipFile(io.BytesIO(good)) as source:
        with zipfile.ZipFile(deflated, 'w', zipfile.ZIP_DEFLATED) as out:
            for name in source.namelist():
                out.writestr(name, source.read(name))
    sheet = 'xl/worksheets/sheet1.xml'
    pipe = tmp_path / 'pipe'
    writers = {
        'empty file': lambda: path.write_bytes(b''),
        'not a zip': lambda: path.write_bytes(b'not a workbook'),
        'truncated': lambda: path.write_bytes(good[: len(good) // 2]),
        'corrupt deflate stream': lambda: path.write_bytes(
            _patch_zip(deflated.read_bytes(), sheet, _flip)
        ),
        'encrypted parts': lambda: path.write_bytes(_headers(good, 6, 8, lambda v: v | 1)),
        'unsupported compression': lambda: path.write_bytes(_headers(good, 8, 10, lambda v: 99)),
        'unpacks past the limit': lambda: _repacked(
            path, good, {'xl/padding.xml': ' ' * (17 << 20)}
        ),
        'unpacks past what it declares': lambda: _understated(path, good),
        'a bzip2 part': lambda: _repacked(path, good, {}, zipfile.ZIP_BZIP2),
        'a format code past any real one': lambda: _repacked(
            path, good, {'xl/styles.xml': _STYLE_BOMB}
        ),
        'a cell reference past any real one': lambda: _repacked(
            path, good, {'xl/worksheets/sheet1.xml': _REF_BOMB}
        ),
        'a malformed merge': lambda: make_xlsx(path, [('Sheet1', _GOLD, ['A1:BAD'])]),
        'a DTD': lambda: _repacked(path, good, {'xl/worksheets/sheet1.xml': _DTD}),
        'infinite number': lambda: make_xlsx(path, [('Sheet1', {**_GOLD, 'C1': math.inf}, [])]),
        'one part named by many sheets': lambda: _one_part_many_sheets(path, tmp_path),
        'used range too large': lambda: make_xlsx(
            path, [('Sheet1', {**_GOLD, 'AMJ1048576': 'x'}, _MERGED)]
        ),
        'a directory': path.mkdir,
        'a FIFO': lambda: os.mkfifo(path),
        'a link to a FIFO': lambda: (os.mkfifo(pipe), path.symlink_to(pipe)),
        'a dangling link': lambda: path.symlink_to(tmp_path / 'gone.xlsx'),
        'unreadable': lambda: (path.write_bytes(good), path.chmod(0)),
    }
    if label != 'missing':
        writers[label]()


@pytest.mark.parametrize(
    'label',
    [
        'missing',
        'empty file',
        'not a zip',
        'truncated',
        'corrupt deflate stream',
        'encrypted parts',
        'unsupported compression',
        'unpacks past the limit',
        'unpacks past what it declares',
        'a bzip2 part',
        'a format code past any real one',
        'a cell reference past any real one',
        'a malformed merge',
        'a DTD',
        'infinite number',
        'used range too large',
        'one part named by many sheets',
        'a directory',
        'a FIFO',
        'a link to a FIFO',
        'a dangling link',
        'unreadable',
    ],
)
def test_a_saved_workbook_the_agent_broke_is_a_judged_zero(
    label: str, tmp_path: Path, work: Path
) -> None:
    """Whatever the agent leaves at the result path, the verifier gives a verdict (it
    exits 0 with protocol JSON) and scores 0, never a crash or a hang (no verdict)."""
    if label == 'unreadable' and os.geteuid() == 0:
        pytest.skip('root reads a file whatever its mode')
    env, _ = _calc_env(tmp_path)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    _plant(label, home / 'Desktop' / FILE, tmp_path)

    result = _run_verifier(env, home, work)

    assert result['score'] == 0.0
    assert 'workbook_saved' in result['details']


def test_a_valid_file_with_700k_styles_is_read_in_time(tmp_path: Path, work: Path) -> None:
    """700,000 cell styles sharing one long format code: each code is classified
    once, so the verifier still finishes within its budget (and the file is correct)."""
    env, _ = _calc_env(tmp_path)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    good = make_xlsx(tmp_path / 'good.xlsx', [('Sheet1', _GOLD, _MERGED)]).read_bytes()
    code = '&quot;' + 'x' * 240 + '&quot; yyyy'
    xfs = '<xf numFmtId="164"/>' * 700_000
    styles = (
        f'<styleSheet xmlns="{_X}"><numFmts><numFmt numFmtId="164" formatCode="{code}"/>'
        f'</numFmts><cellXfs>{xfs}</cellXfs></styleSheet>'
    )
    _repacked(home / 'Desktop' / FILE, good, {'xl/styles.xml': styles})

    assert _run_verifier(env, home, work)['score'] == 1.0


def _slow() -> dict:
    time.sleep(60)
    return {}


def _broken() -> dict:
    raise RuntimeError('judge failed')


@pytest.mark.filterwarnings('ignore:This process .* is multi-threaded')
def test_the_judging_child_returns_its_verdict_or_none() -> None:
    checks = load_checks_module()
    assert checks.bounded_call(lambda: {'ok': True}) == {'ok': True}
    assert checks.bounded_call(_broken) is None
    started = time.monotonic()
    assert checks.bounded_call(_slow, 0.5) is None
    assert time.monotonic() - started < 5


@pytest.mark.filterwarnings('ignore:This process .* is multi-threaded')
def test_a_failing_parent_still_kills_the_judging_child(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = load_checks_module()
    children: list[int] = []
    fork = os.fork

    def recording_fork() -> int:
        pid = fork()
        children.append(pid)
        return pid

    def broken_select(*_: object) -> None:
        raise RuntimeError('select failed')

    monkeypatch.setattr(checks.os, 'fork', recording_fork)
    monkeypatch.setattr(checks.select, 'select', broken_select)

    with pytest.raises(RuntimeError):
        checks.bounded_call(_slow, 30)

    with pytest.raises(ProcessLookupError):  # killed and reaped, not left running
        os.kill(children[0], 0)


def test_a_judge_that_runs_out_of_time_is_a_judged_zero(tmp_path: Path, work: Path) -> None:
    """When judging the file overruns (or dies), the verifier still reports a 0."""
    env, _ = _calc_env(tmp_path)
    home = tmp_path / 'home'
    _save(tmp_path, home, _GOLD, _MERGED)
    assert 'bounded_call(judge)' in env.final_verifier_code
    stalled = env.final_verifier_code.replace(
        'bounded_call(judge)', "bounded_call(lambda: __import__('time').sleep(60), 0.5)"
    )

    result = _run_verifier(env.model_copy(update={'final_verifier_code': stalled}), home, work)

    assert result['score'] == 0.0
    assert 'workbook_saved' in result['details']


def test_unsaved_changes_and_a_wrong_sheet_name_are_judged_zeros(
    tmp_path: Path, work: Path
) -> None:
    """Changes left unsaved leave the starting file, and the right data on a sheet
    with the wrong name fails the rule that names the sheet: both are a scored 0."""
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', _START, [])])
    summary = {'A1': 'Total', 'A2': 1500}
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', _START, []), ('Sheet2', summary, [])])
    rules = [{'type': 'sheet_data', 'sheet_idx0': 'RNSheet2', 'sheet_idx1': 'ENSheet2'}]
    env, _ = _import(tmp_path, _task_config(rules=rules), source, gold)

    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    saved = home / 'Desktop' / FILE
    for label, sheets, score in (
        ('changes left unsaved', [('Sheet1', _START, [])], 0.0),
        ('wrong sheet name', [('Sheet1', _START, []), ('Sheet 2', summary, [])], 0.0),
        ('finished', [('Sheet1', _START, []), ('Sheet2', summary, [])], 1.0),
    ):
        make_xlsx(saved, sheets)
        assert _run_verifier(env, home, work)['score'] == score, label


@pytest.mark.parametrize(
    'label',
    [
        'missing',
        'not a zip',
        'a FIFO',
        'no package',
        'two mains',
        'no manifest',
        'a missing related part',
        'a malformed related part',
        'an inexact main relationship',
        'not a w:document',
    ],
)
def test_a_saved_document_the_agent_broke_is_a_judged_zero(
    label: str, tmp_path: Path, work: Path
) -> None:
    env, _ = _doc_env(tmp_path)
    home = tmp_path / 'home'
    (home / 'Desktop').mkdir(parents=True)
    saved = home / 'Desktop' / DOC
    with zipfile.ZipFile(make_docx(tmp_path / 'gold-copy.docx', _DOC_GOLD)) as gold:
        document = gold.read('word/document.xml')
        rels = gold.read('_rels/.rels').decode()
    if label == 'not a zip':
        saved.write_bytes(b'not a document')
    elif label in (
        'a missing related part',
        'a malformed related part',
        'an inexact main relationship',
        'not a w:document',
    ):
        with zipfile.ZipFile(make_docx(tmp_path / 'right.docx', _DOC_GOLD)) as right:
            parts = {name: right.read(name) for name in right.namelist()}
        if label in ('a missing related part', 'a malformed related part'):
            # python-docx loads and parses every related part
            if label == 'a malformed related part':
                parts['word/styles.xml'] = b'<broken'
            parts['word/_rels/document.xml.rels'] = (
                b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                b'relationships"><Relationship Id="rId1" Target="styles.xml" Type="http://'
                b'schemas.openxmlformats.org/officeDocument/2006/relationships/styles"/>'
                b'</Relationships>'
            )
        elif label == 'an inexact main relationship':
            parts['_rels/.rels'] = parts['_rels/.rels'].replace(
                b'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
                b'officeDocument',
                b'https://invalid/officeDocument',
            )
        else:
            parts['word/document.xml'] = (
                parts['word/document.xml']
                .replace(b'<w:document', b'<w:other')
                .replace(b'</w:document>', b'</w:other>')
            )
        with zipfile.ZipFile(saved, 'w') as archive:
            for name, data in parts.items():
                archive.writestr(name, data)
    elif label == 'no manifest':  # python-docx can't open a package without one
        with zipfile.ZipFile(make_docx(tmp_path / 'right.docx', _DOC_GOLD)) as right:
            with zipfile.ZipFile(saved, 'w') as archive:
                for name in right.namelist():
                    if name != '[Content_Types].xml':
                        archive.writestr(name, right.read(name))
    elif label == 'a FIFO':
        os.mkfifo(saved)
    elif label in ('no package', 'two mains'):
        with zipfile.ZipFile(saved, 'w') as archive:
            archive.writestr('word/document.xml', document)
            if label == 'two mains':
                twice = rels.replace('</Relationships>', rels[rels.index('<Relationship ') :])
                archive.writestr('_rels/.rels', twice.replace('rId1"', 'rId2"', 1))

    result = _run_verifier(env, home, work)

    assert result['score'] == 0.0
    assert 'document_saved' in result['details']


def test_a_rule_that_cannot_be_evaluated_fails_closed() -> None:
    checks = load_checks_module()
    sheets = [('Sheet1', [['Item']], [])]
    holds = {
        'type': 'check_cell',
        'sheet_idx': 0,
        'coordinate': 'A1',
        'props': {'value': {'method': 'eq', 'ref': 'Item'}},
    }
    broken = {'type': 'check_cell', 'sheet_idx': 0, 'coordinate': 'A1', 'props': ['value']}
    assert checks.office_rule_checks([holds], sheets, None) == {'rule0_check_cell': True}
    # One rule that can't be evaluated fails every rule: no partial credit survives it.
    assert checks.office_rule_checks([holds, broken], sheets, None) == {
        'rule0_check_cell': False,
        'rule1_check_cell': False,
    }


def test_the_main_document_part_is_the_one_the_package_names(tmp_path: Path) -> None:
    """python-docx reads the part the officeDocument relationship names, so a stray
    word/document.xml holding the expected text is not what gets scored."""
    checks = load_checks_module()
    doc = make_docx(tmp_path / 'real.docx', _DOC_TEXT)
    with zipfile.ZipFile(doc) as archive:
        real = archive.read('word/document.xml')
    forged = tmp_path / 'forged.docx'
    main = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument'
    with zipfile.ZipFile(forged, 'w') as archive:
        archive.writestr(
            '_rels/.rels',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{main}" Target="word/real.xml"/></Relationships>',
        )
        archive.writestr('[Content_Types].xml', docx_content_types('word/real.xml'))
        archive.writestr('word/real.xml', real)
        with zipfile.ZipFile(make_docx(tmp_path / 'gold.docx', _DOC_GOLD)) as gold:
            archive.writestr('word/document.xml', gold.read('word/document.xml'))

    assert checks.read_docx_text(str(forged)) == checks.read_docx_text(str(doc))


def test_a_sheet_rule_is_one_all_or_nothing_check(tmp_path: Path, work: Path) -> None:
    """compare_table compares a whole sheet, so fixing one cell of it earns nothing;
    and values within the rule's precision match, as pandas' rounding makes them."""
    head = {'A1': 'Name', 'B1': 'Amount', 'A2': 'a', 'A3': 'b', 'A4': 'c'}
    start = {**head, 'B2': 1.0, 'B3': 2.5, 'B4': 3.0}
    done = {**head, 'B2': 1.5, 'B3': 2.5, 'B4': 4.0}
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', start, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', done, [])])
    rules = [{'type': 'sheet_data', 'sheet_idx0': 0, 'sheet_idx1': 'EI0'}]
    env, _ = _import(tmp_path, _task_config(rules=rules), source, gold)

    home = tmp_path / 'home'
    for label, cells, score in (
        ('untouched', start, 0.0),
        ('one of the two cells fixed', {**start, 'B4': 4.0}, 0.0),
        ('within the precision', {**start, 'B2': 1.50001, 'B4': 4.0}, 1.0),
        ('past the precision', {**start, 'B2': 1.5002, 'B4': 4.0}, 0.0),
        ('finished', done, 1.0),
    ):
        _save(tmp_path, home, cells, [])
        assert _run_verifier(env, home, work)['score'] == score, label


def test_text_tasks_compare_the_saved_text(tmp_path: Path, work: Path) -> None:
    env, out = _doc_env(tmp_path)
    variant = json.loads((out / TASK_ID / 'variants.json').read_text())[0]
    assert set(variant) == {'result', 'expected_text'}
    assert variant['result'] == f'Desktop/{DOC}'
    assert variant['expected_text'] == 'presentation instruction describe some customs in china.'

    home = tmp_path / 'home'
    for label, text, score in (
        ('untouched', _DOC_TEXT, 0.0),
        ('halfway', ['presentation instruction', 'Describe some customs in China.'], 0.0),
        ('finished', _DOC_GOLD, 1.0),
    ):
        (home / 'Desktop').mkdir(parents=True, exist_ok=True)
        make_docx(home / 'Desktop' / DOC, text)
        assert _run_verifier(env, home, work)['score'] == score, label

    result = _run_verifier(env, tmp_path / 'empty-home', work)
    assert result['score'] == 0.0 and 'document_saved' in result['details']


# --- Imports that would produce a task the starting state already solves. ---


def test_a_task_the_starting_file_already_solves_is_rejected(tmp_path: Path) -> None:
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', _START, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', _GOLD, _MERGED)])
    with pytest.raises(ValueError, match='needs no work'):
        _import(tmp_path, _task_config(rules=[_rules()[0]]), source, gold)

    with pytest.raises(ValueError, match='needs no work'):
        _doc_env(tmp_path, gold_text=_DOC_TEXT)


def test_a_reference_that_fails_a_rule_is_rejected(tmp_path: Path) -> None:
    source = make_xlsx(tmp_path / 'source.xlsx', [('Sheet1', _START, [])])
    gold = make_xlsx(tmp_path / 'gold.xlsx', [('Sheet1', {**_START, 'B4': 1400}, _MERGED)])
    with pytest.raises(ValueError, match='reference file fails'):
        _import(tmp_path, _task_config(), source, gold)


# --- Task configs the importer does not port. ---


def _updated(config: dict, path: tuple, value: object) -> dict:
    """A deep copy of *config* with the value at the nested *path* replaced."""
    clone = copy.deepcopy(config)
    node = clone
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return clone


def _rule(config: dict, rule: dict) -> dict:
    return _updated(config, ('evaluator', 'options', 'rules'), [rule])


def _cell_rule(method: str, *, sheet_idx: object = 0, **props: object) -> dict:
    return {
        'type': 'check_cell',
        'sheet_idx': sheet_idx,
        'coordinate': 'A1',
        'props': props or {'value': {'method': method, 'ref': 1}},
    }


def _other_id(config: dict) -> dict:
    return _updated(config, ('id',), 'another')


def _no_instruction(config: dict) -> dict:
    return _updated(config, ('instruction',), '  ')


def _no_open(config: dict) -> dict:
    return _updated(config, ('config',), config['config'][:1])


def _two_downloads(config: dict) -> dict:
    return _updated(config, ('config',), [config['config'][0], *config['config']])


def _two_files(config: dict) -> dict:
    other = {'path': f'{DESKTOP}/other.xlsx', 'url': 'https://example.invalid/other.xlsx'}
    files = config['config'][0]['parameters']['files']
    return _updated(config, ('config', 0, 'parameters', 'files'), [*files, other])


def _opens_another(config: dict) -> dict:
    return _updated(config, ('config', 1, 'parameters', 'path'), f'{DESKTOP}/other.xlsx')


def _local_result(config: dict) -> dict:
    return _updated(config, ('evaluator', 'result', 'type'), 'local_file')


def _result_names_another(config: dict) -> dict:
    return _updated(config, ('evaluator', 'result', 'path'), f'{DESKTOP}/other.xlsx')


def _pptx_evaluator(config: dict) -> dict:
    return _updated(config, ('evaluator', 'func'), 'compare_pptx_files')


def _no_rules(config: dict) -> dict:
    return _updated(config, ('evaluator', 'options'), {})


def _chart_rule(config: dict) -> dict:
    return _rule(config, {'type': 'compare_chart'})


def _half_sheet_data(config: dict) -> dict:
    return _rule(config, {'type': 'sheet_data', 'sheet_idx0': 0})


def _bare_cell_rule(config: dict) -> dict:
    return _rule(config, {'type': 'check_cell', 'sheet_idx': 0, 'coordinate': 'A1'})


def _style_prop(config: dict) -> dict:
    return _rule(config, _cell_rule('eq', style={'method': 'eq', 'ref': 1}))


def _contains(config: dict) -> dict:
    return _rule(config, _cell_rule('contains'))


def _bad_tolerance(config: dict) -> dict:
    return _rule(config, _cell_rule('approx:tight'))


def _bad_index(config: dict) -> dict:
    return _rule(config, _cell_rule('eq', sheet_idx='XX0'))


def _unstaged_name(config: dict) -> dict:
    name = 'a;b.xlsx'
    file = {'path': f'{DESKTOP}/{name}', 'url': 'https://example.invalid/x'}
    patched = _updated(config, ('config', 0, 'parameters', 'files'), [file])
    return _updated(patched, ('config', 1, 'parameters', 'path'), f'{DESKTOP}/{name}')


_REJECTED = [
    ('no instruction', _no_instruction, 'no instruction'),
    ('no open step', _no_open, 'one download and one open'),
    ('two downloads', _two_downloads, 'one download and one open'),
    ('two files in one download', _two_files, 'one downloaded file'),
    ('opening another file than the one staged', _opens_another, 'not the downloaded'),
    ('a result that is not the saved file', _local_result, 'is not the saved'),
    ('a result that names another file', _result_names_another, 'is not the saved'),
    ('an evaluator that is not ported', _pptx_evaluator, 'is not ported yet'),
    ('compare_table without a rules list', _no_rules, 'without a rules list'),
    ('a rule that is not ported', _chart_rule, 'is not ported yet'),
    ('sheet_data without its second index', _half_sheet_data, "without 'sheet_idx1'"),
    ('check_cell without props', _bare_cell_rule, 'without props'),
    ('a prop that is not ported', _style_prop, 'is not ported yet'),
    ('a comparison that is not ported', _contains, 'is not ported yet'),
    ('a tolerance that is not a number', _bad_tolerance, 'bad'),
    ('a sheet index that is not ported', _bad_index, 'is not ported yet'),
    ('a file name that cannot be staged', _unstaged_name, 'is not stageable'),
]


@pytest.mark.parametrize(
    ('label', 'mutate', 'match'), _REJECTED, ids=[case[0] for case in _REJECTED]
)
def test_a_task_the_importer_does_not_port_is_rejected(label: str, mutate, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _parse(mutate(_task_config()))


def test_a_config_that_is_not_the_task_asked_for_is_rejected() -> None:
    with pytest.raises(ValueError, match='does not match'):
        parse_task(TASK_ID, CALC, TITLE, _other_id(_task_config()))


def test_a_text_task_without_a_usable_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match='needs a reference file'):
        _parse(_task_config(DOC, func='compare_docx_files', expected=False), domain=WRITER)

    odd = _task_config(DOC, func='compare_docx_files')
    odd['evaluator']['expected'] = {'type': 'vm_file', 'path': f'{DESKTOP}/{DOC}'}
    with pytest.raises(ValueError, match='is not a cloud_file'):
        _parse(odd, domain=WRITER)


def test_the_imported_tasks_are_unique_libreoffice_tasks() -> None:
    ids = [task_id for task_id, _, _ in TASKS]
    assert len(set(ids)) == len(ids)
    assert {domain for _, domain, _ in TASKS} == {CALC, WRITER}
    titles = [title for _, _, title in TASKS]
    assert len(set(titles)) == len(titles) and all(titles)


def test_every_imported_task_ships_a_reference_episode_within_its_budget() -> None:
    episodes = json.loads(_REFERENCE_ACTIONS.read_text(encoding='utf-8'))['episodes']
    assert set(episodes) == {task_id for task_id, _, _ in TASKS}
    for task_id, actions in episodes.items():
        parsed = [Action.model_validate(action) for action in actions]
        assert parsed[-1].action_type == 'done', task_id
        assert len(parsed) <= 30, task_id  # metadata.json's max_steps


# --- Downloads: the fallbacks that make a blocked network work. ---


def test_fetch_task_falls_back_to_the_contents_api(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def fake(url: str, *, accept: str | None = None) -> bytes:
        asked.append(url)
        if url.startswith(_TASK_URL):
            raise OSError('blocked')
        return b'{"id": "abc"}'

    monkeypatch.setattr('import_osworld.download', fake)
    assert fetch_task('abc', CALC) == {'id': 'abc'}
    assert asked == [f'{_TASK_URL}/{CALC}/abc.json', f'{_TASK_API_URL}/{CALC}/abc.json']

    def refuse(url: str, *, accept: str | None = None) -> bytes:
        raise OSError('blocked')

    monkeypatch.setattr('import_osworld.download', refuse)
    with pytest.raises(RuntimeError, match='raw.githubusercontent.com'):
        fetch_task('abc', CALC)


def test_fetch_file_mirrors_huggingface_and_leaves_other_hosts_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def fake(url: str, *, accept: str | None = None) -> bytes:
        asked.append(url)
        if 'hf-mirror' not in url:
            raise OSError('blocked')
        return b'file bytes'

    monkeypatch.setattr('import_osworld.download', fake)
    host = 'https://huggingface.co/datasets/x/y/resolve/main/a.xlsx'
    fetch_file(host, tmp_path / 'a.xlsx')
    assert asked == [host, host.replace('huggingface.co', 'hf-mirror.com')]
    assert (tmp_path / 'a.xlsx').read_bytes() == b'file bytes'

    asked.clear()

    def any_host(url: str, *, accept: str | None = None) -> bytes:
        asked.append(url)
        return b'file bytes'

    monkeypatch.setattr('import_osworld.download', any_host)
    fetch_file('https://example.com/a.xlsx', tmp_path / 'b.xlsx')
    assert asked == ['https://example.com/a.xlsx']
