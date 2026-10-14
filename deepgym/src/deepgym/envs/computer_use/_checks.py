"""Shared helpers for computer-use verifiers.

Besides the app-state helpers, this module carries the readers and rule engine
for office files: `read_sheets` and `read_docx_text` read .xlsx and .docx with
the standard library only, and `office_rule_checks` evaluates the compare_table
rules of an imported OSWorld task against the workbook the agent saved.
"""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
import resource
import select
import signal
import stat
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.parsers import expat


def load_state(path: str = 'state.json') -> dict:
    """Return the app state the page last saved, or {} if it never saved."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def report(progress: dict[str, bool], guards: dict[str, bool]) -> None:
    """Print verifier protocol JSON.

    Score is the fraction of progress checks passed, but only if every guard
    holds (work was saved, nothing else was broken). Otherwise it is 0, so
    neither doing nothing nor saving untouched state earns reward.
    """
    guards_ok = all(guards.values())
    done = sum(progress.values())
    score = done / len(progress) if guards_ok else 0.0
    failed = [name for name, ok in guards.items() if not ok]
    print(
        json.dumps(
            {
                'schema_version': '1.0',
                'score': score,
                'passed': guards_ok and done == len(progress),
                'details': f'{done}/{len(progress)} changes made'
                + (f'; guards failed: {", ".join(failed)}' if failed else ''),
                'cases': [
                    {'id': name, 'passed': ok, 'score': float(ok)}
                    for name, ok in {**guards, **progress}.items()
                ],
            }
        )
    )


# Office files {{{ #
#
# The readers follow what OSWorld's own evaluators see through pandas, openpyxl
# and python-docx: cached formula values (not formula text), empty cells as
# None, numbers int when integral, a number shown as a date marked as one (openpyxl
# reads it as a datetime; check_cell reads the raw serial), and docx text as
# python-docx reads it.
# office_rule_checks() evaluates compare_table's rules with the same semantics,
# including DataFrame.round's rounding (numpy's, not Python's, half-to-even).

_XLSX_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
_REL_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
_PKG_REL_NS = '{http://schemas.openxmlformats.org/package/2006/relationships}'
_W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
_CT_NS = '{http://schemas.openxmlformats.org/package/2006/content-types}'
_WML_MAIN = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'
_WORKBOOK_TYPES = frozenset(
    {
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml',
        'application/vnd.ms-excel.sheet.macroEnabled.main+xml',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml',
        'application/vnd.ms-excel.template.macroEnabled.main+xml',
    }
)
"""The workbook content types openpyxl opens."""
_SHARED_STRINGS_TYPE = (
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml'
)
_OFFICE_DOCUMENT = (
    'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument'
)

_COORD_RE = re.compile(r'([A-Z]{1,3})([1-9][0-9]{0,6})')
_MAX_ROW, _MAX_COL = 1_048_576, 16_384
"""The largest row and column a spreadsheet cell can have."""

_MAX_FILE_BYTES = 16 << 20
"""Largest saved file read, packed or unpacked; a bigger one is unreadable."""
_MAX_CELLS = 1_000_000
"""Most cells (used rows x columns, all sheets together) read from one workbook."""
JUDGE_SECONDS = 20
"""How long judging a saved file may take; a session gives the whole verifier 30 s."""
_JUDGE_MEMORY = 1 << 30
"""Address space the judging process may use, where the platform enforces it."""


def bounded_call(judge: Callable[[], dict], seconds: float = JUDGE_SECONDS) -> dict | None:
    """Return ``judge()``, computed in a forked child under a time and memory limit.

    Reading a file the agent controls can't be made cheap in every case, so the work
    runs where running out of memory or time kills only the child.

    Returns:
        The child's JSON-serializable result, or None if it raised, was killed, or
        ran past *seconds*: the caller scores that a 0.
    """
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
    except OSError:  # our sandbox couldn't start the child: no verdict, nothing left open
        os.close(read_fd)
        os.close(write_fd)
        raise
    if pid == 0:
        status = 1
        try:
            os.close(read_fd)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (_JUDGE_MEMORY, _JUDGE_MEMORY))
            except (ValueError, OSError):  # not enforced on macOS
                pass
            with os.fdopen(write_fd, 'wb') as out:
                out.write(json.dumps(judge()).encode())
            status = 0
        finally:
            os._exit(status)
    os.close(write_fd)
    deadline = time.monotonic() + seconds
    chunks = []
    reaped = False
    try:
        with os.fdopen(read_fd, 'rb') as pipe:
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([pipe], [], [], left)[0]:
                    return None
                chunk = os.read(pipe.fileno(), 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
        _, status = os.waitpid(pid, 0)
        reaped = True
    finally:
        if not reaped:  # past the deadline, or the parent failed: never leave it running
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    return json.loads(b''.join(chunks)) if status == 0 else None


@contextmanager
def _open_package(path: str) -> Iterator[zipfile.ZipFile]:
    """Open the saved file at *path* as a zip package.

    The agent controls this file, so it is opened without blocking (a FIFO planted at
    the path can't stall the verifier) and must be a regular file of at most
    :data:`_MAX_FILE_BYTES`, packed and unpacked, whose parts are stored or deflated
    (the methods :func:`_read_xml` reads with bounded output).

    Raises:
        ValueError: If the file is not a regular file, is too large or uses another
            compression method.
        OSError, zipfile.BadZipFile: If it can't be opened as a zip package.
    """
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('not a regular file')
        if info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f'larger than {_MAX_FILE_BYTES} bytes')
        with zipfile.ZipFile(handle) as archive:
            members = archive.infolist()
            if sum(member.file_size for member in members) > _MAX_FILE_BYTES:
                raise ValueError(f'unpacks to more than {_MAX_FILE_BYTES} bytes')
            if any(
                m.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for m in members
            ):
                raise ValueError('a part uses a compression method other than deflate')
            yield archive


def _read_xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    """Parse one part, reading at most :data:`_MAX_FILE_BYTES` of it.

    A bounded read() caps what is decompressed at the part's declared size, so a
    part that unpacks to more than it declares fails its CRC check instead.
    """
    with archive.open(name) as member:
        data = member.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError(f'{name} unpacks to more than {_MAX_FILE_BYTES} bytes')
    # Office parts never declare a DTD; refusing one rules out entity expansion.
    # Nor a namespace URI longer than any real one, which every name would repeat.
    scan = expat.ParserCreate(namespace_separator=' ')

    def refuse(*_: object) -> None:
        raise ValueError(f'{name} declares a DTD')

    def check_namespace(_: str | None, uri: str) -> None:
        if len(uri) > 200:
            raise ValueError(f'{name} declares a namespace URI of {len(uri)} characters')

    scan.StartDoctypeDeclHandler = refuse
    scan.StartNamespaceDeclHandler = check_namespace
    scan.Parse(data, True)
    return ET.fromstring(data)


def _relationships(archive: zipfile.ZipFile, part: str) -> list[tuple[str, str, str]]:
    """(id, type, target part) of each internal relationship of *part* ('' for the package).

    Targets resolve against the part's folder, as office readers resolve them.
    """
    folder, _, name = part.rpartition('/')
    rels = f'{folder}/_rels/{name}.rels' if folder else f'_rels/{name}.rels'
    if rels not in archive.namelist():
        return []
    found = []
    for rel in _read_xml(archive, rels).findall(f'{_PKG_REL_NS}Relationship'):
        if rel.get('TargetMode') == 'External':
            continue
        target = rel.get('Target', '')
        if target.startswith('/'):
            target = target[1:]
        else:
            target = posixpath.normpath(posixpath.join(folder, target))
        found.append((rel.get('Id', ''), rel.get('Type', ''), target))
    return found


def _check_related_parts(archive: zipfile.ZipFile) -> None:
    """Refuse a package whose relationships name a part it doesn't hold.

    python-docx loads every part reachable from the package's relationships and parses
    its XML parts, failing on a missing or malformed one, so such a document is
    unreadable here too.

    Raises:
        ValueError: If a relationship's target is missing.
        ET.ParseError: If a related XML part doesn't parse.
    """
    names = set(archive.namelist())
    seen, queue = {''}, ['']
    while queue:
        for _, _, target in _relationships(archive, queue.pop()):
            if target not in names:
                raise ValueError(f'a relationship names {target}, which the package lacks')
            if target not in seen:
                if target.lower().endswith('.xml'):  # python-docx parses its XML parts
                    _read_xml(archive, target)
                seen.add(target)
                queue.append(target)


def _main_part(archive: zipfile.ZipFile) -> str:
    """The part the package's officeDocument relationship names, as python-docx and
    openpyxl resolve it, not whatever sits at the usual path.

    Raises:
        ValueError: If the package names no main part or more than one.
    """
    main = [t for _, kind, t in _relationships(archive, '') if kind == _OFFICE_DOCUMENT]
    if len(main) != 1:
        raise ValueError(f'{len(main)} main parts')
    return main[0]


def _coord_to_cell(coord: str) -> tuple[int, int]:
    """'C7' -> (row 7, column 3), both 1-based.

    Raises:
        ValueError: If *coord* is not a cell reference within a sheet's bounds.
    """
    match = _COORD_RE.fullmatch(coord)
    if not match:
        raise ValueError(f'not a cell coordinate: {coord[:20]!r}')
    col = 0
    for ch in match.group(1):
        col = col * 26 + (ord(ch) - ord('A') + 1)
    row = int(match.group(2))
    if row > _MAX_ROW or col > _MAX_COL:
        raise ValueError(f'{coord} is outside a sheet')
    return row, col


def _number(text: str) -> int | float:
    """A numeric cell's value as openpyxl casts it: a float if the text has a '.' or an
    exponent, else an int. pandas types the column from these, so the type matters.

    Raises:
        ValueError: If the text isn't such a number (openpyxl fails on it too) or the
            float isn't finite.
    """
    if '.' in text or 'e' in text.lower():
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f'{text!r} is not a finite number')
        return value
    return int(text)


def _cell_value(cell: ET.Element, shared: list[str]) -> object:
    """The value of one <c> element, as pandas reads it (a date is its serial here)."""
    kind = cell.get('t')
    if kind == 'inlineStr':
        node = cell.find(f'{_XLSX_NS}is')
        if node is None:
            return None
        return ''.join(t.text or '' for t in node.iter(f'{_XLSX_NS}t'))
    value = cell.find(f'{_XLSX_NS}v')
    if value is None or value.text is None:
        return None
    if kind == 's':  # a reference past the shared strings is unreadable, as for openpyxl
        return shared[int(value.text)]
    if kind == 'str':
        return value.text
    if kind == 'b':
        return bool(int(value.text))  # as openpyxl reads it; not an integer is unreadable
    if kind == 'e':
        return value.text
    return _number(value.text)


_DATE_FORMAT_IDS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 46, 47})
_DURATION_FORMAT_IDS = frozenset({46})
"""Built-in formats openpyxl reads as a duration ([h]:mm:ss) rather than a date."""
_TIMEDELTA = re.compile(
    r'\[hh?\](:mm(:ss(\.0*)?)?)?|\[mm?\](:ss(\.0*)?)?|\[ss?\](\.0*)?', re.IGNORECASE
)
"""openpyxl's TIMEDELTA_RE: an elapsed-time section makes a date format a duration."""
"""Built-in number formats that show a date or time."""
_FORMAT_NOISE = re.compile(r'".*?"|\[(?!hh?\]|mm?\]|ss?\])[^\]]*\]')
"""openpyxl's STRIP_RE: quoted text and bracketed sections other than [h], [m], [s]."""


def _date_styles(archive: zipfile.ZipFile, part: str | None) -> tuple[dict[int, str], int]:
    """({style index: 'date' or 'duration'}, how many cell styles there are) for styles
    *part*: the styles whose number format shows a date, time or elapsed time.

    openpyxl reads a number in such a cell as a datetime, a time or a timedelta, so
    pandas never equates it with a plain number or with another kind;
    :func:`_sheet_grid` marks those cells the same way.
    """
    if part is None:
        return {}, 1  # openpyxl's default stylesheet has one cell style
    root = _read_xml(archive, part)
    custom = {
        int(fmt.get('numFmtId', '-1')): fmt.get('formatCode', '')
        for fmt in root.iter(f'{_XLSX_NS}numFmt')
    }
    if any(len(code) > 255 for code in custom.values()):  # longer than any real one
        raise ValueError('a number format code is too long')
    kinds = {fmt_id: 'date' for fmt_id in _DATE_FORMAT_IDS - set(custom)}
    kinds |= dict.fromkeys(_DURATION_FORMAT_IDS - set(custom), 'duration')
    for fmt_id, code in custom.items():  # openpyxl's is_date_format, then is_timedelta_format
        first = code.split(';')[0]
        if re.search(r'(?<![_\\])[dmhysDMHYS]', _FORMAT_NOISE.sub('', first)):
            kinds[fmt_id] = 'duration' if _TIMEDELTA.search(first) else 'date'
    xfs = root.find(f'{_XLSX_NS}cellXfs')
    entries = [] if xfs is None else xfs.findall(f'{_XLSX_NS}xf')
    styles = {}
    for index, xf in enumerate(entries):
        kind = kinds.get(int(xf.get('numFmtId', '0')))
        if kind:
            styles[index] = kind
    return styles, max(len(entries), 1)


def _merge_ranges(root: ET.Element) -> list[str]:
    """The sheet's merged ranges ('A1:C1').

    Raises:
        ValueError: If a range is not two cell references, so a malformed merge makes
            the file unreadable (a 0) instead of failing one rule while others pass.
    """
    merges = root.find(f'{_XLSX_NS}mergeCells')
    if merges is None:
        return []
    refs = [cell.get('ref', '') for cell in merges.findall(f'{_XLSX_NS}mergeCell')]
    for ref in refs:
        start, _, end = ref.partition(':')
        _coord_to_cell(start)
        _coord_to_cell(end)
    return refs


def _trim(grid: list[list[object]]) -> list[list[object]]:
    """Drop trailing rows and columns whose every cell is None (pandas reads no further)."""
    while grid and all(value is None for value in grid[-1]):
        grid.pop()
    while grid and all((row[-1] if row else None) is None for row in grid):
        for row in grid:
            row.pop()
    return grid


def _sheet_grid(
    root: ET.Element, shared: list[str], styles: tuple[dict[int, str], int], shift: int, budget: int
) -> list[list[object]]:
    """Rows of cell values over the sheet's used range; missing cells are None.

    The used range ends at the last non-empty row and column, as after :func:`_trim`.
    *styles* is :func:`_date_styles`' result; a cell naming a style that doesn't exist
    makes the sheet unreadable, as it does for openpyxl. A number in a date- or
    duration-formatted cell is marked
    the way openpyxl decodes it: ``{'duration': serial}``, ``{'time': serial}`` (a date
    format below 1 day) or ``{'date': serial + shift}`` (*shift* puts a 1904-based serial
    on the 1900 calendar), each with the raw ``'serial'`` that check_cell reads. A
    marked cell only equals one of the same kind and value.

    Raises:
        ValueError: If the used range holds more than *budget* cells, or a cell names a
            style that doesn't exist.
    """
    dated, style_count = styles
    cells: dict[tuple[int, int], object] = {}
    max_row = 0
    max_col = 0
    data = root.find(f'{_XLSX_NS}sheetData')
    if data is None:
        return []
    for row in data.findall(f'{_XLSX_NS}row'):
        for cell in row.findall(f'{_XLSX_NS}c'):
            style = int(cell.get('s', '0'))
            if not 0 <= style < style_count:
                raise ValueError(f'a cell names style {style}; there are {style_count}')
            ref = cell.get('r')
            if not ref:
                continue
            value = _cell_value(cell, shared)
            if value is None:
                continue
            kind = dated.get(style) if _is_number(value) else None
            if kind == 'duration':
                value = {'duration': value, 'serial': value}
            elif kind and 0 <= value < 1:
                value = {'time': value, 'serial': value}
            elif kind:
                value = {'date': value + shift, 'serial': value}
            r, c = _coord_to_cell(ref)
            cells[(r, c)] = value
            max_row = max(max_row, r)
            max_col = max(max_col, c)
    if max_row * max_col > budget:
        raise ValueError(f'used range of {max_row} x {max_col} cells is too large')
    grid: list[list[object]] = []
    for r in range(1, max_row + 1):
        grid.append([cells.get((r, c)) for c in range(1, max_col + 1)])
    return _trim(grid)


def _content_types(archive: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    """[Content_Types].xml as ({part name: type}, {extension: type}).

    Raises:
        ValueError: If a part or extension is declared twice: readers would disagree
            on which declaration counts.
    """
    types = _read_xml(archive, '[Content_Types].xml')
    found: tuple[dict[str, str], dict[str, str]] = ({}, {})
    for table, tag, key in ((found[0], 'Override', 'PartName'), (found[1], 'Default', 'Extension')):
        for node in types.findall(f'{_CT_NS}{tag}'):
            name = node.get(key, '').lstrip('/').lower()
            if name in table:
                raise ValueError(f'[Content_Types].xml declares {name} twice')
            table[name] = node.get('ContentType', '')
    return found


def _check_declarations(archive: zipfile.ZipFile, book: str, parts: dict[str, str]) -> None:
    """Refuse a workbook whose content types and relationships name different parts.

    openpyxl finds the workbook and shared strings through [Content_Types].xml and
    reads styles from xl/styles.xml; this reader follows relationships. Where the two
    could disagree, the file is unreadable, so both always read the same parts.

    Raises:
        ValueError: If the declarations disagree.
    """
    declared, _ = _content_types(archive)
    books = {name for name, kind in declared.items() if kind in _WORKBOOK_TYPES}
    strings = {name for name, kind in declared.items() if kind == _SHARED_STRINGS_TYPE}
    if books != {book.lower()}:
        raise ValueError(f'content types declare {sorted(books)}, relationships {book}')
    if strings != ({parts['sharedStrings'].lower()} if 'sharedStrings' in parts else set()):
        raise ValueError('content types and relationships name different shared strings')
    if parts.get('styles', 'xl/styles.xml') != 'xl/styles.xml':
        raise ValueError(f'styles at {parts["styles"]}, not xl/styles.xml')


def read_sheets(path: str) -> list[tuple[str, list[list[object]], list[str]]]:
    """Every sheet of *path*: (name, values grid, merged ranges in A1 notation).

    Args:
        path: Path to an .xlsx workbook.

    Returns:
        Sheets in workbook order. A grid's cells are strings, numbers (int when
        integral), ``{'date': ..., 'serial': ...}`` for a number shown as a date or
        time (see :func:`_sheet_grid`), booleans, error text, or None for an empty cell.

    Raises:
        ValueError: If the file is missing or can't be read for any reason: the
            verifier scores that a 0, since the agent controls the file.
    """
    try:
        with _open_package(path) as archive:
            book_part = _main_part(archive)
            book = _read_xml(archive, book_part)
            related = _relationships(archive, book_part)
            targets = {rel_id: target for rel_id, _, target in related}
            parts = {kind.rsplit('/', 1)[-1]: target for _, kind, target in related}
            _check_declarations(archive, book_part, parts)
            shared_node = (
                _read_xml(archive, parts['sharedStrings']) if 'sharedStrings' in parts else None
            )
            shared = [
                ''.join(t.text or '' for t in si.iter(f'{_XLSX_NS}t'))
                for si in (shared_node if shared_node is not None else [])
            ]
            # openpyxl reads styles from xl/styles.xml whatever the relationships say.
            styles = 'xl/styles.xml' if 'xl/styles.xml' in archive.namelist() else None
            cell_styles = _date_styles(archive, styles)
            properties = book.find(f'{_XLSX_NS}workbookPr')
            epoch = '' if properties is None else properties.get('date1904', '')
            shift = 1462 if epoch.lower() in ('1', 'true') else 0
            node = book.find(f'{_XLSX_NS}sheets')
            sheets = []
            budget = _MAX_CELLS
            parsed: dict[str, tuple[list[list[object]], list[str]]] = {}
            for sheet in [] if node is None else node:
                name, part = sheet.get('name', ''), targets[sheet.get(f'{_REL_NS}id', '')]
                if part not in parsed:  # sheets naming one part parse it once
                    root = _read_xml(archive, part)
                    grid = _sheet_grid(root, shared, cell_styles, shift, budget)
                    parsed[part] = (grid, _merge_ranges(root))
                grid, merges = parsed[part]
                budget -= len(grid) * max((len(row) for row in grid), default=0)
                if budget < 0:
                    raise ValueError(f'more than {_MAX_CELLS} cells')
                sheets.append((name, grid, merges))
            return sheets
    except Exception as exc:  # any way the agent's file breaks reading is a scored 0
        raise ValueError(f'{path} is not a readable workbook: {exc}') from exc


def _run_text(run: ET.Element) -> str:
    """python-docx's `Run.text`: direct t/tab/br/cr/noBreakHyphen/ptab children."""
    parts = []
    for node in run:
        tag = node.tag
        if tag == f'{_W_NS}t':
            parts.append(node.text or '')
        elif tag == f'{_W_NS}tab':
            parts.append('\t')
        elif tag == f'{_W_NS}cr':
            parts.append('\n')
        elif tag == f'{_W_NS}br':
            parts.append('\n' if node.get(f'{_W_NS}type', 'textWrapping') == 'textWrapping' else '')
        elif tag == f'{_W_NS}noBreakHyphen':
            parts.append('-')
        elif tag == f'{_W_NS}ptab':
            parts.append('\t')
    return ''.join(parts)


def read_docx_text(path: str) -> str:
    """The document's text as compare_docx_files reads it: python-docx's
    `paragraph.text` for every body paragraph (direct runs and hyperlink runs), in
    order, whitespace collapsed to single spaces, stripped.

    Args:
        path: Path to a .docx document.

    Returns:
        The normalized text.

    Raises:
        ValueError: If the file is missing or can't be read for any reason: the
            verifier scores that a 0, since the agent controls the file.
    """
    try:
        with _open_package(path) as archive:
            main = _main_part(archive)
            # python-docx opens only a main part whose content type is a Word document.
            declared, defaults = _content_types(archive)
            kind = declared.get(main.lower()) or defaults.get(main.rpartition('.')[2].lower())
            if kind != _WML_MAIN:
                raise ValueError(f'{main} has content type {kind!r}, not a Word document')
            _check_related_parts(archive)
            root = _read_xml(archive, main)
            if root.tag != f'{_W_NS}document':
                raise ValueError(f'{main} is not a w:document')
        body = root.find(f'{_W_NS}body')
        paragraphs = []
        for para in [] if body is None else body:
            if para.tag != f'{_W_NS}p':
                continue
            text_parts = []
            for child in para:
                if child.tag == f'{_W_NS}r':
                    text_parts.append(_run_text(child))
                elif child.tag == f'{_W_NS}hyperlink':
                    for run in child:
                        if run.tag == f'{_W_NS}r':
                            text_parts.append(_run_text(run))
            paragraphs.append(''.join(text_parts))
        return re.sub(r'\s+', ' ', '\n'.join(paragraphs)).strip()
    except Exception as exc:  # any way the agent's file breaks reading is a scored 0
        raise ValueError(f'{path} is not a readable document: {exc}') from exc


def _np_round(value: float, precision: int) -> float:
    """Round like pandas' DataFrame.round, i.e. like numpy: round the value scaled by
    10**precision to nearest-even, then scale back.

    Python's `round(value, precision)` can disagree at exact halves, e.g.
    round(-0.00005, 4) == -0.0001 while numpy rounds it to -0.0.
    """
    scale = 10**precision
    scaled = value * scale
    if not math.isfinite(scaled):  # too large to round; pandas leaves it as it is
        return value
    return round(scaled) / scale


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _same_value(a: object, b: object) -> bool:
    """Elementwise equality as pandas uses it: None only equals None; numbers compare
    across int/float; a boolean only equals a boolean, and a date, time or duration
    only one of the same kind and value; other types must match."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict) or isinstance(b, dict):  # dates, times, durations
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        return {k: v for k, v in a.items() if k != 'serial'} == {
            k: v for k, v in b.items() if k != 'serial'
        }
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def _round_grid(grid: list[list[object]], precision: int) -> list[list[object]]:
    """DataFrame.round(precision)'s effect on a grid: round all-numeric columns' data.

    Row 0 is the header and becomes the frame's column labels, so it never takes
    part in a column's dtype and is never rounded. pandas leaves object columns (a
    column becomes object as soon as one data cell is not a plain number), bools
    and datetimes untouched, so a stray text cell turns off rounding for the whole
    column on that side.
    """
    width = max((len(row) for row in grid), default=0)
    numeric = []
    for column in range(width):
        values = [row[column] for row in grid[1:] if column < len(row) and row[column] is not None]
        numeric.append(bool(values) and all(_is_number(value) for value in values))
    if not any(numeric):
        return grid
    rounded = [list(grid[0])] if grid else []
    for row in grid[1:]:
        rounded.append(
            [
                _np_round(value, precision)
                if column < len(row) and numeric[column] and _is_number(value)
                else value
                for column, value in enumerate(row)
            ]
        )
    return rounded


def _numeric_dtypes(grid: list[list[object]]) -> list[str | None]:
    """The dtype pandas gives each all-number data column: int64 when every cell holds an
    int, float64 once one holds a float or is empty; None for any other column.

    Other dtype differences need values _same_value already tells apart.
    """
    width = max((len(row) for row in grid), default=0)
    dtypes: list[str | None] = []
    for column in range(width):
        values = [row[column] if column < len(row) else None for row in grid[1:]]
        present = [value for value in values if value is not None]
        if not present or not all(_is_number(value) for value in present):
            dtypes.append(None)
        elif len(present) == len(values) and all(isinstance(v, int) for v in present):
            dtypes.append('int64')
        else:
            dtypes.append('float64')
    return dtypes


def sheets_match(
    agent: list[list[object]], expected: list[list[object]], precision: int = 4
) -> bool:
    """Whether two grids are the same sheet, as compare_table's sheet_data sees it.

    Mirrors `pd.read_excel(a).round(precision).equals(pd.read_excel(b).round(precision))`:
    each side is rounded by its own column dtypes first, then shapes, numeric column
    dtypes (int64 against float64 is unequal, as in pandas) and every value must match.
    """
    if len(agent) != len(expected):
        return False
    for row_a, row_e in zip(agent, expected):
        if len(row_a) != len(row_e):
            return False
    if _numeric_dtypes(agent) != _numeric_dtypes(expected):
        return False
    agent = _round_grid(agent, precision)
    expected = _round_grid(expected, precision)
    for row_a, row_e in zip(agent, expected):
        for value_a, value_e in zip(row_a, row_e):
            if not _same_value(value_a, value_e):
                return False
    return True


def is_merged(merges: list[str], coordinate: str) -> bool:
    """True when *coordinate* sits in a merged range but is not its top-left anchor.

    This is what openpyxl reports for a MergedCell, which compare_table's check_cell
    'merge' prop matches against.
    """
    r, c = _coord_to_cell(coordinate)
    for ref in merges:
        if ':' not in ref:
            continue
        start, end = ref.split(':', 1)
        r1, c1 = _coord_to_cell(start)
        r2, c2 = _coord_to_cell(end)
        if r1 <= r <= r2 and c1 <= c <= c2 and not (r == r1 and c == c1):
            return True
    return False


def cell_at(sheet_rows: list[list[object]], coordinate: str) -> object:
    """The value at *coordinate* ('E3'), or None outside the sheet."""
    r, c = _coord_to_cell(coordinate)
    if 1 <= r <= len(sheet_rows) and 1 <= c <= len(sheet_rows[r - 1]):
        return sheet_rows[r - 1][c - 1]
    return None


def build_sheet(rows: list[list[object]]) -> list[list[object]]:
    """Trim a grid built from a literal the way read_sheets trims a file's."""
    return _trim([list(row) for row in rows])


def _parse_sheet_idx(idx: object, names_r: list[str], names_e: list[str]) -> tuple[str, str]:
    """OSWorld's `_parse_sheet_idx`: int/RI/RN select the *result* book (the file the
    agent saved), EI/EN the *expected* one. Returns (side, sheet name); an index that
    cannot be resolved yields name '' which then fails to load, as in OSWorld."""
    if isinstance(idx, int):
        if not names_r or idx >= len(names_r):
            return ('r', '')
        return ('r', names_r[idx])
    if isinstance(idx, str) and idx.startswith('RI'):
        try:
            return ('r', names_r[int(idx[2:])])
        except Exception:
            return ('r', '')
    if isinstance(idx, str) and idx.startswith('RN'):
        return ('r', idx[2:])
    if isinstance(idx, str) and idx.startswith('EI'):
        try:
            return ('e', names_e[int(idx[2:])])
        except Exception:
            return ('e', '')
    if isinstance(idx, str) and idx.startswith('EN'):
        return ('e', idx[2:])
    raise ValueError(f'unrecognized sheet index: {idx!r}')


def _match_rule(value: object, rule: dict) -> bool:
    """OSWorld's `_match_value_to_rule` for the comparison methods the importer
    copies over. A malformed rule or unknown method fails the rule, never passes it."""
    import operator as op

    method = str(rule.get('method', ''))
    if method in {'eq', 'ne', 'le', 'lt', 'ge', 'gt'}:
        try:
            return bool(getattr(op, method)(value, rule['ref']))
        except (KeyError, TypeError):
            return False
    if method.startswith('approx'):
        try:
            return abs(float(value) - rule['ref']) <= float(method.split(':', 1)[1])
        except (KeyError, TypeError, ValueError):
            return False
    return False


def _rule_verdict(
    rule: dict,
    result: list[tuple[str, list[list[object]], list[str]]] | None,
    expected: list[tuple[str, list[list[object]], list[str]]] | None,
    names_r: list[str],
    names_e: list[str],
) -> bool:
    """One compare_table rule against the agent's workbook (and the reference).

    A rule that cannot be resolved — a missing sheet, index or cell — fails;
    no rule of any shape raises."""
    kind = rule.get('type')
    if kind == 'sheet_name':
        return expected is not None and names_r == names_e
    if kind == 'sheet_data':

        def grid_of(idx: object) -> list[list[object]] | None:
            try:
                side, name = _parse_sheet_idx(idx, names_r, names_e)
            except ValueError:
                return None
            sheets = result if side == 'r' else expected
            match = [sheet for sheet in sheets or [] if sheet[0] == name]
            return match[0][1] if match else None

        grid1 = grid_of(rule.get('sheet_idx0'))
        if grid1 is None:
            return False
        grid2 = grid_of(rule.get('sheet_idx1'))
        if grid2 is None:
            return False
        return sheets_match(grid1, grid2, rule.get('precision', 4))
    if kind == 'check_cell':
        try:
            side, name = _parse_sheet_idx(rule.get('sheet_idx'), names_r, names_e)
        except ValueError:
            return False
        sheets = result if side == 'r' else expected
        match = [sheet for sheet in sheets or [] if sheet[0] == name]
        if not match or not rule.get('props'):
            return False
        _, rows, merges = match[0]
        for prop, rule_value in rule['props'].items():
            try:
                if prop == 'value':
                    value = cell_at(rows, rule['coordinate'])
                    if isinstance(value, dict):  # check_cell reads a date's raw serial
                        value = value['serial']
                elif prop == 'merge':
                    value = is_merged(merges, rule['coordinate'])
                else:
                    return False
            except (KeyError, ValueError):
                return False
            if not _match_rule(value, rule_value):
                return False
        return True
    return False


def office_rule_checks(
    rules: list[dict],
    result: list[tuple[str, list[list[object]], list[str]]] | None,
    expected: list[tuple[str, list[list[object]], list[str]]] | None,
) -> dict[str, bool]:
    """Evaluate an imported task's compare_table rules against the agent's workbook.

    Args:
        rules: The task's rules, in order.
        result: `read_sheets()` of the file the agent saved, or None if it could not
            be read.
        expected: `read_sheets()` of the task's reference workbook, or None when the
            task has no reference file.

    Returns:
        {rule name, e.g. 'rule0_sheet_data': passed}; a rule whose sheets, indices
        or cells cannot be resolved fails. A rule's 'name', when it has one, is
        used as its key. If any rule can't be evaluated at all, every rule fails, so
        an evaluation error can never leave partial credit standing.
    """
    names_r = [sheet[0] for sheet in result or []]
    names_e = [sheet[0] for sheet in expected or []]
    names: list[str] = []
    for index, rule in enumerate(rules):
        name = str(rule.get('name') or f'rule{index}_{rule.get("type", "?")}')
        while name in names:  # two rules must never share one verdict slot
            name += '_'
        names.append(name)
    try:
        return {
            name: _rule_verdict(rule, result, expected, names_r, names_e)
            for name, rule in zip(names, rules)
        }
    except Exception:  # a rule that can't decide fails closed, and so does every other
        return dict.fromkeys(names, False)


# }}} Office files #
