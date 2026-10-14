"""Hand-built .xlsx and .docx fixtures for the office reader tests.

Stdlib only, like the readers themselves: these builders write just the parts
`_checks.py` reads, so no office library is needed to exercise it.
"""

import re
import zipfile
from pathlib import Path

_CELL = re.compile(r'([A-Z]+)(\d+)')

_XLSX_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
_REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_PKG_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
_W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


class Inline(str):
    """A string cell stored inline (t="inlineStr") instead of in sharedStrings."""


class Date(int):
    """A serial number in a cell formatted as a date (built-in number format 14)."""


class Duration(int):
    """A serial number in a cell formatted as elapsed time (built-in format 46, [h]:mm:ss)."""


def make_xlsx(path: Path, sheets: list[tuple[str, dict[str, object], list[str]]]) -> Path:
    """Write a minimal workbook: one entry per sheet, (name, cells, merged ranges).

    A cell value may be a number, a bool, a Date, a Duration, a str (shared string) or
    an Inline (an inline string); a cell left out is empty. Cells are 1-based references like 'C7'.
    """
    shared: dict[str, int] = {}
    parts = []
    for index, (name, cells, merges) in enumerate(sheets, start=1):
        parts.append((name, _sheet_xml(cells, merges, shared), index))
    members = {
        '_rels/.rels': _package_rels('xl/workbook.xml'),
        'xl/workbook.xml': _workbook_xml([name for name, _, _ in parts]),
        'xl/_rels/workbook.xml.rels': _rels_xml(len(parts)),
        'xl/styles.xml': _STYLES_XML,
    }
    members.update({f'xl/worksheets/sheet{index}.xml': xml for _, xml, index in parts})
    if shared:
        members['xl/sharedStrings.xml'] = _shared_xml(shared)
    members['[Content_Types].xml'] = _content_types(members)
    with zipfile.ZipFile(path, 'w') as archive:
        for member, xml in members.items():
            archive.writestr(member, xml)
    return path


def _cell_xml(ref: str, value: object, shared: dict[str, int]) -> str:
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, Date):
        return f'<c r="{ref}" s="1"><v>{int(value)}</v></c>'
    if isinstance(value, Duration):
        return f'<c r="{ref}" s="2"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value!r}</v></c>'
    if isinstance(value, Inline):
        return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
    if isinstance(value, str):
        return f'<c r="{ref}" t="s"><v>{shared.setdefault(value, len(shared))}</v></c>'
    raise TypeError(f'cannot write {value!r}')


def _sheet_xml(cells: dict[str, object], merges: list[str], shared: dict[str, int]) -> str:
    rows: dict[int, list[str]] = {}
    for ref, value in cells.items():
        rows.setdefault(int(_CELL.fullmatch(ref).group(2)), []).append(
            _cell_xml(ref, value, shared)
        )
    body = ''.join(
        f'<row r="{number}">{"".join(row)}</row>' for number, row in sorted(rows.items())
    )
    merge = (
        f'<mergeCells count="{len(merges)}">'
        + ''.join(f'<mergeCell ref="{ref}"/>' for ref in merges)
        + '</mergeCells>'
        if merges
        else ''
    )
    return f'<worksheet xmlns="{_XLSX_NS}"><sheetData>{body}</sheetData>{merge}</worksheet>'


_STYLES_XML = (
    f'<styleSheet xmlns="{_XLSX_NS}"><cellXfs count="3"><xf numFmtId="0"/>'
    '<xf numFmtId="14" applyNumberFormat="1"/><xf numFmtId="46" applyNumberFormat="1"/>'
    '</cellXfs></styleSheet>'
)
"""Style 0 is General, style 1 a date, style 2 elapsed time."""


def _shared_xml(shared: dict[str, int]) -> str:
    items = sorted(shared, key=shared.get)
    body = ''.join(f'<si><t>{text}</t></si>' for text in items)
    count = len(items)
    return f'<sst xmlns="{_XLSX_NS}" count="{count}" uniqueCount="{count}">{body}</sst>'


def _workbook_xml(names: list[str]) -> str:
    sheets = ''.join(
        f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(names, start=1)
    )
    return f'<workbook xmlns="{_XLSX_NS}" xmlns:r="{_REL_NS}"><sheets>{sheets}</sheets></workbook>'


_SHEET_TYPES = 'application/vnd.openxmlformats-officedocument.spreadsheetml'
_CT_NS = 'http://schemas.openxmlformats.org/package/2006/content-types'


def _content_types(members: dict[str, str]) -> str:
    """[Content_Types].xml declaring the workbook, sheets, shared strings and styles."""
    kinds = {
        'xl/workbook.xml': 'sheet.main+xml',
        'xl/sharedStrings.xml': 'sharedStrings+xml',
        'xl/styles.xml': 'styles+xml',
    }
    overrides = ''.join(
        f'<Override PartName="/{name}" ContentType="{_SHEET_TYPES}.'
        f'{kinds.get(name, "worksheet+xml")}"/>'
        for name in members
        if name.startswith('xl/') and name.endswith('.xml') and '_rels' not in name
    )
    return (
        f'<Types xmlns="{_CT_NS}"><Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'{overrides}</Types>'
    )


def _package_rels(main: str) -> str:
    """The package's relationships: its main part, at *main*."""
    return (
        f'<Relationships xmlns="{_PKG_REL_NS}"><Relationship Id="rId1" '
        f'Type="{_REL_NS}/officeDocument" Target="{main}"/></Relationships>'
    )


def _rels_xml(count: int) -> str:
    """The workbook's relationships: its sheets, shared strings and styles."""
    rels = ''.join(
        f'<Relationship Id="rId{index}" Type="{_REL_NS}/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, count + 1)
    )
    rels += (
        f'<Relationship Id="rIdS" Type="{_REL_NS}/sharedStrings" Target="sharedStrings.xml"/>'
        f'<Relationship Id="rIdT" Type="{_REL_NS}/styles" Target="styles.xml"/>'
    )
    return f'<Relationships xmlns="{_PKG_REL_NS}">{rels}</Relationships>'


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    """Write a minimal document; a paragraph starting with '<w:p' is taken as raw XML."""
    body = ''.join(
        para if para.startswith('<w:p') else f'<w:p><w:r><w:t>{para}</w:t></w:r></w:p>'
        for para in paragraphs
    )
    document = f'<w:document xmlns:w="{_W_NS}"><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('[Content_Types].xml', docx_content_types('word/document.xml'))
        archive.writestr('_rels/.rels', _package_rels('word/document.xml'))
        archive.writestr('word/document.xml', document)
    return path


def docx_content_types(main: str) -> str:
    """[Content_Types].xml declaring *main* as a Word document."""
    wml = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'
    return (
        f'<Types xmlns="{_CT_NS}"><Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'<Override PartName="/{main}" ContentType="{wml}"/></Types>'
    )
