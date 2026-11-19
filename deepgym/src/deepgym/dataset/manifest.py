"""Write a dataset to a folder with its manifest, and verify a folder against its manifest.

:func:`write_dataset` exports the kept traces of each split in the formats asked for,
with each split's verdicts in ``<split>.verdicts.jsonl``, and writes ``manifest.json``;
the SHA-256 of that file (:func:`manifest_sha256`) is the identifier to publish, and
:func:`verify_dataset` checks a folder against it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from deepgym import __version__
from deepgym.dataset.build import (
    Dataset,
    DatasetCounts,
    DatasetEntry,
    DatasetSpec,
    DropReason,
    DuplicateGroup,
    Sha256,
    Split,
    TaskBand,
    canonical_sha256,
    trace_failure,
)
from deepgym.exceptions import DeepGymError
from deepgym.labels import Name
from deepgym.passk import task_difficulty
from deepgym.traces import Trace
from deepgym.traces.export import ExportFormat, export_traces
from deepgym.traces.schema import json_text

MANIFEST = 'manifest.json'

_PATHS: dict[ExportFormat, str] = {
    'jsonl': '{split}.jsonl',
    'parquet': '{split}.parquet',
    'atif': 'atif/{split}',
}
VERDICTS = '{split}.verdicts.jsonl'
"""Each split's verdicts, one JSON object per kept trace: what to read rewards from."""

IGNORED = frozenset({'.DS_Store'})
"""File names :func:`verify_dataset` skips: folder metadata macOS Finder writes."""

_NAME = TypeAdapter(Name)
_FORMATS = TypeAdapter(tuple[ExportFormat, ...])


class DatasetManifest(BaseModel):
    """What a dataset folder holds.

    ``content_sha256`` is the SHA-256 of every other field as Python's
    ``json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=False)`` writes
    it; it catches edits, not a forger, who can recompute it. The identifier to publish is
    the SHA-256 of the ``manifest.json`` file (:func:`manifest_sha256`), which pins every
    file through ``files``. ``created_at`` is the time of writing, so rebuilding the same
    dataset never reproduces a published SHA-256: compare the manifests' other fields.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: Name
    version: Name
    created_at: AwareDatetime
    deepgym_version: str
    formats: tuple[ExportFormat, ...] = Field(min_length=1)
    spec: DatasetSpec
    counts: DatasetCounts
    entries: dict[str, DatasetEntry]
    dropped: dict[DropReason, tuple[str, ...]]
    duplicates: tuple[DuplicateGroup, ...]
    bands: dict[str, TaskBand]
    files: dict[str, Sha256]
    """SHA-256 of every file in the folder but the manifest, by path relative to it."""

    content_sha256: Sha256


def write_dataset(
    dataset: Dataset,
    out: Path,
    *,
    name: str,
    version: str,
    formats: Sequence[ExportFormat] = ('jsonl',),
) -> str:
    """Export each split of *dataset* in each of *formats* into *out*, with its manifest.

    Files, for each split that has traces: ``<split>.jsonl``, ``<split>.parquet`` and
    ``atif/<split>/`` (whole traces, see :func:`deepgym.traces.export_traces`), always
    ``<split>.verdicts.jsonl`` (each trace's verdict, whose it is, the verifier score that
    stands and whether a human reversed the verifier: what to read rewards from, since the
    traces keep the verifier's result),
    and JSONL with ATIF, so the traces' hashes can be checked; then ``manifest.json``.
    Everything is checked first, then written into a new folder next to *out*, verified
    with :func:`verify_dataset` and renamed to *out*, so a failure leaves nothing behind.

    Args:
        dataset: What :func:`build_dataset` returned.
        out: A new or empty directory.
        name: Dataset name: one line, no whitespace at either end.
        version: Dataset version, the same way.
        formats: Export formats, at least one.

    Returns:
        The SHA-256 of the ``manifest.json`` written: the identifier to publish.

    Raises:
        ValueError: If *name*, *version* or *formats* isn't valid.
        DeepGymError: If *out* isn't a new or empty directory, Parquet is asked for
            without pyarrow, or the folder written fails verification.
    """
    name, version, unique = checked_write(out, name=name, version=version, formats=formats)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f'.{out.name}-{uuid4().hex}'
    staging.mkdir()  # under the umask, so the dataset gets the usual permissions
    try:
        published = _sha256(_write(dataset, staging, name, version, unique))
        verify_dataset(staging, sha256=published)
        if out.exists():
            out.rmdir()
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return published


def checked_write(
    out: Path, *, name: str, version: str, formats: Sequence[ExportFormat]
) -> tuple[str, str, tuple[ExportFormat, ...]]:
    """Check what :func:`write_dataset` is given before anything is read or written.

    Returns:
        The name, the version and the formats to write: each once, JSONL added to ATIF.

    Raises:
        ValueError: If *name*, *version* or *formats* isn't valid.
        DeepGymError: If *out* isn't a new or empty directory, or Parquet is asked for
            without pyarrow.
    """
    if isinstance(formats, str):
        raise ValueError(f"formats is a list of formats such as ['jsonl'], not {formats!r}")
    unique = tuple(dict.fromkeys(_FORMATS.validate_python(tuple(formats))))
    if not unique:
        raise ValueError('write_dataset needs at least one format')
    if 'atif' in unique and 'jsonl' not in unique:
        unique = ('jsonl', *unique)
    name, version = _NAME.validate_python(name), _NAME.validate_python(version)
    if 'parquet' in unique:
        try:
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise DeepGymError("Parquet needs pyarrow: pip install 'deepgym[traces]'") from exc
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise DeepGymError(f'{out} is not a new or empty directory')
    return name, version, unique


def manifest_sha256(folder: Path) -> str:
    """Return the SHA-256 of *folder*'s ``manifest.json``: the dataset's identifier."""
    return _file_sha256(folder / MANIFEST)


def verify_dataset(folder: Path, *, sha256: str | None = None) -> DatasetManifest:
    """Check *folder* against its manifest, and the manifest against *sha256* if given.

    Checks that ``manifest.json`` is what *sha256* names; that the manifest is consistent
    (its content hash, counts recomputed from the entries and dropped ids, no id both kept
    and dropped, each entry in the split its prompt's hash and the spec give, each prompt
    in one split, one kept copy per trajectory, duplicate groups that match the kept and
    duplicate traces); that the folder holds exactly the files the manifest lists, as
    regular files with their hashes, and no symlink, other file or empty directory
    (regular ``.DS_Store`` files are skipped); that those include the files each format
    and each split with traces need; that each JSONL and Parquet file holds exactly that
    split's traces, each once, each matching its entry (hash, trajectory, prompt and
    environment hashes, split, and the verifier's verdict and score the entry records;
    in Parquet, a ``trace_id`` column that is the trace's) and passing the spec's checks
    that need no labels (agents, models, environments, verdict, usage, cost, score range,
    and band by the manifest's bands); that each ``<split>.verdicts.jsonl`` is, byte for
    byte, the lines its entries give; and that each ATIF folder has one document per trace
    of that split.

    Without *sha256*, this only shows the folder is self-consistent: someone who rewrote
    the files and the manifest together would pass. Parquet summary columns and the body
    of ATIF documents are covered by their file hashes only, so by *sha256*.

    Args:
        folder: A folder written by :func:`write_dataset`.
        sha256: The identifier the dataset was published with (:func:`manifest_sha256`),
            in either case.

    Returns:
        The manifest.

    Raises:
        DeepGymError: If the manifest is missing or isn't JSON with unique keys, or
            listing every problem found.
        pydantic.ValidationError: If the manifest doesn't have a manifest's fields.
    """
    path = folder / MANIFEST
    if not path.is_file():
        raise DeepGymError(f'{path} not found')
    try:
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_keys)
    except ValueError as exc:
        raise DeepGymError(f'{path} is not JSON with unique keys: {exc}') from exc
    manifest = DatasetManifest.model_validate(data)
    problems: list[str] = []
    if sha256 is not None and sha256.lower() != (published := manifest_sha256(folder)):
        problems.append(f'{MANIFEST} has sha256 {published}, not the published {sha256}')
    if _content_sha256(data) != manifest.content_sha256:
        problems.append(f'{MANIFEST} was changed: its content_sha256 no longer matches')
    problems += _consistency_problems(manifest)
    present, walked = _walk(folder)
    problems += walked
    problems += _file_problems(folder, manifest, set(present) - {MANIFEST})
    if problems:
        listing = ''.join(f'\n- {problem}' for problem in problems)
        raise DeepGymError(f'{folder} does not match its manifest:{listing}')
    return manifest


def _write(
    dataset: Dataset, folder: Path, name: str, version: str, formats: tuple[ExportFormat, ...]
) -> bytes:
    """Write the exports, verdicts and manifest into the empty *folder*; return the manifest."""
    for split, traces in _by_split(dataset).items():
        for fmt in formats:
            export_traces(traces, fmt, folder / _PATHS[fmt].format(split=split))
        lines = [
            json_text(_verdict_record(trace.trace_id, dataset.entries[trace.trace_id])) + '\n'
            for trace in traces
        ]
        (folder / VERDICTS.format(split=split)).write_bytes(''.join(lines).encode())
    files, _ = _walk(folder)
    fields = {
        'name': name,
        'version': version,
        'created_at': datetime.now(timezone.utc),
        'deepgym_version': __version__,
        'formats': formats,
        'spec': dataset.spec,
        'counts': dataset.counts,
        'entries': dataset.entries,
        'dropped': dataset.dropped,
        'duplicates': dataset.duplicates,
        'bands': dataset.bands,
        'files': {path: _file_sha256(folder / path) for path in files},
    }
    data = json.loads(json_text(fields))
    data['content_sha256'] = _content_sha256(data)
    DatasetManifest.model_validate(data)
    manifest = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    (folder / MANIFEST).write_bytes(manifest)
    return manifest


def _verdict_record(trace_id: str, entry: DatasetEntry) -> dict[str, JsonValue]:
    """One line of ``<split>.verdicts.jsonl``."""
    return {
        'trace_id': trace_id,
        'passed': entry.passed,
        'verdict_source': entry.verdict_source,
        'score': entry.score,
        'overridden': entry.overridden,
    }


def _by_split(dataset: Dataset) -> dict[Split, list[Trace]]:
    """The kept traces of each split that has any, in the dataset's order."""
    by_split: dict[Split, list[Trace]] = {}
    for trace in dataset.traces:
        by_split.setdefault(dataset.entries[trace.trace_id].split, []).append(trace)
    return {split: by_split[split] for split in get_args(Split) if split in by_split}


def _consistency_problems(manifest: DatasetManifest) -> list[str]:
    """Check the manifest's counts, ids, splits and duplicate groups against each other."""
    problems: list[str] = []
    counts, entries = manifest.counts, manifest.entries
    if counts != DatasetCounts.of(
        entries, manifest.dropped, counts.unread, counts.unmatched_labels
    ):
        problems.append('counts do not match the entries and dropped ids')
    dropped = Counter(trace_id for ids in manifest.dropped.values() for trace_id in ids)
    if both := sorted(entries.keys() & dropped.keys()):
        problems.append(f'traces both kept and dropped: {both}')
    if twice := sorted(trace_id for trace_id, count in dropped.items() if count > 1):
        problems.append(f'traces dropped more than once: {twice}')
    spec = manifest.spec
    if moved := sorted(i for i, e in entries.items() if e.split != spec.split_of(e.task_sha256)):
        problems.append(f'traces not in the split their prompt and the spec give: {moved}')
    splits: dict[str, set[Split]] = {}
    for entry in entries.values():
        splits.setdefault(entry.task_sha256, set()).add(entry.split)
    if crossing := sorted(task for task, found in splits.items() if len(found) > 1):
        problems.append(f'prompts in more than one split: {crossing}')
    problems += _band_problems(manifest)
    if 'atif' in manifest.formats and 'jsonl' not in manifest.formats:
        problems.append("ATIF without JSONL: the traces' hashes could not be checked")
    contents = Counter(entry.content_sha256 for entry in entries.values())
    if repeated := sorted(content for content, count in contents.items() if count > 1):
        problems.append(f'more than one kept copy of trajectories {repeated}')
    copies = {trace_id for group in manifest.duplicates if group.kept for trace_id in group.copies}
    kept_ok = all(
        group.kept is None
        or (
            group.kept in group.copies
            and group.kept in entries
            and entries[group.kept].content_sha256 == group.content_sha256
        )
        for group in manifest.duplicates
    )
    if not kept_ok or not set(manifest.dropped.get('duplicate', ())) <= copies:
        problems.append('duplicate groups do not match the kept and duplicate traces')
    return problems


def _band_problems(manifest: DatasetManifest) -> list[str]:
    """Check each task's band against its counts and the spec, and its kept traces against
    its counts: every kept trace with a verdict is one of its samples."""
    spec, problems = manifest.spec, []
    for task, record in manifest.bands.items():
        expected: tuple[object, object] = (None, None)
        if record.samples >= spec.min_samples:
            try:
                difficulty = task_difficulty(record.samples, record.solved, *spec.band_range)
            except ValueError:
                expected = ('impossible', 'counts')
            else:
                expected = (difficulty.band, difficulty.outside_band)
        if (record.band, record.outside_band) != expected:
            problems.append(f'task {task} has the wrong band for its counts and the spec')
    judged = Counter(
        e.environment_sha256 for e in manifest.entries.values() if e.passed is not None
    )
    passed = Counter(e.environment_sha256 for e in manifest.entries.values() if e.passed)
    for task, count in judged.items():
        record = manifest.bands.get(task)
        if record is None or count > record.samples or passed[task] > record.solved:
            problems.append(f'task {task} keeps more verdicts or passes than its band counts')
    return problems


def _file_problems(folder: Path, manifest: DatasetManifest, present: set[str]) -> list[str]:
    """Check the files against the manifest's list and the traces each one must hold."""
    ids: dict[Split, set[str]] = {}
    for trace_id, entry in manifest.entries.items():
        ids.setdefault(entry.split, set()).add(trace_id)
    required: set[str] = set()
    images: list[str] = []
    for split, split_ids in ids.items():
        required.add(VERDICTS.format(split=split))
        for fmt in manifest.formats:
            if fmt == 'atif':
                required |= {f'atif/{split}/{trace_id}.json' for trace_id in split_ids}
                images += [f'atif/{split}/images/{trace_id}/' for trace_id in split_ids]
            else:
                required.add(_PATHS[fmt].format(split=split))
    listed = manifest.files.keys()
    problems = [f'the manifest lists no {name}' for name in sorted(required - listed)]
    problems += [
        f'the manifest lists a file no format writes: {name}'
        for name in sorted(listed - required)
        if not name.startswith(tuple(images))
    ]
    problems += [f'missing: {name}' for name in sorted((listed | required) - present)]
    problems += [f'not in the manifest: {name}' for name in sorted(present - listed)]
    problems += [
        f'changed: {name}'
        for name in sorted(listed & present)
        if _file_sha256(folder / name) != manifest.files[name]
    ]
    for split in ids:
        for fmt in ('jsonl', 'parquet'):
            name = _PATHS[fmt].format(split=split)
            if fmt not in manifest.formats or name not in present:
                continue
            try:
                raws = (
                    [(None, line) for line in _jsonl_lines(folder / name)]
                    if fmt == 'jsonl'
                    else _parquet_rows(folder / name)
                )
            except DeepGymError as exc:
                problems.append(str(exc))
                continue
            rows, refused = _trace_rows(raws, manifest, columns=fmt == 'parquet')
            problems += _row_problems(name, split, rows, manifest.entries)
            if refused:
                problems.append(f'{name} holds traces its spec drops: {sorted(refused)}')
        name = VERDICTS.format(split=split)
        if name in present:
            try:
                rows = _verdict_rows(_jsonl_lines(folder / name), manifest.entries)
            except DeepGymError as exc:
                problems.append(str(exc))
                continue
            problems += _row_problems(name, split, rows, manifest.entries)
    return problems


def _jsonl_lines(path: Path) -> list[bytes]:
    """Return each line without its newline; lines end at b'\\n' only, as written.

    Raises:
        DeepGymError: If the last line has no newline.
    """
    with path.open('rb') as file:
        lines = file.readlines()
    if lines and not lines[-1].endswith(b'\n'):
        raise DeepGymError(f'{path.name} does not end with a newline')
    return [line.rstrip(b'\n') for line in lines]


def _parquet_rows(path: Path) -> list[tuple[object, bytes | None]]:
    """Return each row's ``trace_id`` column and its ``trace`` column, the trace's JSON.

    Raises:
        DeepGymError: If pyarrow isn't installed or can't read the file.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DeepGymError(
            f"Checking {path.name} needs pyarrow: pip install 'deepgym[traces]'"
        ) from exc
    try:
        table = pq.read_table(path, columns=['trace_id', 'trace'])
    except (pa.ArrowException, OSError) as exc:
        raise DeepGymError(f'{path.name} is not a trace Parquet file: {exc}') from exc
    columns = zip(table['trace_id'].to_pylist(), table['trace'].to_pylist())
    return [
        (trace_id, trace.encode() if isinstance(trace, str) else None)
        for trace_id, trace in columns
    ]


def _trace_rows(
    raws: Iterable[tuple[object, bytes | None]], manifest: DatasetManifest, *, columns: bool
) -> tuple[list[tuple[str | None, bool]], list[str]]:
    """Return each row's trace id and whether the trace matches its entry exactly (hash,
    keys, split, and the verifier's verdict and score the entry records), and the rows the
    spec's trace checks drop, given the entry's verdict and the manifest's bands. With
    *columns*, a row's ``trace_id`` column must be its trace's (a null one is not)."""
    rows: list[tuple[str | None, bool]] = []
    refused: list[str] = []
    for column, raw in raws:
        claimed = column if isinstance(column, str) else None
        try:
            trace = None if raw is None else Trace.model_validate_json(raw)
        except ValueError:
            trace = None
        if trace is None or raw is None:
            rows.append((claimed, False))
            continue
        entry = manifest.entries.get(trace.trace_id)
        matches = (
            entry is not None
            and (not columns or claimed == trace.trace_id)
            and entry
            == DatasetEntry.of(
                trace, manifest.spec, entry.human, _sha256(raw), entry.scores_disagree
            )
        )
        rows.append((trace.trace_id, matches))
        if entry is not None:
            reason = trace_failure(trace, manifest.spec, entry.passed, entry.score, manifest.bands)
            if reason is not None:
                refused.append(f'{trace.trace_id} ({reason})')
    return rows, refused


def _verdict_rows(
    lines: Iterable[bytes], entries: Mapping[str, DatasetEntry]
) -> list[tuple[str | None, bool]]:
    """Return each verdict line's trace id and whether it is, byte for byte, the line its
    entry gives."""
    rows: list[tuple[str | None, bool]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        trace_id = record.get('trace_id') if isinstance(record, dict) else None
        if not isinstance(trace_id, str):
            rows.append((None, False))
            continue
        entry = entries.get(trace_id)
        expected = None if entry is None else json_text(_verdict_record(trace_id, entry)).encode()
        rows.append((trace_id, line == expected))
    return rows


def _row_problems(
    name: str,
    split: Split,
    rows: Iterable[tuple[str | None, bool]],
    entries: Mapping[str, DatasetEntry],
) -> list[str]:
    """Check that *rows* are exactly the traces of *split*, each once, each matching."""
    seen: Counter[str | None] = Counter()
    wrong: set[str | None] = set()
    for trace_id, matches in rows:
        seen[trace_id] += 1
        entry = entries.get(trace_id) if trace_id is not None else None
        if not matches or entry is None or entry.split != split:
            wrong.add(trace_id)
    expected = {trace_id for trace_id, entry in entries.items() if entry.split == split}
    problems = []
    if wrong:
        problems.append(f'{name} holds traces that are not its {split} entries: {_ids(wrong)}')
    if twice := {trace_id for trace_id, count in seen.items() if count > 1}:
        problems.append(f'{name} holds traces more than once: {_ids(twice)}')
    if missing := expected - seen.keys():
        problems.append(f'{name} lacks {split} traces: {_ids(missing)}')
    return problems


def _ids(ids: Iterable[str | None]) -> list[str]:
    return sorted(str(trace_id) for trace_id in ids)


def _walk(folder: Path) -> tuple[list[str], list[str]]:
    """Return the regular files under *folder* as sorted POSIX paths relative to it, and
    what else is there: symlinks (never followed), other non-regular files, empty folders.
    Regular files named in :data:`IGNORED` are skipped; a folder, link or anything else by
    that name is reported like any other entry, so the name can't hide content."""
    files: list[str] = []
    problems: list[str] = []
    pending = [folder]
    while pending:
        directory = pending.pop()
        children = sorted(
            path
            for path in directory.iterdir()
            if not (path.name in IGNORED and stat.S_ISREG(path.lstat().st_mode))
        )
        if not children and directory != folder:
            problems.append(f'empty directory: {directory.relative_to(folder).as_posix()}')
        for path in children:
            mode = path.lstat().st_mode
            name = path.relative_to(folder).as_posix()
            if stat.S_ISLNK(mode):
                problems.append(f'symlink: {name}')
            elif stat.S_ISDIR(mode):
                pending.append(path)
            elif stat.S_ISREG(mode):
                files.append(name)
            else:
                problems.append(f'not a regular file: {name}')
    return sorted(files), problems


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing a key given twice (a reader could take either)."""
    keys = Counter(key for key, _ in pairs)
    if repeated := sorted(key for key, count in keys.items() if count > 1):
        raise ValueError(f'duplicate keys {repeated}')
    return dict(pairs)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(data: Mapping[str, object]) -> str:
    """The manifest's content hash: of every field but ``content_sha256``."""
    return canonical_sha256({key: value for key, value in data.items() if key != 'content_sha256'})
