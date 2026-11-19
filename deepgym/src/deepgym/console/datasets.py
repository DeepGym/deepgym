"""Dataset routes: build datasets from the console's traces and labels into the datasets
directory chosen at launch, list them, and verify one against its manifest.

Clients name a dataset by its folder in that directory, never by a path, and no answer
holds a path on the server: the directory's own path reads ``<datasets>``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from deepgym.console.deps import DatasetsDirDep, EpisodeStoreDep, LabelStoreDep, strict_query
from deepgym.console.schemas import (
    DatasetBuild,
    DatasetBuilt,
    DatasetListing,
    DatasetPage,
    DatasetVerification,
    DatasetVerify,
    LabelQuery,
)
from deepgym.dataset import (
    MANIFEST,
    DatasetManifest,
    DatasetMismatchError,
    build_dataset,
    checked_write,
    manifest_sha256,
    verify_dataset,
    write_dataset,
)
from deepgym.exceptions import DeepGymError
from deepgym.labels import SQLITE_MAX_INT

DEFAULT_DATASETS_DIR = Path.home() / '.deepgym' / 'datasets'

router = APIRouter(prefix='/datasets', tags=['datasets'])
_NO_QUERY = [Depends(strict_query())]


@router.post(
    '/build', response_model=DatasetBuilt, status_code=status.HTTP_201_CREATED,
    dependencies=_NO_QUERY,
)  # fmt: skip
def build(
    body: DatasetBuild, episodes: EpisodeStoreDep, labels: LabelStoreDep, root: DatasetsDirDep
) -> DatasetBuilt:
    """Build a dataset from every trace the console holds that reads as one and every label
    record, into the folder ``<name>-<version>`` of the datasets directory. Stored traces
    that don't read as one are counted in ``counts.unread`` and listed.

    Raises:
        HTTPException: 409 if that folder exists and isn't empty (or another build wrote it
            meanwhile), the datasets directory can't be written, Parquet is asked for
            without pyarrow, no stored trace reads as one, or the build refuses the traces
            (see ``build_dataset``).
    """
    out = root / body.folder
    try:
        checked_write(out, name=body.name, version=body.version, formats=body.formats)
        held = episodes.held_traces()
        if not held.traces:
            left_out = len(held.conflicts) + len(held.unreadable)
            detail = 'The console holds no traces' + (
                f' that read as one ({left_out} left out)' if left_out else ''
            )
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
        dataset = build_dataset(
            held.traces,
            body.spec,
            labels.page(LabelQuery(), 0, None).labels,
            {episode: trace_id for trace_id, ids in held.episodes.items() for episode in ids},
            {
                'episode_without_trace': held.untraced,
                'merged_copy': held.merged,
                'conflicting_copy': held.conflicting,
                'unreadable_trace': len(held.unreadable),
            },
        )
        sha256 = write_dataset(
            dataset, out, name=body.name, version=body.version, formats=body.formats
        )
    except (DeepGymError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=_public(str(exc), root)
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Can't write {body.folder} in the datasets directory: {_reason(exc)}",
        ) from exc
    return DatasetBuilt(
        dataset=body.folder,
        sha256=sha256,
        counts=dataset.counts,
        conflicts=list(held.conflicts),
        unreadable=list(held.unreadable),
    )


@router.get('', response_model=DatasetPage, dependencies=[Depends(strict_query('offset', 'limit'))])
def list_datasets(
    root: DatasetsDirDep,
    offset: int = Query(0, ge=0, le=SQLITE_MAX_INT),
    limit: int | None = Query(None, ge=1, le=SQLITE_MAX_INT),
) -> DatasetPage:
    """List the folders of the datasets directory by name, each with its manifest as read
    (not verified), a page at a time (every one without *limit*). Hidden folders, such as
    a build's staging folder, and symlinks are left out.

    Raises:
        HTTPException: 409 if the datasets directory can't be read.
    """
    try:
        entries = list(root.iterdir()) if root.is_dir() else []
        folders = sorted(
            path.name
            for path in entries
            if not path.name.startswith('.') and stat.S_ISDIR(path.lstat().st_mode)
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Can't read the datasets directory: {_reason(exc)}",
        ) from exc
    page = folders[offset:] if limit is None else folders[offset : offset + limit]
    return DatasetPage(datasets=[_listing(root, name) for name in page], total=len(folders))


def _listing(root: Path, name: str) -> DatasetListing:
    """Return folder *name* with its manifest, or why it has none that reads."""
    folder = root / name
    sha256: str | None = None
    try:
        sha256 = manifest_sha256(folder)
        manifest = DatasetManifest.model_validate_json((folder / MANIFEST).read_bytes())
    except DeepGymError as exc:
        error = _public(str(exc), root)
    except ValidationError as exc:
        error = (
            f'{MANIFEST} is not a dataset manifest ({exc.error_count()} problems); '
            'verify the dataset to list them'
        )
    except OSError as exc:
        error = f"Can't read {MANIFEST}: {_reason(exc)}"
    else:
        return DatasetListing(dataset=name, sha256=sha256, manifest=manifest)
    return DatasetListing(dataset=name, sha256=sha256, manifest=None, error=error)


@router.post('/verify', response_model=DatasetVerification, dependencies=_NO_QUERY)
def verify(body: DatasetVerify, root: DatasetsDirDep) -> DatasetVerification:
    """Check a dataset folder against its manifest, and the manifest against ``sha256``
    when given; a mismatch is a result (``valid`` false, with every problem), not an error.

    Raises:
        HTTPException: 404 if the datasets directory has no such folder (a symlink isn't
            one), 409 if the datasets directory can't be read.
    """
    folder = root / body.dataset
    try:
        is_folder = stat.S_ISDIR(folder.lstat().st_mode)
    except (FileNotFoundError, NotADirectoryError):
        is_folder = False
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Can't read the datasets directory: {_reason(exc)}",
        ) from exc
    if not is_folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Dataset {body.dataset} not found'
        )
    sha256: str | None = None
    try:
        sha256 = manifest_sha256(folder)
        manifest = verify_dataset(folder, sha256=body.sha256)
    except DatasetMismatchError as exc:
        problems = [_public(problem, root) for problem in exc.problems]
    except DeepGymError as exc:
        problems = [_public(str(exc), root)]
    except ValidationError as exc:
        problems = [
            f'{MANIFEST} is not a dataset manifest: {".".join(map(str, error["loc"]))}: '
            f'{error["msg"]}'
            for error in exc.errors(include_url=False)
        ]
    except OSError as exc:
        problems = [f"Can't read the dataset: {_reason(exc)}"]
    else:
        return DatasetVerification(
            dataset=body.dataset, valid=True, sha256=sha256, manifest=manifest, problems=[]
        )
    return DatasetVerification(
        dataset=body.dataset, valid=False, sha256=sha256, manifest=None, problems=problems
    )


def _public(text: str, root: Path) -> str:
    """Return *text* with paths in the datasets directory relative to it, and the
    directory itself as ``<datasets>``, so no server path reaches a client."""
    return text.replace(f'{root}{os.sep}', '').replace(str(root), '<datasets>')


def _reason(exc: OSError) -> str:
    """The operating system's reason for *exc*, without the path it names."""
    return exc.strerror or type(exc).__name__
