"""Versioned datasets of graded traces: choose, write and verify them.

:func:`build_dataset` keeps the traces a :class:`DatasetSpec` asks for and records why it
dropped each other one (see :mod:`deepgym.dataset.build`); :func:`write_dataset` exports
them per split with a manifest, and :func:`verify_dataset` checks a folder against it (see
:mod:`deepgym.dataset.manifest`).
"""

from deepgym.dataset.build import (
    Dataset,
    DatasetCounts,
    DatasetEntry,
    DatasetSpec,
    DropReason,
    DuplicateGroup,
    LabelMatch,
    Split,
    TaskBand,
    UnreadReason,
    build_dataset,
)
from deepgym.dataset.manifest import (
    MANIFEST,
    DatasetManifest,
    DatasetMismatchError,
    checked_write,
    manifest_sha256,
    verify_dataset,
    write_dataset,
)

__all__ = [
    'MANIFEST',
    'Dataset',
    'DatasetCounts',
    'DatasetEntry',
    'DatasetManifest',
    'DatasetMismatchError',
    'DatasetSpec',
    'DropReason',
    'DuplicateGroup',
    'LabelMatch',
    'Split',
    'TaskBand',
    'UnreadReason',
    'build_dataset',
    'checked_write',
    'manifest_sha256',
    'verify_dataset',
    'write_dataset',
]
