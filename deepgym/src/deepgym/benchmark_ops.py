"""Benchmark manifest and contamination audit utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from deepgym.models import Environment
from deepgym.reward_qa import fingerprint_verifier, resolve_verifier_source

BenchmarkSplit = Literal['public_train', 'public_eval', 'private_holdout', 'canary']


class BenchmarkManifestEntry(BaseModel):
    """One environment entry in a benchmark manifest."""

    env_id: str
    split: BenchmarkSplit
    task_hash: str
    verifier_hash: str
    test_cases_hash: str
    content_hash: str
    provenance: str = 'registered'


class BenchmarkLeak(BaseModel):
    """Potential contamination or leakage finding across benchmark splits."""

    leak_type: Literal['task', 'verifier', 'content']
    fingerprint: str
    env_ids: list[str] = Field(default_factory=list)
    splits: list[BenchmarkSplit] = Field(default_factory=list)
    severity: Literal['warning', 'high'] = 'warning'
    details: str


class BenchmarkAuditReport(BaseModel):
    """Audit report for benchmark split hygiene and provenance."""

    benchmark: str
    total_environments: int
    split_counts: dict[BenchmarkSplit, int]
    contamination_risk: bool
    duplicate_task_groups: list[list[str]] = Field(default_factory=list)
    duplicate_verifier_groups: list[list[str]] = Field(default_factory=list)
    leaks: list[BenchmarkLeak] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    manifest: list[BenchmarkManifestEntry] = Field(default_factory=list)


def load_environments_from_dir(root: Path) -> dict[str, Environment]:
    """Recursively load environment directories from disk."""
    environments: dict[str, Environment] = {}
    for task_path in sorted(root.rglob('task.md')):
        env_dir = task_path.parent
        verifier_path = env_dir / 'verifier.py'
        if not verifier_path.exists():
            continue
        env_id = env_dir.relative_to(root).as_posix()
        environments[env_id] = Environment(
            task=task_path.read_text(encoding='utf-8'),
            verifier_path=verifier_path,
        )
    return environments


def assign_splits(
    environment_ids: list[str],
    *,
    seed: int = 0,
    public_eval_ratio: float = 0.2,
    holdout_ratio: float = 0.1,
    canary_ratio: float = 0.05,
    split_overrides: dict[str, BenchmarkSplit] | None = None,
) -> dict[str, BenchmarkSplit]:
    """Assign deterministic splits across a set of environment IDs."""
    if public_eval_ratio < 0 or holdout_ratio < 0 or canary_ratio < 0:
        raise ValueError('Split ratios must be non-negative.')
    if public_eval_ratio + holdout_ratio + canary_ratio >= 1.0:
        raise ValueError('Split ratios must sum to less than 1.0.')

    split_overrides = split_overrides or {}
    unknown = set(split_overrides) - set(environment_ids)
    if unknown:
        raise ValueError(f'Unknown environment ids in split_overrides: {sorted(unknown)}')

    ordered = sorted(
        (env_id for env_id in environment_ids if env_id not in split_overrides),
        key=lambda env_id: hashlib.sha256(f'{seed}:{env_id}'.encode()).hexdigest(),
    )
    total = len(ordered)

    canary_count = _bucket_size(total, canary_ratio)
    holdout_count = _bucket_size(total - canary_count, holdout_ratio)
    eval_count = _bucket_size(total - canary_count - holdout_count, public_eval_ratio)

    assigned: dict[str, BenchmarkSplit] = dict(split_overrides)
    index = 0

    for _ in range(canary_count):
        assigned[ordered[index]] = 'canary'
        index += 1
    for _ in range(holdout_count):
        assigned[ordered[index]] = 'private_holdout'
        index += 1
    for _ in range(eval_count):
        assigned[ordered[index]] = 'public_eval'
        index += 1
    while index < total:
        assigned[ordered[index]] = 'public_train'
        index += 1

    return assigned


def build_benchmark_audit(
    environments: dict[str, Environment],
    *,
    benchmark: str = 'custom',
    provenance: str = 'registered',
    seed: int = 0,
    public_eval_ratio: float = 0.2,
    holdout_ratio: float = 0.1,
    canary_ratio: float = 0.05,
    split_overrides: dict[str, BenchmarkSplit] | None = None,
) -> BenchmarkAuditReport:
    """Build a manifest and audit it for split leakage risks."""
    env_ids = list(environments)
    splits = assign_splits(
        env_ids,
        seed=seed,
        public_eval_ratio=public_eval_ratio,
        holdout_ratio=holdout_ratio,
        canary_ratio=canary_ratio,
        split_overrides=split_overrides,
    )

    manifest: list[BenchmarkManifestEntry] = []
    task_groups: dict[str, list[str]] = {}
    verifier_groups: dict[str, list[str]] = {}
    content_groups: dict[str, list[str]] = {}

    for env_id in env_ids:
        env = environments[env_id]
        task_hash = _hash_text(_normalize_text(env.task))
        verifier_hash = fingerprint_verifier(resolve_verifier_source(env))
        test_cases_hash = _hash_json(env.test_cases or [])
        content_hash = _hash_text(f'{task_hash}:{verifier_hash}:{test_cases_hash}')

        manifest.append(
            BenchmarkManifestEntry(
                env_id=env_id,
                split=splits[env_id],
                task_hash=task_hash,
                verifier_hash=verifier_hash,
                test_cases_hash=test_cases_hash,
                content_hash=content_hash,
                provenance=provenance,
            )
        )
        task_groups.setdefault(task_hash, []).append(env_id)
        verifier_groups.setdefault(verifier_hash, []).append(env_id)
        content_groups.setdefault(content_hash, []).append(env_id)

    duplicate_task_groups = _sorted_duplicate_groups(task_groups)
    duplicate_verifier_groups = _sorted_duplicate_groups(verifier_groups)
    leaks = _find_leaks(manifest, task_groups, verifier_groups, content_groups)

    recommendations = []
    if duplicate_task_groups:
        recommendations.append('Deduplicate tasks across public and private splits.')
    if duplicate_verifier_groups:
        recommendations.append('Rotate or diversify verifier logic across split boundaries.')
    if leaks:
        recommendations.append(
            'Regenerate holdout and canary splits after removing leaked content.'
        )
    if not recommendations:
        recommendations.append(
            'Keep private holdouts hidden and rerun this audit after each import.'
        )

    split_counts: dict[BenchmarkSplit, int] = {
        'public_train': 0,
        'public_eval': 0,
        'private_holdout': 0,
        'canary': 0,
    }
    for entry in manifest:
        split_counts[entry.split] += 1

    return BenchmarkAuditReport(
        benchmark=benchmark,
        total_environments=len(manifest),
        split_counts=split_counts,
        contamination_risk=bool(leaks),
        duplicate_task_groups=duplicate_task_groups,
        duplicate_verifier_groups=duplicate_verifier_groups,
        leaks=leaks,
        recommendations=recommendations,
        manifest=sorted(manifest, key=lambda entry: entry.env_id),
    )


def _bucket_size(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    size = int(total * ratio)
    return max(1, size)


def _normalize_text(text: str) -> str:
    return ' '.join(text.split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _hash_json(value: object) -> str:
    return _hash_text(json.dumps(value, sort_keys=True, separators=(',', ':')))


def _sorted_duplicate_groups(groups: dict[str, list[str]]) -> list[list[str]]:
    duplicates = [sorted(env_ids) for env_ids in groups.values() if len(env_ids) > 1]
    duplicates.sort()
    return duplicates


def _find_leaks(
    manifest: list[BenchmarkManifestEntry],
    task_groups: dict[str, list[str]],
    verifier_groups: dict[str, list[str]],
    content_groups: dict[str, list[str]],
) -> list[BenchmarkLeak]:
    manifest_by_id = {entry.env_id: entry for entry in manifest}
    leaks: list[BenchmarkLeak] = []

    for leak_type, groups in (
        ('task', task_groups),
        ('verifier', verifier_groups),
        ('content', content_groups),
    ):
        for fingerprint, env_ids in groups.items():
            if len(env_ids) < 2:
                continue
            entries = [manifest_by_id[env_id] for env_id in env_ids]
            splits = {entry.split for entry in entries}
            private = {'private_holdout', 'canary'}
            public = {'public_train', 'public_eval'}
            if splits & private and splits & public:
                leaks.append(
                    BenchmarkLeak(
                        leak_type=leak_type,
                        fingerprint=fingerprint,
                        env_ids=sorted(env_ids),
                        splits=sorted(splits),
                        severity='high' if leak_type in {'task', 'content'} else 'warning',
                        details=(
                            f'{leak_type} fingerprint {fingerprint} appears in both '
                            'public and private splits.'
                        ),
                    )
                )

    leaks.sort(key=lambda leak: (leak.leak_type, leak.fingerprint, leak.env_ids))
    return leaks
