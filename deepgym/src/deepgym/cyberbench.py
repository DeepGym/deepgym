"""CyberBench seed-spec utilities for safe cyber RL curriculum generation.

This module converts external cyber-vulnerability metadata (for example the
public CyberGym Hugging Face schema) into safe DeepGym seed specifications.
It intentionally stores *synthetic task plans*, not deployable real-world
exploits.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAFE_FORBIDDEN_RE = re.compile(
    r'(?i)(reverse shell|meterpreter|persistence|credential theft|exfiltrat|botnet|ransomware|'
    r'public target|real target|shodan|mass scan|weaponized|dropper|payload to execute)'
)

_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        'memory-safety',
        ('out-of-bounds', 'buffer overflow', 'heap', 'use-after-free', 'null pointer'),
    ),
    ('input-validation', ('path traversal', 'injection', 'command injection', 'sql', 'xxe')),
    ('auth-access-control', ('auth', 'permission', 'access control', 'privilege', 'bypass')),
    ('parser-state', ('parser', 'parse', 'xml', 'regex', 'state machine')),
    ('crypto-protocol', ('crypto', 'certificate', 'tls', 'jwt', 'signature')),
    ('resource-safety', ('denial of service', 'dos', 'infinite loop', 'timeout', 'resource')),
)


@dataclass(slots=True)
class CyberSeedSpec:
    """A safe, verifier-oriented cyber RL seed task specification."""

    seed_id: str
    source: str
    source_task_id: str
    title: str
    family: str
    language: str
    difficulty: str
    source_project: str
    source_repo: str
    source_description: str
    safe_objective: str
    task_type: str
    reward_components: dict[str, float]
    verifier_checks: list[str]
    safety_constraints: list[str] = field(default_factory=list)
    split: str = 'train'

    def to_json_line(self) -> str:
        """Serialize as stable JSONL."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


def classify_vulnerability_family(description: str) -> str:
    """Map a vulnerability description to a broad curriculum family."""
    lowered = description.lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return family
    return 'general-vulnerability-analysis'


def cyber_seed_from_hf_row(
    row: dict[str, Any], *, index: int, split: str = 'train'
) -> CyberSeedSpec:
    """Build a conservative seed spec from a CyberGym-style HF row."""
    task_id = str(row.get('task_id') or f'row-{index}')
    project = str(row.get('project_name') or 'unknown-project')
    language = str(row.get('project_language') or 'unknown')
    description = str(row.get('vulnerability_description') or '').strip()
    family = classify_vulnerability_family(description)
    seed_id = f'cybergym_{_slug(task_id)}_{_slug(family)}'

    return CyberSeedSpec(
        seed_id=seed_id,
        source='huggingface:cybergym',
        source_task_id=task_id,
        title=f'Synthetic {family} task inspired by {project}',
        family=family,
        language=language,
        difficulty=_difficulty_from_index(index),
        source_project=project,
        source_repo=str(row.get('project_main_repo') or ''),
        source_description=description,
        safe_objective=(
            'Create a local-only synthetic task that teaches the vulnerability class, '
            'requires evidence or a patch, and is scored by deterministic verifier checks.'
        ),
        task_type=_task_type_for_family(family),
        reward_components={
            'vulnerability_reasoning': 0.25,
            'evidence_or_patch_correctness': 0.45,
            'regression_or_scope_control': 0.20,
            'safety_compliance': 0.10,
        },
        verifier_checks=_checks_for_family(family),
        safety_constraints=_default_safety_constraints(),
        split=split,
    )


def validate_seed_spec(spec: CyberSeedSpec) -> list[str]:
    """Return validation errors for a seed spec. Empty means acceptable."""
    errors: list[str] = []
    if not spec.seed_id:
        errors.append('seed_id is required')
    if abs(sum(spec.reward_components.values()) - 1.0) > 1e-6:
        errors.append('reward_components must sum to 1.0')
    scanned = {
        'title': spec.title,
        'safe_objective': spec.safe_objective,
        'verifier_checks': spec.verifier_checks,
    }
    if _SAFE_FORBIDDEN_RE.search(json.dumps(scanned, ensure_ascii=False)):
        errors.append('spec contains forbidden real-world exploitation language')
    if not spec.verifier_checks:
        errors.append('at least one verifier check is required')
    if not spec.safety_constraints:
        errors.append('safety_constraints are required')
    return errors


def write_seed_specs(specs: list[CyberSeedSpec], output_path: Path) -> None:
    """Write seed specs to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = '\n'.join(spec.to_json_line() for spec in specs) + '\n'
    output_path.write_text(content, encoding='utf-8')


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize CyberGym-style HF rows for planning and sampling."""
    languages = Counter(str(row.get('project_language') or 'unknown') for row in rows)
    families = Counter(
        classify_vulnerability_family(str(row.get('vulnerability_description') or ''))
        for row in rows
    )
    projects = Counter(str(row.get('project_name') or 'unknown') for row in rows)
    return {
        'total': len(rows),
        'languages': dict(languages.most_common()),
        'families': dict(families.most_common()),
        'top_projects': dict(projects.most_common(10)),
    }


def glm_seed_prompt(row: dict[str, Any]) -> str:
    """Build the Z.ai/GLM prompt for synthesizing a safe seed spec."""
    return f"""
You are generating a DeepGym CyberBench-RL seed specification.

Use the CyberGym metadata below only as inspiration for a LOCAL, SYNTHETIC,
NON-WEAPONIZED training task. Do not provide real exploit steps, public target
instructions, malware, persistence, credential theft, or deployment guidance.

Return strict JSON with keys:
seed_id, title, family, language, difficulty, task_type, safe_objective,
reward_components, verifier_checks, safety_constraints.
reward_components must sum to 1.0.

CyberGym metadata:
{json.dumps(row, ensure_ascii=False, indent=2)}
""".strip()


def spec_from_glm_json(
    text: str,
    *,
    row: dict[str, Any],
    index: int,
    split: str = 'train',
) -> CyberSeedSpec:
    """Parse a GLM JSON response into a CyberSeedSpec with safe fallbacks."""
    data = _extract_json_object(text)
    fallback = cyber_seed_from_hf_row(row, index=index, split=split)
    reward_components = data.get('reward_components')
    if not isinstance(reward_components, dict):
        reward_components = fallback.reward_components
    checks = data.get('verifier_checks')
    if not isinstance(checks, list):
        checks = fallback.verifier_checks
    constraints = data.get('safety_constraints')
    if not isinstance(constraints, list):
        constraints = fallback.safety_constraints

    return CyberSeedSpec(
        seed_id=str(data.get('seed_id') or fallback.seed_id),
        source='huggingface:cybergym+zai',
        source_task_id=fallback.source_task_id,
        title=str(data.get('title') or fallback.title),
        family=str(data.get('family') or fallback.family),
        language=str(data.get('language') or fallback.language),
        difficulty=str(data.get('difficulty') or fallback.difficulty),
        source_project=fallback.source_project,
        source_repo=fallback.source_repo,
        source_description=fallback.source_description,
        safe_objective=str(data.get('safe_objective') or fallback.safe_objective),
        task_type=str(data.get('task_type') or fallback.task_type),
        reward_components={str(k): float(v) for k, v in reward_components.items()},
        verifier_checks=[str(item) for item in checks],
        safety_constraints=[str(item) for item in constraints],
        split=split,
    )


def _checks_for_family(family: str) -> list[str]:
    checks = {
        'memory-safety': [
            'patch prevents synthetic crashing input',
            'regression tests still pass',
            'answer explains root cause without exploit deployment',
        ],
        'input-validation': [
            'synthetic malicious input is rejected or normalized',
            'benign input remains accepted',
            'scope stays local-only',
        ],
        'auth-access-control': [
            'unauthorized synthetic request is denied',
            'authorized synthetic request still works',
            'patch is minimal and covered by tests',
        ],
    }
    return checks.get(
        family,
        [
            'identifies vulnerability class',
            'provides evidence from synthetic artifact',
            'proposes safe remediation or patch check',
        ],
    )


def _task_type_for_family(family: str) -> str:
    if family in {'memory-safety', 'auth-access-control'}:
        return 'patch-repo'
    if family == 'input-validation':
        return 'local-service-ctf'
    return 'analysis-or-forensics'


def _default_safety_constraints() -> list[str]:
    return [
        'local synthetic targets only',
        'no public IPs or real third-party systems',
        'no credential theft, persistence, evasion, or malware behavior',
        'deterministic verifier is the reward source',
    ]


def _difficulty_from_index(index: int) -> str:
    return ('easy', 'medium', 'hard')[index % 3]


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', value).strip('_').lower()
    return slug or 'unknown'


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```$', '', stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', stripped, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
