"""Tests for CyberBench seed-spec generation utilities."""

from __future__ import annotations

import json
from pathlib import Path

from deepgym.cyberbench import (
    classify_vulnerability_family,
    cyber_seed_from_hf_row,
    glm_seed_prompt,
    spec_from_glm_json,
    summarize_rows,
    validate_seed_spec,
    write_seed_specs,
)


def _row(description: str = 'A heap buffer overflow occurs in parser input handling.') -> dict:
    return {
        'task_id': 'arvo:1',
        'project_name': 'toyproj',
        'project_main_repo': 'https://example.invalid/toy.git',
        'project_language': 'c',
        'vulnerability_description': description,
    }


def test_classifies_common_vulnerability_families() -> None:
    assert classify_vulnerability_family('heap buffer overflow in parser') == 'memory-safety'
    assert classify_vulnerability_family('SQL injection in query handler') == 'input-validation'
    assert classify_vulnerability_family('authentication bypass') == 'auth-access-control'


def test_builds_valid_seed_from_cybergym_row() -> None:
    spec = cyber_seed_from_hf_row(_row(), index=0)

    assert spec.seed_id == 'cybergym_arvo_1_memory_safety'
    assert spec.family == 'memory-safety'
    assert spec.reward_components['evidence_or_patch_correctness'] == 0.45
    assert validate_seed_spec(spec) == []


def test_glm_prompt_keeps_generation_synthetic() -> None:
    prompt = glm_seed_prompt(_row())

    assert 'LOCAL, SYNTHETIC' in prompt
    assert 'Do not provide real exploit steps' in prompt
    assert 'CyberGym metadata' in prompt


def test_parses_glm_json_with_fallback_source_fields() -> None:
    text = json.dumps(
        {
            'seed_id': 'custom_seed',
            'title': 'Synthetic parser patch task',
            'family': 'parser-state',
            'language': 'c',
            'difficulty': 'medium',
            'task_type': 'patch-repo',
            'safe_objective': 'Patch a synthetic local parser bug.',
            'reward_components': {'reasoning': 0.3, 'patch': 0.5, 'safety': 0.2},
            'verifier_checks': ['synthetic crash fixed', 'regression passes'],
            'safety_constraints': ['local synthetic targets only'],
        }
    )

    spec = spec_from_glm_json(text, row=_row(), index=2)

    assert spec.seed_id == 'custom_seed'
    assert spec.source == 'huggingface:cybergym+zai'
    assert spec.source_task_id == 'arvo:1'
    assert validate_seed_spec(spec) == []


def test_write_seed_specs_jsonl(tmp_path: Path) -> None:
    spec = cyber_seed_from_hf_row(_row(), index=0)
    output = tmp_path / 'seeds.jsonl'

    write_seed_specs([spec], output)

    loaded = json.loads(output.read_text(encoding='utf-8').strip())
    assert loaded['seed_id'] == spec.seed_id


def test_summarize_rows_counts_families() -> None:
    summary = summarize_rows([
        _row('heap buffer overflow'),
        _row('path traversal in file endpoint'),
    ])

    assert summary['total'] == 2
    assert summary['families']['memory-safety'] == 1
    assert summary['families']['input-validation'] == 1
