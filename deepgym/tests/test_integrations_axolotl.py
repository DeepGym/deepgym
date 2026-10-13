"""Tests for deepgym.integrations.axolotl.

Axolotl itself needs Python 3.12+ and a CUDA-era dependency stack, so it isn't installed
here. These tests follow its code instead: it resolves ``trl.reward_funcs`` entries with
importlib (GRPOStrategy.get_reward_func, axolotl 0.20), loads a local .jsonl dataset with
``datasets.load_dataset('json', ...)`` and leaves a GRPO dataset without ``type`` as is
(utils/data/shared.py, utils/data/rl.py), and TRL then passes every dataset column to the
reward function (GRPOTrainer._calculate_rewards).
"""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml

from deepgym import load_environment
from deepgym.core import DeepGym
from deepgym.integrations import axolotl
from deepgym.integrations.axolotl import (
    generate_axolotl_config,
    make_axolotl_async_reward_fn,
    make_axolotl_reward_fn,
    make_axolotl_shaped_reward_fn,
    row_reward_fn,
)
from deepgym.integrations.verl import verl_rows
from deepgym.models import Environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
COIN_CHANGE = (ENVS_DIR / 'coin_change' / 'reference_solution.py').read_text(encoding='utf-8')
REVERSE = 'def reverse_string(s: str) -> str:\n    return s[::-1]\n'


def _conversation(answer: str) -> list[dict]:
    """A TRL conversational completion: the assistant's turns as message dicts."""
    return [{'role': 'assistant', 'content': f'Here it is:\n```python\n{answer}```'}]


def _call_like_trl(reward_func: Callable, rows: Sequence[dict], completions: list) -> list:
    """Call *reward_func* as TRL's GRPOTrainer._calculate_rewards does."""
    columns = [key for key in rows[0] if key not in ('prompt', 'completion', 'completion_ids')]
    reward_kwargs = {key: [row[key] for row in rows] for key in columns}
    return reward_func(
        prompts=[row['prompt'] for row in rows],
        completions=completions,
        completion_ids=[[1, 2, 3]] * len(rows),
        trainer_state=None,
        log_extra=lambda *args, **kwargs: None,
        log_metric=lambda *args, **kwargs: None,
        **reward_kwargs,
    )


@pytest.fixture()
def reverse_env() -> Environment:
    return load_environment('reverse_string')  # type: ignore[return-value]


async def test_reward_fns_score_conversational_completions(
    reverse_env: Environment, local_dg: DeepGym
) -> None:
    """TRL hands conversational datasets' completions over as message lists."""
    completions = [_conversation(REVERSE), REVERSE]

    sync_scores = make_axolotl_reward_fn(reverse_env, local_dg)(completions=completions)
    async_scores = await make_axolotl_async_reward_fn(reverse_env, local_dg)(
        completions=completions
    )
    shaped = make_axolotl_shaped_reward_fn(reverse_env, local_dg)(completions=completions)

    assert sync_scores == async_scores == shaped == [1.0, 1.0]


def test_no_verdict_is_none(local_dg: DeepGym) -> None:
    broken = Environment(task='t', verifier_code='raise SystemExit(3)\n', verifier_mode='script')

    assert make_axolotl_reward_fn(broken, local_dg)(completions=['x']) == [None]


def test_generated_grpo_config_scores_with_deepgym() -> None:
    config = yaml.safe_load(generate_axolotl_config(base_model='Qwen/Qwen2.5-0.5B-Instruct'))

    assert config['rl'] == 'grpo'
    reward_funcs = config['trl']['reward_funcs']
    assert reward_funcs == ['deepgym.integrations.axolotl.row_reward_fn']
    for fqn in reward_funcs:  # as GRPOStrategy.get_reward_func resolves them
        module_name, _, name = fqn.rpartition('.')
        func = getattr(importlib.import_module(module_name), name)
        assert len(inspect.signature(func).parameters) >= 2
        assert func is row_reward_fn
    # verl_rows already have a chat prompt; a `type` would send them through a transform.
    assert config['datasets'] == [{'path': 'data/train.jsonl', 'split': 'train'}]


def test_prm_config_keeps_its_stepwise_dataset() -> None:
    config = yaml.safe_load(generate_axolotl_config(base_model='m', method='prm'))

    assert config['datasets'][0]['type'] == 'stepwise_supervised'
    assert 'trl' not in config


def test_row_reward_fn_scores_each_row_with_its_environment(
    tmp_path: Path, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows saved as JSON lines and loaded the way Axolotl loads them, then scored like TRL."""
    datasets = pytest.importorskip('datasets')
    monkeypatch.setattr(axolotl, '_client', lambda: local_dg)
    path = tmp_path / 'train.jsonl'
    rows = verl_rows(['coin_change', 'reverse_string', 'coin_change'])
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    dataset = datasets.load_dataset('json', data_files=str(path), split='train')

    completions = [_conversation(COIN_CHANGE), _conversation(REVERSE), 'pass']
    scores = _call_like_trl(row_reward_fn, list(dataset), completions)

    assert scores[:2] == [1.0, 1.0]
    assert scores[2] == 0.0
