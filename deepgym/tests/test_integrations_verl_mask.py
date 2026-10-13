"""DeepGym's single-turn verl agent loop against a stand-in for verl (verl isn't installed here).

The stand-in SingleTurnAgentLoop does what verl's does for this loop's purposes: one
generation, returned as token ids with a response_mask of ones, and the trainer config
as ``self.config``. verl's ``get_custom_reward_fn`` returns the configured reward function
with its ``reward_kwargs`` bound; the stand-in config holds that function as is. Scoring is
real (local DeepGym).
"""

import asyncio
import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from deepgym import load_environment
from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError
from deepgym.integrations import verl as verl_reward
from deepgym.integrations.dapo import generate_dapo_verl_config
from deepgym.models import MODEL_FAULT_ERRORS, Environment


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return ''.join(chr(i) for i in ids)


class _TrainerConfig:
    """verl's trainer config (a DictConfig, hashable) as far as the loop reads it."""

    def __init__(self, use_v1: bool, reward_fn: Callable | None) -> None:
        # verl >= 0.9 defaults trainer.use_v1 to true.
        self.trainer = {'use_v1': use_v1}
        self.reward = {'custom_reward_function': reward_fn}


class _SingleTurnAgentLoop:
    """Mimic verl's SingleTurnAgentLoop: one response, every token trainable."""

    def __init__(self, response: str, use_v1: bool, reward_fn: Callable | None = None) -> None:
        self.tokenizer = _Tokenizer()
        self.response = response
        self.config = _TrainerConfig(use_v1, reward_fn)  # AgentLoopBase keeps it

    async def run(self, sampling_params: dict, **kwargs: object) -> types.SimpleNamespace:
        ids = self.tokenizer.encode(self.response)
        return types.SimpleNamespace(
            response_ids=ids, response_mask=[1] * len(ids), reward_score=None, extra_fields={}
        )


@pytest.fixture()
def verl(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    modules = {
        'verl.experimental.agent_loop.agent_loop': {
            'AgentLoopOutput': types.SimpleNamespace,
            'register': lambda name: lambda cls: cls,
        },
        'verl.experimental.agent_loop.single_turn_agent_loop': {
            'SingleTurnAgentLoop': _SingleTurnAgentLoop
        },
        'verl.trainer.ppo.reward': {
            'get_custom_reward_fn': lambda config: config.reward['custom_reward_function']
        },
    }
    for name in (
        'verl',
        'verl.experimental',
        'verl.experimental.agent_loop',
        'verl.trainer',
        'verl.trainer.ppo',
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, 'deepgym.integrations.verl_mask', raising=False)
    return importlib.import_module('deepgym.integrations.verl_mask')


def _rollout(
    verl: types.ModuleType, response: str, use_v1: bool = True, reward_fn: Callable | None = None
) -> types.SimpleNamespace:
    loop = verl.DeepGymSingleTurnAgentLoop(response, use_v1, reward_fn, mode='local')
    row = {'data_source': 'coin_change', 'extra_info': {'environment': 'coin_change'}}
    return asyncio.run(loop.run({}, raw_prompt=[], priority=0, **row))


@pytest.fixture()
def broken_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Score every row with a verifier that exits without a verdict (a verifier_error)."""
    broken = Environment(task='t', verifier_code='raise SystemExit(3)\n', verifier_mode='script')
    monkeypatch.setattr(verl_reward, '_environment', lambda name, variant: broken)


@pytest.mark.parametrize('use_v1', [True, False])
def test_scored_sample_keeps_its_mask(verl: types.ModuleType, use_v1: bool) -> None:
    reference = load_environment('coin_change').verifier_path.with_name('reference_solution.py')
    output = _rollout(verl, f'```python\n{reference.read_text()}```', use_v1)

    assert output.reward_score == 1.0
    assert output.extra_fields['reward_extra_info'] == {'score': 1.0, 'passed': 1.0, 'error': 0.0}
    assert set(output.response_mask) == {1}


def test_wrong_answer_is_trained_on(verl: types.ModuleType) -> None:
    output = _rollout(verl, '```python\ndef coin_change(coins, amount):\n    return -7\n```')

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0
    assert set(output.response_mask) == {1}


@pytest.mark.usefixtures('broken_verifier')
def test_failed_scoring_drops_the_sample_under_the_v1_trainer(verl: types.ModuleType) -> None:
    """verl v1 records the raising session's group as failed and trains on the rest.

    The failed sample writes no trajectory, so its 0.0 can't enter the group's GRPO mean
    and std (verl/trainer/ppo/v1/agent_loop_tq.py:136-146, utils.py:148).
    """
    with pytest.raises(DeepGymError, match='No verdict'):
        _rollout(verl, 'anything', use_v1=True)


@pytest.mark.usefixtures('broken_verifier')
def test_failed_scoring_masks_the_sample_under_the_legacy_trainer(verl: types.ModuleType) -> None:
    output = _rollout(verl, 'anything', use_v1=False)

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 1.0
    assert output.response_mask == [0] * len('anything')


@pytest.mark.parametrize('error_type', sorted(MODEL_FAULT_ERRORS))
def test_broken_patch_is_trained_on_as_zero(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch, error_type: str
) -> None:
    verdict = f"{{'score': 0.0, 'passed': False, 'details': 'x', 'error_type': '{error_type}'}}"
    broken_patch = Environment(task='t', verifier_code=f'return {verdict}\n')
    monkeypatch.setattr(verl_reward, '_environment', lambda name, variant: broken_patch)

    output = _rollout(verl, 'not a diff')

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0
    assert set(output.response_mask) == {1}


def test_the_loop_scores_with_the_configured_reward_function(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reward.custom_reward_function may hold one environment (make_verl_compute_score).

    The loop scores with it, as verl's reward step would, keeps its failures out too, and
    reports each sample to the masked-rate warning.
    """
    reports: list[bool] = []
    recorder = types.SimpleNamespace(record=lambda *, masked: reports.append(masked))
    monkeypatch.setattr(verl, '_MASKED', recorder)
    local = DeepGym(mode='local')
    always = Environment(task='t', verifier_code='return 1.0')
    passes = verl_reward.make_verl_compute_score(always, local)
    broken = Environment(task='t', verifier_code='raise SystemExit(3)\n', verifier_mode='script')
    fails = verl_reward.make_verl_compute_score(broken, local)

    scored = _rollout(verl, 'anything', reward_fn=passes)
    masked = _rollout(verl, 'anything', use_v1=False, reward_fn=fails)

    assert scored.extra_fields['reward_extra_info'] == {'score': 1.0, 'passed': 1.0, 'error': 0.0}
    assert masked.extra_fields['reward_extra_info'] == {'score': 0.0, 'passed': 0.0, 'error': 1.0}
    assert masked.response_mask == [0] * len('anything')
    with pytest.raises(DeepGymError, match='No verdict'):
        _rollout(verl, 'anything', use_v1=True, reward_fn=fails)
    assert reports == [False, True, True]


async def _async_reward(**kwargs: object) -> dict:
    return {'score': 1.0, 'passed': True}


@pytest.mark.parametrize(
    ('reward_fn', 'expected'),
    [
        (lambda **kwargs: 0.5, {'score': 0.5, 'passed': 0.0, 'error': 0.0}),
        (lambda **kwargs: {'score': 1.0, 'acc': 1.0}, {'score': 1.0, 'passed': 1.0, 'error': 0.0}),
        (_async_reward, {'score': 1.0, 'passed': 1.0, 'error': 0.0}),
    ],
    ids=['float', 'other_keys', 'async'],
)
def test_every_reward_function_verl_allows_gives_one_schema(
    verl: types.ModuleType, reward_fn: Callable, expected: dict
) -> None:
    """verl takes a float or a dict with any keys, from a sync or async function.

    It batches reward_extra_info by the first sample's keys, so a scored sample must have
    exactly the keys a sample kept out of training gets.
    """
    output = _rollout(verl, 'anything', reward_fn=reward_fn)

    assert output.reward_score == expected['score']
    assert output.extra_fields['reward_extra_info'] == expected


def test_generated_dapo_config_scores_with_this_loop(verl: types.ModuleType) -> None:
    """The DAPO config must pick the agent loop that masks, not verl's own reward step."""
    config = yaml.safe_load(generate_dapo_verl_config(train_files='train.parquet'))
    agent = config['actor_rollout_ref']['rollout']['agent']
    loops = yaml.safe_load(Path(agent['agent_loop_config_path']).read_text(encoding='utf-8'))
    target = {loop['name']: loop['_target_'] for loop in loops}[agent['default_agent_loop']]
    module, _, name = target.rpartition('.')

    assert getattr(importlib.import_module(module), name) is verl.DeepGymSingleTurnAgentLoop
