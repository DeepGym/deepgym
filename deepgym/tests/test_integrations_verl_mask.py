"""DeepGym's masking verl agent loop against a stand-in for verl (verl isn't installed here).

The stand-in SingleTurnAgentLoop does what verl's does for this loop's purposes: one
generation, returned as token ids with a response_mask of ones. Scoring is real
(local DeepGym).
"""

import asyncio
import importlib
import sys
import types

import pytest

from deepgym import load_environment
from deepgym.integrations import verl as verl_reward
from deepgym.models import Environment


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return ''.join(chr(i) for i in ids)


class _SingleTurnAgentLoop:
    """Mimic verl's SingleTurnAgentLoop: one response, every token trainable."""

    def __init__(self, response: str) -> None:
        self.tokenizer = _Tokenizer()
        self.response = response

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
    }
    for name in ('verl', 'verl.experimental', 'verl.experimental.agent_loop'):
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


def _rollout(verl: types.ModuleType, response: str) -> types.SimpleNamespace:
    loop = verl.DeepGymSingleTurnAgentLoop(response, mode='local')
    row = {'data_source': 'coin_change', 'extra_info': {'environment': 'coin_change'}}
    return asyncio.run(loop.run({}, raw_prompt=[], priority=0, **row))


def test_scored_sample_keeps_its_mask(verl: types.ModuleType) -> None:
    reference = load_environment('coin_change').verifier_path.with_name('reference_solution.py')
    output = _rollout(verl, f'```python\n{reference.read_text()}```')

    assert output.reward_score == 1.0
    assert output.extra_fields['reward_extra_info'] == {'score': 1.0, 'passed': 1.0, 'error': 0.0}
    assert set(output.response_mask) == {1}


def test_wrong_answer_is_trained_on(verl: types.ModuleType) -> None:
    output = _rollout(verl, '```python\ndef coin_change(coins, amount):\n    return -7\n```')

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0
    assert set(output.response_mask) == {1}


def test_failed_scoring_masks_the_sample_out(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exits without printing a verdict: run_batch reports a verifier_error.
    broken = Environment(task='t', verifier_code='raise SystemExit(3)\n', verifier_mode='script')
    monkeypatch.setattr(verl_reward, '_environment', lambda name, variant: broken)

    output = _rollout(verl, 'anything')

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 1.0
    assert output.response_mask == [0] * len('anything')


@pytest.mark.parametrize('error_type', ['invalid_patch', 'patch_apply_failed'])
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
