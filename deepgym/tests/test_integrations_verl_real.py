"""DeepGym's verl reward function, agent loop and DAPO config under verl's own code.

Needs verl (written against 0.9.1) with its ray, tensordict, omegaconf and hydra-core
dependencies. CI doesn't install verl, so this runs where it is installed; the stand-in
tests cover the agent loops everywhere.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip('verl')

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.utils import get_class  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop  # noqa: E402
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager  # noqa: E402
from verl.trainer.ppo.reward import get_custom_reward_fn  # noqa: E402
from verl.workers.config import AgentLoopConfig  # noqa: E402

import deepgym.integrations  # noqa: E402
from deepgym.exceptions import NoVerdictError  # noqa: E402
from deepgym.integrations import verl as verl_reward  # noqa: E402
from deepgym.integrations.dapo import generate_dapo_verl_config  # noqa: E402
from deepgym.models import Environment  # noqa: E402

MIXED = Environment(
    task='Say right.',
    verifier_code=(
        'source = open(solution_path).read()\n'
        "if 'boom' in source:\n"
        "    raise RuntimeError('verifier bug')\n"
        "return 1.0 if 'right' in source else 0.0\n"
    ),
)
"""Judges 'wrong' 0.0 (a real zero) and crashes on 'boom': no verdict."""


@pytest.fixture()
def verl_mask(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """deepgym.integrations.verl_mask on the real verl, not a copy the stand-in tests left."""
    monkeypatch.delitem(sys.modules, 'deepgym.integrations.verl_mask', raising=False)
    monkeypatch.delattr(deepgym.integrations, 'verl_mask', raising=False)
    return importlib.import_module('deepgym.integrations.verl_mask')


class _Tokenizer:
    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = False) -> str:
        return ''.join(chr(int(i)) for i in ids)


def _sample(response: str) -> DataProto:
    """One rollout as verl's agent loop hands it to the reward manager."""
    responses = torch.tensor([[ord(char) for char in response]])
    prompts = torch.tensor([[1, 2]])
    return DataProto.from_dict(
        tensors={
            'prompts': prompts,
            'responses': responses,
            'attention_mask': torch.ones(1, prompts.size(1) + responses.size(1), dtype=torch.long),
        },
        non_tensors={
            'data_source': np.array(['coin_change'], dtype=object),
            'reward_model': np.array([{'ground_truth': ''}], dtype=object),
            'extra_info': np.array([{'environment': 'coin_change'}], dtype=object),
        },
    )


async def test_verls_reward_manager_gets_no_reward_for_a_run_without_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """verl loads compute_score from the package, as the README's config says, and calls it.

    Nothing in run_single catches the raise, so under verl's v1 trainer it fails the
    sample's agent-loop session, which drops the sample.
    """
    monkeypatch.setattr(verl_reward, '_environment', lambda name, variant: MIXED)
    config = OmegaConf.create(
        {
            'reward': {
                'custom_reward_function': {
                    'path': 'pkg://deepgym.integrations.verl',
                    'name': 'compute_score',
                    'reward_kwargs': {'mode': 'local'},
                }
            }
        }
    )
    manager = NaiveRewardManager(config, _Tokenizer(), get_custom_reward_fn(config))

    judged = await manager.run_single(_sample('wrong'))

    assert judged == {
        'reward_score': 0.0,
        'reward_extra_info': {'score': 0.0, 'passed': 0.0, 'error': 0.0},
    }
    with pytest.raises(NoVerdictError, match='runtime_error'):
        await manager.run_single(_sample('boom'))


def test_generated_dapo_config_selects_an_agent_loop_verl_loads(
    verl_mask: types.ModuleType,
) -> None:
    """verl's own config schema takes the agent section, and its file names the loop."""
    config = OmegaConf.create(generate_dapo_verl_config(train_files='train.parquet'))
    agent = config.actor_rollout_ref.rollout.agent
    AgentLoopConfig(**agent)
    # As AgentLoopWorker loads agent_loop_config_path and instantiates the default loop.
    loops = {loop.name: loop for loop in OmegaConf.load(agent.agent_loop_config_path)}

    loop = get_class(loops[agent.default_agent_loop]['_target_'])

    assert loop is verl_mask.DeepGymSingleTurnAgentLoop
    assert issubclass(loop, SingleTurnAgentLoop)


def test_a_reward_module_file_runs_once_for_every_loop(
    verl_mask: types.ModuleType, tmp_path: Path
) -> None:
    """verl builds an agent loop per sample and runs a reward module file on every load."""
    module = tmp_path / 'reward_module.py'
    module.write_text(
        'import pathlib\n'
        "with pathlib.Path(__file__).with_suffix('.runs').open('a') as runs:\n"
        "    runs.write('run\\n')\n"
        'from deepgym import DeepGym\n'
        'from deepgym.integrations.verl import make_verl_compute_score\n'
        'from deepgym.models import Environment\n'
        "always = Environment(task='t', verifier_code='return 1.0')\n"
        "compute_score = make_verl_compute_score(always, DeepGym(mode='local'))\n",
        encoding='utf-8',
    )
    reward = {'custom_reward_function': {'path': str(module), 'name': 'compute_score'}}

    scorers = [
        verl_mask._reward_function(OmegaConf.create({'reward': reward}), 'auto') for _ in range(3)
    ]

    assert scorers[0] is scorers[1] is scorers[2]
    assert module.with_suffix('.runs').read_text(encoding='utf-8') == 'run\n'
    assert scorers[0](data_source='x', solution_str='y') == {
        'score': 1.0,
        'passed': 1.0,
        'error': 0.0,
    }
