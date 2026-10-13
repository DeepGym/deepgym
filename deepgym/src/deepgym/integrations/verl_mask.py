"""verl single-turn agent loop that keeps unscored samples out of training.

verl's reward-manager contract can't mark a sample as unscored: ``run_single`` returns
only ``reward_score`` and ``reward_extra_info``
(verl/experimental/reward_loop/reward_manager/naive.py), and only those reach the
trajectory (``AgentLoopWorker._compute_score``). So DeepGym's reward functions
(:func:`~deepgym.integrations.verl.compute_score` and those from
:func:`~deepgym.integrations.verl.make_verl_compute_score`) raise
:class:`~deepgym.exceptions.NoVerdictError` for a run whose sandbox or verifier failed.
verl's v1 trainer computes the reward inside the sample's agent-loop session
(``AgentLoopWorkerTQ._agent_loop_postprocess``), so the raise drops the sample as
described below; the legacy trainer can't drop a sample, so it stops.

This agent loop scores the response itself, with the reward function verl is configured
with (``reward.custom_reward_function`` and its ``reward_kwargs``, as verl's own reward
step would; DeepGym's ``compute_score`` if none is set). verl then skips its own reward
step because ``reward_score`` is set. A sample whose reward function raised
``NoVerdictError`` is kept out of training:

- With verl's v1 trainer (``trainer.use_v1``, the default since verl 0.9) the loop
  raises. verl marks the prompt's group as failed (verl/trainer/ppo/v1/agent_loop_tq.py:143)
  and the failed sample writes no trajectory, so GRPO computes its group's mean and std
  from the samples that got a verdict (verl/trainer/ppo/v1/utils.py:148). The sync trainer
  trains on the rest of the group; the async trainers drop the group and sample a new
  prompt (verl/trainer/ppo/v1/replay_buffer.py:107-113).
- With the legacy trainer, where an exception would stop training, the loop zeroes the
  sample's ``response_mask`` instead. The mask is also the ``loss_mask``, so the sample
  adds nothing to the policy, entropy or KL loss (verl/workers/utils/losses.py:98) and is
  left out of their normalizers (verl/workers/engine/fsdp/transformer_impl.py:734,
  verl/trainer/ppo/core_algos.py:1183); its GRPO advantage is zero (core_algos.py:329).
  But its 0.0 still counts toward its group's GRPO mean and std (core_algos.py:313-321),
  so its siblings' advantages shift as if it had failed.

Either way a policy that learns to trigger failures escapes their penalty, so keep
watching the ``error`` metric and the failed-sample logs; the loop logs a warning when more
than 5% of a batch is kept out (:class:`~deepgym.integrations.reward.MaskedRate`). Line
numbers in the two items are from verl main at 76e837e; the v1 ones still match verl 0.9.1.

Agent loop config (``actor_rollout_ref.rollout.agent.agent_loop_config_path``)::

    - name: deepgym_single_turn
      _target_: deepgym.integrations.verl_mask.DeepGymSingleTurnAgentLoop
      mode: daytona   # DeepGym backend, used when reward.custom_reward_function isn't set

then set ``actor_rollout_ref.rollout.agent.default_agent_loop=deepgym_single_turn``.
Dataset rows are the ones :func:`~deepgym.integrations.verl.verl_rows` builds.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.trainer.ppo.reward import get_custom_reward_fn

from deepgym.exceptions import DeepGymError, NoVerdictError
from deepgym.integrations.reward import MaskedRate
from deepgym.integrations.verl import compute_score
from deepgym.sandboxes import Provider

if TYPE_CHECKING:
    from omegaconf import DictConfig

_MASKED = MaskedRate('verl deepgym_single_turn')


def exclude_unscored(output: AgentLoopOutput, trainer_config: DictConfig, why: str) -> None:
    """Keep a sample that got no verdict out of training (see the module docstring).

    Args:
        output: The sample's agent loop output.
        trainer_config: verl's trainer config (the agent loop's ``self.config``).
        why: What failed, for the error.

    Raises:
        DeepGymError: Under verl's v1 trainer, which then drops the sample from its group.
    """
    if trainer_config.trainer.get('use_v1', False):
        raise DeepGymError(f'No verdict, so verl drops this sample: {why}')
    output.response_mask = [0] * len(output.response_mask)


@functools.cache
def _reward_function(
    trainer_config: DictConfig, mode: Literal['auto', 'daytona', 'local'] | Provider
) -> Callable[..., object]:
    """Return the reward function verl is configured with, loaded once per trainer config.

    verl builds an agent loop for every sample, and loading a reward module from a file
    runs it again each time, which would rebuild its environment and DeepGym client.
    """
    return get_custom_reward_fn(trainer_config) or functools.partial(compute_score, mode=mode)


@register('deepgym_single_turn')
class DeepGymSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn loop scored by DeepGym; samples that got no verdict are kept out of training."""

    def __init__(
        self,
        *args: Any,
        mode: Literal['auto', 'daytona', 'local'] | Provider = 'auto',
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.compute_score = _reward_function(self.config, mode)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Generate one response and score it; keep it out of training if scoring failed.

        Raises:
            DeepGymError: If scoring failed under verl's v1 trainer (see :func:`exclude_unscored`).
        """
        output = await super().run(sampling_params, **kwargs)
        try:
            result = await self._score(output.response_ids, kwargs)
        except NoVerdictError as exc:
            _MASKED.record(masked=True)
            output.reward_score = 0.0
            output.extra_fields['reward_extra_info'] = {'score': 0.0, 'passed': 0.0, 'error': 1.0}
            exclude_unscored(output, self.config, f'scoring {kwargs["data_source"]}: {exc.reason}')
            return output
        _MASKED.record(masked=False)
        output.reward_score = result['score']
        output.extra_fields['reward_extra_info'] = result
        return output

    async def _score(self, response_ids: list[int], row: dict[str, Any]) -> dict[str, float]:
        """Decode the response and score it as verl's reward step would.

        Returns:
            The same keys for every sample, kept out or not, since verl batches
            ``reward_extra_info`` by the first sample's keys: ``score``, ``passed`` (the
            result's own, else whether the score is full) and ``error`` 0.0. A reward function
            may return a float or a dict and may be async, as verl allows; other keys it
            returns are dropped.
        """
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        score = functools.partial(
            self.compute_score,
            data_source=row['data_source'],
            solution_str=response,
            ground_truth=(row.get('reward_model') or {}).get('ground_truth'),
            extra_info=row.get('extra_info'),
        )
        if inspect.iscoroutinefunction(self.compute_score):
            result = await score()
        else:
            result = await asyncio.to_thread(score)
        if not isinstance(result, Mapping):
            result = {'score': result}
        value = float(result['score'])
        return {'score': value, 'passed': float(result.get('passed', value == 1.0)), 'error': 0.0}
