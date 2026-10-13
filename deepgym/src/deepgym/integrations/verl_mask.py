"""verl single-turn agent loop that keeps unscored samples out of training.

When a sandbox or verifier fails, :func:`deepgym.integrations.verl.compute_score`
returns 0.0 with ``error: 1.0``, and verl trains on that 0.0. A reward function or
reward manager can't drop the sample: ``run_single`` returns only ``reward_score`` and
``reward_extra_info`` (verl/experimental/reward_loop/reward_manager/naive.py:99), and
only those two reach the trajectory (verl/experimental/agent_loop/agent_loop.py:1044).

An agent loop can, so this loop scores the response itself (verl then skips its own
reward step because ``reward_score`` is set, agent_loop.py:989) and keeps a sample with
``error`` 1.0 out of training:

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
watching the ``error`` metric and the failed-sample logs. Line numbers are from verl
main at 76e837e.

Agent loop config (``actor_rollout_ref.rollout.agent.agent_loop_config_path``)::

    - name: deepgym_single_turn
      _target_: deepgym.integrations.verl_mask.DeepGymSingleTurnAgentLoop
      mode: daytona   # DeepGym backend, as in compute_score's reward_kwargs

then set ``actor_rollout_ref.rollout.agent.default_agent_loop=deepgym_single_turn``.
Dataset rows are the ones :func:`~deepgym.integrations.verl.verl_rows` builds.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from deepgym.exceptions import DeepGymError
from deepgym.integrations.verl import compute_score
from deepgym.sandboxes import Provider

if TYPE_CHECKING:
    from omegaconf import DictConfig


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
        self.mode = mode

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Generate one response and score it; keep it out of training if scoring failed.

        Raises:
            DeepGymError: If scoring failed under verl's v1 trainer (see :func:`exclude_unscored`).
        """
        output = await super().run(sampling_params, **kwargs)
        result = await asyncio.to_thread(self._score, output.response_ids, kwargs)
        output.reward_score = result['score']
        output.extra_fields['reward_extra_info'] = result
        if result['error'] == 1.0:
            exclude_unscored(output, self.config, f'scoring {kwargs["data_source"]} failed')
        return output

    def _score(self, response_ids: list[int], row: dict[str, Any]) -> dict[str, float]:
        """Decode the response and score it with the environment its dataset row names."""
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return compute_score(
            row['data_source'], response, extra_info=row.get('extra_info'), mode=self.mode
        )
