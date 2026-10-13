"""Axolotl fine-tuning framework integration.

Axolotl is a config-driven LLM post-training framework that supports GRPO,
DPO, SFT, LoRA, PRM training, and more -- all from a single YAML file.

This module provides:
- Reward functions for Axolotl's GRPO/RL training loops
- Config generation helpers for Axolotl YAML files

Since Axolotl wraps TRL's GRPOTrainer internally, the reward function interface
matches TRL's signature.

Usage (GRPO reward function):
    from deepgym.integrations.axolotl import make_axolotl_reward_fn

    reward_fn = make_axolotl_reward_fn(env=env)

    # In your Axolotl custom training script:
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn])

Usage (config generation):
    from deepgym.integrations.axolotl import generate_axolotl_config

    config = generate_axolotl_config(
        base_model='Qwen/Qwen3.5-Coder-7B',
        env=env,
        method='grpo',
    )

Sources:
    - https://github.com/axolotl-ai-cloud/axolotl
    - https://axolotlai.substack.com/p/process-reward-models
    - https://docs.axolotl.ai/
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal

from deepgym.core import DeepGym
from deepgym.models import Environment

# ---------------------------------------------------------------------------
# GRPO / RL reward functions
# ---------------------------------------------------------------------------


def make_axolotl_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
    max_parallel: int = 32,
) -> Callable[..., list[float]]:
    """Create a reward function for Axolotl's GRPO training.

    Axolotl uses TRL's GRPOTrainer internally, so the reward function
    follows TRL's interface: ``reward_func(completions, **kwargs) -> list[float]``.

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).
        max_parallel: Maximum concurrent sandbox executions.

    Returns:
        Callable matching Axolotl/TRL reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(completions: list[str], **kwargs: object) -> list[float]:
        """Score completions via sandboxed execution.

        Args:
            completions: List of decoded model output strings.
            **kwargs: Additional args from Axolotl/TRL (prompts, etc.)

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        batch = _dg.run_batch(
            env,
            completions,
            max_parallel=min(len(completions), max_parallel),
            **kwargs,
        )
        return [r.score for r in batch.results]

    return reward_fn


def make_axolotl_async_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
    max_parallel: int = 32,
) -> Callable[..., Awaitable[list[float]]]:
    """Create an async reward function for Axolotl's GRPO training.

    Async reward functions run concurrently when Axolotl provides multiple
    reward functions, reducing latency for I/O-bound sandbox execution.

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).
        max_parallel: Maximum concurrent sandbox executions.

    Returns:
        Async callable matching Axolotl/TRL reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    async def reward_fn(completions: list[str], **kwargs: object) -> list[float]:
        """Score completions asynchronously via sandboxed execution.

        Args:
            completions: List of decoded model output strings.
            **kwargs: Additional args from Axolotl/TRL.

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        loop = asyncio.get_running_loop()
        batch = await loop.run_in_executor(
            None,
            lambda: _dg.run_batch(
                env,
                completions,
                max_parallel=min(len(completions), max_parallel),
                **kwargs,
            ),
        )
        return [r.score for r in batch.results]

    return reward_fn


def make_axolotl_shaped_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
    max_parallel: int = 32,
    component: str | None = None,
) -> Callable[..., list[float]]:
    """Create a reward function that returns a specific reward component.

    When verifiers emit shaped reward components (correctness, efficiency,
    style), this function extracts a single component for use as a dedicated
    reward signal. Axolotl supports multiple reward functions -- use one
    per component for multi-signal GRPO training.

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).
        max_parallel: Maximum concurrent sandbox executions.
        component: Reward component key to extract (e.g. 'correctness').
            If None, returns the aggregate score.

    Returns:
        Callable matching Axolotl/TRL reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(completions: list[str], **kwargs: object) -> list[float]:
        """Score completions and extract a specific reward component.

        Args:
            completions: List of decoded model output strings.
            **kwargs: Additional args from Axolotl/TRL.

        Returns:
            List of component scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        batch = _dg.run_batch(
            env,
            completions,
            max_parallel=min(len(completions), max_parallel),
            **kwargs,
        )
        if component is None:
            return [r.score for r in batch.results]
        return [(r.reward_components or {}).get(component, r.score) for r in batch.results]

    return reward_fn


# ---------------------------------------------------------------------------
# Axolotl YAML config generation
# ---------------------------------------------------------------------------

_GRPO_CONFIG_TEMPLATE = """\
# Axolotl GRPO config with DeepGym reward scoring
# Generated by deepgym.integrations.axolotl
#
# Usage:
#   pip install axolotl[flash-attn,deepspeed] deepgym
#   axolotl train {config_filename}

base_model: {base_model}

rl: grpo

# DeepGym reward function -- loaded via custom reward module
# See: https://github.com/DeepGym/deepgym/wiki/Integrations#axolotl
reward_model: null

datasets:
  - path: {dataset_path}
    type: {dataset_type}
    split: train

val_set_size: 0.02

sequence_len: {sequence_len}
sample_packing: false

micro_batch_size: {micro_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
num_epochs: {num_epochs}
learning_rate: {learning_rate}
lr_scheduler: cosine

optimizer: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0
warmup_ratio: 0.03

bf16: auto
tf32: true

flash_attention: true
gradient_checkpointing: true

logging_steps: 1
evals_per_epoch: 4
save_strategy: epoch

deepspeed: null
fsdp: null
"""

_PRM_CONFIG_TEMPLATE = """\
# Axolotl PRM (Process Reward Model) config
# Trains on a stepwise_supervised dataset (prompt, completions, labels) you provide
# Generated by deepgym.integrations.axolotl
#
# Usage:
#   pip install axolotl[flash-attn,deepspeed] deepgym
#   axolotl train {config_filename}

base_model: {base_model}

model_type: AutoModelForTokenClassification
num_labels: 2
process_reward_model: true

datasets:
  - path: {dataset_path}
    type: stepwise_supervised
    step_separator: "{step_separator}"
    train_on_last_step_only: false
    split: train

val_set_size: 0.01
evals_per_epoch: 10

sequence_len: {sequence_len}
sample_packing: false

micro_batch_size: {micro_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
num_epochs: {num_epochs}
learning_rate: {learning_rate}
lr_scheduler: cosine

optimizer: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0
warmup_ratio: 0.03

bf16: auto
tf32: true

flash_attention: true
gradient_checkpointing: true

logging_steps: 1
save_strategy: epoch
"""


def generate_axolotl_config(
    base_model: str,
    method: Literal['grpo', 'prm'] = 'grpo',
    dataset_path: str = 'data/train.jsonl',
    dataset_type: str = 'completion',
    step_separator: str = '\\n\\n',
    sequence_len: int = 4096,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 5e-6,
    config_filename: str = 'axolotl_config.yaml',
) -> str:
    """Generate an Axolotl YAML config for DeepGym-powered training.

    Args:
        base_model: HuggingFace model identifier (e.g. 'Qwen/Qwen3.5-Coder-7B').
        method: Training method -- 'grpo' for RL or 'prm' for process reward model.
        dataset_path: Path to the training dataset.
        dataset_type: Axolotl dataset type (e.g. 'completion', 'stepwise_supervised').
        step_separator: PRM step separator string (only for method='prm').
        sequence_len: Maximum sequence length.
        micro_batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        num_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        config_filename: Filename for the generated config (used in comments).

    Returns:
        YAML config string ready to write to a file.
    """
    params = {
        'base_model': base_model,
        'dataset_path': dataset_path,
        'dataset_type': dataset_type,
        'step_separator': step_separator,
        'sequence_len': sequence_len,
        'micro_batch_size': micro_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'config_filename': config_filename,
    }

    if method == 'prm':
        return _PRM_CONFIG_TEMPLATE.format(**params)
    return _GRPO_CONFIG_TEMPLATE.format(**params)
