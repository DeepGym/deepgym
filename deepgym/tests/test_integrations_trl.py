"""DeepGym's TRL reward function inside a real GRPOTrainer step (needs trl>=1.6 and torch).

A scripted rollout gives one prompt four conversational completions: two right, one
wrong and one the verifier crashes on. TRL must train on a real 0 for the wrong one and
leave the crash out: no advantage for it, and no say in its siblings' baseline.

TRL 1.15 computes log-probs with Triton kernels, so its GRPO step needs Linux and a GPU;
on CPU this runs with TRL 1.14, whose reward and advantage code is the same.
"""

from __future__ import annotations

import math
from itertools import cycle
from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.integrations.trl import make_trl_reward_fn
from deepgym.models import Environment

trl = pytest.importorskip('trl')
datasets = pytest.importorskip('datasets')

TINY_MODEL = 'trl-internal-testing/tiny-Qwen2ForCausalLM-2.5'
ANSWERS = ['right', 'right', 'wrong', 'boom']

MIXED = Environment(
    task='Say right.',
    verifier_code=(
        'source = open(solution_path).read()\n'
        "if 'boom' in source:  # a crash after reading the answer would be the answer's 0\n"
        "    return {'score': 0.0, 'passed': False, 'error_type': 'runtime_error'}\n"
        "return 1.0 if 'right' in source else 0.0\n"
    ),
)
"""Judges 'right' 1.0 and 'wrong' 0.0, and reports its own error on 'boom': no verdict."""


def _scripted_rollout(prompts: list, trainer: object) -> dict:
    """Answer the (repeated) prompt with ANSWERS in order, as tokens of the trainer's model."""
    tokenizer = trainer.processing_class  # type: ignore[attr-defined]
    answers = cycle(ANSWERS)
    texts = [
        tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    prompt_ids = [tokenizer(text, add_special_tokens=False)['input_ids'] for text in texts]
    completion_ids = [
        tokenizer.encode(next(answers), add_special_tokens=False) + [tokenizer.eos_token_id]
        for _ in prompts
    ]
    logprobs = [[0.0] * len(ids) for ids in completion_ids]
    return {'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'logprobs': logprobs}


def test_grpo_trains_on_a_real_zero_and_leaves_a_crash_out(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    reward_fn = make_trl_reward_fn(MIXED, local_dg)
    dataset = datasets.Dataset.from_list(
        [{'prompt': [{'role': 'user', 'content': 'Say right.'}]}] * len(ANSWERS)
    )
    args = trl.GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=len(ANSWERS),
        num_generations=len(ANSWERS),
        max_completion_length=8,
        max_steps=1,
        report_to='none',
        use_cpu=True,
    )
    trainer = trl.GRPOTrainer(
        model=TINY_MODEL,
        reward_funcs=[reward_fn],
        args=args,
        train_dataset=dataset,
        rollout_func=_scripted_rollout,
    )

    trainer.train()

    rewards = list(trainer._logs['rewards']['reward_fn'])
    advantages = list(trainer._logs['advantages'])
    assert rewards[:3] == [1.0, 1.0, 0.0]
    assert math.isnan(rewards[3])
    # The baseline is the mean and std of [1, 1, 0]; the unjudged answer gets no advantage.
    mean, std = 2 / 3, math.sqrt(1 / 3)
    expected = [(1 - mean) / (std + 1e-4)] * 2 + [(0 - mean) / (std + 1e-4), 0.0]
    assert advantages == pytest.approx(expected, abs=1e-4)
