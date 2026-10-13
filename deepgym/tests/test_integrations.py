"""Tests for deepgym.integrations -- TRL, verl, DAPO, OpenRLHF integration wrappers."""

import asyncio
import logging
import math
import threading
import time
from pathlib import Path
from typing import Literal

import httpx
import pytest
from anyio import to_thread
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
from deepgym.exceptions import NoVerdictError, SandboxError, VerifierError
from deepgym.exceptions import TimeoutError as DeepGymTimeout
from deepgym.integrations import openrlhf
from deepgym.integrations import verl as verl_module
from deepgym.integrations.dapo import (
    generate_dapo_reward_module,
    generate_dapo_verl_config,
    make_dapo_async_reward_fn,
    make_dapo_reward_fn,
    make_dapo_shaped_reward_fn,
)
from deepgym.integrations.openrlhf import (
    OpenRLHFRewardRequest,
    OpenRLHFRewardResponse,
    create_openrlhf_router,
)
from deepgym.integrations.reward import (
    MASKED_BATCH,
    AsyncRewardFunction,
    MaskedRate,
    RewardFunction,
)
from deepgym.integrations.trl import make_trl_async_reward_fn, make_trl_reward_fn
from deepgym.integrations.verl import (
    compute_score,
    extract_code,
    make_verl_compute_score,
    make_verl_reward_fn,
    verl_rows,
)
from deepgym.models import Environment, RunResult

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
REVERSE_DIR = ENVS_DIR / 'reverse_string'

GOOD_SOLUTION = 'def reverse_string(s: str) -> str:\n    return s[::-1]\n'
BAD_SOLUTION = '# no function\npass\n'
FENCED = f'Here you go:\n```python\n{GOOD_SOLUTION}```\n'

MIXED = Environment(
    task='Say right.',
    verifier_code=(
        'source = open(solution_path).read()\n'
        "if 'boom' in source:\n"
        "    raise RuntimeError('verifier bug')\n"
        "return {'score': 1.0 if 'right' in source else 0.0, 'passed': 'right' in source,\n"
        "        'reward_components': {'style': 0.5}}\n"
    ),
)
"""Judges 'right' 1.0 and 'wrong' 0.0 (a real zero), and crashes on 'boom': no verdict."""


@pytest.fixture()
def env() -> Environment:
    """Return the reverse_string environment for integration tests."""
    return Environment(
        task=(REVERSE_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=REVERSE_DIR / 'verifier.py',
        difficulty='easy',
    )


@pytest.fixture()
def local_dg() -> DeepGym:
    """Return a local-mode DeepGym client."""
    return DeepGym(mode='local')


class TestTRLIntegration:
    """Verify TRL reward function wrappers."""

    def test_make_trl_reward_fn_returns_callable(self, env: Environment) -> None:
        fn = make_trl_reward_fn(env=env)
        assert callable(fn)

    def test_trl_reward_fn_empty_list(self, env: Environment) -> None:
        fn = make_trl_reward_fn(env=env)
        assert fn(completions=[]) == []

    def test_trl_reward_fn_scores_good_solution(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_trl_reward_fn(env=env, dg=local_dg)
        scores = fn(completions=[GOOD_SOLUTION])
        assert len(scores) == 1
        assert scores[0] >= 0.9

    def test_trl_reward_fn_scores_bad_lower(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_trl_reward_fn(env=env, dg=local_dg)
        scores = fn(completions=[GOOD_SOLUTION, BAD_SOLUTION])
        assert scores[0] > scores[1]

    def test_make_trl_async_reward_fn_returns_callable(self, env: Environment) -> None:
        fn = make_trl_async_reward_fn(env=env)
        assert callable(fn)

    async def test_trl_reward_fns_score_every_completion_format_alike(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        """Plain, fenced, conversational and tool-calling completions of one answer score alike."""
        fenced = f'Here you go:\n```python\n{GOOD_SOLUTION}```'
        tool_call = {'type': 'function', 'function': {'name': 'run', 'arguments': {}}}
        completions = [
            GOOD_SOLUTION,
            fenced,
            [{'role': 'assistant', 'content': GOOD_SOLUTION}],
            [{'role': 'assistant', 'content': [{'type': 'text', 'text': fenced}]}],
            [
                {'role': 'assistant', 'content': '', 'tool_calls': [tool_call]},
                {'role': 'tool', 'name': 'run', 'content': 'ok'},
                {'role': 'assistant', 'content': fenced},
            ],
        ]

        sync_scores = make_trl_reward_fn(env=env, dg=local_dg)(completions=completions)
        async_scores = await make_trl_async_reward_fn(env=env, dg=local_dg)(completions=completions)

        assert sync_scores == async_scores == [1.0] * len(completions)

    async def test_no_verdict_is_none_and_a_judged_zero_stays_zero(self, local_dg: DeepGym) -> None:
        """TRL 1.6+ leaves None out of the group baseline; a 0.0 would read as a failed answer."""
        completions = ['right', 'wrong', 'boom']

        sync_scores = make_trl_reward_fn(MIXED, local_dg)(completions=completions)
        async_scores = await make_trl_async_reward_fn(MIXED, local_dg)(completions=completions)
        styles = make_trl_reward_fn(MIXED, local_dg, component='style')(completions=completions)

        assert sync_scores == async_scores == [1.0, 0.0, None]
        assert styles == [0.5, 0.5, None]


class TestVerlIntegration:
    """Verify verl reward function wrappers."""

    def test_make_verl_compute_score_returns_callable(self, env: Environment) -> None:
        fn = make_verl_compute_score(env=env)
        assert callable(fn)

    def test_verl_compute_score_returns_what_compute_score_does(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        fn = make_verl_compute_score(env=env, dg=local_dg)
        assert fn('dataset', GOOD_SOLUTION, '') == {'score': 1.0, 'passed': 1.0, 'error': 0.0}

    def test_no_verdict_never_reaches_verl_as_a_reward(
        self, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verl's reward functions can't mark a sample unscored, so a failed run raises.

        verl's v1 trainer then drops the sample; the batch function marks it None.
        """
        monkeypatch.setattr(verl_module, '_environment', lambda name, variant: MIXED)
        monkeypatch.setattr(verl_module, '_client', lambda mode: local_dg)
        fixed = make_verl_compute_score(MIXED, local_dg)

        judged_zero = {'score': 0.0, 'passed': 0.0, 'error': 0.0}
        assert fixed('ds', 'wrong') == compute_score('ds', 'wrong') == judged_zero
        for score in (fixed, compute_score):
            with pytest.raises(NoVerdictError, match='runtime_error') as raised:
                score('ds', 'boom')
            # verl's legacy trainer stops on the raise; the message says what to use instead.
            assert 'deepgym_single_turn' in str(raised.value)
        batch = make_verl_reward_fn(MIXED, local_dg)({'responses': ['right', 'wrong', 'boom']})
        assert batch == [1.0, 0.0, None]

    def test_a_solution_that_hangs_is_a_judged_zero(
        self, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hang is the policy's fault, so the bare reward functions score it 0.

        They raise only for a run without a verdict, which would stop verl's legacy
        trainer; a policy must not be able to cause that by hanging.
        """
        hangs = Environment(
            task='Return 42 from solve().',
            verifier_code=(
                'from deepgym_solution import load_solution\n\n'
                'def verify(solution_path, test_cases_path=None):\n'
                '    try:\n'
                '        return float(load_solution(solution_path).solve() == 42)\n'
                '    except Exception:  # the solution failed: here it ran out of time\n'
                '        return 0.0\n'
            ),
            verifier_mode='verify_function',
            case_timeout=1,
        )
        monkeypatch.setattr(verl_module, '_environment', lambda name, variant: hangs)
        monkeypatch.setattr(verl_module, '_client', lambda mode: local_dg)
        solution = 'def solve():\n    while True:\n        pass\n'
        judged_zero = {'score': 0.0, 'passed': 0.0, 'error': 0.0}

        assert compute_score('ds', solution) == judged_zero
        assert make_verl_compute_score(hangs, local_dg)('ds', solution) == judged_zero

    def test_make_verl_reward_fn_returns_callable(self, env: Environment) -> None:
        fn = make_verl_reward_fn(env=env)
        assert callable(fn)

    def test_verl_reward_fn_empty_batch(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_verl_reward_fn(env=env, dg=local_dg)
        assert fn({}) == []

    def test_verl_reward_fn_with_responses_key(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_verl_reward_fn(env=env, dg=local_dg)
        scores = fn({'responses': [GOOD_SOLUTION]})
        assert len(scores) == 1
        assert scores[0] >= 0.5

    def test_compute_score_accepts_verl_extra_kwargs(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        fn = make_verl_compute_score(env=env, dg=local_dg)
        score = fn(
            data_source='ds',
            solution_str=f'Here you go:\n```python\n{GOOD_SOLUTION}```\n',
            ground_truth='',
            extra_info={},
            reward_router_address=None,
        )
        assert score['score'] >= 0.5

    def test_extract_code_takes_last_block_or_whole_text(self) -> None:
        assert extract_code('a\n```python\nx = 1\n```\nb\n```\ny = 2\n```') == 'y = 2\n'
        assert extract_code('def f(): pass') == 'def f(): pass'

    @pytest.mark.parametrize(
        'tokens',
        [
            '<|im_end|>',
            '<|endoftext|>',
            '</s>',
            '<|eot_id|>',
            '<|eot_id|>\n<|end_of_text|>',
            '<end_of_turn>',
            '<｜end▁of▁sentence｜>',
        ],
    )
    def test_extract_code_drops_trailing_special_tokens(self, tokens: str) -> None:
        """Decoding with skip_special_tokens=False leaves EOS tokens after unfenced code."""
        assert extract_code(f'{GOOD_SOLUTION}{tokens}\n') == GOOD_SOLUTION
        assert extract_code(f'```python\n{GOOD_SOLUTION}```\n{tokens}') == GOOD_SOLUTION

    def test_unfenced_code_before_an_eos_token_scores_in_every_adapter(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        trl = make_trl_reward_fn(env=env, dg=local_dg)(completions=[f'{GOOD_SOLUTION}</s>'])
        verl = make_verl_compute_score(env, local_dg)('ds', f'{GOOD_SOLUTION}<|eot_id|>')['score']
        query = f'{OPENRLHF_PROMPT}{GOOD_SOLUTION}<|im_end|>'
        openrlhf = _openrlhf_score(env, local_dg, [query])['rewards']

        assert (trl, verl, openrlhf) == ([1.0], 1.0, [1.0])

    def test_fenced_code_scores_the_same_in_every_entry_point(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        """Raw fenced text is a syntax error; every entry point must run the block inside."""
        scores = {
            'trl': make_trl_reward_fn(env=env, dg=local_dg)(completions=[FENCED]),
            'dapo': make_dapo_reward_fn(env, local_dg)(completions=[FENCED]),
            'verl': [make_verl_compute_score(env, local_dg)('ds', FENCED)['score']],
            'verl_batch': make_verl_reward_fn(env, local_dg)({'responses': [FENCED]}),
            'reward_function': RewardFunction(env, local_dg)([FENCED]),
            'openrlhf': _openrlhf_score(env, local_dg, [OPENRLHF_PROMPT + FENCED])['rewards'],
        }

        assert scores == dict.fromkeys(scores, [1.0])

    def test_package_compute_score_picks_env_per_row(self) -> None:
        reference = (
            Path(__file__).resolve().parent.parent
            / 'src/deepgym/envs/coin_change/reference_solution.py'
        ).read_text()
        row = verl_rows(['coin_change'])[0]
        kwargs = {
            'data_source': row['data_source'],
            'ground_truth': row['reward_model']['ground_truth'],
            'extra_info': {**row['extra_info'], 'num_turns': None},
            'mode': 'local',
        }
        good = compute_score(solution_str=f'```python\n{reference}```', **kwargs)
        bad = compute_score(solution_str='nothing useful', **kwargs)
        assert good == {'score': 1.0, 'passed': 1.0, 'error': 0.0}
        assert bad.keys() == good.keys()
        assert bad['score'] < 1.0

    def test_verl_rows_have_verl_fields(self) -> None:
        row = verl_rows(['coin_change'])[0]
        assert row['prompt'][0]['role'] == 'user'
        assert row['extra_info'] == {'environment': 'coin_change'}
        assert set(row) == {'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'}


class TestDAPOIntegration:
    """Verify thin DAPO integration helpers."""

    def test_make_dapo_reward_fn_returns_callable(self, env: Environment) -> None:
        fn = make_dapo_reward_fn(env=env)
        assert callable(fn)

    def test_dapo_reward_fn_scores_good_solution(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_dapo_reward_fn(env=env, dg=local_dg)
        scores = fn(completions=[GOOD_SOLUTION])
        assert len(scores) == 1
        assert scores[0] >= 0.9

    def test_make_dapo_async_reward_fn_returns_callable(self, env: Environment) -> None:
        fn = make_dapo_async_reward_fn(env=env)
        assert callable(fn)

    def test_dapo_shaped_reward_fn_returns_component(
        self,
        local_dg: DeepGym,
    ) -> None:
        shaped_env = Environment(
            task='Return anything',
            verifier_code=(
                'return {"score": 0.4, "passed": False, '
                '"reward_components": {"correctness": 0.8, "style": 0.2}}\n'
            ),
        )
        fn = make_dapo_shaped_reward_fn(env=shaped_env, dg=local_dg, component='correctness')
        scores = fn(completions=['print("hi")\n'])
        assert scores == [0.8]

    async def test_dapo_reward_fns_reward_no_verdict_as_none(self, local_dg: DeepGym) -> None:
        completions = ['right', 'wrong', 'boom']

        sync_scores = make_dapo_reward_fn(MIXED, local_dg)(completions=completions)
        async_scores = await make_dapo_async_reward_fn(MIXED, local_dg)(completions=completions)
        shaped = make_dapo_shaped_reward_fn(MIXED, local_dg, component='style')(
            completions=completions
        )

        assert sync_scores == async_scores == [1.0, 0.0, None]
        assert shaped == [0.5, 0.5, None]

    def test_generate_dapo_verl_config_contains_expected_fields(self) -> None:
        config = generate_dapo_verl_config(train_files='data/train.parquet')
        assert 'adv_estimator: grpo' in config
        assert '    path: pkg://deepgym.integrations.verl\n    name: compute_score\n' in config
        assert 'actor_rollout_ref:\n  rollout:\n    n: 8\n' in config
        assert '      default_agent_loop: deepgym_single_turn\n' in config

    def test_generate_dapo_reward_module_defines_per_sample_compute_score(self) -> None:
        module_text = generate_dapo_reward_module('coin_change')
        assert "load_environment('coin_change')" in module_text
        assert 'compute_score = make_verl_compute_score(' in module_text


class TestRewardFunction:
    """Verify the universal RewardFunction class."""

    def test_callable_interface(self, env: Environment, local_dg: DeepGym) -> None:
        rf = RewardFunction(env=env, dg=local_dg)
        scores = rf([GOOD_SOLUTION])
        assert len(scores) == 1
        assert scores[0] >= 0.5

    def test_empty_list_returns_empty(self, env: Environment, local_dg: DeepGym) -> None:
        rf = RewardFunction(env=env, dg=local_dg)
        assert rf([]) == []

    def test_shaped_rewards_returns_dicts(self, env: Environment, local_dg: DeepGym) -> None:
        rf = RewardFunction(env=env, dg=local_dg)
        shaped = rf.shaped_rewards([GOOD_SOLUTION])
        assert len(shaped) == 1
        assert isinstance(shaped[0], dict)

    def test_shaped_rewards_empty_returns_empty(self, env: Environment, local_dg: DeepGym) -> None:
        rf = RewardFunction(env=env, dg=local_dg)
        assert rf.shaped_rewards([]) == []

    def test_call_with_details(self, env: Environment, local_dg: DeepGym) -> None:
        rf = RewardFunction(env=env, dg=local_dg)
        batch = rf.call_with_details([GOOD_SOLUTION])
        assert batch.total == 1

    async def test_no_verdict_is_none_in_every_method(self, local_dg: DeepGym) -> None:
        outputs = ['right', 'boom']
        rf = RewardFunction(MIXED, local_dg)
        arf = AsyncRewardFunction(MIXED, AsyncDeepGym(mode='local'))

        assert rf(outputs) == await arf(outputs) == [1.0, None]
        assert rf.shaped_rewards(outputs) == await arf.shaped_rewards(outputs)
        assert rf.shaped_rewards(outputs) == [{'style': 0.5}, None]
        assert rf.per_test_rewards(outputs) == await arf.per_test_rewards(outputs)
        assert rf.per_test_rewards(outputs) == [{'overall': 1.0}, None]


def test_masked_rate_warns_when_a_batch_passes_five_percent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A policy that finds a way to get masked must show up, not escape quietly."""
    rate = MaskedRate('test trainer')

    with caplog.at_level(logging.WARNING, logger='deepgym.integrations.reward'):
        for i in range(MASKED_BATCH):
            rate.record(masked=i < 5)  # 5%: fine
        assert not caplog.records
        for i in range(MASKED_BATCH):
            rate.record(masked=i < 6)

    assert f'test trainer: 6 of the last {MASKED_BATCH} samples got no verdict' in caplog.text
    assert f'(11 of {2 * MASKED_BATCH} this run)' in caplog.text


OPENRLHF_CONTEXT = '<|im_start|>user\n' + 'Strings are sequences of characters. ' * 12
OPENRLHF_PROMPT = (
    f'{OPENRLHF_CONTEXT}\nWrite reverse_string(s). For example:\n'
    f'```python\n{GOOD_SOLUTION}```<|im_end|>\n<|im_start|>assistant\n'
)
"""A chat-templated prompt whose fenced example would pass if it were ever scored."""


def _openrlhf_score(env: Environment, dg: DeepGym, queries: list[str]) -> dict:
    """POST *queries* as OpenRLHF's SingleTurnAgentExecutor._fetch_rewards_via_http does.

    openrlhf/utils/agent.py sends {'query', 'prompts', 'labels'}, where each query is
    ``hf_tokenizer.decode(prompt_ids + action_ids, skip_special_tokens=False)``: the
    templated prompt and the response decoded together, special tokens included.
    """
    payload = {
        'query': queries,
        'prompts': [OPENRLHF_PROMPT] * len(queries),
        'labels': [None] * len(queries),
    }
    app = FastAPI()
    app.include_router(create_openrlhf_router(env, dg, infra_failure_score='zero'))
    response = TestClient(app).post('/reward/score', json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _post(
    env: Environment,
    dg: DeepGym,
    payload: dict,
    infra_failure_score: Literal['zero', 'nan'] = 'zero',
    max_parallel: int = 32,
) -> dict:
    """POST *payload* to a fresh reward router; the request must succeed."""
    app = FastAPI()
    router = create_openrlhf_router(
        env, dg, infra_failure_score=infra_failure_score, max_parallel=max_parallel
    )
    app.include_router(router)
    response = TestClient(app).post('/reward/score', json=payload)
    assert response.status_code == 200, response.text
    return response.json()  # json.loads, which reads NaN as OpenRLHF's client does


class _ScriptedDeepGym(DeepGym):
    """A client that runs nothing: each run of an output does the next step of its script.

    A step that is an exception is raised; a number is seconds to wait before judging.
    Every run first waits *seconds*. A judged output scores 1.0 if it says 'right', else
    0.0. The client records each run's output, the most runs going at once, and when the
    first run started.
    """

    def __init__(
        self, script: dict[str, list[BaseException | float]] | None = None, seconds: float = 0.0
    ) -> None:
        super().__init__(mode='local')
        self.script = {output: list(steps) for output, steps in (script or {}).items()}
        self.seconds = seconds
        self.calls: list[str] = []
        self.running = self.most_running = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def run(self, env: Environment, model_output: str, **kwargs: object) -> RunResult:
        with self._lock:
            self.calls.append(model_output)
            steps = self.script.get(model_output)
            step = steps.pop(0) if steps else 0.0
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        self.started.set()
        try:
            time.sleep(self.seconds)
            if isinstance(step, BaseException):
                raise step
            time.sleep(step)
            judged = 'right' in model_output
            return RunResult(
                score=float(judged),
                passed=judged,
                output='',
                stderr='',
                exit_code=0,
                execution_time_ms=0.0,
                sandbox_id='fake',
            )
        finally:
            with self._lock:
                self.running -= 1


class _GatedDeepGym(DeepGym):
    """Local client that scores the 'slow' output only after another output was scored."""

    def __init__(self) -> None:
        super().__init__(mode='local')
        self.slow_started = threading.Event()
        self.other_scored = threading.Event()
        self.finished: list[str] = []

    def run(self, env: Environment, model_output: str, **kwargs: object) -> RunResult:
        if model_output == 'slow':
            self.slow_started.set()
            self.other_scored.wait(timeout=10)
        result = super().run(env, model_output, **kwargs)
        self.finished.append(model_output)
        if model_output != 'slow':
            self.other_scored.set()
        return result


class TestOpenRLHFIntegration:
    """Verify OpenRLHF router creation and endpoint."""

    def test_create_router_returns_api_router(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg, infra_failure_score='nan')
        assert router is not None

    @pytest.mark.parametrize(
        'kwargs', [{}, {'infra_failure_score': 'skip'}], ids=['unset', 'other']
    )
    def test_infra_failure_score_is_a_choice_made_at_setup(
        self, env: Environment, local_dg: DeepGym, kwargs: dict
    ) -> None:
        """No default: what OpenRLHF trains on for our failures is decided by whoever sets it up."""
        with pytest.raises(ValueError, match='infra_failure_score') as refused:
            create_openrlhf_router(env, local_dg, **kwargs)

        assert "'nan' (recommended)" in str(refused.value)
        assert '--algo.dynamic_filtering_enable' in str(refused.value)
        assert "'zero' trains on it as a failed answer" in str(refused.value)

    def test_score_endpoint(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg, infra_failure_score='zero')
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            '/reward/score',
            json={'prompts': ['test'], 'outputs': [GOOD_SOLUTION]},
        )
        assert response.status_code == 200
        data = response.json()
        assert 'rewards' in data
        assert len(data['rewards']) == 1
        assert data['rewards'][0] >= 0.5

    def test_score_endpoint_accepts_the_official_agent_payload(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        responses = [f'Sure:\n```python\n{GOOD_SOLUTION}```<|im_end|>', 'No idea.<|im_end|>']

        data = _openrlhf_score(env, local_dg, [OPENRLHF_PROMPT + r for r in responses])

        # The second response has no code; the example in its prompt must not be scored.
        assert data['rewards'] == [1.0, 0.0]

    def test_score_endpoint_never_scores_prompt_text(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        """A too-long prompt loses tokens from its front, but OpenRLHF sends the whole string.

        The response starts after the prompt's tail.
        """
        truncated = OPENRLHF_PROMPT[len(OPENRLHF_CONTEXT) // 2 :]
        queries = [
            truncated + 'No idea.<|im_end|>',
            truncated + f'```python\n{GOOD_SOLUTION}```<|im_end|>',
        ]

        data = _openrlhf_score(env, local_dg, queries)

        assert data == {'rewards': [0.0, 1.0], 'error_types': [None, None]}

    def test_failures_the_model_triggers_never_fail_the_request(self, local_dg: DeepGym) -> None:
        """A failed request stops OpenRLHF's training step, so no output may cause one.

        The verifier crashes on 'boom', and the last query lost its prompt: each scores 0.
        """
        payload = {
            'prompts': [OPENRLHF_PROMPT] * 3,
            'query': [OPENRLHF_PROMPT + 'right', OPENRLHF_PROMPT + 'boom', 'A lost prompt.'],
        }

        data = _post(MIXED, local_dg, payload)

        assert data == {
            'rewards': [1.0, 0.0, 0.0],
            'error_types': [None, 'runtime_error', 'prompt_mismatch'],
        }

    def test_only_provider_failures_get_nan(self, local_dg: DeepGym) -> None:
        """NaN drops the sample's group; a verifier the model's output broke must score 0."""
        payload = {'prompts': ['p'], 'outputs': ['boom']}

        data = _post(MIXED, local_dg, payload, infra_failure_score='nan')

        assert data == {'rewards': [0.0], 'error_types': ['runtime_error']}

    @pytest.mark.parametrize(
        'failure',
        [ConnectionError('provider unavailable'), TimeoutError('provider call timed out')],
        ids=['unavailable', 'call_timed_out'],
    )
    def test_a_provider_failure_is_graded_once_more(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """A failure is_infrastructure_failure calls ours gets one more grading of the response.

        A provider call that times out raises the built-in TimeoutError, which since Python
        3.11 is also what waiting too long for a run raises; it's still graded again.
        """
        monkeypatch.setattr(openrlhf, '_RETRY_DELAY_S', 0.0)
        dg = _ScriptedDeepGym({'right': [failure]})

        data = _post(MIXED, dg, {'prompts': ['p'] * 2, 'outputs': ['right', 'wrong']})

        assert data == {'rewards': [1.0, 0.0], 'error_types': [None, None]}
        assert sorted(dg.calls) == ['right', 'right', 'wrong']

    @pytest.mark.parametrize(
        'failure',
        [
            SandboxError('Sandbox 7f3a is not running'),
            DeepGymTimeout('Command exceeded 30s timeout'),
            VerifierError('Verifier did not produce valid JSON'),
        ],
        ids=['sandbox_killed', 'hung', 'verifier_failed'],
    )
    def test_a_failure_the_model_may_have_caused_is_never_graded_again(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception
    ) -> None:
        """It scores 0, even with NaN asked for: a second grading could hand it a pass."""
        monkeypatch.setattr(openrlhf, '_RETRY_DELAY_S', 0.0)
        dg = _ScriptedDeepGym({'right': [failure]})

        data = _post(MIXED, dg, {'prompts': ['p'], 'outputs': ['right']}, infra_failure_score='nan')

        assert data['rewards'] == [0.0]
        assert dg.calls == ['right']

    @pytest.mark.parametrize('infra_failure_score', ['zero', 'nan'])
    @pytest.mark.parametrize(
        ('second', 'budget'),
        [(ConnectionError('still unavailable'), 150.0), (2.0, 0.5)],
        ids=['fails_too', 'out_of_time'],
    )
    def test_a_second_grading_that_fails_gets_the_infrastructure_score(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        infra_failure_score: str,
        second: BaseException | float,
        budget: float,
    ) -> None:
        """Failing again, or passing too late (2 s against 0.5 s), is never a pass or an error.

        0 in zero mode; NaN in nan mode, which OpenRLHF's dynamic filtering drops.
        """
        monkeypatch.setattr(openrlhf, '_RETRY_DELAY_S', 0.0)
        monkeypatch.setattr(openrlhf, '_REQUEST_BUDGET_S', budget)
        dg = _ScriptedDeepGym({'right': [ConnectionError('provider unavailable'), second]})
        payload = {'prompts': ['p'], 'outputs': ['right']}

        with caplog.at_level(logging.WARNING, logger='deepgym.integrations.openrlhf'):
            data = _post(MIXED, dg, payload, infra_failure_score=infra_failure_score)

        (reward,) = data['rewards']
        assert (reward == 0.0) if infra_failure_score == 'zero' else math.isnan(reward)
        assert data['error_types'] == ['unknown_error']
        assert dg.calls == ['right', 'right']
        assert 'no verdict (unknown_error)' in caplog.text

    def test_second_gradings_take_the_same_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every run, first or second, waits for one of max_parallel slots: never more."""
        monkeypatch.setattr(openrlhf, '_RETRY_DELAY_S', 0.0)
        outputs = [f'right {i}' for i in range(4)]
        dg = _ScriptedDeepGym({o: [ConnectionError('unavailable')] for o in outputs}, 0.05)

        data = _post(MIXED, dg, {'prompts': outputs, 'outputs': outputs}, max_parallel=2)

        assert data['rewards'] == [1.0] * 4
        assert len(dg.calls) == 8
        assert dg.most_running == 2

    def test_a_run_its_request_gave_up_on_keeps_its_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Answering at the deadline never frees a slot the run still holds.

        The run still going scores 0, as the model's code may be what slows it. The next
        request's run can't start at all, so it gets the infrastructure score (NaN here).
        """
        monkeypatch.setattr(openrlhf, '_REQUEST_BUDGET_S', 0.3)
        dg = _ScriptedDeepGym({'right': [1.0]})
        app = FastAPI()
        app.include_router(
            create_openrlhf_router(MIXED, dg, infra_failure_score='nan', max_parallel=1)
        )
        client = TestClient(app)

        first = client.post('/reward/score', json={'prompts': ['p'], 'outputs': ['right']})
        second = client.post('/reward/score', json={'prompts': ['p'], 'outputs': ['wrong']})

        assert first.json() == {'rewards': [0.0], 'error_types': ['deadline_exceeded']}
        assert math.isnan(second.json()['rewards'][0])
        assert second.json()['error_types'] == ['deadline_exceeded']
        assert dg.calls == ['right']
        assert dg.most_running == 1

    def test_lasting_failures_count_toward_the_masked_rate_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(openrlhf, '_RETRY_DELAY_S', 0.0)
        failing = [f'right {i} outage' for i in range(6)]
        outputs = failing + ['right'] * (MASKED_BATCH - 6)
        down = [ConnectionError('down'), ConnectionError('down')]
        dg = _ScriptedDeepGym({output: list(down) for output in failing})

        with caplog.at_level(logging.WARNING, logger='deepgym.integrations.reward'):
            _post(MIXED, dg, {'prompts': outputs, 'outputs': outputs})

        assert f'6 of the last {MASKED_BATCH} samples got no verdict' in caplog.text

    async def test_a_slow_verification_does_not_hold_up_other_requests(
        self, env: Environment
    ) -> None:
        """OpenRLHF posts one request per rollout, concurrently, each with a 180 s timeout.

        The 'slow' request's scoring waits until another request has been scored. If
        scoring ran on the event loop, the second request couldn't even start.
        """
        dg = _GatedDeepGym()
        app = FastAPI()
        app.include_router(create_openrlhf_router(env, dg, infra_failure_score='zero'))
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            slow = asyncio.create_task(
                client.post('/reward/score', json={'prompts': ['p'], 'outputs': ['slow']})
            )
            await asyncio.to_thread(dg.slow_started.wait, 10)
            fast = await client.post('/reward/score', json={'prompts': ['p'], 'outputs': [FENCED]})
            await slow

        assert fast.json()['rewards'] == [1.0]
        assert dg.finished == [extract_code(FENCED), 'slow']

    async def test_a_request_holds_no_worker_thread_while_it_is_graded(self) -> None:
        """Requests are graded together even with one worker thread, so none waits for one.

        A request's 150 s would only start once it got a thread, and OpenRLHF gives up at 180.
        """
        to_thread.current_default_thread_limiter().total_tokens = 1
        dg = _ScriptedDeepGym(seconds=0.5)
        app = FastAPI()
        app.include_router(
            create_openrlhf_router(MIXED, dg, infra_failure_score='nan', max_parallel=2)
        )
        transport = httpx.ASGITransport(app=app)
        payload = {'prompts': ['p'], 'outputs': ['right']}

        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            replies = await asyncio.gather(
                *(client.post('/reward/score', json=payload) for _ in range(2))
            )

        assert [reply.json()['rewards'] for reply in replies] == [[1.0], [1.0]]
        assert dg.most_running == 2

    async def test_runs_score_as_they_stood_at_the_deadline_however_late_it_is_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A busy event loop that reads the outcomes late changes none of them.

        With one slot and the loop held past the 0.3 s deadline, the first run ends after it
        and scores 0, and the second, whose slot only came after it, never starts.
        """
        monkeypatch.setattr(openrlhf, '_REQUEST_BUDGET_S', 0.3)
        dg = _ScriptedDeepGym(seconds=0.6)
        app = FastAPI()
        app.include_router(
            create_openrlhf_router(MIXED, dg, infra_failure_score='nan', max_parallel=1)
        )
        transport = httpx.ASGITransport(app=app)
        payload = {'prompts': ['p', 'p'], 'outputs': ['right', 'right too']}

        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            reply = asyncio.create_task(client.post('/reward/score', json=payload))
            assert await asyncio.to_thread(dg.started.wait, 5)
            time.sleep(1.0)  # holds the event loop past the deadline and both slots' turns
            data = (await reply).json()

        assert data['rewards'][0] == 0.0
        assert math.isnan(data['rewards'][1])
        assert data['error_types'] == ['deadline_exceeded', 'deadline_exceeded']
        assert dg.calls == ['right']

    async def test_a_cancelled_request_starts_none_of_its_waiting_runs(self) -> None:
        """A request cancelled while it's graded, as at shutdown, leaves no run queued."""
        dg = _ScriptedDeepGym(seconds=0.2)
        app = FastAPI()
        app.include_router(
            create_openrlhf_router(MIXED, dg, infra_failure_score='nan', max_parallel=1)
        )
        transport = httpx.ASGITransport(app=app)
        payload = {'prompts': ['p', 'p'], 'outputs': ['first', 'waiting']}

        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            reply = asyncio.create_task(client.post('/reward/score', json=payload))
            assert await asyncio.to_thread(dg.started.wait, 5)
            reply.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reply
            await asyncio.sleep(0.4)

        assert dg.calls == ['first']

    def test_score_endpoint_empty(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg, infra_failure_score='zero')
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.post(
            '/reward/score',
            json={'prompts': [], 'outputs': []},
        )
        assert response.status_code == 200
        assert response.json()['rewards'] == []


class TestUntrustedInputSize:
    """The reward server takes untrusted bodies: matching and extraction stay linear and capped."""

    def test_one_megabyte_prompt_and_query_match_fast(self) -> None:
        size = 1_000_000
        repetitive = OpenRLHFRewardRequest(prompts=['a' * size + 'c'], query=['a' * size + 'b'])
        prompt = 'ab' * (size // 2)
        cut = OpenRLHFRewardRequest(prompts=[prompt], query=[prompt[size // 2 :] + 'answer'])

        start = time.perf_counter()
        responses = repetitive.responses + cut.responses
        elapsed = time.perf_counter() - start

        assert responses == [None, 'answer']
        assert elapsed < 0.5

    def test_one_megabyte_of_trailing_tokens_extracts_fast(self) -> None:
        start = time.perf_counter()
        code = extract_code('x = 1\n' + '<|im_end|> ' * 100_000)
        elapsed = time.perf_counter() - start

        assert code == 'x = 1\n'
        assert elapsed < 0.5

    def test_oversized_texts_are_rejected_before_matching(
        self, env: Environment, local_dg: DeepGym
    ) -> None:
        from deepgym.integrations.openrlhf import MAX_TEXT_CHARS

        app = FastAPI()
        app.include_router(create_openrlhf_router(env, local_dg, infra_failure_score='zero'))
        too_long = 'x' * (MAX_TEXT_CHARS + 1)

        response = TestClient(app).post(
            '/reward/score', json={'query': [too_long], 'prompts': ['p'], 'labels': [None]}
        )

        assert response.status_code == 422
        assert 'at most' in response.text


class TestOpenRLHFModels:
    """Verify OpenRLHF Pydantic models."""

    def test_request_model(self) -> None:
        req = OpenRLHFRewardRequest(prompts=['p1'], outputs=['o1'])
        assert req.prompts == ['p1']

    def test_response_model(self) -> None:
        resp = OpenRLHFRewardResponse(rewards=[0.5, 0.8])
        assert len(resp.rewards) == 2
