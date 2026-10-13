"""Tests for deepgym.integrations -- TRL, verl, DAPO, OpenRLHF integration wrappers."""

import asyncio
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
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
from deepgym.integrations.reward import AsyncRewardFunction, RewardFunction
from deepgym.integrations.trl import make_trl_async_reward_fn, make_trl_reward_fn
from deepgym.integrations.verl import (
    compute_score,
    extract_code,
    make_verl_compute_score,
    make_verl_reward_fn,
    verl_rows,
)
from deepgym.models import Environment

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

    def test_verl_compute_score_returns_float(self, env: Environment, local_dg: DeepGym) -> None:
        fn = make_verl_compute_score(env=env, dg=local_dg)
        score = fn('dataset', GOOD_SOLUTION, '')
        assert isinstance(score, float)
        assert score >= 0.5

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
        assert score >= 0.5

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
        verl = make_verl_compute_score(env=env, dg=local_dg)('ds', f'{GOOD_SOLUTION}<|eot_id|>')
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
            'verl': [make_verl_compute_score(env, local_dg)('ds', FENCED)],
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
        assert 'reward:\n  reward_manager:\n    name: dapo\n' in config
        assert '    path: pkg://deepgym.integrations.verl\n    name: compute_score\n' in config
        assert 'actor_rollout_ref:\n  rollout:\n    n: 8\n' in config

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
    app.include_router(create_openrlhf_router(env, dg))
    response = TestClient(app).post('/reward/score', json=payload)
    assert response.status_code == 200, response.text
    return response.json()


class _GatedDeepGym(DeepGym):
    """Local client that scores the 'slow' output only after another output was scored."""

    def __init__(self) -> None:
        super().__init__(mode='local')
        self.slow_started = threading.Event()
        self.other_scored = threading.Event()
        self.finished: list[str] = []

    def run_batch(self, env: Environment, outputs: list[str], **kwargs: object) -> object:
        if outputs == ['slow']:
            self.slow_started.set()
            self.other_scored.wait(timeout=10)
        batch = super().run_batch(env, outputs, **kwargs)
        self.finished.extend(outputs)
        if outputs != ['slow']:
            self.other_scored.set()
        return batch


class TestOpenRLHFIntegration:
    """Verify OpenRLHF router creation and endpoint."""

    def test_create_router_returns_api_router(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg)
        assert router is not None

    def test_score_endpoint(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg)
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

    @pytest.mark.parametrize(
        ('payload', 'status', 'error_types'),
        [
            # The verifier crashes on 'boom': no verdict for that sample.
            ({'prompts': ['p'] * 3, 'outputs': ['right', 'wrong', 'boom']}, 503,
             [None, None, 'runtime_error']),
            # A query that doesn't contain its prompt can't be split into a response.
            ({'prompts': [OPENRLHF_PROMPT] * 2,
              'query': [OPENRLHF_PROMPT + 'right', 'A query that lost its prompt.']}, 422,
             [None, 'prompt_mismatch']),
        ],
        ids=['verifier_failed', 'prompt_mismatch'],
    )  # fmt: skip
    def test_a_sample_without_a_verdict_fails_the_whole_request(
        self, local_dg: DeepGym, payload: dict, status: int, error_types: list
    ) -> None:
        """OpenRLHF can't skip a sample and trains on whatever ``rewards`` holds.

        Its agent calls ``raise_for_status()``, retries, then leaves the rollout without a
        reward, so a failure must be an HTTP error, never a 200 carrying a 0.
        """
        app = FastAPI()
        app.include_router(create_openrlhf_router(MIXED, local_dg))

        response = TestClient(app).post('/reward/score', json=payload)

        assert response.status_code == status
        assert 'rewards' not in response.json()
        assert response.json()['detail']['error_types'] == error_types

    async def test_a_slow_verification_does_not_hold_up_other_requests(
        self, env: Environment
    ) -> None:
        """OpenRLHF posts one request per rollout, concurrently, each with a 180 s timeout.

        The 'slow' request's scoring waits until another request has been scored. If
        scoring ran on the event loop, the second request couldn't even start.
        """
        dg = _GatedDeepGym()
        app = FastAPI()
        app.include_router(create_openrlhf_router(env, dg))
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

    def test_score_endpoint_empty(self, env: Environment, local_dg: DeepGym) -> None:
        router = create_openrlhf_router(env, local_dg)
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
        app.include_router(create_openrlhf_router(env, local_dg))
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
