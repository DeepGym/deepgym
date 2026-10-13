"""Tests for deepgym.integrations -- TRL, verl, OpenRLHF integration wrappers."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from deepgym.integrations.reward import RewardFunction
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
        """Built like OpenRLHF's SingleTurnAgentExecutor._fetch_rewards_via_http does it.

        openrlhf/utils/agent.py posts {'query', 'prompts', 'labels'}, where each query is
        ``hf_tokenizer.decode(prompt_ids + action_ids, skip_special_tokens=False)``: the
        templated prompt and the response decoded together, special tokens included.
        """
        prompt = (
            '<|im_start|>user\nWrite reverse_string(s). For example:\n'
            f'```python\n{GOOD_SOLUTION}```<|im_end|>\n<|im_start|>assistant\n'
        )
        responses = [f'Sure:\n```python\n{GOOD_SOLUTION}```<|im_end|>', 'No idea.<|im_end|>']
        payload = {
            'query': [prompt + response for response in responses],
            'prompts': [prompt] * len(responses),
            'labels': [None] * len(responses),
        }
        app = FastAPI()
        app.include_router(create_openrlhf_router(env, local_dg))

        response = TestClient(app).post('/reward/score', json=payload)

        assert response.status_code == 200, response.text
        # The second response has no code; the example in its prompt must not be scored.
        assert response.json()['rewards'] == [1.0, 0.0]

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


class TestOpenRLHFModels:
    """Verify OpenRLHF Pydantic models."""

    def test_request_model(self) -> None:
        req = OpenRLHFRewardRequest(prompts=['p1'], outputs=['o1'])
        assert req.prompts == ['p1']

    def test_response_model(self) -> None:
        resp = OpenRLHFRewardResponse(rewards=[0.5, 0.8])
        assert len(resp.rewards) == 2
