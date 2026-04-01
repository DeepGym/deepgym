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
from deepgym.integrations.verl import make_verl_compute_score, make_verl_reward_fn
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
        config = generate_dapo_verl_config(
            train_files='data/train.parquet',
            reward_module_path='reward_module.py',
        )
        assert 'adv_estimator: dapo' in config
        assert 'custom_reward_function:' in config
        assert 'reward_module.py' in config

    def test_generate_dapo_reward_module_uses_dapo_reward_fn(self) -> None:
        module_text = generate_dapo_reward_module('coin_change')
        assert "load_environment('coin_change')" in module_text
        assert 'make_dapo_reward_fn' in module_text


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
