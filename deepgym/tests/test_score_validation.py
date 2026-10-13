"""Invalid scores (NaN, inf, out of range) are verifier errors, never reward."""

import math

import pytest

from deepgym import DeepGym
from deepgym.exceptions import InvalidScoreError, VerifierError
from deepgym.integrations.dapo import make_dapo_reward_fn, make_dapo_shaped_reward_fn
from deepgym.integrations.trl import make_trl_reward_fn
from deepgym.integrations.verl import make_verl_compute_score
from deepgym.models import Action, CaseResult, Environment, MultiTurnEnvironment, checked_score

INVALID = ['float("nan")', 'float("inf")', 'float("-inf")', '-1', '7.5']


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='local')


def _body(score_expr: str) -> Environment:
    return Environment(task='t', verifier_code=f'return {score_expr}\n')


def _dict_body(score_expr: str) -> Environment:
    return Environment(
        task='t', verifier_code=f'return {{"score": {score_expr}, "passed": True}}\n'
    )


def _script(score_expr: str, components: str = 'None') -> Environment:
    return Environment(
        task='t',
        verifier_code=(
            'import json\n'
            'if __name__ == "__main__":\n'
            f'    print(json.dumps({{"score": {score_expr}, "passed": True, '
            f'"reward_components": {components}}}))\n'
        ),
    )


@pytest.mark.parametrize('make_env', [_body, _dict_body, _script], ids=['body', 'dict', 'script'])
@pytest.mark.parametrize('score', INVALID)
class TestInvalidScores:
    def test_single_run_raises(self, dg: DeepGym, make_env, score: str) -> None:
        with pytest.raises(InvalidScoreError):
            dg.run(make_env(score), '')

    def test_batch_scores_zero(self, dg: DeepGym, make_env, score: str) -> None:
        results = dg.run_batch(make_env(score), ['', '']).results
        assert [(r.score, r.passed, r.error_type) for r in results] == [
            (0.0, False, 'invalid_score')
        ] * 2

    def test_reward_fns_never_hand_out_the_invalid_score(
        self, dg: DeepGym, make_env, score: str
    ) -> None:
        """An invalid score is no verdict: None for TRL-style callbacks, 0.0 for verl's."""
        env = make_env(score)
        assert make_trl_reward_fn(env, dg)(completions=['']) == [None]
        assert make_dapo_reward_fn(env, dg)(completions=['']) == [None]
        assert make_verl_compute_score(env, dg)('ds', '', '') == 0.0


@pytest.mark.parametrize('make_env', [_body, _script], ids=['body', 'script'])
def test_float_noise_is_clamped(dg: DeepGym, make_env) -> None:
    result = dg.run(make_env('1.0000000001'), '')
    assert result.score == 1.0


def test_non_finite_reward_components_become_zero(dg: DeepGym) -> None:
    env = _script('0.5', '{"good": 0.5, "nan": float("nan"), "inf": float("inf")}')
    result = dg.run(env, '')
    assert result.reward_components == {'good': 0.5, 'nan': 0.0, 'inf': 0.0}
    shaped = make_dapo_shaped_reward_fn(env, dg, component='nan')
    assert shaped(completions=['']) == [0.0]


def test_invalid_case_score_fails_that_case() -> None:
    case = CaseResult(id='c', passed=True, score=float('nan'))
    assert (case.score, case.passed) == (0.0, False)


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -1, 7.5, 'abc'])
def test_checked_score_rejects(value: object) -> None:
    with pytest.raises(InvalidScoreError):
        checked_score(value)


def test_checked_score_clamps_noise() -> None:
    assert checked_score(-1e-9) == 0.0
    assert checked_score(1.0000000001) == 1.0
    assert checked_score(0.25) == 0.25


@pytest.mark.parametrize('printed', ['nan', 'inf', '-1', '7.5'])
def test_session_step_verifier_rejects_invalid(dg: DeepGym, printed: str) -> None:
    env = MultiTurnEnvironment(
        task='t', max_steps=3, step_verifier_code=f'print(float("{printed}"))'
    )
    with dg.session(env) as session:
        session.start()
        with pytest.raises(VerifierError):
            session.step(Action(content='print(1)'))
        assert session.done
