"""CyberBench-style built-in environment tests."""

import json
from pathlib import Path

import pytest

from deepgym import DeepGym, load_environment, load_suite

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
CYBER_ENVS = [
    'cyber/cve_2021_44228_log_triage',
    'cyber/web_path_traversal_toy',
]


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='local')


@pytest.mark.parametrize('env_name', CYBER_ENVS)
def test_cyberbench_reference_solution_passes(env_name: str, dg: DeepGym) -> None:
    env = load_environment(env_name)
    solution = (ENVS_DIR / env_name / 'reference_solution.py').read_text(encoding='utf-8')

    result = dg.run(env, solution)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    assert result.reward_components
    assert result.cases


def test_cyberbench_suite_loads_new_family() -> None:
    envs = load_suite('cyberbench')

    names = {env.name for env in envs}
    assert {'cve_2021_44228_log_triage', 'web_path_traversal_toy'} <= names
    assert all(env.domain == 'cyber' for env in envs)


def test_cyberbench_bad_answer_gets_partial_or_zero_reward(dg: DeepGym) -> None:
    env = load_environment('cyber/cve_2021_44228_log_triage')

    result = dg.run(env, '{"source_ips": ["198.51.100.10"], "vulnerability": "none"}')

    assert result.passed is False
    assert result.score < 0.5


_TRIAGE = {
    'cve': 'CVE-2021-44228',
    'vulnerability': 'Log4Shell JNDI injection',
    'source_ips': ['198.51.100.77', '203.0.113.42'],
    'payload_indicators': ['jndi:ldap lookup', 'jndi:dns lookup'],
    'recommended_action': 'Patch Log4j, block both sources and search the logs for ${jndi:.',
}


@pytest.mark.parametrize('extra_ip', ['8.8.8.8', '198.51.100.10'])
def test_log_triage_penalizes_every_ip_that_did_not_probe(dg: DeepGym, extra_ip: str) -> None:
    env = load_environment('cyber/cve_2021_44228_log_triage')
    padded = {**_TRIAGE, 'source_ips': [*_TRIAGE['source_ips'], extra_ip]}

    correct = dg.run(env, json.dumps(_TRIAGE))
    wrong = dg.run(env, json.dumps(padded))

    assert correct.score == pytest.approx(1.0)
    assert wrong.passed is False
    assert wrong.score < correct.score


def test_log_triage_scores_the_answer_fields_not_keywords_elsewhere(dg: DeepGym) -> None:
    """Wrong fields padded with the right keywords used to earn 1.0."""
    env = load_environment('cyber/cve_2021_44228_log_triage')
    padded = {
        'cve': 'CVE-2021-45046',
        'vulnerability': 'SQL injection',
        'source_ips': _TRIAGE['source_ips'],
        'payload_indicators': [],
        'notes': 'CVE-2021-44228 log4shell jndi ldap dns',
        'recommended_action': 'Do not patch, block or search the logs.',
    }

    result = dg.run(env, json.dumps(padded))

    assert result.passed is False
    assert result.score == pytest.approx(0.4)  # the IPs, nothing else


@pytest.mark.parametrize(
    ('action', 'score'),
    [
        ('Do not patch, block or search the logs.', 0.85),
        ("Don't patch Log4j; there is no need to block or search logs.", 0.85),
        ('Patching Log4j, blocking the sources and searching logs are not needed.', 0.85),
        ('Patch Log4j, block both sources and search the logs. Do not expose JNDI.', 1.0),
        ('Do not delay: patch Log4j, block both sources and search the logs.', 1.0),
    ],
)
def test_log_triage_negated_actions_earn_nothing(dg: DeepGym, action: str, score: float) -> None:
    env = load_environment('cyber/cve_2021_44228_log_triage')

    result = dg.run(env, json.dumps({**_TRIAGE, 'recommended_action': action}))

    assert result.score == pytest.approx(score)
