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
