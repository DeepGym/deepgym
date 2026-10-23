"""Tests for verifier audit and benchmark audit utilities."""

from pathlib import Path

import pytest

from deepgym.adversarial import environment_id, grading_fingerprint
from deepgym.benchmark_ops import build_benchmark_audit, load_environments_from_dir
from deepgym.core import DeepGym
from deepgym.exploit_db import ExploitDB
from deepgym.models import Environment
from deepgym.reward_qa import RewardAuditor
from tests.test_adversarial import CRASHING_VERIFIER, ROBUST_VERIFIER

STANDALONE_PASSING_VERIFIER = (
    'import sys, json\n'
    'if __name__ == "__main__":\n'
    '    print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,'
    '"details":None,"truncated":False}))\n'
    '    sys.exit(0)\n'
)


@pytest.fixture()
def dg() -> DeepGym:
    """Return a local-mode DeepGym client."""
    return DeepGym(mode='local')


class TestRewardAuditor:
    """Verify reward QA audits and persistence hooks."""

    def test_audit_detects_exploitable_verifier(self, dg: DeepGym) -> None:
        env = Environment(task='Do anything', verifier_code=STANDALONE_PASSING_VERIFIER)
        auditor = RewardAuditor(dg)

        report = auditor.audit(
            env,
            verifier_id='weak-verifier',
            strategies=['empty', 'trivial'],
            persist=False,
        )

        assert report.verifier_id == 'weak-verifier'
        assert report.exploitable is True
        assert report.risk_level in {'high', 'critical'}
        assert report.verifier_hash == grading_fingerprint(env)
        assert 'few-test-cases' in report.patterns
        assert report.recommendations
        assert report.stored is False

    def test_attacks_that_cannot_run_are_not_exploits(self, dg: DeepGym) -> None:
        env = Environment(task='Do anything', verifier_code=CRASHING_VERIFIER)

        report = RewardAuditor(dg).audit(env, strategies=['empty'])

        assert report.exploitable is False
        assert (report.report.is_robust, report.report.inconclusive) == (False, 1)

    def test_inconclusive_audits_are_never_stored_as_tested(
        self, tmp_path: Path, dg: DeepGym
    ) -> None:
        """Otherwise a provider failure marks the verifier tested and later scans skip it."""
        env = Environment(task='Do anything', verifier_code=CRASHING_VERIFIER)
        db_path = tmp_path / 'audits.db'

        report = RewardAuditor(dg).audit(
            env, verifier_id='flaky', strategies=['empty'], persist=True, db_path=db_path
        )

        db = ExploitDB(db_path=db_path)
        try:
            assert (report.stored, db.get('flaky', benchmark='custom')) == (False, None)
        finally:
            db.close()

    def test_an_inconclusive_retest_of_a_changed_verifier_drops_its_old_result(
        self, tmp_path: Path, dg: DeepGym
    ) -> None:
        """The old verifier's robust result must not stand in for the changed one."""
        db_path = tmp_path / 'audits.db'
        auditor = RewardAuditor(dg)
        reports = [
            auditor.audit(
                Environment(task='Write compute(a, b).', verifier_code=verifier),
                verifier_id='v',
                strategies=['empty'],
                persist=True,
                db_path=db_path,
            )
            for verifier in (ROBUST_VERIFIER, CRASHING_VERIFIER)
        ]

        assert [report.status for report in reports] == ['robust', 'inconclusive']
        db = ExploitDB(db_path=db_path)
        try:
            assert db.get('v', benchmark='custom') is None
        finally:
            db.close()

    def test_audit_persists_to_exploit_db(self, tmp_path: Path, dg: DeepGym) -> None:
        env = Environment(task='Do anything', verifier_code=STANDALONE_PASSING_VERIFIER)
        auditor = RewardAuditor(dg)
        db_path = tmp_path / 'audits.db'

        report = auditor.audit(
            env,
            verifier_id='weak-verifier',
            strategies=['empty'],
            persist=True,
            db_path=db_path,
        )

        assert report.stored is True
        db = ExploitDB(db_path=db_path)
        try:
            record = db.get('weak-verifier', benchmark='custom')
            assert record is not None
            assert record.exploitable is True
            assert record.num_attacks == 1
        finally:
            db.close()


class TestBenchmarkAudit:
    """Verify benchmark split and contamination audit helpers."""

    def test_build_benchmark_audit_flags_public_private_leak(self) -> None:
        shared_task = 'Sort a list of integers'
        shared_verifier = 'return 1.0\n'
        environments = {
            'train/env_a': Environment(task=shared_task, verifier_code=shared_verifier),
            'holdout/env_b': Environment(task=shared_task, verifier_code=shared_verifier),
            'unique/env_c': Environment(task='Reverse a string', verifier_code='return 0.0\n'),
        }

        report = build_benchmark_audit(
            environments,
            benchmark='demo',
            split_overrides={
                'train/env_a': 'public_train',
                'holdout/env_b': 'private_holdout',
                'unique/env_c': 'public_eval',
            },
        )

        assert report.contamination_risk is True
        assert any(
            set(group) == {'holdout/env_b', 'train/env_a'} for group in report.duplicate_task_groups
        )
        assert any(
            leak.leak_type == 'task' and set(leak.env_ids) == {'holdout/env_b', 'train/env_a'}
            for leak in report.leaks
        )

    def test_load_environments_from_dir_recurses(self, tmp_path: Path) -> None:
        env_a = tmp_path / 'env_a'
        env_a.mkdir()
        (env_a / 'task.md').write_text('Task A', encoding='utf-8')
        (env_a / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')

        env_b = tmp_path / 'nested' / 'env_b'
        env_b.mkdir(parents=True)
        (env_b / 'task.md').write_text('Task B', encoding='utf-8')
        (env_b / 'verifier.py').write_text('return 0.0\n', encoding='utf-8')

        environments = load_environments_from_dir(tmp_path)

        assert set(environments) == {'env_a', 'nested/env_b'}


def test_an_audit_that_judged_nothing_has_no_risk_rating(tmp_path: Path, dg: DeepGym) -> None:
    env = Environment(task='Do anything', verifier_code=CRASHING_VERIFIER)

    report = RewardAuditor(dg).audit(env, strategies=['empty', 'trivial'])

    assert report.status == 'inconclusive'
    assert (report.risk_score, report.risk_level) == (None, None)  # never a reassuring LOW


def test_audit_ids_are_identity_and_hashes_cover_all_grading() -> None:
    """Two tasks sharing a verifier keep their own ids and grade differently; a changed
    verifier keeps its task's id, so its old stored result is found and dropped."""
    verifier = 'return 1.0\n'
    first = Environment(name='first', task='t', verifier_code=verifier, test_cases=[{'x': 1}])
    second = first.model_copy(update={'name': 'second', 'test_cases': [{'x': 2}]})
    changed = first.model_copy(update={'verifier_code': verifier + '# edited\n'})
    prefix = 'x' * 80
    unnamed = [Environment(task=prefix + end, verifier_code=verifier) for end in ('a', 'b')]

    assert environment_id(first) != environment_id(second)
    assert grading_fingerprint(first) != grading_fingerprint(second)
    assert environment_id(changed) == environment_id(first)
    assert grading_fingerprint(changed) != grading_fingerprint(first)
    assert environment_id(unnamed[0]) != environment_id(unnamed[1])  # no 80-character prefix


def test_a_harbor_task_whose_tests_change_is_a_new_version() -> None:
    """Harbor grades with its test scripts, not its placeholder verifier: changing them must
    invalidate a stored result, or an old robust verdict outlives an unjudged retest."""
    from pathlib import Path

    from deepgym.harbor.environment import HarborEnvironment

    env = HarborEnvironment.from_dir(Path(__file__).parent / 'fixtures/harbor/tasks/hello')
    tests = {name: body + b'\n# changed\n' for name, body in env.harbor_task.tests.items()}
    changed = env.model_copy(
        update={'harbor_task': env.harbor_task.model_copy(update={'tests': tests})}
    )

    assert environment_id(changed) == environment_id(env)
    assert grading_fingerprint(changed) != grading_fingerprint(env)
