"""Tests for verifier audit and benchmark audit utilities."""

from pathlib import Path

import pytest

from deepgym.benchmark_ops import build_benchmark_audit, load_environments_from_dir
from deepgym.core import DeepGym
from deepgym.exploit_db import ExploitDB
from deepgym.models import Environment
from deepgym.reward_qa import RewardAuditor, fingerprint_verifier

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
        assert report.verifier_hash == fingerprint_verifier(STANDALONE_PASSING_VERIFIER)
        assert 'few-test-cases' in report.patterns
        assert report.recommendations
        assert report.stored is False

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
            record = db.get('weak-verifier', 'white-box')
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
