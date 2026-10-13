"""scripts/run_blackbox_scan.py saves only conclusive scans and reuses only identical ones."""

import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import run_blackbox_scan  # noqa: E402

from deepgym.adversarial import BlackBoxReport  # noqa: E402
from deepgym.exploit_db import ExploitDB  # noqa: E402
from deepgym.models import Environment  # noqa: E402


class _Tester:
    """A stand-in BlackBoxAdversarialTester that returns a fixed report."""

    _model, _max_rounds, _attempts_per_round = 'm', 2, 3

    def __init__(self, report: BlackBoxReport) -> None:
        self.report = report
        self.calls = 0

    def test(self, env: Environment) -> BlackBoxReport:
        self.calls += 1
        return self.report


def _report(status: str) -> BlackBoxReport:
    return BlackBoxReport(
        verifier_id='v',
        status=status,
        exploitable=status == 'exploitable',
        problems=['round 1: overloaded'] if status == 'inconclusive' else [],
    )


def test_an_inconclusive_scan_is_not_saved_so_the_next_scan_retries(tmp_path: Path) -> None:
    db = ExploitDB(db_path=tmp_path / 'e.db')
    env = Environment(task='t', verifier_code='return 0.0\n')
    try:
        line = run_blackbox_scan._scan_one(db, _Tester(_report('inconclusive')), 'v', 'b', env)

        assert line.startswith('INCONCLUSIVE, not saved (round 1: overloaded')
        assert db.get('v', benchmark='b', attack_type='black-box') is None
    finally:
        db.close()


def test_a_saved_scan_is_reused_until_the_verifier_changes(tmp_path: Path) -> None:
    db = ExploitDB(db_path=tmp_path / 'e.db')
    tester = _Tester(_report('robust'))
    env = Environment(task='t', verifier_code='return 0.0\n')
    changed = Environment(task='t', verifier_code='return 0.1\n')
    try:
        assert run_blackbox_scan._scan_one(db, tester, 'v', 'b', env).startswith('ROBUST')
        assert 'skipping' in run_blackbox_scan._scan_one(db, tester, 'v', 'b', env)
        assert run_blackbox_scan._scan_one(db, tester, 'v', 'b', changed).startswith('ROBUST')
        assert tester.calls == 2  # the changed verifier was tested again
    finally:
        db.close()
