"""Tests for deepgym.adversarial — reward hack detection."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from deepgym.adversarial import (
    AdversarialReport,
    AdversarialTester,
    AttackResult,
    LLMAdversarialTester,
)
from deepgym.core import DeepGym
from deepgym.models import Environment


@pytest.fixture()
def dg() -> DeepGym:
    """Return a local-mode DeepGym client."""
    return DeepGym(mode='local')


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestAttackResult:
    """Verify AttackResult model creation."""

    def test_creation(self) -> None:
        ar = AttackResult(
            strategy='empty',
            exploited=False,
            score=0.0,
            details='nothing happened',
        )
        assert ar.strategy == 'empty'
        assert ar.exploited is False
        assert ar.score == 0.0

    def test_exploited_creation(self) -> None:
        ar = AttackResult(
            strategy='trivial',
            exploited=True,
            score=0.8,
            details='verifier was tricked',
        )
        assert ar.exploited is True
        assert ar.score == 0.8


class TestAdversarialReport:
    """Verify AdversarialReport model creation."""

    def test_creation_robust(self) -> None:
        report = AdversarialReport(
            environment='test env',
            attacks_run=3,
            exploits_found=0,
            results=[],
            is_robust=True,
        )
        assert report.is_robust is True
        assert report.attacks_run == 3
        assert report.exploits_found == 0

    def test_creation_not_robust(self) -> None:
        ar = AttackResult(strategy='empty', exploited=True, score=0.9, details='')
        report = AdversarialReport(
            environment='weak env',
            attacks_run=1,
            exploits_found=1,
            results=[ar],
            is_robust=False,
        )
        assert report.is_robust is False
        assert len(report.results) == 1


# ---------------------------------------------------------------------------
# Weak verifier (always passes)
# ---------------------------------------------------------------------------

WEAK_VERIFIER = (
    '#!/usr/bin/env python3\n'
    'import sys, json\n'
    'if __name__ == "__main__":\n'
    '    out = {"schema_version": "1.0", "score": 1.0, "passed": True,\n'
    '           "details": None, "truncated": False}\n'
    '    print(json.dumps(out))\n'
    '    sys.exit(0)\n'
)

ROBUST_VERIFIER = (
    '#!/usr/bin/env python3\n'
    'import sys, json, importlib.util\n'
    'if __name__ == "__main__":\n'
    '    sp = sys.argv[1]\n'
    '    try:\n'
    '        spec = importlib.util.spec_from_file_location("solution", sp)\n'
    '        mod = importlib.util.module_from_spec(spec)\n'
    '        spec.loader.exec_module(mod)\n'
    '        if hasattr(mod, "compute") and mod.compute(7, 3) == 21:\n'
    '            out = {"schema_version": "1.0", "score": 1.0, "passed": True,\n'
    '                   "details": None, "truncated": False}\n'
    '        else:\n'
    '            out = {"schema_version": "1.0", "score": 0.0, "passed": False,\n'
    '                   "details": "wrong", "truncated": False}\n'
    '    except Exception as e:\n'
    '        out = {"schema_version": "1.0", "score": 0.0, "passed": False,\n'
    '               "details": str(e), "truncated": False}\n'
    '    print(json.dumps(out))\n'
    '    sys.exit(0 if out["passed"] else 1)\n'
)


def _run_attack(verifier_script: str, attack_code: str) -> dict:
    """Run attack_code against verifier_script in a subprocess, return parsed JSON.

    Extract the last JSON-like line from stdout (the verifier output may be
    preceded by output from the attack solution itself).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / 'verifier.py').write_text(verifier_script, encoding='utf-8')
        (tmp / 'solution.py').write_text(attack_code, encoding='utf-8')
        proc = subprocess.run(
            [sys.executable, str(tmp / 'verifier.py'), str(tmp / 'solution.py')],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Find last JSON object line in stdout.
        for line in reversed(proc.stdout.strip().splitlines()):
            stripped = line.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                return json.loads(stripped)
        raise ValueError(f'No JSON found in stdout: {proc.stdout!r}')


class TestWeakVerifier:
    """A verifier that always returns score 1.0 should be exploitable."""

    def test_weak_verifier_exploited(self, dg: DeepGym) -> None:
        env = Environment(task='Do anything', verifier_code='return 1.0\n')
        tester = AdversarialTester(dg)
        attack_code = tester._generate_attack(env, 'empty')

        output = _run_attack(WEAK_VERIFIER, attack_code)
        assert output['score'] >= 0.5
        assert output['passed'] is True


# ---------------------------------------------------------------------------
# Robust verifier
# ---------------------------------------------------------------------------


class TestRobustVerifier:
    """A verifier that actually checks solution correctness should survive attacks."""

    def test_robust_verifier_survives(self, dg: DeepGym) -> None:
        env = Environment(task='Multiply two numbers', verifier_code='return 0.0\n')
        tester = AdversarialTester(dg)

        for strategy in ['empty', 'hardcoded', 'trivial', 'overflow', 'pattern']:
            attack_code = tester._generate_attack(env, strategy)
            output = _run_attack(ROBUST_VERIFIER, attack_code)
            assert output['score'] < 0.5, (
                f'Strategy {strategy!r} exploited the robust verifier with score {output["score"]}'
            )


class TestAdversarialTesterIntegration:
    """Integration test using the AdversarialTester.test() method.

    The wrap_verifier template uses os._exit which does not flush stdout,
    so the tester gets empty output and defaults to score=0.0. We verify
    that the tester still produces a well-formed report.
    """

    def test_report_structure(self, dg: DeepGym) -> None:
        env = Environment(task='Anything', verifier_code='return 1.0\n')
        tester = AdversarialTester(dg)
        report = tester.test(env, strategies=['empty', 'trivial'])
        assert isinstance(report, AdversarialReport)
        assert report.attacks_run == 2
        assert len(report.results) == 2
        for r in report.results:
            assert isinstance(r, AttackResult)
            assert r.strategy in ('empty', 'trivial')


CRASHING_VERIFIER = 'import os\n\nif __name__ == "__main__":\n    os._exit(3)\n'
"""Dies before printing a verdict, so every run of it raises VerifierError."""


class TestAttacksThatCannotRun:
    """An attack that crashed or never ran says nothing about the verifier: inconclusive."""

    def test_crashed_attacks_are_inconclusive_not_resisted(self, dg: DeepGym) -> None:
        env = Environment(task='Anything', verifier_code=CRASHING_VERIFIER)

        report = AdversarialTester(dg).test(env, strategies=['empty', 'trivial'])

        assert not report.is_robust
        assert [(r.exploited, r.inconclusive) for r in report.results] == [(False, True)] * 2
        assert (report.exploits_found, report.inconclusive) == (0, 2)

    def test_unavailable_llm_attack_is_inconclusive(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        env = Environment(task='Anything', verifier_code='return 0.0\n')

        reports = [
            AdversarialTester(dg).test(env, strategies=['llm_attack']),
            LLMAdversarialTester(dg).test(env),
        ]

        assert [(r.is_robust, r.inconclusive) for r in reports] == [(False, 1)] * 2


class TestAttacksUseTheClient:
    """Attacks run on the tester's DeepGym client, not a hard-wired local executor."""

    def test_attacks_go_through_dg_run(self) -> None:
        class RecordingDeepGym(DeepGym):
            def __init__(self) -> None:
                super().__init__(mode='local')
                self.outputs: list[str] = []

            def run(self, env, model_output, **kwargs):
                self.outputs.append(model_output)
                return super().run(env, model_output, **kwargs)

        dg = RecordingDeepGym()
        env = Environment(task='Anything', verifier_code='return 1.0\n')

        report = AdversarialTester(dg).test(env, strategies=['empty', 'trivial'])

        assert len(dg.outputs) == 2
        assert report.exploits_found == 2
