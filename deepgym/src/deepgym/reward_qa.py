"""Verifier audit utilities for reward QA and red-teaming workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from deepgym.adversarial import AdversarialReport, AdversarialTester
from deepgym.exploit_db import ExploitDB, ExploitRecord
from deepgym.models import Environment

RiskLevel = Literal['low', 'medium', 'high', 'critical']


class VerifierAuditReport(BaseModel):
    """Structured report for a verifier audit."""

    verifier_id: str
    benchmark: str = 'custom'
    attack_type: Literal['white-box'] = 'white-box'
    verifier_hash: str
    exploitable: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    patterns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    attacks_run: int = Field(ge=0)
    exploits_found: int = Field(ge=0)
    report: AdversarialReport
    stored: bool = False


def fingerprint_verifier(verifier_code: str) -> str:
    """Return a stable short hash for a verifier."""
    return hashlib.sha256(verifier_code.encode('utf-8')).hexdigest()[:16]


def classify_patterns(
    verifier_code: str,
    report: AdversarialReport | None = None,
) -> list[str]:
    """Classify likely verifier weaknesses from source and attack outcomes."""
    patterns: list[str] = []
    code_lower = verifier_code.lower()

    if 'random' not in code_lower and 'seed' not in code_lower:
        patterns.append('static-inputs')

    assert_count = code_lower.count('assert') + code_lower.count('==')
    if assert_count < 5:
        patterns.append('few-test-cases')

    if all(token not in code_lower for token in ('inspect', 'source', 'ast')):
        patterns.append('output-only-check')

    if all(token not in code_lower for token in ('type(', 'isinstance')):
        patterns.append('no-type-validation')

    if all(token not in code_lower for token in ('forbidden', 'subprocess', 'os.', 'builtins')):
        patterns.append('no-import-restriction')

    if report is not None and not report.is_robust:
        for result in report.results:
            if not result.exploited:
                continue
            label = result.strategy.lower()
            if 'hardcoded' in label:
                patterns.append('predictable-tests')
            if 'empty' in label or 'trivial' in label:
                patterns.append('shallow-symbol-check')
            if 'overflow' in label:
                patterns.append('numeric-edge-acceptance')
            if 'pattern' in label:
                patterns.append('structure-leakage')

    # Preserve order while deduplicating.
    return list(dict.fromkeys(patterns))


def recommend_fixes(patterns: list[str], report: AdversarialReport) -> list[str]:
    """Map detected patterns to actionable remediations."""
    suggestions: list[str] = []

    mapping = {
        'static-inputs': 'Add randomized or parameterized cases and log the verifier seed.',
        'few-test-cases': 'Increase test coverage with edge cases and anti-cheat assertions.',
        'output-only-check': (
            'Check implementation behavior or intermediate artifacts, not only output.'
        ),
        'no-type-validation': 'Validate return types and schema before assigning reward.',
        'no-import-restriction': 'Block reflection, dangerous imports, and side-effectful modules.',
        'predictable-tests': 'Move high-value checks into private holdouts or rotating canaries.',
        'shallow-symbol-check': (
            'Assert semantic correctness instead of only checking symbol presence.'
        ),
        'numeric-edge-acceptance': (
            'Guard against NaN, Inf, and extreme numeric values explicitly.'
        ),
        'structure-leakage': (
            'Avoid exposing easily echoed test structure or deterministic templates.'
        ),
    }

    for pattern in patterns:
        recommendation = mapping.get(pattern)
        if recommendation is not None:
            suggestions.append(recommendation)

    if report.exploits_found > 0:
        suggestions.append(
            'Rerun the verifier against private holdouts after patching the weaknesses.'
        )

    return list(dict.fromkeys(suggestions))


def compute_risk_score(patterns: list[str], report: AdversarialReport) -> float:
    """Combine exploit rate and heuristic weakness signals into a single score."""
    attacks_run = max(report.attacks_run, 1)
    exploit_rate = report.exploits_found / attacks_run
    pattern_bonus = min(len(patterns) * 0.05, 0.25)
    score = min(1.0, exploit_rate + pattern_bonus)
    if report.exploits_found > 0:
        score = max(score, 0.55)
    return round(score, 4)


def risk_level_for_score(score: float) -> RiskLevel:
    """Map a risk score to a severity bucket."""
    if score >= 0.85:
        return 'critical'
    if score >= 0.65:
        return 'high'
    if score >= 0.35:
        return 'medium'
    return 'low'


def build_exploit_record(
    *,
    verifier_id: str,
    benchmark: str,
    verifier_hash: str,
    patterns: list[str],
    report: AdversarialReport,
) -> ExploitRecord:
    """Convert an audit report into an ExploitDB record."""
    return ExploitRecord(
        verifier_id=verifier_id,
        benchmark=benchmark,
        exploitable=not report.is_robust,
        patterns=patterns,
        num_attacks=report.attacks_run,
        num_exploits=report.exploits_found,
        max_exploit_score=max((r.score for r in report.results if r.exploited), default=0.0),
        attacks=[
            {
                'strategy': r.strategy,
                'score': r.score,
                'exploited': r.exploited,
                'details': r.details,
            }
            for r in report.results
        ],
        verifier_hash=verifier_hash,
        attack_type='white-box',
    )


def resolve_verifier_source(env: Environment) -> str:
    """Return the verifier source code for an environment."""
    if env.verifier_code:
        return env.verifier_code
    if env.verifier_path is not None:
        return env.verifier_path.read_text(encoding='utf-8')
    return ''


class RewardAuditor:
    """Run white-box verifier audits and optionally persist the results."""

    def __init__(
        self,
        dg,
        *,
        pass_threshold: float = 0.5,
        db: ExploitDB | None = None,
    ) -> None:
        self._tester = AdversarialTester(dg, pass_threshold=pass_threshold)
        self._db = db

    def audit(
        self,
        env: Environment,
        *,
        verifier_id: str = '',
        benchmark: str = 'custom',
        strategies: list[str] | None = None,
        persist: bool = False,
        db_path: Path | None = None,
    ) -> VerifierAuditReport:
        """Run a verifier audit and return a structured QA report."""
        report = self._tester.test(env, strategies=strategies)
        verifier_source = resolve_verifier_source(env)
        resolved_verifier_id = verifier_id or env.task[:80]
        verifier_hash = fingerprint_verifier(verifier_source)
        patterns = classify_patterns(verifier_source, report)
        risk_score = compute_risk_score(patterns, report)

        stored = False
        if persist:
            db, should_close = self._open_db(db_path)
            try:
                db.save(
                    build_exploit_record(
                        verifier_id=resolved_verifier_id,
                        benchmark=benchmark,
                        verifier_hash=verifier_hash,
                        patterns=patterns,
                        report=report,
                    )
                )
                stored = True
            finally:
                if should_close:
                    db.close()

        return VerifierAuditReport(
            verifier_id=resolved_verifier_id,
            benchmark=benchmark,
            attack_type='white-box',
            verifier_hash=verifier_hash,
            exploitable=not report.is_robust,
            risk_score=risk_score,
            risk_level=risk_level_for_score(risk_score),
            patterns=patterns,
            recommendations=recommend_fixes(patterns, report),
            attacks_run=report.attacks_run,
            exploits_found=report.exploits_found,
            report=report,
            stored=stored,
        )

    def _open_db(self, db_path: Path | None) -> tuple[ExploitDB, bool]:
        if self._db is not None:
            return self._db, False
        return ExploitDB(db_path=db_path), True
