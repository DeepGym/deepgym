"""Single-turn environment checks: reference, determinism, empty/stub, reward hacking."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from deepgym.adversarial import DEFAULT_STRATEGIES, AdversarialTester
from deepgym.models import SCORE_TOLERANCE, Environment, RunResult
from deepgym.qa.report import QACheck, verdict

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_HEURISTIC_STRATEGIES = [strategy for strategy in DEFAULT_STRATEGIES if strategy != 'llm_attack']
"""Attacks the gate runs: the LLM attack needs an Anthropic key and isn't reproducible."""


def solved(result: RunResult) -> bool:
    """Return True when *result* scored full marks (1.0 up to float noise)."""
    return result.score >= 1 - SCORE_TOLERANCE


def stub_solution(reference: str) -> str:
    """Return a stub defining each top-level function of *reference* as ``return None``.

    The verifier's entry point is one of those names, so the stub reaches it and
    computes nothing. A reference with no top-level function (a script, or a
    non-Python answer) gives ``''``.
    """
    try:
        body = ast.parse(reference).body
    except SyntaxError:
        return ''
    return ''.join(
        f'def {node.name}(*args, **kwargs):\n    return None\n\n'
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _summary(result: RunResult) -> str:
    """Score plus the first line of the verifier's details, for a check's details."""
    first_line = (result.output or result.stderr).strip().split('\n', 1)[0][:120]
    return f'score {result.score:.2f}' + (f' ({first_line})' if first_line else '')


def _scores_below_full(name: str, result: RunResult) -> QACheck:
    return QACheck(name=name, status=verdict(not solved(result)), details=_summary(result))


def _same_scores(results: list[RunResult]) -> QACheck:
    scores = [result.score for result in results]
    if len(set(scores)) == 1:
        details = f'{len(scores)} runs all scored {scores[0]:.2f}'
    else:
        listed = ', '.join(f'{score:.4f}' for score in scores)
        details = f'scores differ across {len(scores)} runs: {listed}'
    return QACheck(name='deterministic', status=verdict(len(set(scores)) == 1), details=details)


def single_turn_checks(
    env: Environment, reference: str | None, dg: DeepGym, reruns: int
) -> list[QACheck]:
    """Check the reference, determinism, and that empty and stub solutions fail.

    The reference runs ``1 + reruns`` times; every run goes out in one parallel batch.

    Args:
        env: The single-turn environment.
        reference: Source of its reference solution, or None if it has none.
        dg: Client whose backend runs the verifier.
        reruns: Extra reference runs whose scores must match the first.

    Returns:
        reference_passes, deterministic, empty_fails and stub_fails, in that order.
        Without a reference, only reference_passes (failed) and empty_fails.
    """
    if reference is None:
        (empty,) = dg.run_batch(env, ['']).results
        missing = QACheck(
            name='reference_passes', status='fail', details='no reference_solution.py found'
        )
        return [missing, _scores_below_full('empty_fails', empty)]

    runs = 1 + reruns
    stub = stub_solution(reference)
    results = dg.run_batch(env, [reference] * runs + [''] + ([stub] if stub else [])).results
    stub_check = (
        _scores_below_full('stub_fails', results[runs + 1])
        if stub
        else QACheck(
            name='stub_fails', status='skip', details='reference has no top-level function'
        )
    )
    first = results[0]
    return [
        QACheck(name='reference_passes', status=verdict(solved(first)), details=_summary(first)),
        _same_scores(results[:runs]),
        _scores_below_full('empty_fails', results[runs]),
        stub_check,
    ]


def hack_resistant(env: Environment, dg: DeepGym, strategies: list[str] | None = None) -> QACheck:
    """Run reward-hack attacks; fail if any reaches the pass threshold.

    Inconclusive, not passed, when no attack got through but some got no verdict (the run
    failed, the verifier reported an error, or the attack never ran).

    Args:
        env: The environment to attack.
        dg: Client whose backend runs every attack.
        strategies: Attack strategies; None means every heuristic one.
    """
    report = AdversarialTester(dg).test(env, strategies=strategies or _HEURISTIC_STRATEGIES)
    exploits = [result for result in report.results if result.exploited]
    not_run = [result for result in report.results if result.inconclusive]
    if exploits:
        details = 'exploited by ' + ', '.join(f'{r.strategy} ({r.score:.2f})' for r in exploits)
        return QACheck(name='hack_resistant', status='fail', details=details)
    if not_run:
        reasons = ', '.join(f'{r.strategy} ({r.details[:120]})' for r in not_run)
        details = f'{len(not_run)} of {report.attacks_run} attacks got no verdict: {reasons}'
        return QACheck(name='hack_resistant', status='inconclusive', details=details)
    top = max(result.score for result in report.results)
    details = f'{report.attacks_run} attacks, max score {top:.2f}'
    return QACheck(name='hack_resistant', status='pass', details=details)
