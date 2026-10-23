"""Environment checks: reference, determinism, empty/stub, reward hacking.

Single-turn envs run solutions through the verifier; multi-turn envs replay a
scripted reference episode and an agent that quits at once.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING

from deepgym.adversarial import DEFAULT_STRATEGIES, PASS_THRESHOLD, AdversarialTester
from deepgym.exceptions import DeepGymError, error_type_for
from deepgym.models import (
    MODEL_FAULT_ERRORS,
    SCORE_TOLERANCE,
    Action,
    Environment,
    MultiTurnEnvironment,
    RunResult,
)
from deepgym.qa.report import QACheck, verdict

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_HEURISTIC_STRATEGIES = [strategy for strategy in DEFAULT_STRATEGIES if strategy != 'llm_attack']
"""Attacks the gate runs: the LLM attack needs an Anthropic key and isn't reproducible."""

_DONE = Action(action_type='done')


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
    """Score plus the verifier's details, whole, for a check's details."""
    details = (result.output or result.stderr).strip()
    return f'score {result.score:.2f}' + (f' ({details})' if details else '')


def earns_nothing(name: str, result: RunResult) -> QACheck:
    """Pass if *result*, a run that does nothing toward the task, earned no reward.

    Any reward above float noise fails: a verifier that pays for an empty answer, a
    stub or an idle agent teaches a policy to do nothing. Only a real verdict counts:
    a verifier that crashed or printed no protocol output also scores 0, which says
    nothing about the run, so that is inconclusive.
    """
    if not result.has_verdict:
        details = f'the verifier errored instead of judging the solution: {_summary(result)}'
        return QACheck(name=name, status='inconclusive', details=details)
    if result.score > SCORE_TOLERANCE:
        details = f'{_summary(result)}: earned reward without doing the task'
        return QACheck(name=name, status='fail', details=details)
    return QACheck(name=name, status='pass', details=_summary(result))


def _same_scores(results: Sequence[RunResult]) -> QACheck:
    scores = [result.score for result in results]
    if len(scores) < 2:
        details = 'one run says nothing about determinism; rerun the reference (--reruns 1+)'
        return QACheck(name='deterministic', status='skip', details=details)
    if not all(result.has_verdict for result in results):
        details = 'a reference run got no verdict (the verifier errored), so this is unknown'
        return QACheck(name='deterministic', status='inconclusive', details=details)
    if len(set(scores)) == 1:
        details = f'{len(scores)} runs all scored {scores[0]:.2f}'
    else:
        listed = ', '.join(f'{score:.4f}' for score in scores)
        details = f'scores differ across {len(scores)} runs: {listed}'
    return QACheck(name='deterministic', status=verdict(len(set(scores)) == 1), details=details)


def run_verdicts(results: Sequence[RunResult], runs: int) -> list[QACheck]:
    """Judge reference runs ``results[:runs]``, the empty run after them, then any stub run.

    Returns:
        reference_passes, deterministic, empty_fails, and stub_fails if a stub ran.
    """
    first = results[0]
    reference = QACheck(
        name='reference_passes', status=verdict(solved(first)), details=_summary(first)
    )
    if not first.has_verdict:
        details = f'the verifier errored instead of judging the reference: {_summary(first)}'
        reference = QACheck(name='reference_passes', status='inconclusive', details=details)
    return [
        reference,
        _same_scores(results[:runs]),
        earns_nothing('empty_fails', results[runs]),
        *(earns_nothing('stub_fails', stub) for stub in results[runs + 1 :]),
    ]


def single_turn_results(
    env: Environment, reference: str, dg: DeepGym, reruns: int
) -> list[RunResult]:
    """Run the reference ``1 + reruns`` times, then '' and its stub, in one parallel batch."""
    stub = stub_solution(reference)
    return dg.run_batch(env, [reference] * (1 + reruns) + [''] + ([stub] if stub else [])).results


def single_turn_verdicts(results: Sequence[RunResult], reruns: int) -> list[QACheck]:
    """Judge :func:`single_turn_results`; stub_fails is 'skip' when there was no stub."""
    checks = run_verdicts(results, 1 + reruns)
    if len(results) == reruns + 2:
        details = 'reference has no top-level function'
        checks.append(QACheck(name='stub_fails', status='skip', details=details))
    return checks


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
        return [missing, earns_nothing('empty_fails', empty)]
    return single_turn_verdicts(single_turn_results(env, reference, dg, reruns), reruns)


def episode(env: MultiTurnEnvironment, actions: Sequence[Action], dg: DeepGym) -> RunResult:
    """Replay *actions* (then 'done') as one episode; a run that errors has no verdict."""
    replay = iter(actions)
    try:
        return dg.run_episode(env, lambda _: next(replay, _DONE))[1]
    except DeepGymError as exc:
        return RunResult(
            score=0.0,
            passed=False,
            output=f'{type(exc).__name__}: {exc}',
            stderr='',
            exit_code=2,
            execution_time_ms=0.0,
            sandbox_id='error',
            error_type=error_type_for(exc),
        )


def episode_results(
    env: MultiTurnEnvironment, actions: Sequence[Action], dg: DeepGym, reruns: int
) -> list[RunResult]:
    """Replay the reference actions ``1 + reruns`` times, then an agent that quits at once.

    Episodes run one after another on *dg*'s backend. Local sessions are not isolated
    (they run in a host temp directory), so the reference actions run as the author.
    """
    return [*(episode(env, actions, dg) for _ in range(1 + reruns)), episode(env, [], dg)]


def hack_resistant(env: Environment, dg: DeepGym, strategies: list[str] | None = None) -> QACheck:
    """Run reward-hack attacks; fail if any reaches the exploit threshold (0.50).

    Inconclusive, not passed, when no attack got through but some got no verdict (the run
    failed, the verifier reported an error, or the attack never ran). Skipped, not passed,
    when the verifier rejected every default attack as a malformed answer (an error_type
    such as ``patch_apply_failed``): those attacks are Python solutions, and an env
    graded on another answer format (a repo patch's diff) was never exercised by them.
    The verifier's own verdicts decide this, never a guess from the env's files.

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
    malformed = all(r.error_type in MODEL_FAULT_ERRORS for r in report.results)
    if strategies is None and report.results and malformed:
        details = (
            f'the verifier rejected every attack as a malformed answer '
            f'({report.results[0].error_type}): the attacks are Python solutions, and this '
            'environment takes another answer format, so nothing was attacked'
        )
        return QACheck(name='hack_resistant', status='skip', details=details)
    if not_run:
        reasons = ', '.join(f'{r.strategy} ({r.details})' for r in not_run)
        details = f'{len(not_run)} of {report.attacks_run} attacks got no verdict: {reasons}'
        return QACheck(name='hack_resistant', status='inconclusive', details=details)
    top = max(result.score for result in report.results)
    details = (
        f'{report.attacks_run} attacks, max score {top:.2f} (an exploit is {PASS_THRESHOLD:.2f})'
    )
    return QACheck(name='hack_resistant', status='pass', details=details)
