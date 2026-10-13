"""Run every quality check on one environment and collect the verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepgym.models import MultiTurnEnvironment
from deepgym.qa.checks import hack_resistant, single_turn_checks
from deepgym.qa.difficulty import difficulty_check
from deepgym.qa.report import SKIPPED, QACheck, QAReport
from deepgym.registry import load_environment

if TYPE_CHECKING:
    from deepgym.core import DeepGym
    from deepgym.integrations.zai import ZaiChatClient


def run_qa(
    name: str,
    dg: DeepGym,
    reruns: int = 3,
    client: ZaiChatClient | None = None,
    samples: int = 8,
) -> QAReport:
    """Quality-gate one registered environment.

    Multi-turn and computer-use envs (they need an agent) and benchmark-backed
    envs such as swebench_pro (they score through their own runner) get a single
    'skipped' check with status 'skip', so the report is neither a pass nor a failure.
    Self-contained patch tasks such as mimo_code have no reference patch, so they only
    get the patch-tampering attacks. Harbor tasks with a solution/solve.sh are checked
    with it as the reference; their hack_resistant check is 'skip', because its attacks
    are Python solutions and prove nothing about a bash runner. An environment that
    fails to load gets a failed 'load' check.

    Args:
        name: Registered environment name, e.g. 'coin_change'.
        dg: Client whose backend runs every verifier run, attacks included.
        reruns: Extra reference runs for the determinism check.
        client: Model chat client for the difficulty probe; None skips it.
        samples: Completions to sample for the difficulty probe.

    Returns:
        The environment's QAReport.

    Example:
        >>> run_qa('coin_change', DeepGym(mode='local')).passed
        True
    """
    try:
        env = load_environment(name)
    except ValueError as exc:
        check = QACheck(name='load', status='fail', details=str(exc))
        return QAReport(environment=name, checks=[check])
    if isinstance(env, MultiTurnEnvironment):
        return _skipped(name, 'multi-turn/computer-use env needs an agent')
    # The tests core.run_batch and core.run use to route a run to a benchmark runner. One
    # with a reference solution of its own (a Harbor task's solve.sh) is still checkable.
    reference = env.reference_solution
    if hasattr(env, 'prepare_batch_requests') or (
        hasattr(env, 'run_with_deepgym') and reference is None
    ):
        # A self-contained patch task (e.g. mimo_code) has no reference patch to check,
        # but its runner must still refuse diffs that add test infrastructure.
        if 'patch' in env.tags and not hasattr(env, 'prepare_batch_requests'):
            check = hack_resistant(env, dg, ['patch_tampering'])
            return QAReport(environment=name, checks=[check])
        return _skipped(name, 'benchmark-backed env scores through its own runner')

    checks = single_turn_checks(env, reference, dg, reruns)
    # The heuristic attacks are Python solutions. Any other runner (a Harbor task's bash)
    # would reject them for the wrong reason, so the check is skipped, never passed.
    if env.language == 'python':
        checks.append(hack_resistant(env, dg))
    else:
        details = f'the attacks are Python solutions; this environment runs {env.language}'
        checks.append(QACheck(name='hack_resistant', status='skip', details=details))
    if client is not None:
        checks.append(difficulty_check(env, dg, client, samples))
    return QAReport(environment=name, checks=checks)


def _skipped(name: str, reason: str) -> QAReport:
    """Return a skipped report whose only check says why *name* wasn't checked."""
    details = f'{reason}; the gate checks single-turn envs'
    check = QACheck(name=SKIPPED, status='skip', details=details)
    return QAReport(environment=name, checks=[check])
