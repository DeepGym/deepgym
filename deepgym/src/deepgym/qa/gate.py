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
    envs such as swebench_pro or mimo_code (they score through their own runner)
    get a single 'skipped' check with status 'skip', so the report is neither a
    pass nor a failure. An environment that fails to load gets a failed 'load' check.

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
    # The tests core.run_batch and core.run use to route a run to a benchmark runner.
    if hasattr(env, 'prepare_batch_requests') or hasattr(env, 'run_with_deepgym'):
        return _skipped(name, 'benchmark-backed env scores through its own runner')

    checks = single_turn_checks(env, env.reference_solution, dg, reruns)
    checks.append(hack_resistant(env, dg))
    if client is not None:
        checks.append(difficulty_check(env, dg, client, samples))
    return QAReport(environment=name, checks=checks)


def _skipped(name: str, reason: str) -> QAReport:
    """Return a skipped report whose only check says why *name* wasn't checked."""
    details = f'{reason}; the gate checks single-turn envs'
    check = QACheck(name=SKIPPED, status='skip', details=details)
    return QAReport(environment=name, checks=[check])

