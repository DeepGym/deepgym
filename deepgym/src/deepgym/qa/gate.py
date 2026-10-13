"""Run every quality check on one environment and collect the verdict."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import REFERENCE_ACTIONS, MultiTurnEnvironment
from deepgym.qa.agents import do_everything, random_agent
from deepgym.qa.checks import (
    earns_nothing,
    episode,
    episode_results,
    hack_resistant,
    run_verdicts,
    single_turn_checks,
)
from deepgym.qa.difficulty import difficulty_check
from deepgym.qa.mutants import mutation_check
from deepgym.qa.report import DIFFICULTY_CHECK, MUTANTS_CHECK, SKIPPED, QACheck, QAReport
from deepgym.registry import load_environment, load_environment_dir

if TYPE_CHECKING:
    from deepgym.core import DeepGym


class QAOptions(BaseModel):
    """How hard run_qa tries, and which optional checks it adds."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    reruns: int = Field(default=3, ge=0)
    """Extra reference runs for the determinism check."""
    client: ZaiChatClient | None = None
    """Model chat client for the difficulty probe; None skips it."""
    samples: int = Field(default=8, ge=2)
    """Completions to sample for the difficulty probe."""
    mutants: bool = False
    """Add the mutation check (see :mod:`deepgym.qa.mutants`)."""


def run_qa(target: str | Path, dg: DeepGym, options: QAOptions | None = None) -> QAReport:
    """Quality-gate one registered environment, or the environment folder *target*.

    Multi-turn and computer-use envs with a reference_actions.json replay it and
    run scripted agents (one that quits at once, one that does everything, one that
    acts at random), each of which must earn nothing. Without one, or for a
    computer-use env off Daytona (it needs a desktop), and for benchmark-backed envs
    such as swebench_pro (they score through their own runner), the report has a
    single 'skipped' check with status 'skip': neither a pass nor a failure.
    Self-contained patch tasks such as mimo_code have no reference patch, so they
    only get the patch-tampering attacks. Harbor tasks with a solution/solve.sh
    are checked with it as the reference; their hack_resistant and mutants_killed
    checks are 'skip', because both run Python solutions and prove nothing about a
    bash runner. A Python env whose verifier rejects every attack as a malformed
    answer (a repo-patch env, which takes diffs) gets hack_resistant 'skip' too. A
    skipped check counts neither way and is listed in the report's ``not_run``. An
    environment that fails to load gets a failed 'load' check.

    Args:
        target: Registered environment name, e.g. 'coin_change', or a folder Path.
        dg: Client whose backend runs every verifier run, attacks included.
        options: Reruns, difficulty probe and mutation testing; defaults if None.

    Returns:
        The environment's QAReport, named after the folder for a Path.

    Example:
        >>> run_qa('coin_change', DeepGym(mode='local')).passed
        True
    """
    options = options or QAOptions()
    label = target.resolve().name if isinstance(target, Path) else target
    try:
        env = load_environment_dir(target) if isinstance(target, Path) else load_environment(target)
    except (ValueError, OSError) as exc:
        check = QACheck(name='load', status='fail', details=str(exc))
        return QAReport(environment=label, checks=[check])
    if isinstance(env, MultiTurnEnvironment):
        if env.reference_actions is None:
            return _skipped(label, f'multi-turn env without {REFERENCE_ACTIONS} needs an agent')
        if env.desktop and dg.backend != 'daytona':
            return _skipped(label, 'computer-use env needs a desktop: run with --mode daytona')
        return QAReport(environment=label, checks=_multi_turn_checks(env, dg, options))
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
            return QAReport(environment=label, checks=[check])
        return _skipped(label, 'benchmark-backed env scores through its own runner')

    checks = single_turn_checks(env, reference, dg, options.reruns)
    # The heuristic attacks and the mutants are Python solutions. A runner for another
    # language (a Harbor task's bash) would reject them for the wrong reason, so both
    # are skipped there; for a Python env, the attacks' own verdicts decide (see
    # hack_resistant), so QA never classifies an env differently from how it is graded.
    python = env.language == 'python'
    if python:
        checks.append(hack_resistant(env, dg))
    else:
        details = f'the attacks are Python solutions; this environment runs {env.language}'
        checks.append(QACheck(name='hack_resistant', status='skip', details=details))
    if options.mutants:
        checks.append(mutation_check(env, (reference or '') if python else '', dg))
    if options.client is not None:
        checks.append(difficulty_check(env, dg, options.client, options.samples))
    return QAReport(environment=label, checks=checks)


def _multi_turn_checks(env: MultiTurnEnvironment, dg: DeepGym, options: QAOptions) -> list[QACheck]:
    """Replay the reference episode, then scripted agents that must earn nothing.

    empty_fails is an agent that quits at once, busy_fails one that uses every
    allowed action in turn and random_fails one that acts at random (seeded), each for
    the whole step budget without working on the task.
    """
    actions = env.reference_actions or []
    checks = run_verdicts(episode_results(env, actions, dg, options.reruns), 1 + options.reruns)
    checks.append(earns_nothing('busy_fails', episode(env, do_everything(env), dg)))
    checks.append(earns_nothing('random_fails', episode(env, random_agent(env), dg)))
    # Named as skipped, so the report's not_run says what this evidence leaves out.
    skipped = {
        'stub_fails': 'a multi-turn reference is a list of actions; there is no stub',
        'hack_resistant': 'the attacks are single-turn Python solutions; this env needs an agent',
    }
    if options.mutants:
        skipped[MUTANTS_CHECK] = 'a multi-turn reference is a list of actions, not code'
    if options.client is not None:
        skipped[DIFFICULTY_CHECK] = 'the probe samples single-turn answers; this env needs an agent'
    checks += [QACheck(name=name, status='skip', details=why) for name, why in skipped.items()]
    return checks


def _skipped(name: str, reason: str) -> QAReport:
    """Return a skipped report whose only check says why *name* wasn't checked."""
    details = f'{reason}; the gate checks single-turn envs and replayable multi-turn ones'
    check = QACheck(name=SKIPPED, status='skip', details=details)
    return QAReport(environment=name, checks=[check])
