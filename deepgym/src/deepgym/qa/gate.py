"""Run every quality check on one environment and collect the verdict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import Action, MultiTurnEnvironment
from deepgym.qa.checks import episode_results, hack_resistant, run_verdicts, single_turn_checks
from deepgym.qa.difficulty import difficulty_check
from deepgym.qa.mutants import mutation_check
from deepgym.qa.report import DIFFICULTY_CHECK, MUTANTS_CHECK, SKIPPED, QACheck, QAReport
from deepgym.registry import load_environment, load_environment_dir

if TYPE_CHECKING:
    from deepgym.core import DeepGym

REFERENCE_ACTIONS = 'reference_actions.json'
"""A multi-turn env's scripted reference episode: a JSON list of actions."""


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


def read_reference_actions(path: Path) -> list[Action]:
    """Parse a reference_actions.json file into actions.

    Raises:
        ValueError: If it isn't a JSON list of valid actions.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('expected a JSON list of actions')
    return [Action.model_validate(item) for item in data]


def run_qa(target: str | Path, dg: DeepGym, options: QAOptions | None = None) -> QAReport:
    """Quality-gate one registered environment, or the environment folder *target*.

    Computer-use envs (they need a desktop agent), multi-turn envs without a
    reference_actions.json, and benchmark-backed envs such as swebench_pro (they
    score through their own runner) get a single 'skipped' check with status 'skip',
    so the report is neither a pass nor a failure. Self-contained patch tasks such as
    mimo_code have no reference patch, so they only get the patch-tampering attacks.
    A multi-turn folder with reference actions replays them. Harbor tasks with a solution/solve.sh
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
        actions = target / REFERENCE_ACTIONS if isinstance(target, Path) else None
        if env.desktop or actions is None or not actions.is_file():
            return _skipped(label, 'multi-turn/computer-use env needs an agent')
        return QAReport(environment=label, checks=_multi_turn_checks(env, actions, dg, options))
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


def _multi_turn_checks(
    env: MultiTurnEnvironment, actions_path: Path, dg: DeepGym, options: QAOptions
) -> list[QACheck]:
    """Replay the reference episode and an agent that quits at once."""
    try:
        actions = read_reference_actions(actions_path)
    except ValueError as exc:
        details = f'{REFERENCE_ACTIONS}: {exc}'
        return [QACheck(name='reference_passes', status='fail', details=details)]
    checks = run_verdicts(episode_results(env, actions, dg, options.reruns), 1 + options.reruns)
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
    details = f'{reason}; the gate checks single-turn envs and multi-turn folders'
    check = QACheck(name=SKIPPED, status='skip', details=details)
    return QAReport(environment=name, checks=[check])
