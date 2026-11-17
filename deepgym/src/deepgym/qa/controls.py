"""Controls: answers an environment's author knows the verifier must pass or fail.

An env folder may ship ``controls/must_pass/*`` and ``controls/must_fail/*``. Each file
is an answer in the reference's form (a solution file, a diff for a repo patch, an
episode like reference_actions.json for a multi-turn env), and it runs exactly as the
reference does. Every run of a must_pass control must score 1.0, and every run of a
must_fail one must score below it, each with a verdict: a run the verifier never judged
fails the check. A control may start with a line ``# expect: TEXT``; every run's details,
or one of its cases' errors, must then contain TEXT, so a must_fail control that fails
for another reason (a typo in it, say) doesn't count. That line is removed before the
control runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import BaseModel, ConfigDict

from deepgym.models import (
    CONTROLS_DIR,
    REFERENCE_ACTIONS,
    Action,
    Environment,
    MultiTurnEnvironment,
    RunResult,
)
from deepgym.qa.checks import _summary, episode, solved
from deepgym.qa.report import CONTROLS_CHECK, QACheck
from deepgym.registry import parse_actions

if TYPE_CHECKING:
    from deepgym.core import DeepGym

ControlKind = Literal['must_pass', 'must_fail']
_KINDS: tuple[ControlKind, ...] = get_args(ControlKind)

_EXPECT = '# expect:'
"""Prefix of a control's optional first line, naming text its runs' details must contain."""


class Control(BaseModel):
    """One file under controls/: an answer, and what every run of it must show."""

    model_config = ConfigDict(frozen=True)

    kind: ControlKind
    name: str
    answer: str
    """The file's text, without its expect line."""
    expect: str | None = None
    """Text every run's details or case errors must contain, if the file names one."""

    @property
    def label(self) -> str:
        """Where the control lives under controls/, e.g. 'must_fail/nan_string.py'."""
        return f'{self.kind}/{self.name}'


def _read_controls(folder: Path) -> tuple[list[Control], list[str]]:
    """Read the controls in ``folder/controls``, and list whatever else is in there.

    Hidden files (``.DS_Store``) are ignored. Anything else that isn't a file directly in
    must_pass/ or must_fail/ is stray: a misnamed folder would otherwise never run.

    Returns:
        The controls, sorted by path, and the stray paths relative to controls/.
    """
    root = folder / CONTROLS_DIR
    controls: list[Control] = []
    stray: list[str] = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part.startswith('.') for part in rel.parts):
            continue
        if rel.parts[0] not in _KINDS or path.is_file() != (len(rel.parts) == 2):
            stray.append(rel.as_posix())
        elif path.is_file():
            text = path.read_text(encoding='utf-8')
            first, _, rest = text.partition('\n')
            expect = first.removeprefix(_EXPECT).strip() if first.startswith(_EXPECT) else None
            answer = text if expect is None else rest
            control = Control(
                kind=rel.parts[0], name=path.name, answer=answer, expect=expect or None
            )
            controls.append(control)
    return controls, stray


def controls_check(
    env: Environment | MultiTurnEnvironment, folder: Path, dg: DeepGym, runs: int
) -> QACheck | None:
    """Run every control in ``folder/controls`` *runs* times, the way the reference runs.

    A single-turn env runs them all in one parallel batch; a multi-turn env replays each
    as an episode.

    Args:
        env: The environment, loaded from *folder*.
        folder: The environment's folder.
        dg: Client whose backend runs the verifier.
        runs: Runs per control.

    Returns:
        None when the folder has no controls/ (the check is opt-in); 'skip' when it holds
        none; 'fail' naming every control a run went against, and any stray path or
        unreadable episode; else 'pass'.
    """
    root = folder / CONTROLS_DIR
    if not root.is_dir():
        return None
    links = [path.relative_to(folder).as_posix() for path in (root, *root.rglob('*'))]
    links = [rel for rel in links if (folder / rel).is_symlink()]
    if links:  # a link can point at, or from, a file the agent is given
        details = f'{", ".join(links)} is a link; controls must be plain files'
        return QACheck(name=CONTROLS_CHECK, status='fail', details=details)
    controls, stray = _read_controls(folder)
    if stray:
        details = (
            f'{CONTROLS_DIR}/ holds {", ".join(stray)}; put each control directly in '
            f'{CONTROLS_DIR}/must_pass/ or {CONTROLS_DIR}/must_fail/'
        )
        return QACheck(name=CONTROLS_CHECK, status='fail', details=details)
    if not controls:
        details = f'{CONTROLS_DIR}/ holds no files in must_pass/ or must_fail/'
        return QACheck(name=CONTROLS_CHECK, status='skip', details=details)
    if isinstance(env, MultiTurnEnvironment):
        try:
            episodes = [_episode(control) for control in controls]
        except ValueError as exc:
            return QACheck(name=CONTROLS_CHECK, status='fail', details=str(exc))
        outcomes = [[episode(env, actions, dg) for _ in range(runs)] for actions in episodes]
    else:
        results = dg.run_batch(env, [c.answer for c in controls for _ in range(runs)]).results
        outcomes = [results[i * runs : (i + 1) * runs] for i in range(len(controls))]
    problems = [p for c, rs in zip(controls, outcomes) if (p := _problem(c, rs))]
    if problems:
        return QACheck(name=CONTROLS_CHECK, status='fail', details='; '.join(problems))
    passing = sum(control.kind == 'must_pass' for control in controls)
    details = (
        f'{passing} must_pass and {len(controls) - passing} must_fail controls, '
        f'{runs} runs each: every run as stated'
    )
    return QACheck(name=CONTROLS_CHECK, status='pass', details=details)


def _episode(control: Control) -> list[Action]:
    """*control*'s actions.

    Raises:
        ValueError: Naming the control, if it isn't an episode like reference_actions.json.
    """
    try:
        return parse_actions(control.answer)
    except ValueError as exc:
        raise ValueError(
            f'{control.label} is not an episode like {REFERENCE_ACTIONS}: {exc}'
        ) from exc


def _problem(control: Control, results: Sequence[RunResult]) -> str | None:
    """Say which of *control*'s runs went against it, or None if none did."""
    wrong = [(why, result) for result in results if (why := _wrong(control, result))]
    if not wrong:
        return None
    why, result = wrong[0]
    count = f'{len(wrong)} of {len(results)} runs wrong'
    return f'{control.label}: {count}, the first {why}: {_summary(result)}'


def _wrong(control: Control, result: RunResult) -> str | None:
    """Say how *result* goes against *control*, or None if it is what the control says."""
    if not result.has_verdict:
        return 'got no verdict (the verifier errored)'
    if solved(result) != (control.kind == 'must_pass'):
        return 'scored full marks' if solved(result) else 'scored below full marks'
    said = '\n'.join([result.output, *(case.error or '' for case in result.cases or [])])
    if control.expect is not None and control.expect not in said:
        return f'did not say {control.expect!r}'
    return None
