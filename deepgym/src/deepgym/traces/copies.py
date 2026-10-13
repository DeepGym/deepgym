"""Reconcile copies of one trace stored in several places, such as the console and disk."""

from __future__ import annotations

from collections.abc import Sequence

from deepgym.models import RunResult
from deepgym.traces.capture import UNFINISHED
from deepgym.traces.schema import Trace


def merge_copies(traces: Sequence[Trace]) -> list[Trace]:
    """Keep one copy of each trace, the one that got furthest, if its copies agree.

    A trace can be stored twice, e.g. in the console mid-episode and on disk once it
    ended. Copies agree when each is the furthest one cut short: the same start, steps
    and grades that begin the furthest one's, and, once finished, nothing different.

    Raises:
        ValueError: If copies of a trace disagree: keeping either would drop the other.
    """
    copies: dict[str, list[Trace]] = {}
    for trace in traces:
        copies.setdefault(trace.trace_id, []).append(trace)
    kept: list[Trace] = []
    conflicts: list[str] = []
    for trace_id, group in copies.items():
        furthest = max(group, key=_progress)
        if all(_cut_short(copy, furthest) for copy in group):
            kept.append(furthest)
        else:
            conflicts.append(trace_id)
    if conflicts:
        raise ValueError(f'Copies of traces {conflicts} disagree; keep the right copy of each')
    return kept


def _grades(trace: Trace) -> tuple[RunResult, ...]:
    return () if trace.result is None else (*trace.prior_results, trace.result)


def _progress(trace: Trace) -> tuple[int, int, bool, bool, int]:
    """How far a copy got: steps, grades, whether it started and finished, then what it says
    stopped it (a real error over the loader's guess that the process died)."""
    error = trace.error
    return (
        len(trace.steps),
        len(_grades(trace)),
        trace.initial_observation is not None,
        trace.finished_at is not None,
        0 if error is None else 1 if error == UNFINISHED else 2,
    )


def _cut_short(copy: Trace, furthest: Trace) -> bool:
    """Whether *copy* is *furthest* recorded earlier, or the same trace.

    A traced episode takes no action once scored, so a graded copy has every step, and it
    never clears an error, so one the copy records stays (or a later one replaces it); the
    loader's crash marker doesn't show it, so keeping that copy would drop it.
    """
    if copy.schema_version == '1.0' and copy.result is not None and not copy.prior_results:
        # Schema 1.0 kept only the last grade in memory, so a console copy can lack the
        # earlier grades its event log has: take those from the furthest copy.
        history = _grades(furthest)
        if copy.result not in history:
            return False
        finished = copy.finished_at is not None
        position = len(history) - 1 if finished else history.index(copy.result)
        copy = copy.model_copy(update={'prior_results': history[:position]})
    if copy.finished_at is not None:
        return copy == furthest
    if copy.error != furthest.error and _progress(copy) == _progress(furthest):
        return False  # neither got further, so neither error is known to be the later one
    grades = _grades(copy)
    steps = furthest.steps if grades else furthest.steps[: len(copy.steps)]
    erred = copy.error not in (None, UNFINISHED)
    return (
        (copy.schema_version, copy.task, copy.provenance)
        == (furthest.schema_version, furthest.task, furthest.provenance)
        and copy.initial_observation in (None, furthest.initial_observation)
        and copy.steps == steps
        and grades == _grades(furthest)[: len(grades)]
        and not (erred and furthest.error in (None, UNFINISHED))
    )
