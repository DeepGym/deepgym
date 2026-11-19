"""Tests for the dataset builder: filters, verdicts, bands, labels, dedup, splits, manifest, CLI."""

import hashlib
import json
import os
import random
import stat
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.cli import main
from deepgym.console.episode_store import read_trace_ids, record_episode
from deepgym.console.label_store import LabelStore, load_jsonl, read_labels
from deepgym.console.schemas import LabelQuery
from deepgym.dataset import (
    MANIFEST,
    Dataset,
    DatasetSpec,
    LabelMatch,
    TaskBand,
    build_dataset,
    manifest_sha256,
    verify_dataset,
    write_dataset,
)
from deepgym.dataset.build import copy_sha256
from deepgym.exceptions import DeepGymError
from deepgym.labels import Label, LabelCreate
from deepgym.models import Action, MultiTurnEnvironment, Observation
from deepgym.traces import EnvFingerprint, Trace
from tests.test_traces_export import _event_log
from tests.trace_samples import STARTED, graded_trace

BASE = graded_trace()
SPLITS = ('train', 'val', 'test')


def _trace(trace_id: str, task: str = 'a', *, passed: bool = True, **changes: object) -> Trace:
    """A graded trace of prompt *task* with its own trajectory."""
    env = MultiTurnEnvironment(
        name=changes.pop('env_name', 'write_notes'),
        task=f'Task {task}',
        final_verifier_code='print(1)\n',
    )
    fingerprint = EnvFingerprint.of(env, variant=changes.pop('variant', 2))
    started = STARTED + timedelta(seconds=changes.pop('started', len(trace_id)))
    agent = BASE.provenance.agent.model_copy(
        update={'name': changes.pop('agent', 'react-shell'), 'model': changes.pop('model', 'm1')}
    )
    provenance = BASE.provenance.model_copy(
        update={'env': fingerprint, 'agent': agent, 'started_at': started}
    )
    action = Action(action_type='bash', content=changes.pop('command', f'echo {trace_id}'))
    first = BASE.steps[0].model_copy(update={'action': action})
    score = changes.pop('score', 1.0 if passed else 0.0)
    result = BASE.result.model_copy(update={'score': score, 'passed': passed})
    fields = {'trace_id': trace_id, 'task': env.task, 'provenance': provenance,
              'steps': (first, *BASE.steps[1:]), 'result': result}  # fmt: skip
    return BASE.model_copy(update=fields | changes)


def _unjudged(trace_id: str, task: str = 'a', **changes: object) -> Trace:
    trace = _trace(trace_id, task, passed=False, **changes)
    failure = {'error_type': 'sandbox_error', 'exit_code': 2}
    return trace.model_copy(update={'result': trace.result.model_copy(update=failure)})


def _kept(dataset: Dataset) -> list[str]:
    return list(dataset.entries)


def _build(traces: list[Trace], labels: list[Label] | None = None, **spec: object) -> Dataset:
    episodes = spec.pop('episodes', None)
    spec.setdefault('test_ratio', 0.0)
    spec.setdefault('val_ratio', 0.0)
    return build_dataset(traces, DatasetSpec.model_validate(spec), labels or [], episodes)


def _store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / 'console.db')


def _labels(store: LabelStore) -> list[Label]:
    return store.page(LabelQuery(), 0, None).labels


def _invalidate(target_id: str, kind: str = 'trace') -> LabelCreate:
    return LabelCreate(target={'kind': kind, 'id': target_id}, type='invalidate',
                       source='human', author='ana', comment='verifier was broken')  # fmt: skip


def _verdict(target_id: str, passed: bool, **changes: object) -> LabelCreate:
    fields = {'target': {'kind': 'trace', 'id': target_id}, 'name': 'passed', 'value': passed,
              'source': 'human', 'author': 'ana', 'comment': 'checked by hand'}  # fmt: skip
    return LabelCreate.model_validate(fields | changes)


def _retract(label: Label, author: str = 'ana') -> LabelCreate:
    return LabelCreate(target=label.target, type='retract', supersedes=label.id,
                       source='human', author=author, comment='withdrawn')  # fmt: skip


def _u(seed: int, task: str) -> float:
    """Where the documented split hash puts prompt *task* (as ``_trace`` writes it)."""
    key = hashlib.sha256(f'Task {task}'.encode()).hexdigest()
    return int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), 16) / 2**256


class TestSpec:
    def test_round_trips_as_json(self) -> None:
        spec = DatasetSpec(
            min_score=0.5,
            passed=True,
            agents=('react-shell',),
            models=('m1',),
            bands=('learnable',),
            min_samples=4,
            band_range=(0.2, 0.8),
            inside_band=True,
            require_labels=(LabelMatch(name='passed', value=True, source='human'),),
            env_names=('write_notes',),
            max_cost_usd=0.01,
            seed=3,
        )

        assert DatasetSpec.model_validate_json(spec.model_dump_json()) == spec

    @pytest.mark.parametrize(
        'fields',
        [
            {'min_scor': 0.5},
            {'min_score': 0.8, 'max_score': 0.2},
            {'min_score': 1.5},
            {'test_ratio': 0.5, 'val_ratio': 0.5},
            {'bands': []},
            {'bands': ['easy']},
            {'band_range': [0.9, 0.1]},
            {'min_samples': 0},
            {'max_cost_usd': -1},
            {'models': 'm1'},
        ],
    )
    def test_rejects_bad_specs(self, fields: dict) -> None:
        with pytest.raises(ValidationError):
            DatasetSpec.model_validate(fields)


class TestFilters:
    def test_drops_traces_without_a_verdict_by_default(self) -> None:
        traces = [_trace('t1'), _trace('t2', passed=False), _unjudged('t3'),
                  _trace('t4', result=None)]  # fmt: skip

        dataset = _build(traces)

        assert _kept(dataset) == ['t1', 't2']
        assert dataset.dropped == {'incomplete': ('t4',), 'no_verdict': ('t3',)}
        assert dataset.counts.sources == 4

    def test_keeps_traces_without_a_verdict_when_asked(self) -> None:
        traces = [_trace('t1'), _unjudged('t2'), _trace('t3', result=None)]

        dataset = _build(traces, require_verdict=False)

        assert _kept(dataset) == ['t1', 't2', 't3']
        assert dataset.entries['t2'].passed is None

    def test_a_trace_without_a_verdict_never_counts_as_a_fail(self) -> None:
        traces = [_trace('t1'), _trace('t2', passed=False), _unjudged('t3')]

        dataset = _build(traces, require_verdict=False, passed=False)

        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'no_verdict': ('t3',), 'passed': ('t1',)}

    def test_score_range_is_inclusive(self) -> None:
        traces = [_trace('t1', score=0.2, passed=False), _trace('t2', score=0.5),
                  _trace('t3', score=0.9)]  # fmt: skip

        dataset = _build(traces, min_score=0.5, max_score=0.9)

        assert _kept(dataset) == ['t2', 't3']
        assert dataset.dropped == {'score': ('t1',)}

    @pytest.mark.parametrize(('passed', 'kept', 'reason'), [(True, 't1', 'failed'),
                                                             (False, 't2', 'passed')])  # fmt: skip
    def test_passed_or_failed_only(self, passed: bool, kept: str, reason: str) -> None:
        dataset = _build([_trace('t1'), _trace('t2', passed=False)], passed=passed)

        assert _kept(dataset) == [kept]
        assert list(dataset.dropped) == [reason]

    def test_agents_and_models(self) -> None:
        traces = [_trace('t1'), _trace('t2', agent='other'), _trace('t3', model='m2'),
                  _trace('t4', model=None)]  # fmt: skip

        dataset = _build(traces, agents=['react-shell'], models=['m1'])

        assert _kept(dataset) == ['t1']
        assert dataset.dropped == {'agent': ('t2',), 'model': ('t3', 't4')}

    def test_environment_name_and_variant(self) -> None:
        traces = [_trace('t1'), _trace('t2', env_name='other'), _trace('t3', variant=None)]

        dataset = _build(traces, env_names=['write_notes'], env_variants=[2])

        assert _kept(dataset) == ['t1']
        assert dataset.dropped == {'env_name': ('t2',), 'env_variant': ('t3',)}

    def test_usage_complete(self) -> None:
        traces = [_trace('t1'), _trace('t2', usage_complete=False)]

        dataset = _build(traces, require_usage_complete=True)

        assert _kept(dataset) == ['t1']
        assert dataset.dropped == {'usage_incomplete': ('t2',)}

    def test_max_cost_drops_costlier_and_unknown_costs(self) -> None:
        cheap = _trace('t1')
        assert cheap.cost_usd == pytest.approx(0.007)
        unpriced = _trace('t2', usage_complete=False)  # its cost total is unknown
        costly = _trace('t3', unattached_cost_usd=1.0)

        dataset = _build([cheap, unpriced, costly], max_cost_usd=0.01)

        assert _kept(dataset) == ['t1']
        assert dataset.dropped == {'cost_unknown': ('t2',), 'over_cost': ('t3',)}

    def test_each_dropped_trace_counts_once_under_its_first_reason(self) -> None:
        traces = [_trace('t1', model='m2', result=None), _trace('t2', result=None)]

        dataset = _build(traces, models=['m1'])

        assert dataset.dropped == {'model': ('t1',), 'incomplete': ('t2',)}
        assert dataset.counts.kept + sum(dataset.counts.dropped.values()) == 2

    def test_refuses_traces_sharing_an_id(self) -> None:
        with pytest.raises(ValueError, match='share ids'):
            _build([_trace('t1'), _trace('T1', 'b')])


class TestVerdicts:
    def test_a_human_verdict_overrides_the_verifier(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', False))
        store.add(_verdict('t2', True))
        store.add(_verdict('t3', False, source='judge', author='gpt'))  # judges don't override
        traces = [_trace('t1'), _unjudged('t2'), _trace('t3')]

        dataset = _build(traces, _labels(store), passed=True)

        assert _kept(dataset) == ['t2', 't3']
        assert dataset.dropped == {'failed': ('t1',)}
        # A human verdict where the verifier gave none fills it; it reverses nothing.
        assert dataset.entries['t2'].model_dump(
            include={'passed', 'verdict_source', 'score', 'overridden'}
        ) == {'passed': True, 'verdict_source': 'human', 'score': None, 'overridden': False}
        assert dataset.entries['t3'].verdict_source == 'verifier'
        assert dataset.counts.overridden == 0

    def test_humans_who_disagree_contest_a_trace(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', True))
        store.add(_verdict('t1', False, author='bo'))
        store.add(_verdict('t2', True))
        store.add(_verdict('t2', True, author='bo'))

        dataset = _build([_trace('t1'), _trace('t2')], _labels(store))

        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'contested': ('t1',)}

    def test_a_human_verdict_alone_does_not_pass_a_score_filter(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', True))

        dataset = _build([_unjudged('t1')], _labels(store), min_score=0.5)

        assert dataset.dropped == {'no_score': ('t1',)}

    def test_an_overridden_verifier_score_does_not_stand(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', False))

        dataset = _build([_trace('t1'), _trace('t2', 'b')], _labels(store), min_score=0.5)

        assert dataset.dropped == {'no_score': ('t1',)}
        assert dataset.entries['t2'].score == 1.0


class TestBands:
    @staticmethod
    def _traces() -> list[Trace]:
        return [
            _trace('a1', 'a'), _trace('a2', 'a'),                       # always solved
            _trace('b1', 'b'), _trace('b2', 'b', passed=False),         # learnable
            _trace('c1', 'c', passed=False), _trace('c2', 'c', passed=False),  # never solved
            _trace('d1', 'd'), _unjudged('d2', 'd'),                    # always solved
        ]  # fmt: skip

    def test_keeps_tasks_in_the_bands_asked_for(self) -> None:
        dataset = _build(self._traces(), bands=['learnable'])

        assert _kept(dataset) == ['b1', 'b2']
        assert dataset.dropped['band'] == ('a1', 'a2', 'c1', 'c2', 'd1')
        task = _trace('b1', 'b').provenance.env.environment_sha256
        assert dataset.bands[task] == TaskBand(
            samples=2, solved=1, band='learnable', outside_band=False
        )

    def test_no_verdict_does_not_count_as_a_failed_sample(self) -> None:
        dataset = _build(self._traces(), bands=['always_solved'], require_verdict=False)

        assert _kept(dataset) == ['a1', 'a2', 'd1', 'd2']

    def test_bands_come_from_every_selected_trace(self) -> None:
        dataset = _build(self._traces(), bands=['learnable'], passed=True)

        assert _kept(dataset) == ['b1']

    def test_bands_use_the_human_verdict(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('a2', False))

        dataset = _build(self._traces(), _labels(store), bands=['learnable'])

        assert _kept(dataset) == ['a1', 'a2', 'b1', 'b2']

    def test_traces_left_out_by_labels_or_model_are_not_samples(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_invalidate('b2'))
        store.add(LabelCreate(target={'kind': 'trace', 'id': 'c1'}, name='mark',
                              value='reward-hack', source='human', author='ana'))  # fmt: skip
        traces = [*self._traces(), _trace('c3', 'c', model='m2')]

        dataset = _build(traces, _labels(store), bands=['always_solved', 'never_solved'],
                         models=['m1'], forbid_labels=[{'name': 'mark'}])  # fmt: skip

        assert _kept(dataset) == ['a1', 'a2', 'b1', 'c2', 'd1']
        bands = {key: (band.samples, band.band) for key, band in dataset.bands.items()}
        b, c = (_trace('x', task).provenance.env.environment_sha256 for task in 'bc')
        assert bands[b] == (1, 'always_solved') and bands[c] == (1, 'never_solved')

    def test_required_labels_select_after_bands(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        for trace_id in ('h1', 'h2', 'a1'):
            store.add(_verdict(trace_id, True))
        half = [_trace('h1', 'h'), _trace('h2', 'h'), _trace('h3', 'h', passed=False),
                _trace('h4', 'h', passed=False)]  # fmt: skip
        traces = [*half, _trace('a1', 'a'), _trace('a2', 'a')]

        dataset = _build(traces, _labels(store), bands=['learnable'], require_labels=[
            {'name': 'passed', 'value': True, 'source': 'human'}])  # fmt: skip

        assert _kept(dataset) == ['h1', 'h2']
        assert dataset.dropped == {'band': ('a1', 'a2'), 'missing_label': ('h3', 'h4')}
        task = half[0].provenance.env.environment_sha256
        assert (dataset.bands[task].samples, dataset.bands[task].band) == (4, 'learnable')

    def test_bands_are_per_environment(self) -> None:
        traces = [_trace('t1', 'a'), _trace('t2', 'a', passed=False, env_name='other')]

        dataset = _build(traces, bands=['always_solved', 'never_solved'])

        assert _kept(dataset) == ['t1', 't2']

    def test_min_samples_and_outside_band(self) -> None:
        traces = [_trace(f'e{index}', 'e', passed=index < 1) for index in range(10)]
        traces += [_trace('f1', 'f'), _trace('f2', 'f', passed=False)]

        dataset = _build(traces, min_samples=3, inside_band=True, band_range=(0.2, 0.8))

        assert dataset.dropped == {'no_band': ('f1', 'f2'), 'outside_band': tuple(
            f'e{index}' for index in range(10))}  # fmt: skip
        assert {band.solved / band.samples for band in dataset.bands.values()} == {0.1, 0.5}

    def test_a_task_without_verdicts_has_no_band(self) -> None:
        dataset = _build([_unjudged('t1'), _trace('t2', 'b')], require_verdict=False,
                         bands=['always_solved'])  # fmt: skip

        assert dataset.dropped == {'no_band': ('t1',)}


class TestLabels:
    def test_invalidated_traces_are_dropped_and_lifted_ones_kept(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_invalidate('t1'))
        store.add(_retract(store.add(_invalidate('t2')), author='bo'))

        dataset = _build([_trace('t1'), _trace('t2')], _labels(store))

        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'invalidated': ('t1',)}

    def test_episode_labels_reach_the_trace_they_were_recorded_from(self, tmp_path: Path) -> None:
        db = tmp_path / 'console.db'
        episode = {trace_id: record_episode(_trace(trace_id), db_path=db)
                   for trace_id in ('t1', 't2', 't3', 't4')}  # fmt: skip
        store = LabelStore(db)
        store.add(_invalidate(episode['t1'], kind='episode'))
        store.add(_verdict(episode['t2'], False, target={'kind': 'episode', 'id': episode['t2']}))
        # A label on a replay frame counts on the trace, for the label filters.
        store.add(LabelCreate(target={'kind': 'episode', 'id': episode['t3'], 'step': 1},
                              name='mark', value='bad', source='human', author='ana'))  # fmt: skip
        store.add(LabelCreate(target={'kind': 'episode', 'id': episode['t4']}, name='mark',
                              value='bad', source='human', author='ana'))  # fmt: skip
        store.add(_verdict('elsewhere', True))
        traces = [_trace('t1'), _trace('t2'), _trace('t3'), _trace('t4')]
        episodes = read_trace_ids(db)

        dataset = _build(traces, _labels(store), episodes=episodes, passed=True,
                         forbid_labels=[{'name': 'mark'}])  # fmt: skip

        assert episodes == {episode[trace_id]: trace_id for trace_id in episode}
        assert _kept(dataset) == []
        assert dataset.dropped == {'invalidated': ('t1',), 'forbidden_label': ('t3', 't4'),
                                   'failed': ('t2',)}  # fmt: skip
        assert [label.target.id for label in dataset.unmatched] == ['elsewhere']

    def test_require_an_expert_pass(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', True))
        store.add(_verdict('t2', False))
        store.add(_retract(store.add(_verdict('t3', True))))
        corrected = store.add(_verdict('t4', True))
        store.add(_verdict('t4', False, supersedes=corrected.id))
        store.add(_verdict('t5', True, source='judge', author='gpt'))
        traces = [_trace(f't{index}') for index in range(1, 7)]

        dataset = _build(traces, _labels(store), require_labels=[
            {'name': 'passed', 'value': True, 'source': 'human'}])  # fmt: skip

        assert _kept(dataset) == ['t1']
        assert dataset.dropped == {'missing_label': ('t2', 't3', 't4', 't5', 't6')}

    def test_forbid_a_mark_on_any_step(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't1', 'step': 2}, name='mark',
                              value='reward-hack', source='human', author='ana'))  # fmt: skip
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't2', 'step': 1}, name='mark',
                              value='fine', source='human', author='ana'))  # fmt: skip

        dataset = _build([_trace('t1'), _trace('t2')], _labels(store),
                         forbid_labels=[{'name': 'mark', 'value': 'reward-hack'}])  # fmt: skip

        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'forbidden_label': ('t1',)}

    def test_a_bool_value_never_matches_a_number(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't1'}, name='rating', value=1.0,
                              source='human', author='ana'))  # fmt: skip
        label = _labels(store)[0]

        assert LabelMatch(name='rating', value=1).matches(label)
        assert not LabelMatch(name='rating', value=True).matches(label)
        assert not LabelMatch(name='rating', author='bo').matches(label)

    def test_an_older_database_reads_as_empty_and_unchanged(self, tmp_path: Path) -> None:
        import sqlite3

        db = tmp_path / 'old.db'
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE episodes (id TEXT PRIMARY KEY)')
        before = db.read_bytes()

        assert read_labels(db) == [] and read_trace_ids(db) == {}
        assert db.read_bytes() == before

    def test_reads_a_wal_database_in_a_folder_it_cannot_write(self, tmp_path: Path) -> None:
        folder = tmp_path / 'readonly'
        folder.mkdir()
        db = folder / 'console.db'
        record_episode(_trace('t1'), db_path=db)
        LabelStore(db).add(_invalidate('t1'))
        assert sorted(path.name for path in folder.iterdir()) == ['console.db']
        folder.chmod(0o555)
        try:
            labels, traces = read_labels(db), read_trace_ids(db)
        finally:
            folder.chmod(0o755)

        assert [label.type for label in labels] == ['invalidate']
        assert list(traces.values()) == ['t1']

    def test_load_jsonl_refuses_other_files(self, tmp_path: Path) -> None:
        path = tmp_path / 'trace.jsonl'
        path.write_text(_event_log(graded_trace()))

        with pytest.raises(ValueError, match='line 1 is not a label record'):
            load_jsonl(path)


class TestDuplicates:
    def test_keeps_the_earliest_copy_of_a_trajectory(self) -> None:
        late = _trace('t1', command='echo same', started=30)
        early = _trace('t2', command='echo same', started=10)
        # Same task and trajectory; times, usage and token data don't count.
        steps = tuple(step.model_copy(update={'usage': None, 'cost_usd': None, 'duration_ms': 9})
                      for step in early.steps)  # fmt: skip
        middle = _trace('t3', command='echo same', started=20).model_copy(update={'steps': steps})
        other = _trace('t4', command='echo other')

        dataset = _build([late, early, middle, other])

        assert _kept(dataset) == ['t2', 't4']
        assert [group.model_dump(exclude={'content_sha256'}) for group in dataset.duplicates] == [
            {'copies': ('t1', 't2', 't3'), 'kept': 't2', 'verdicts_disagree': False}
        ]
        assert dataset.dropped == {'duplicate': ('t1', 't3')}

    def test_copy_hash_is_pinned(self) -> None:
        # A published content_sha256 must not change when a defaulted field is added.
        assert copy_sha256(graded_trace()) == (
            'fbc8335603c4123da8c23dfc10b671adc27056751d199c688726fecb2c1939ea'
        )

    @pytest.mark.parametrize(
        'change',
        [
            {'initial_observation': Observation(content='another start', step=0)},
            {'steps': tuple(step.model_copy(update={'applied': True}) for step in BASE.steps)},
            {'steps': tuple(step.model_copy(update={'error': None}) for step in BASE.steps)},
        ],
    )
    def test_start_applied_and_errors_count(self, change: dict) -> None:
        first = _trace('t1', command=BASE.steps[0].action.content, started=1)
        second = first.model_copy(update={'trace_id': 't2', **change})

        assert _kept(_build([first, second])) == ['t1', 't2']

    def test_copies_the_verifier_both_passed_and_failed_are_inconsistent(self) -> None:
        traces = [
            _trace('t1', command='echo same'),
            _trace('t2', command='echo same', passed=False),
        ]

        dataset = _build(traces, require_verdict=False)

        assert dataset.dropped == {'inconsistent': ('t1', 't2')}
        assert dataset.duplicates[0].model_dump(include={'kept', 'verdicts_disagree'}) == {
            'kept': None, 'verdicts_disagree': True}  # fmt: skip

    def test_a_copy_dropped_by_another_check_leaves_the_next(self) -> None:
        early = _trace('t1', command='echo same', started=10, usage_complete=False)
        late = _trace('t2', command='echo same', started=20)

        dataset = _build([early, late], require_usage_complete=True)

        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'usage_incomplete': ('t1',)}
        assert dataset.duplicates[0].copies == ('t1', 't2')
        assert dataset.duplicates[0].kept == 't2'


class TestCopies:
    """Labels about a trajectory reach every copy of it."""

    @staticmethod
    def _copies() -> list[Trace]:
        return [_trace('t1', command='echo same', started=10),
                _trace('t2', command='echo same', started=20)]  # fmt: skip

    def test_a_human_fail_on_one_copy_fails_them_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', False))

        dataset = _build(self._copies(), _labels(store), passed=True)

        assert dataset.entries == {}
        assert dataset.dropped == {'failed': ('t1', 't2')}

    def test_a_human_verdict_on_a_later_copy_ships_on_the_kept_one(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', False))

        dataset = _build(self._copies(), _labels(store))

        assert _kept(dataset) == ['t1']
        assert dataset.entries['t1'].model_dump(
            include={'passed', 'verdict_source', 'score', 'overridden'}
        ) == {'passed': False, 'verdict_source': 'human', 'score': None, 'overridden': True}
        assert dataset.counts.overridden == 1

    def test_humans_who_disagree_across_copies_contest_them_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t1', True))
        store.add(_verdict('t2', False, author='bo'))

        dataset = _build(self._copies(), _labels(store))

        assert dataset.dropped == {'contested': ('t1', 't2')}
        assert dataset.duplicates[0].verdicts_disagree

    def test_a_human_verdict_settles_inconsistent_copies(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', True))
        traces = [self._copies()[0].model_copy(update={'result': BASE.result.model_copy(
            update={'score': 0.0, 'passed': False})}), self._copies()[1]]  # fmt: skip

        dataset = _build(traces, _labels(store), passed=True)

        # The copy whose verifier agrees with the human is kept, though it started later.
        assert _kept(dataset) == ['t2']
        assert not dataset.entries['t2'].overridden
        assert dataset.dropped == {'duplicate': ('t1',)}

    def test_a_forbidden_mark_on_one_copy_drops_them_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't1', 'step': 1}, name='mark',
                              value='reward-hack', source='human', author='ana'))  # fmt: skip

        dataset = _build(self._copies(), _labels(store), forbid_labels=[{'name': 'mark'}])

        assert dataset.dropped == {'forbidden_label': ('t1', 't2')}

    def test_a_required_label_on_one_copy_counts_for_all(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', True))

        dataset = _build(self._copies(), _labels(store), require_labels=[{'name': 'passed'}])

        assert _kept(dataset) == ['t1']

    def test_an_invalidation_stays_on_its_own_run(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_invalidate('t1'))
        traces = [self._copies()[0].model_copy(update={'result': BASE.result.model_copy(
            update={'score': 0.0, 'passed': False})}), self._copies()[1]]  # fmt: skip

        dataset = _build(traces, _labels(store))

        # t1's failed grade is discredited with it, so t2's pass is not inconsistent.
        assert _kept(dataset) == ['t2']
        assert dataset.dropped == {'invalidated': ('t1',)}
        assert not dataset.duplicates[0].verdicts_disagree


class TestKeptCopy:
    """What ships never depends on which copy of a trajectory ran first."""

    def test_scores_within_tolerance_agree(self) -> None:
        exact = _trace('t1', command='echo same', score=0.6, started=10)
        noisy = _trace('t2', command='echo same', score=0.1 + 0.2 + 0.3, started=20)

        dataset = _build([exact, noisy], min_score=0.5)

        assert _kept(dataset) == ['t2']  # the higher score, though it started later
        assert dataset.entries['t2'].score == 0.1 + 0.2 + 0.3
        assert not dataset.entries['t2'].scores_disagree
        assert dataset.dropped == {'duplicate': ('t1',)}

    @pytest.mark.parametrize('first', ['low', 'high'])
    def test_copies_whose_scores_disagree_ship_no_score(self, first: str) -> None:
        low = _trace('low', command='echo same', score=0.8, started=10 if first == 'low' else 20)
        high = _trace('high', command='echo same', score=0.9, started=20 if first == 'low' else 10)

        dataset = _build([low, high], passed=True)
        scored = _build([low, high], min_score=0.5)

        assert _kept(dataset) == ['high']
        shipped = dataset.entries['high'].model_dump(include={'passed', 'score', 'scores_disagree'})
        assert shipped == {'passed': True, 'score': None, 'scores_disagree': True}
        assert dataset.duplicates[0].verdicts_disagree
        assert scored.dropped == {'no_score': ('high', 'low')}

    @pytest.mark.parametrize('first', ['fail', 'pass'])
    def test_keeps_the_copy_whose_verifier_the_human_agrees_with(
        self, tmp_path: Path, first: str
    ) -> None:
        store = _store(tmp_path)
        store.add(_verdict('fail', True))
        failed = _trace('fail', command='echo same', passed=False,
                        started=10 if first == 'fail' else 20)  # fmt: skip
        passed = _trace('pass', command='echo same', started=20 if first == 'fail' else 10)

        dataset = _build([failed, passed], _labels(store))

        assert _kept(dataset) == ['pass']
        assert dataset.entries['pass'].model_dump(
            include={'passed', 'verdict_source', 'score', 'overridden'}
        ) == {'passed': True, 'verdict_source': 'human', 'score': 1.0, 'overridden': False}
        assert dataset.dropped == {'duplicate': ('fail',)}

    def test_keeps_a_judged_copy_over_an_earlier_unjudged_one(self) -> None:
        unjudged = _unjudged('t1', command='echo same', started=10)
        judged = _trace('t2', command='echo same', started=20)

        dataset = _build([unjudged, judged], require_verdict=False)

        assert _kept(dataset) == ['t2']
        assert dataset.entries['t2'].passed is True
        assert dataset.dropped == {'duplicate': ('t1',)}


class TestPools:
    """Only copies in the population asked for, and not invalidated, pool their verdicts."""

    def test_a_copy_in_another_environment_is_another_trajectory(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', False))
        here = _trace('t1', command='echo same')
        there = _trace('t2', command='echo same', passed=False, env_name='other')

        dataset = _build([here, there], _labels(store), env_names=['write_notes'])

        assert _kept(dataset) == ['t1']
        assert dataset.entries['t1'].verdict_source == 'verifier'
        assert dataset.dropped == {'env_name': ('t2',)}
        assert dataset.duplicates == ()

    @pytest.mark.parametrize(
        ('changes', 'spec', 'reason'),
        [
            ({'variant': 3}, {'env_variants': [2]}, 'env_variant'),
            ({'model': 'm2'}, {'models': ['m1']}, 'model'),
            ({'agent': 'other'}, {'agents': ['react-shell']}, 'agent'),
        ],
    )
    def test_an_excluded_copy_neither_conflicts_nor_votes(
        self, tmp_path: Path, changes: dict, spec: dict, reason: str
    ) -> None:
        store = _store(tmp_path)
        store.add(_verdict('t2', False))
        asked = _trace('t1', command='echo same', started=10)
        excluded = _trace('t2', command='echo same', passed=False, started=5, **changes)

        dataset = _build([asked, excluded], _labels(store), **spec)

        assert _kept(dataset) == ['t1']
        assert dataset.entries['t1'].model_dump(include={'passed', 'verdict_source'}) == {
            'passed': True, 'verdict_source': 'verifier'}  # fmt: skip
        assert dataset.dropped == {reason: ('t2',)}
        assert dataset.duplicates[0].copies == ('t1', 't2')

    def test_an_invalidated_copy_vouches_for_none(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_invalidate('t1'))
        store.add(_verdict('t1', False))
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't1'}, name='reviewed',
                              value=True, source='human', author='ana'))  # fmt: skip
        copies = [_trace('t1', command='echo same'), _trace('t2', command='echo same')]

        kept = _build(copies, _labels(store))
        required = _build(copies, _labels(store), require_labels=[{'name': 'reviewed'}])

        assert _kept(kept) == ['t2']
        assert kept.entries['t2'].verdict_source == 'verifier'
        assert required.dropped == {'invalidated': ('t1',), 'missing_label': ('t2',)}

    def test_an_invalidated_copy_still_forbids(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.add(_invalidate('t1'))
        store.add(LabelCreate(target={'kind': 'trace', 'id': 't1'}, name='mark',
                              value='reward-hack', source='human', author='ana'))  # fmt: skip
        copies = [_trace('t1', command='echo same'), _trace('t2', command='echo same')]

        dataset = _build(copies, _labels(store), forbid_labels=[{'name': 'mark'}])

        assert dataset.dropped == {'invalidated': ('t1',), 'forbidden_label': ('t2',)}


class TestSplits:
    @staticmethod
    def _traces(tasks: int) -> list[Trace]:
        return [
            _trace(f'{task}-{sample}', f'{task}') for task in range(tasks) for sample in range(3)
        ]

    def test_follows_the_documented_hash(self) -> None:
        dataset = _build(self._traces(30), test_ratio=0.2, val_ratio=0.3, seed=4)

        for trace_id, entry in dataset.entries.items():
            u = _u(4, trace_id.split('-')[0])
            assert entry.split == ('test' if u < 0.2 else 'val' if u < 0.5 else 'train')
            assert entry.task_sha256 == hashlib.sha256(
                f'Task {trace_id.split("-")[0]}'.encode()).hexdigest()  # fmt: skip
        assert all(dataset.counts.splits.values())

    def test_a_prompt_lands_in_one_split_across_environments(self) -> None:
        traces = [_trace(f'{task}-{env}', f'{task}', env_name=f'env{env}')
                  for task in range(30) for env in (1, 2)]  # fmt: skip

        dataset = _build(traces, test_ratio=0.3, val_ratio=0.3)

        splits: dict[str, set[str]] = {}
        for entry in dataset.entries.values():
            splits.setdefault(entry.task_sha256, set()).add(entry.split)
        assert len(splits) == 30 and all(len(found) == 1 for found in splits.values())
        assert len({entry.environment_sha256 for entry in dataset.entries.values()}) == 60

    def test_adding_prompts_never_moves_one(self) -> None:
        before = _build(self._traces(20), test_ratio=0.2, val_ratio=0.2)
        after = _build(self._traces(60), test_ratio=0.2, val_ratio=0.2)

        assert all(after.entries[trace_id] == entry for trace_id, entry in before.entries.items())

    def test_splits_are_deterministic_by_seed(self) -> None:
        traces = self._traces(20)
        shuffled = random.Random(1).sample(traces, len(traces))

        first = _build(traces, seed=5, test_ratio=0.2, val_ratio=0.2)

        assert _build(shuffled, seed=5, test_ratio=0.2, val_ratio=0.2) == first
        assert _build(traces, seed=6, test_ratio=0.2, val_ratio=0.2).entries != first.entries

    def test_refuses_an_empty_train_split(self) -> None:
        seed = next(seed for seed in range(100) if _u(seed, 'a') < 0.5)

        with pytest.raises(ValueError, match='No trace landed in train'):
            _build([_trace('t1')], seed=seed, test_ratio=0.5)


def _dataset(*extra: Trace) -> Dataset:
    traces = [_trace(f'{task}-{sample}', f'{task}') for task in range(12) for sample in range(2)]
    dataset = _build([*traces, *extra, _unjudged('x1')], test_ratio=0.3, val_ratio=0.3)
    assert all(dataset.counts.splits.values())
    return dataset


def _write(dataset: Dataset, out: Path, *formats: str) -> str:
    return write_dataset(dataset, out, name='notes', version='1.0.0', formats=formats or ['jsonl'])


def _reseal(out: Path, change: Callable[[dict], object] = lambda data: None) -> None:
    """Apply *change* to the manifest, then rehash every file and the content as a forger
    would, with the documented recipe."""
    path = out / MANIFEST
    data = json.loads(path.read_text())
    change(data)
    files = [file for file in out.rglob('*') if file.is_file() and file.name != MANIFEST]
    data['files'] = {file.relative_to(out).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in files}  # fmt: skip
    data['content_sha256'] = _content_hash(data)
    path.write_text(json.dumps(data))


def _content_hash(data: dict) -> str:
    body = {key: value for key, value in data.items() if key != 'content_sha256'}
    text = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _split_ids(dataset: Dataset, split: str) -> list[str]:
    return [trace_id for trace_id, entry in dataset.entries.items() if entry.split == split]


def _drop_line(path: Path) -> None:
    path.write_bytes(b''.join(path.read_bytes().splitlines(keepends=True)[1:]))


def _join_prompts(data: dict) -> None:
    """Give a train entry the prompt of a test entry."""
    entries = data['entries'].values()
    test = next(entry for entry in entries if entry['split'] == 'test')
    next(entry for entry in entries if entry['split'] == 'train')['task_sha256'] = test[
        'task_sha256'
    ]


def _edit_line(path: Path, trace_id: str, **changes: object) -> None:
    """Change the fields of *trace_id*'s line in a verdicts file."""
    lines = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        lines.append(json.dumps(record | changes if record['trace_id'] == trace_id else record))
    path.write_text(''.join(f'{line}\n' for line in lines))


def _edit_first_line(path: Path, **changes: object) -> None:
    _edit_line(path, json.loads(path.read_text().splitlines()[0])['trace_id'], **changes)


class TestManifest:
    def test_round_trip_in_every_format(self, tmp_path: Path) -> None:
        pytest.importorskip('pyarrow')
        odd = _trace('odd', 'odd\u2028line', command='printf "a\rb\u2028c"')
        dataset = _dataset(odd)
        out = tmp_path / 'ds'

        published = _write(dataset, out, 'jsonl', 'parquet', 'atif')
        manifest = verify_dataset(out, sha256=published.upper())

        assert published == hashlib.sha256((out / MANIFEST).read_bytes()).hexdigest()
        assert published == manifest_sha256(out)
        assert (manifest.name, manifest.version, manifest.formats) == (
            'notes', '1.0.0', ('jsonl', 'parquet', 'atif'))  # fmt: skip
        assert manifest.entries == dataset.entries and manifest.spec == dataset.spec
        assert manifest.counts == dataset.counts and manifest.bands == dataset.bands
        assert manifest.dropped == {'no_verdict': ('x1',)}
        assert _content_hash(json.loads((out / MANIFEST).read_text())) == manifest.content_sha256
        for split in SPLITS:
            lines = (out / f'{split}.jsonl').read_bytes().split(b'\n')[:-1]
            ids = _split_ids(dataset, split)
            assert sorted(Trace.model_validate_json(line).trace_id for line in lines) == ids
            # A trace's hash is its JSONL line's, so a buyer can check it with sha256 alone.
            assert sorted(hashlib.sha256(line).hexdigest() for line in lines) == sorted(
                dataset.entries[trace_id].sha256 for trace_id in ids
            )
            assert (out / f'{split}.parquet').exists()
            assert sorted(path.stem for path in (out / 'atif' / split).glob('*.json')) == ids
        written = {path.relative_to(out).as_posix() for path in out.rglob('*') if path.is_file()}
        assert set(manifest.files) == written - {MANIFEST}
        assert not [path for path in tmp_path.iterdir() if path.name != 'ds']

    def test_records_what_the_sources_left_out(self, tmp_path: Path) -> None:
        unread = {'episode_without_trace': 2, 'not_a_trace_file': 1, 'merged_copy': 3}
        dataset = build_dataset([_trace('t1')], DatasetSpec(test_ratio=0, val_ratio=0),
                                unread=unread)  # fmt: skip
        _write(dataset, tmp_path / 'ds')

        assert verify_dataset(tmp_path / 'ds').counts.unread == unread

    def test_content_hash_does_not_depend_on_layout_or_input_order(self, tmp_path: Path) -> None:
        dataset = _dataset()
        _write(dataset, tmp_path / 'a')
        traces = random.Random(2).sample(list(dataset.traces), len(dataset.traces))
        _write(build_dataset([*traces, _unjudged('x1')], dataset.spec), tmp_path / 'b')
        path = tmp_path / 'a' / MANIFEST
        data = json.loads(path.read_text())
        path.write_text(json.dumps(dict(reversed(data.items()))))  # another layout, same content

        first = verify_dataset(tmp_path / 'a')
        second = verify_dataset(tmp_path / 'b')

        changing = {'created_at', 'content_sha256'}
        assert first.model_dump(exclude=changing) == second.model_dump(exclude=changing)
        with pytest.raises(DeepGymError, match='not the published'):
            verify_dataset(tmp_path / 'a', sha256=manifest_sha256(tmp_path / 'b'))

    @pytest.mark.parametrize(
        ('tamper', 'problem'),
        [
            (lambda out: (out / 'train.jsonl').write_text('{}\n'), 'changed: train.jsonl'),
            (lambda out: (out / 'test.jsonl').unlink(), 'missing: test.jsonl'),
            (lambda out: (out / 'extra.jsonl').write_text(''), 'not in the manifest: extra'),
            (lambda out: (out / 'notes').mkdir(), 'empty directory: notes'),
            (lambda out: (out / 'link.jsonl').symlink_to(out / 'train.jsonl'), 'symlink: link'),
            (lambda out: os.mkfifo(out / 'pipe'), 'not a regular file: pipe'),
            (
                lambda out: (out / MANIFEST).write_text(
                    (out / MANIFEST).read_text().replace('"kept": ', '"kept": 1') + ' '
                ),
                'content_sha256 no longer matches',
            ),
        ],
    )
    def test_detects_tampering(self, tmp_path: Path, tamper, problem: str) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        tamper(out)

        with pytest.raises(DeepGymError, match=problem):
            verify_dataset(out)

    def test_refuses_duplicate_keys(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        path = out / MANIFEST
        path.write_text(path.read_text().replace('{', '{"name": "other", ', 1))

        with pytest.raises(DeepGymError, match="duplicate keys \\['name'\\]"):
            verify_dataset(out)

    def test_refuses_formats_as_a_string(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        _reseal(out, lambda data: data.update(formats='jsonl'))

        with pytest.raises(ValidationError, match='formats'):
            verify_dataset(out)

    def test_published_sha256_catches_a_resealed_folder(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        published = _write(_dataset(), out)
        _reseal(out, lambda data: data.update(name='renamed'))

        assert verify_dataset(out).name == 'renamed'  # self-consistent
        with pytest.raises(DeepGymError, match='not the published'):
            verify_dataset(out, sha256=published)

    @pytest.mark.parametrize(
        ('forge', 'problem'),
        [
            (lambda out, ids: _drop_line(out / 'train.jsonl'), 'lacks train traces'),
            (
                lambda out, ids: (out / 'val.jsonl').write_bytes(
                    (out / 'val.jsonl').read_bytes() + (out / 'train.jsonl').read_bytes()
                ),
                'holds traces that are not its val entries',
            ),
            (
                lambda out, ids: (out / 'train.jsonl').write_bytes(
                    (out / 'train.jsonl').read_bytes() * 2
                ),
                'holds traces more than once',
            ),
            (
                lambda out, ids: (out / 'atif' / 'test' / f'{ids["test"][0]}.json').unlink(),
                'the manifest lists no atif/test/',
            ),
            (
                lambda out, ids: (out / 'atif' / 'train' / 'stray.json').write_text('{}'),
                'lists a file no format writes: atif/train/stray.json',
            ),
            (lambda out, ids: (out / 'val.jsonl').unlink(), 'the manifest lists no val.jsonl'),
        ],
    )
    def test_detects_forged_contents(self, tmp_path: Path, forge, problem: str) -> None:
        out = tmp_path / 'ds'
        dataset = _dataset()
        _write(dataset, out, 'jsonl', 'atif')
        forge(out, {split: _split_ids(dataset, split) for split in SPLITS})
        _reseal(out)

        with pytest.raises(DeepGymError, match=problem):
            verify_dataset(out)

    def test_detects_a_forged_parquet_trace(self, tmp_path: Path) -> None:
        pa = pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')
        out = tmp_path / 'ds'
        _write(_dataset(), out, 'parquet')
        path = out / 'train.parquet'
        table = pq.read_table(path)
        traces = table['trace'].to_pylist()
        traces[0] = traces[0].replace('"passed":false', '"passed":true')
        traces[0] = traces[0].replace('echo', 'ECHO')
        table = table.set_column(table.schema.get_field_index('trace'), 'trace', pa.array(traces))
        pq.write_table(table, path)
        _reseal(out)

        with pytest.raises(DeepGymError, match='train.parquet holds traces that are not its'):
            verify_dataset(out)

    @pytest.mark.parametrize(
        ('change', 'problem'),
        [
            (lambda data: data['counts'].update(kept=1), 'counts do not match'),
            (
                lambda data: data['dropped'].update(failed=[next(iter(data['entries']))]),
                'traces both kept and dropped',
            ),
            (_join_prompts, 'prompts in more than one split'),
        ],
    )
    def test_detects_an_inconsistent_manifest(self, tmp_path: Path, change, problem: str) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        _reseal(out, change)

        with pytest.raises(DeepGymError, match=problem):
            verify_dataset(out)

    def test_verdicts_sidecar_carries_the_shipped_verdicts(self, tmp_path: Path) -> None:
        dataset = _dataset()
        overridden = _split_ids(dataset, 'train')[0]
        store = _store(tmp_path)
        store.add(_verdict(overridden, False))
        dataset = build_dataset([*dataset.traces, _unjudged('x1')], dataset.spec, _labels(store))
        out = tmp_path / 'ds'

        _write(dataset, out)

        for split in SPLITS:
            lines = (out / f'{split}.verdicts.jsonl').read_text().splitlines()
            assert [json.loads(line) for line in lines] == [
                {'trace_id': trace.trace_id, **dataset.entries[trace.trace_id].model_dump(
                    include={'passed', 'verdict_source', 'score', 'overridden'})}
                for trace in dataset.traces if dataset.entries[trace.trace_id].split == split
            ]  # fmt: skip
        assert dataset.entries[overridden].model_dump(include={'passed', 'score'}) == {
            'passed': False, 'score': None}  # fmt: skip
        assert verify_dataset(out).counts.overridden == 1

    @pytest.mark.parametrize(
        ('forge', 'problem'),
        [
            (
                lambda out: _edit_first_line(out / 'train.verdicts.jsonl', passed=False),
                'train.verdicts.jsonl holds traces that are not its train entries',
            ),
            (
                lambda out: (out / 'test.verdicts.jsonl').unlink(),
                'the manifest lists no test.verdicts.jsonl',
            ),
        ],
    )
    def test_detects_a_forged_verdicts_file(self, tmp_path: Path, forge, problem: str) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        forge(out)
        _reseal(out)

        with pytest.raises(DeepGymError, match=problem):
            verify_dataset(out)

    @pytest.mark.parametrize('field', ['environment_sha256', 'task_sha256', 'passed', 'score'])
    def test_rows_must_match_their_entry(self, tmp_path: Path, field: str) -> None:
        pytest.importorskip('pyarrow')
        out = tmp_path / 'ds'
        dataset = _dataset()
        _write(dataset, out, 'jsonl', 'parquet')
        trace_id = _split_ids(dataset, 'train')[0]
        forged = {'environment_sha256': 'f' * 64, 'task_sha256': 'f' * 64, 'passed': False,
                  'score': 0.5}[field]  # fmt: skip

        def change(data: dict) -> None:
            data['entries'][trace_id][field] = forged

        if field in ('passed', 'score'):
            _edit_line(out / 'train.verdicts.jsonl', trace_id, **{field: forged})
        _reseal(out, change)

        with pytest.raises(DeepGymError) as raised:
            verify_dataset(out)

        for name in ('train.jsonl', 'train.parquet'):
            assert f"{name} holds traces that are not its train entries: ['{trace_id}']" in str(
                raised.value
            )

    def test_recomputes_each_split_from_the_spec(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        _reseal(out, lambda data: data['spec'].update(seed=99))

        with pytest.raises(DeepGymError, match='not in the split their prompt and the spec give'):
            verify_dataset(out)

    def test_skips_finder_metadata(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        published = _write(_dataset(), out, 'jsonl', 'atif')
        (out / '.DS_Store').write_bytes(b'\0')
        (out / 'atif' / 'train' / '.DS_Store').write_bytes(b'\0')

        verify_dataset(out, sha256=published)

    @pytest.mark.parametrize('kind', ['dir', 'link'])
    def test_finder_name_never_hides_a_folder_or_link(self, tmp_path: Path, kind: str) -> None:
        out = tmp_path / 'ds'
        published = _write(_dataset(), out)
        hidden = out / '.DS_Store'
        if kind == 'dir':
            hidden.mkdir()
            (hidden / 'extra.jsonl').write_text('{}\n')
        else:
            hidden.symlink_to(tmp_path)

        with pytest.raises(DeepGymError, match='DS_Store'):
            verify_dataset(out, sha256=published)

    @pytest.mark.parametrize(
        ('change', 'reason'),
        [
            ({'models': ['other']}, 'model'),
            ({'env_names': ['other']}, 'env_name'),
            ({'passed': False}, 'passed'),
            ({'max_score': 0.5}, 'score'),
            ({'require_usage_complete': True, 'max_cost_usd': 0.001}, 'over_cost'),
            ({'bands': ['never_solved']}, 'band'),
        ],
    )
    def test_rechecks_the_spec_against_the_rows(
        self, tmp_path: Path, change: dict, reason: str
    ) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        _reseal(out, lambda data: data['spec'].update(change))

        with pytest.raises(DeepGymError, match=f'holds traces its spec drops: .*\\({reason}\\)'):
            verify_dataset(out)

    def test_refuses_two_kept_copies_of_one_trajectory(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        dataset = _dataset()
        _write(dataset, out)
        first, second = _split_ids(dataset, 'train')[:2]

        def change(data: dict) -> None:
            entries = data['entries']
            entries[second]['content_sha256'] = entries[first]['content_sha256']

        _reseal(out, change)

        with pytest.raises(DeepGymError, match='more than one kept copy of trajectories'):
            verify_dataset(out)

    def test_verdict_lines_must_match_byte_for_byte(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        path = out / 'train.verdicts.jsonl'
        path.write_text(''.join(f'{json.dumps(json.loads(line))}\n'
                                for line in path.read_text().splitlines()))  # fmt: skip
        _reseal(out)

        with pytest.raises(DeepGymError, match='train.verdicts.jsonl holds traces that are not'):
            verify_dataset(out)

    def test_a_null_parquet_trace_id_is_a_mismatch(self, tmp_path: Path) -> None:
        pa = pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')
        out = tmp_path / 'ds'
        _write(_dataset(), out, 'parquet')
        path = out / 'train.parquet'
        table = pq.read_table(path)
        ids = [None, *table['trace_id'].to_pylist()[1:]]
        index = table.schema.get_field_index('trace_id')
        pq.write_table(table.set_column(index, 'trace_id', pa.array(ids, pa.string())), path)
        _reseal(out)

        with pytest.raises(DeepGymError, match='train.parquet holds traces that are not'):
            verify_dataset(out)

    @pytest.mark.parametrize(
        ('change', 'problem'),
        [
            (lambda band: band.update(band='never_solved'), 'has the wrong band'),
            (lambda band: band.update(solved=band['samples'] + 1), 'has the wrong band'),
            (lambda band: band.update(samples=1, solved=1), 'keeps more verdicts or passes'),
        ],
    )
    def test_rechecks_the_bands(self, tmp_path: Path, change, problem: str) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        _reseal(out, lambda data: change(next(iter(data['bands'].values()))))

        with pytest.raises(DeepGymError, match=problem):
            verify_dataset(out)

    def test_a_parquet_trace_that_is_no_string_is_a_mismatch(self, tmp_path: Path) -> None:
        pa = pytest.importorskip('pyarrow')
        pq = pytest.importorskip('pyarrow.parquet')
        out = tmp_path / 'ds'
        _write(_dataset(), out, 'parquet')
        path = out / 'train.parquet'
        table = pq.read_table(path)
        index = table.schema.get_field_index('trace')
        numbers = pa.array(range(table.num_rows), pa.int64())
        pq.write_table(table.set_column(index, 'trace', numbers), path)
        _reseal(out)

        with pytest.raises(DeepGymError, match='train.parquet holds traces that are not'):
            verify_dataset(out)

    def test_a_verdict_file_ends_with_a_newline(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'
        _write(_dataset(), out)
        path = out / 'train.verdicts.jsonl'
        path.write_bytes(path.read_bytes().rstrip(b'\n'))
        _reseal(out)

        with pytest.raises(DeepGymError, match='train.verdicts.jsonl does not end with a newline'):
            verify_dataset(out)

    def test_needs_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(DeepGymError, match='not found'):
            verify_dataset(tmp_path)


class TestWrite:
    @pytest.mark.parametrize(
        ('changes', 'error'),
        [
            ({'formats': 'jsonl'}, "not 'jsonl'"),
            ({'formats': []}, 'at least one format'),
            ({'formats': ['csv']}, 'jsonl'),
            ({'name': ' notes'}, 'pattern'),
            ({'version': ''}, 'pattern'),
        ],
    )
    def test_validates_before_writing(self, tmp_path: Path, changes: dict, error: str) -> None:
        args = {'name': 'notes', 'version': '1', 'formats': ['jsonl']} | changes

        with pytest.raises(ValueError, match=error):
            write_dataset(_dataset(), tmp_path / 'ds', **args)

        assert list(tmp_path.iterdir()) == []

    def test_parquet_needs_pyarrow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'pyarrow.parquet', None)

        with pytest.raises(DeepGymError, match='needs pyarrow'):
            _write(_dataset(), tmp_path / 'ds', 'jsonl', 'parquet')

        assert list(tmp_path.iterdir()) == []

    def test_a_failure_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*args: object) -> None:
            raise OSError('disk full')

        monkeypatch.setattr('deepgym.traces.export.write_atif', fail)

        with pytest.raises(OSError, match='disk full'):
            _write(_dataset(), tmp_path / 'ds', 'jsonl', 'atif')

        assert list(tmp_path.iterdir()) == []

    def test_atif_comes_with_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / 'ds'

        _write(_dataset(), out, 'atif')

        assert verify_dataset(out).formats == ('jsonl', 'atif')
        assert all((out / f'{split}.jsonl').exists() for split in SPLITS)

    def test_verifies_the_staging_folder_before_renaming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deepgym.dataset import manifest

        checked: list[tuple[Path, str | None]] = []
        real = manifest.verify_dataset

        def spy(folder: Path, *, sha256: str | None = None) -> object:
            checked.append((folder, sha256))
            return real(folder, sha256=sha256)

        monkeypatch.setattr(manifest, 'verify_dataset', spy)
        out = tmp_path / 'ds'

        published = _write(_dataset(), out)

        ((folder, sha256),) = checked
        assert folder.parent == tmp_path and folder.name.startswith('.ds-')
        assert sha256 == published and not folder.exists()

    def test_a_failed_verification_leaves_nothing_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(folder: Path, *, sha256: str | None = None) -> None:
            raise DeepGymError('does not match its manifest')

        monkeypatch.setattr('deepgym.dataset.manifest.verify_dataset', refuse)

        with pytest.raises(DeepGymError, match='does not match'):
            _write(_dataset(), tmp_path / 'ds')

        assert list(tmp_path.iterdir()) == []

    def test_the_folder_gets_the_usual_permissions(self, tmp_path: Path) -> None:
        umask = os.umask(0o022)
        try:
            _write(_dataset(), tmp_path / 'ds')
        finally:
            os.umask(umask)

        assert stat.S_IMODE((tmp_path / 'ds').stat().st_mode) == 0o755

    def test_writes_into_an_empty_directory(self, tmp_path: Path) -> None:
        (tmp_path / 'ds').mkdir()

        verify_dataset(tmp_path / 'ds', sha256=_write(_dataset(), tmp_path / 'ds'))

    def test_refuses_a_folder_with_files(self, tmp_path: Path) -> None:
        (tmp_path / 'old.jsonl').write_text('')

        with pytest.raises(DeepGymError, match='not a new or empty directory'):
            _write(_dataset(), tmp_path)


class TestCli:
    @pytest.fixture(autouse=True)
    def no_default_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('deepgym.console.db.DEFAULT_DB_PATH', tmp_path / 'missing.db')

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
        monkeypatch.setattr(sys, 'argv', ['deepgym', 'dataset', *args])
        main()

    @staticmethod
    def _spec(tmp_path: Path, **fields: object) -> str:
        spec = tmp_path / 'spec.json'
        spec.write_text(json.dumps({'test_ratio': 0, 'val_ratio': 0, **fields}))
        return str(spec)

    def test_build_reads_the_default_console_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        sources = tmp_path / 'traces.db'
        for trace in [_trace('t1', 'a'), _trace('t2', 'b'), _trace('t3', 'c'), _unjudged('t4')]:
            record_episode(trace, db_path=sources)
        default = tmp_path / 'console.db'
        monkeypatch.setattr('deepgym.console.db.DEFAULT_DB_PATH', default)
        episode = record_episode(_trace('t2', 'b'), db_path=default)
        LabelStore(default).add(_invalidate(episode, kind='episode'))
        LabelStore(default).add(_verdict('nowhere', True))
        other = LabelStore(tmp_path / 'other.db')
        other.add(_verdict('t3', False))
        other.export_jsonl(tmp_path / 'labels.jsonl')
        out = tmp_path / 'ds'

        self._run(monkeypatch, 'build', '--db', str(sources), '--labels',
                  str(tmp_path / 'labels.jsonl'), '--spec', self._spec(tmp_path),
                  '--name', 'notes', '--version', '2', '--out', str(out),
                  '--format', 'jsonl', 'atif')  # fmt: skip
        built = capsys.readouterr()
        self._run(monkeypatch, 'verify', str(out))
        checked = capsys.readouterr().out
        sha256 = manifest_sha256(out)
        self._run(monkeypatch, 'verify', str(out), '--sha256', sha256.upper())

        assert 'Kept 2 of 4 traces' in built.out and '(train 2, val 0, test 0)' in built.out
        for reason in ('invalidated', 'no_verdict'):
            assert f'dropped 1: {reason}' in built.out
        assert '1 kept traces have a verifier verdict a human reversed' in built.out
        assert f'3 label records read from {sources}, {default}, ' in built.out
        assert '1 current label records matched none of these traces' in built.out
        assert f'Publish this sha256 of {out}/manifest.json: {sha256}' in built.out
        nowhere = next(label for label in _labels(LabelStore(default)) if label.type == 'label')
        assert 'WARNING: 1 invalidations, human verdicts and labels the spec' in built.err
        assert f'passed=True {nowhere.id} on trace nowhere' in built.err
        assert 'is self-consistent (notes 2, 2 traces)' in checked and sha256 in checked
        assert f'matches the published sha256 {sha256}' in capsys.readouterr().out
        assert verify_dataset(out).counts.unmatched_labels == 1

    def test_warns_when_no_labels_were_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        traces = tmp_path / 'traces'
        traces.mkdir()
        (traces / 'a.jsonl').write_text(_event_log(graded_trace()))
        (traces / 'b.jsonl').write_text('{"not": "a trace"}\n')

        self._run(monkeypatch, 'build', '--traces', str(traces), '--spec', self._spec(tmp_path),
                  '--name', 'notes', '--version', '1', '--out', str(tmp_path / 'ds'))  # fmt: skip

        captured = capsys.readouterr()
        assert 'WARNING: no label records were read' in captured.err
        assert '0 label records read from nowhere' in captured.out
        assert verify_dataset(tmp_path / 'ds').counts.unread == {
            'episode_without_trace': 0, 'not_a_trace_file': 1, 'merged_copy': 0}  # fmt: skip

    def test_checks_the_output_before_reading_traces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit):
            self._run(monkeypatch, 'build', '--traces', str(tmp_path / 'missing'), '--spec',
                      self._spec(tmp_path), '--name', ' padded', '--version', '1', '--out',
                      str(tmp_path / 'ds'))  # fmt: skip

        err = capsys.readouterr().err
        assert 'pattern' in err and 'not found' not in err

    def test_warns_when_a_database_holds_no_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db = tmp_path / 'console.db'
        record_episode(_trace('t1'), db_path=db)

        self._run(monkeypatch, 'build', '--db', str(db), '--spec', self._spec(tmp_path),
                  '--name', 'n', '--version', '1', '--out', str(tmp_path / 'ds'))  # fmt: skip

        assert f'WARNING: no label records were read ({db})' in capsys.readouterr().err

    def test_reads_the_default_database_read_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        sources = tmp_path / 'traces.db'
        record_episode(_trace('t1'), db_path=sources)
        default = tmp_path / 'console.db'
        LabelStore(default).add(LabelCreate(target={'kind': 'run', 'id': 'r1'}, name='mark',
                                            value='bad', source='human', author='ana'))  # fmt: skip
        before = default.read_bytes()
        monkeypatch.setattr('deepgym.console.db.DEFAULT_DB_PATH', default)
        spec = self._spec(tmp_path, forbid_labels=[{'name': 'mark'}])

        self._run(monkeypatch, 'build', '--db', str(sources), '--spec', spec, '--name', 'n',
                  '--version', '1', '--out', str(tmp_path / 'ds'))  # fmt: skip

        assert default.read_bytes() == before
        err = capsys.readouterr().err
        assert 'labels the spec requires or forbids matched none' in err
        assert 'mark=bad' in err and 'on run r1' in err

    def test_reads_an_older_trace_database_without_migrating_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import sqlite3

        old = tmp_path / 'old.db'
        with sqlite3.connect(old) as conn:
            conn.execute('CREATE TABLE episodes (id TEXT PRIMARY KEY, environment TEXT)')
            conn.execute("INSERT INTO episodes VALUES ('e1', 'legacy')")
        before = old.read_bytes()
        (tmp_path / 'a.jsonl').write_text(_event_log(graded_trace()))

        self._run(monkeypatch, 'build', '--db', str(old), '--traces', str(tmp_path / 'a.jsonl'),
                  '--spec', self._spec(tmp_path), '--name', 'n', '--version', '1',
                  '--out', str(tmp_path / 'ds'))  # fmt: skip

        assert old.read_bytes() == before
        assert '1 console episodes were recorded without a trace' in capsys.readouterr().out

    def test_leaves_a_missing_default_database_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / 'console.db'
        record_episode(_trace('t1'), db_path=db)

        self._run(monkeypatch, 'build', '--db', str(db), '--spec', self._spec(tmp_path),
                  '--name', 'n', '--version', '1', '--out', str(tmp_path / 'ds'))  # fmt: skip

        assert not (tmp_path / 'missing.db').exists()

    def test_labels_must_be_a_label_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db = tmp_path / 'console.db'
        record_episode(_trace('t1'), db_path=db)
        (tmp_path / 'trace.jsonl').write_text(_event_log(graded_trace()))

        with pytest.raises(SystemExit) as exit_info:
            self._run(monkeypatch, 'build', '--db', str(db), '--labels',
                      str(tmp_path / 'trace.jsonl'), '--spec', self._spec(tmp_path), '--name',
                      'n', '--version', '1', '--out', str(tmp_path / 'ds'))  # fmt: skip

        assert exit_info.value.code == 1
        assert 'is not a label record' in capsys.readouterr().err
        assert not (tmp_path / 'ds').exists()

    def test_verify_fails_on_a_changed_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        out = tmp_path / 'ds'
        _write(_build([_trace('t1')]), out)
        (out / 'train.jsonl').write_text('\n')

        with pytest.raises(SystemExit) as exit_info:
            self._run(monkeypatch, 'verify', str(out), '--sha256', '0' * 64)

        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert 'changed: train.jsonl' in err and 'not the published' in err
