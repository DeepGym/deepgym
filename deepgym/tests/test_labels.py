"""Tests for labels: the record's validation, supersede, retract and invalidate, and the store."""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.console import label_store
from deepgym.console.label_store import LabelStore
from deepgym.console.schemas import LabelQuery
from deepgym.labels import Label, LabelCreate, current_labels, excluded

SHA = 'ab' * 32
WHOLE = {'name': None, 'value': None}
STEP_2 = {'kind': 'trace', 'id': 't1', 'step': 2}


def _label(**changes: object) -> LabelCreate:
    """A human's 'tone' mark on trace t1, with *changes*."""
    fields = {
        'target': {'kind': 'trace', 'id': 't1'},
        'name': 'tone',
        'value': 'clear',
        'source': 'human',
        'author': 'ana',
    }
    return LabelCreate.model_validate({**fields, **changes})


def _passed(**changes: object) -> LabelCreate:
    """A human's passed verdict on trace t1, with *changes*."""
    return _label(**{'name': 'passed', 'value': True, 'comment': 'Solves it.', **changes})


def _invalidate(**changes: object) -> LabelCreate:
    return _label(**{'type': 'invalidate', **WHOLE, 'comment': 'verifier was broken', **changes})


def _retract(supersedes: str, **changes: object) -> LabelCreate:
    fields = {'type': 'retract', **WHOLE, 'comment': 'wrong trace', 'supersedes': supersedes}
    return _label(**{**fields, **changes})


@pytest.fixture()
def store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / 'console.db')


def _all(store: LabelStore, **filters: object) -> list[Label]:
    return store.page(LabelQuery.model_validate(filters), 0, None).labels


class TestTarget:
    @pytest.mark.parametrize(
        'target',
        [
            {'kind': 'job', 'id': 't1'},
            {'kind': 'trace', 'id': ''},
            {'kind': 'trace', 'id': ' t1'},
            {'kind': 'trace', 'id': 't\n1'},
            {'kind': 'trace', 'id': 't\x001'},
            {'kind': 'trace', 'id': 't1', 'extra': 1},
            {'kind': 'run', 'id': 'r1', 'step': 1},
            {'kind': 'trace', 'id': 't1', 'step': 0},
            {'kind': 'episode', 'id': 'e1', 'step': -1},
            {'kind': 'trace', 'id': 't1', 'step': 2**63},
            {'kind': 'trace', 'id': 't1', 'step': '3'},
            {'kind': 'trace', 'id': 't1', 'step': 3.0},
            {'kind': 'trace', 'id': 't1', 'step': 1, 'span': [0, 4]},
            {'kind': 'trace', 'id': 't1', 'span': [0, 4]},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [4, 4]},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [-1, 4]},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [0, 2**63]},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': ['0', 4]},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [0, 4, 8]},
            {'kind': 'run', 'id': 'r1', 'field': 'action'},
            {'kind': 'trace', 'id': 't1', 'step': 1, 'field': 'answer'},
            {'kind': 'trace', 'id': 't1', 'step': 1, 'field': 'reply'},
            {'kind': 'trace', 'id': 't1', 'step': 1, 'field': 'message:01'},
            {'kind': 'episode', 'id': 'e1', 'step': 0, 'field': 'message:0'},
            {'kind': 'run', 'id': 'r1', 'quote': 'x'},
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [0, 3], 'quote': 'ab'},
        ],
    )
    def test_bad(self, target: dict) -> None:
        with pytest.raises(ValidationError):
            _label(target=target)

    @pytest.mark.parametrize(
        'target',
        [
            {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [3, 9]},
            {'kind': 'episode', 'id': 'e1', 'field': 'answer'},
            {'kind': 'trace', 'id': 't1', 'field': 'answer'},
            {'kind': 'episode', 'id': 'e1', 'step': 0, 'field': 'observation', 'span': [0, 1]},
            {'kind': 'trace', 'id': 't1', 'step': 2, 'field': 'action'},
            {'kind': 'trace', 'id': 't1', 'step': 2, 'field': 'message:12', 'span': [5, 7]},
        ],
    )
    def test_good(self, target: dict) -> None:
        assert _label(target=target).target.field == target['field']

    def test_quote_counts_code_points(self) -> None:
        text = 'fix 🐛 now'
        start = text.index('🐛')
        target = {'kind': 'run', 'id': 'r1', 'field': 'answer', 'quote': '🐛 now'}

        label = _label(target={**target, 'span': [start, len(text)]})

        assert text[slice(*label.target.span)] == label.target.quote
        with pytest.raises(ValidationError, match='code points'):
            _label(target={**target, 'span': [start, start + 6]})  # UTF-16 units


class TestFields:
    @pytest.mark.parametrize(
        'changes',
        [
            {'name': ''},
            {'name': 'tone '},
            {'name': 'to\nne'},
            {'name': 'to\x00ne'},
            {'name': None},
            {'value': None},
            {'value': float('nan')},
            {'value': float('inf')},
            {'value': ''},
            {'value': 'clear\nshort'},
            {'value': 'clear short'},
            {'value': ['clear']},
            {'source': 'expert'},
            {'author': ''},
            {'author': '  '},
            {'author': 'ana '},
            {'author': 'a‮na'},
            {'author': 'a\x7fna'},
            {'blind': 'true'},
            {'blind': 1},
            {'rubric_sha256': 'AB' * 32},
            {'rubric_sha256': 'ab'},
            {'rubric_version': ''},
            {'group_id': ''},
            {'group_id': 'g\x001'},
            {'id': ''},
            {'type': 'delete'},
            {'supersedes': ''},
            {'score': 1.0},
        ],
    )
    def test_bad(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            _label(**changes)

    @pytest.mark.parametrize(
        ('value', 'kept'),
        [(3, 3.0), (0.25, 0.25), ('minor', 'minor'), ('1', '1'), (True, True)],
    )
    def test_value_keeps_its_type(self, value: object, kept: object) -> None:
        label = _label(name='rating', value=value)

        assert label.value == kept
        assert type(label.value) is type(kept)

    @pytest.mark.parametrize(
        ('name', 'value'), [('passed', 1.0), ('passed', 'yes'), ('c2', 'UNMET'), ('c10', 0.0)]
    )
    def test_verdicts_are_bools(self, name: str, value: object) -> None:
        with pytest.raises(ValidationError, match='true or false'):
            _label(name=name, value=value, source='judge', author='judge-a')

    @pytest.mark.parametrize(
        'target',
        [STEP_2, {'kind': 'run', 'id': 'r1', 'field': 'answer'},
         {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [0, 3]}],
    )  # fmt: skip
    @pytest.mark.parametrize('name', ['passed', 'c1'])
    def test_verdicts_are_on_whole_targets(self, target: dict, name: str) -> None:
        with pytest.raises(ValidationError, match='whole target'):
            _passed(name=name, target=target)

    def test_other_names_take_any_value_anywhere(self) -> None:
        assert _label(name='c0', value='UNMET').value == 'UNMET'
        assert _label(name='mark', value=0.5, target=STEP_2).value == 0.5

    def test_optional_fields(self) -> None:
        label = _label(
            id='client-1',
            comment='Clear.\nA little long, though.',
            blind=True,
            rubric_version='rubric-judge/1',
            rubric_sha256=SHA,
            group_id='call-7',
        )

        assert (label.id, label.blind, label.rubric_sha256, label.group_id) == (
            'client-1', True, SHA, 'call-7'
        )  # fmt: skip

    @pytest.mark.parametrize('comment', [None, '', '  \n'])
    def test_a_human_passed_verdict_says_why(self, comment: str | None) -> None:
        with pytest.raises(ValidationError, match='comment saying why'):
            _passed(comment=comment)

        assert _passed(comment=comment, source='judge', author='judge-a').comment == comment
        assert _passed(name='c1', comment=comment).comment == comment
        assert _label(comment=comment).comment == comment
        span = {'kind': 'run', 'id': 'r1', 'field': 'answer', 'span': [0, 3]}
        assert _label(target=span, comment=comment).comment == comment

    def test_invalidation_is_a_reason_about_a_whole_target(self) -> None:
        with pytest.raises(ValidationError, match='comment saying why'):
            _invalidate(source='verifier', comment='  ')
        with pytest.raises(ValidationError, match='no name or value'):
            _label(type='invalidate', value=None, comment='broken')
        with pytest.raises(ValidationError, match='whole target'):
            _invalidate(target=STEP_2)
        with pytest.raises(ValidationError, match='whole target'):
            _invalidate(target={'kind': 'run', 'id': 'r1', 'field': 'answer'})

        assert _invalidate().type == 'invalidate'

    def test_retraction_names_what_it_retracts(self) -> None:
        with pytest.raises(ValidationError, match='supersedes'):
            _label(type='retract', **WHOLE, comment='wrong trace')
        with pytest.raises(ValidationError, match='comment saying why'):
            _retract('x', source='judge', author='judge-a', comment=None)
        with pytest.raises(ValidationError, match='no name or value'):
            _label(type='retract', comment='wrong trace', supersedes='x')
        with pytest.raises(ValidationError, match='whole target'):
            _retract('x', target={'kind': 'trace', 'id': 't1', 'step': 1})


class TestCurrent:
    def test_supersede_chains(self, store: LabelStore) -> None:
        first = store.add(_label())
        other = store.add(_label(name='mood'))
        second = store.add(_label(value='terse', supersedes=first.id))
        third = store.add(_label(value='blunt', supersedes=second.id))

        assert current_labels([first, other, second, third]) == [other, third]
        assert store.current('trace', 't1') == [other, third]

    def test_retraction_drops_itself_and_its_label(self, store: LabelStore) -> None:
        kept = store.add(_label(name='mood'))
        wrong = store.add(_label())
        store.add(_retract(wrong.id))

        assert store.current('trace', 't1') == [kept]

    def test_retracting_a_correction_withdraws_the_chain(self, store: LabelStore) -> None:
        first = store.add(_label())
        fixed = store.add(_label(value='terse', supersedes=first.id))
        store.add(_retract(fixed.id))

        assert store.current('trace', 't1') == []
        assert current_labels(_all(store)) == []

    def test_current_filter_combines_with_any_other(self, store: LabelStore) -> None:
        renamed = store.add(_label(name='tone'))
        store.add(_label(name='mood', supersedes=renamed.id))
        retracted = store.add(_label(name='tone', value='warm'))
        store.add(_retract(retracted.id))
        kept = store.add(_label(name='tone', value='dry', author='bo'))

        tone = _all(store, name='tone')

        assert current_labels(tone) == [renamed, retracted, kept]  # misses rename, retraction
        assert _all(store, name='tone', current=True) == [kept]
        assert store.page(LabelQuery(name='tone', current=True), 0, 1).total == 1

    def test_invalidation_excludes_until_retracted(self, store: LabelStore) -> None:
        mark = store.add(_label(target=STEP_2))
        store.add(_label(target={'kind': 'run', 'id': 'r1'}))
        first = store.add(_invalidate())
        assert excluded(_all(store)) == store.excluded() == {('trace', 't1')}

        reworded = store.add(_invalidate(comment='verifier image was stale', supersedes=first.id))
        assert store.current('trace', 't1') == [mark, reworded]
        assert excluded(_all(store)) == store.excluded() == {('trace', 't1')}

        store.add(_retract(reworded.id, comment='verifier was fine'))
        assert store.current('trace', 't1') == [mark]
        assert excluded(_all(store)) == store.excluded() == set()


class TestSupersede:
    def test_unknown_label(self, store: LabelStore) -> None:
        with pytest.raises(ValueError, match='not stored'):
            store.add(_label(supersedes='nope'))

    @pytest.mark.parametrize(
        'target', [{'kind': 'trace', 'id': 't2'}, {'kind': 'episode', 'id': 't1'}]
    )
    def test_same_target(self, store: LabelStore, target: dict) -> None:
        first = store.add(_label())

        with pytest.raises(ValueError, match='another target'):
            store.add(_label(target=target, supersedes=first.id))
        with pytest.raises(ValueError, match='another target'):
            store.add(_retract(first.id, target=target))

    @pytest.mark.parametrize('changes', [{'source': 'judge'}, {'author': 'lead'}])
    def test_same_source_and_author(self, store: LabelStore, changes: dict) -> None:
        first = store.add(_label())

        with pytest.raises(ValueError, match="another labeler's view is a new label"):
            store.add(_label(value='terse', **changes, supersedes=first.id))
        with pytest.raises(ValueError, match="another labeler's view is a new label"):
            store.add(_retract(first.id, **changes))

    def test_any_human_retracts_an_invalidation(self, store: LabelStore) -> None:
        by_bot = store.add(_invalidate(source='verifier', author='qa-bot'))
        by_ana = store.add(_invalidate(target={'kind': 'run', 'id': 'r1'}))

        with pytest.raises(ValueError, match="another labeler's view"):
            store.add(_retract(by_bot.id, source='judge', author='judge-a'))
        with pytest.raises(ValueError, match="another labeler's view"):
            store.add(_invalidate(author='lead', supersedes=by_bot.id))
        store.add(_retract(by_bot.id, author='lead', comment='the verifier was fine'))
        store.add(_retract(by_ana.id, author='bo', target={'kind': 'run', 'id': 'r1'}))

        assert store.excluded() == set()

    def test_correction_may_move_the_step(self, store: LabelStore) -> None:
        first = store.add(_label(target={'kind': 'trace', 'id': 't1', 'step': 1}))

        fixed = store.add(_label(target=STEP_2, supersedes=first.id))

        assert store.current('trace', 't1') == [fixed]

    def test_correction_keeps_the_type(self, store: LabelStore) -> None:
        mark = store.add(_label())
        excluding = store.add(_invalidate())

        with pytest.raises(ValueError, match='keeps its type'):
            store.add(_invalidate(supersedes=mark.id))
        with pytest.raises(ValueError, match='keeps its type'):
            store.add(_label(supersedes=excluding.id))

    def test_retraction_is_final(self, store: LabelStore) -> None:
        retraction = store.add(_retract(store.add(_label()).id))

        with pytest.raises(ValueError, match='final'):
            store.add(_retract(retraction.id))

    def test_a_label_is_superseded_once(self, store: LabelStore) -> None:
        first = store.add(_label())
        store.add(_label(value='terse', supersedes=first.id))

        with pytest.raises(ValueError, match='already superseded'):
            store.add(_retract(first.id))


class TestOneCurrentVerdict:
    @pytest.mark.parametrize('name', ['passed', 'c3'])
    def test_second_verdict_must_supersede(self, store: LabelStore, name: str) -> None:
        first = store.add(_passed(name=name))

        with pytest.raises(ValueError, match='supersede it'):
            store.add(_passed(name=name, value=False))
        fixed = store.add(_passed(name=name, value=False, supersedes=first.id))

        assert store.current('trace', 't1') == [fixed]

    def test_retracted_verdict_can_be_given_again(self, store: LabelStore) -> None:
        store.add(_retract(store.add(_passed()).id))

        assert store.add(_passed(value=False)).value is False

    def test_others_repeats_tags_and_spans_stay_legal(self, store: LabelStore) -> None:
        span = {'kind': 'trace', 'id': 't1', 'step': 1, 'field': 'action', 'span': [0, 2]}
        judge = {'source': 'judge', 'author': 'judge-a', 'comment': None}
        labels = [
            _passed(),
            _passed(author='bo', value=False),
            _passed(**judge, group_id='repeat-1'),
            _passed(**judge, group_id='repeat-2', value=False),
            _passed(target={'kind': 'trace', 'id': 't2'}),
            _passed(name='c1'),
            _label(),
            _label(value='terse'),
            _label(name='mark', value='typo', target=span),
            _label(name='mark', value='typo', target=span),
        ]

        assert len(store.add_many(labels)) == len(labels)

    def test_two_verdicts_in_one_batch_store_none(self, store: LabelStore) -> None:
        with pytest.raises(ValueError, match='supersede it'):
            store.add_many([_passed(), _passed(value=False)])

        assert _all(store) == []

    def test_a_rename_into_a_held_verdict_is_refused(self, store: LabelStore) -> None:
        store.add(_passed(name='c2'))
        first = store.add(_passed(name='c1'))

        with pytest.raises(ValueError, match="current 'c2'"):
            store.add(_passed(name='c2', supersedes=first.id))


class TestClientIds:
    def test_a_retry_returns_the_stored_record(self, store: LabelStore) -> None:
        first = store.add(_passed(id='ana-1'))

        again = store.add(_passed(id='ana-1'))

        assert again == first
        assert _all(store) == [first]

    def test_a_retried_batch_stores_nothing_new(self, store: LabelStore) -> None:
        batch = [_passed(id='a'), _label(id='b'), _label(name='mood', id='c')]
        stored = store.add_many(batch)

        assert store.add_many([*batch, _label(name='size', id='d')])[:3] == stored
        assert [label.seq for label in _all(store)] == [1, 2, 3, 4]

    def test_same_id_other_body_is_refused(self, store: LabelStore) -> None:
        store.add(_label(id='ana-1'))

        with pytest.raises(ValueError, match='another body'):
            store.add(_label(id='ana-1', value='terse'))
        with pytest.raises(ValueError, match='another body'):
            store.add_many([_label(name='mood'), _label(id='ana-1', comment='hm')])

        assert len(_all(store)) == 1


class TestBatch:
    def test_one_transaction_one_time(self, store: LabelStore) -> None:
        labels = store.add_many([_passed(name=f'c{n}') for n in range(1, 4)])

        assert [label.name for label in labels] == ['c1', 'c2', 'c3']
        assert [label.seq for label in labels] == [1, 2, 3]
        assert len({label.created_at for label in labels}) == 1
        assert len({label.id for label in labels}) == 3
        assert _all(store) == labels

    def test_a_bad_record_writes_none(self, store: LabelStore) -> None:
        first = store.add(_label())

        with pytest.raises(ValueError, match='not stored'):
            store.add_many([_label(name='mood'), _label(supersedes='nope')])
        with pytest.raises(ValueError, match='already superseded'):
            store.add_many([_label(value='terse', supersedes=first.id), _retract(first.id)])

        assert _all(store) == [first]
        assert store.add(_label(name='mood')).seq == 2


def test_time_never_runs_back(store: LabelStore, monkeypatch: pytest.MonkeyPatch) -> None:
    first = store.add(_label())

    class SlowClock(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return datetime(2020, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(label_store, 'datetime', SlowClock)
    second = store.add(_label(name='mood'))

    assert second.seq == 2
    assert second.created_at == first.created_at


def test_seq_and_time_increase_together_across_writers(tmp_path: Path) -> None:
    stores = [LabelStore(tmp_path / 'console.db') for _ in range(4)]

    def write(store: LabelStore) -> None:
        for _ in range(15):
            store.add(_label())

    threads = [threading.Thread(target=write, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    labels = _all(stores[0])
    assert [label.seq for label in labels] == list(range(1, 61))
    assert [label.created_at for label in labels] == sorted(label.created_at for label in labels)


class TestQuery:
    def test_paging_counts_every_match(self, store: LabelStore) -> None:
        labels = [store.add(_passed(name=f'c{n}')) for n in range(1, 6)]

        page = store.page(LabelQuery(), 1, 2)
        rest = store.page(LabelQuery(), 3, None)

        assert (page.labels, page.total) == (labels[1:3], 5)
        assert (rest.labels, rest.total) == (labels[3:], 5)
        assert store.page(LabelQuery(), 9, None).model_dump() == {'labels': [], 'total': 5}

    def test_count_and_page_read_one_snapshot(
        self, store: LabelStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.add(_label())
        writer = LabelStore(tmp_path / 'console.db')
        real_connect = label_store.connect
        raced: list[bool] = []

        @contextmanager
        def racing_connect(path: Path) -> Iterator:
            with real_connect(path) as conn:

                def between(sql: str) -> None:
                    if sql.startswith('SELECT record') and not raced:
                        raced.append(True)
                        writer.add(_label(name='mood'))  # lands after the COUNT

                conn.set_trace_callback(between)
                yield conn

        monkeypatch.setattr(label_store, 'connect', racing_connect)

        page = store.page(LabelQuery(), 0, None)

        assert raced
        assert (len(page.labels), page.total) == (1, 1)
        assert store.page(LabelQuery(), 0, None).total == 2

    @pytest.mark.parametrize(
        ('query', 'expected'),
        [
            ({'target_kind': 'run'}, [1]),
            ({'target_id': 't1'}, [0, 3, 4, 5]),
            ({'target_kind': 'trace', 'target_id': 't1', 'name': 'c1'}, [3]),
            ({'name': 'passed'}, [0, 1, 2]),
            ({'source': 'judge'}, [2, 3]),
            ({'author': 'ana'}, [0, 1, 4, 5]),
            ({'source': 'judge', 'author': 'ana'}, []),
            ({'type': 'invalidate'}, [5]),
            ({'type': 'label', 'target_id': 't1'}, [0, 3, 4]),
            ({'step': 2}, [4]),
            ({'step': 0}, []),
            ({'blind': True}, [3]),
            ({'blind': False}, [2]),
            ({'rubric_sha256': SHA}, [2, 3]),
            ({'rubric_version': 'rubric-judge/1'}, [3]),
            ({'group_id': 'call-1'}, [2, 3]),
            ({'after_seq': 4}, [4, 5]),
            ({'after_seq': 0}, [0, 1, 2, 3, 4, 5]),
            ({'current': True}, [0, 1, 2, 3, 4, 5]),
        ],
    )
    def test_filters(self, store: LabelStore, query: dict, expected: list[int]) -> None:
        judged = {'source': 'judge', 'author': 'j', 'rubric_sha256': SHA, 'group_id': 'call-1'}
        labels = [
            store.add(_passed()),
            store.add(_passed(target={'kind': 'run', 'id': 'r1'})),
            store.add(_passed(target={'kind': 'trace', 'id': 't2'}, blind=False, **judged)),
            store.add(
                _passed(name='c1', value=False, blind=True, rubric_version='rubric-judge/1',
                        **judged)
            ),
            store.add(_label(name='mark', target=STEP_2)),
            store.add(_invalidate()),
        ]  # fmt: skip

        page = store.page(LabelQuery.model_validate(query), 0, None)

        assert page.labels == [labels[i] for i in expected]
        assert page.total == len(expected)

    def test_created_range(self, store: LabelStore) -> None:
        first, second, third = (store.add(_label(name=f'm{n}')) for n in range(1, 4))
        india = timezone(timedelta(hours=5, minutes=30))

        def names(since: object = None, until: object = None) -> list[str]:
            return [label.name for label in _all(store, since=since, until=until)]

        assert names(since=second.created_at) == ['m2', 'm3']
        assert names(until=second.created_at) == ['m1']
        assert names(since=second.created_at.astimezone(india)) == ['m2', 'm3']
        assert names(since=first.created_at, until=third.created_at) == ['m1', 'm2']

    @pytest.mark.parametrize(
        'query',
        [
            {'since': '2026-11-19T10:00:00'},
            {'until': '9999-12-31T23:59:59-10:00'},
            {'since': '0001-01-01T00:00:00+05:00'},
            {'after_seq': 2**63},
            {'step': -1},
            {'stepp': 1},
        ],
    )
    def test_bad_query(self, query: dict) -> None:
        with pytest.raises(ValidationError):
            LabelQuery.model_validate(query)


def test_jsonl_round_trip(store: LabelStore, tmp_path: Path) -> None:
    first = store.add(_passed(id='ana-1', blind=True, rubric_sha256=SHA))
    store.add_many(
        [
            _passed(value=False, comment='Misses a case.', supersedes=first.id),
            _label(target={'kind': 'trace', 'id': 't1', 'step': 2, 'field': 'message:0',
                           'span': [0, 5], 'quote': 'think'},
                   name='mark', value='rambling', source='judge', author='judge-a',
                   rubric_version='judge/1', group_id='call-1'),
            _label(target={'kind': 'run', 'id': 'r1'}, name='score', value=0.75,
                   source='verifier', author='fizzbuzz'),
            _invalidate(target={'kind': 'run', 'id': 'r2'}),
        ]
    )  # fmt: skip
    out = tmp_path / 'out' / 'labels.jsonl'

    written = store.export_jsonl(out)

    lines = out.read_text(encoding='utf-8').splitlines()
    assert written == len(lines) == 5
    assert [Label.model_validate_json(line) for line in lines] == _all(store)
    assert [json.loads(line)['value'] for line in lines] == [True, False, 'rambling', 0.75, None]
    assert [json.loads(line)['seq'] for line in lines] == [1, 2, 3, 4, 5]
    assert json.loads(lines[0])['id'] == 'ana-1'
