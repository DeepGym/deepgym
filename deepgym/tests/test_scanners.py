"""Tests for trace scanners: rule and model scanners, scan runs, groups, labels, agreement, CLI."""

import json
import re
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.cli import main
from deepgym.console.label_store import LabelStore, read_labels
from deepgym.dataset import DatasetSpec, LabelMatch, build_dataset
from deepgym.labels import LabelCreate, LabelTarget
from deepgym.models import Action, Observation
from deepgym.rubric import ContextOverflowError
from deepgym.scanners import (
    SCAN_ERROR,
    SCAN_PROMPT,
    SCAN_PROMPT_VERSION,
    Finding,
    HumanMark,
    LLMScanner,
    PatternScanner,
    ProtectedPathScanner,
    ScanError,
    ScannerInfo,
    ScanReport,
    finding_labels,
    group_findings,
    load_rule_scanners,
    scan_traces,
    scanner_agreement,
    scanner_forbids,
)
from deepgym.scanners.writes import cli_writes, python_writes, shell_writes
from deepgym.traces import Trace, TraceMessage, TraceStep
from deepgym.traces.schema import json_text, sha256_text
from tests.test_traces_export import _event_log
from tests.trace_samples import STARTED, graded_trace

WIDE = '\U0001d518'  # one code point, two UTF-16 units, four UTF-8 bytes
BASE = graded_trace()


def _step(
    index: int,
    action: Action,
    observation: str | None = 'ok',
    *,
    messages: tuple[TraceMessage, ...] = (),
    applied: bool = True,
    exit_code: int | None = None,
) -> TraceStep:
    shown = None
    if observation is not None:
        metadata = None if exit_code is None else {'exit_code': exit_code}
        shown = Observation(content=observation, step=index, metadata=metadata)
    return TraceStep(
        index=index,
        action=action,
        observation=shown,
        messages=messages,
        applied=applied,
        timestamp=STARTED,
        duration_ms=1.0,
    )


def _trace(trace_id: str, *steps: TraceStep) -> Trace:
    return BASE.model_copy(update={'trace_id': trace_id, 'steps': steps})


def _bash(*commands: str) -> Trace:
    steps = [_step(i, Action(action_type='bash', content=c)) for i, c in enumerate(commands, 1)]
    return _trace('t1', *steps)


def _scan_one(scanner: object, trace: Trace) -> list[Finding]:
    report = scan_traces([trace], [scanner])
    assert report.errors == ()
    return list(report.findings)


PROTECTED = ProtectedPathScanner(name='protected', globs=('tests/**',))


class TestPatternScanner:
    def test_cites_matches_in_code_points(self) -> None:
        content = f'echo "{WIDE} café" && rm -rf /tmp/x'
        scanner = PatternScanner(
            name='danger', patterns=[{'regex': r'rm -rf \S+', 'message': 'Deletes recursively'}]
        )

        (found,) = _scan_one(scanner, _bash(content))

        start = content.index('rm -rf')
        assert found.span == (start, len(content)) and start == 17  # 18 in UTF-16 units
        assert found.quote == 'rm -rf /tmp/x' == content[slice(*found.span)]
        assert (found.step, found.field, found.severity) == (1, 'action', 'medium')
        assert found.message == 'Deletes recursively'

    def test_scans_the_fields_chosen(self) -> None:
        messages = (
            TraceMessage(role='user', content='never rm -rf here'),
            TraceMessage(role='assistant', content=f'{WIDE} I will rm -rf b'),
        )
        step = _step(1, Action(action_type='bash', content='rm -rf a'), 'did rm -rf a',
                     messages=messages)  # fmt: skip
        scanner = PatternScanner(
            name='danger',
            patterns=[{'regex': 'rm -rf', 'message': 'Deletes', 'severity': 'high'}],
            fields=('observation', 'assistant'),
        )

        found = _scan_one(scanner, _trace('t1', step))

        assert [(f.field, f.span) for f in found] == [
            ('observation', (4, 10)),
            ('message:1', (9, 15)),
        ]
        assert {f.severity for f in found} == {'high'}

    def test_no_match_and_empty_matches_find_nothing(self) -> None:
        scanner = PatternScanner(
            name='p',
            patterns=[{'regex': 'zzz', 'message': 'Never'}, {'regex': 'x*', 'message': 'Empty'}],
        )

        assert _scan_one(scanner, _bash('echo abc')) == []

    def test_config_round_trips_and_refuses_bad_scanners(self) -> None:
        text = json.dumps([
            {'kind': 'pattern', 'name': 'p', 'patterns': [{'regex': '(?i)RM', 'message': 'Rm'}]},
            {'kind': 'protected_paths', 'name': 'q', 'globs': ['tests/**'], 'severity': 'low'},
        ])  # fmt: skip

        scanners = load_rule_scanners(text)

        assert [type(s) for s in scanners] == [PatternScanner, ProtectedPathScanner]
        assert scanners[0].patterns[0].regex.search('rm')
        dumped = json.dumps([s.model_dump(mode='json') for s in scanners])
        assert load_rule_scanners(dumped) == scanners
        for bad in (
            [{'kind': 'pattern', 'name': 'p', 'patterns': [{'regex': '(', 'message': 'x'}]}],
            [{'kind': 'nope', 'name': 'p'}],
            [{'kind': 'protected_paths', 'name': 'q', 'globs': ['a'], 'extra': 1}],
            [{'kind': 'protected_paths', 'name': 'q', 'globs': []}],
            [{'kind': 'pattern', 'name': ' p', 'patterns': [{'regex': 'a', 'message': 'x'}]}],
        ):
            with pytest.raises(ValidationError):
                load_rule_scanners(json.dumps(bad))


class TestProtectedPaths:
    @pytest.mark.parametrize(
        ('command', 'quotes'),
        [
            ('echo hi > tests/a.py', ['tests/a.py']),
            ('echo hi >>tests/a.py', ['tests/a.py']),
            ("sed -i 's/a/b/' tests/x.py", ['tests/x.py']),
            ('sed s/a/b/ tests/x.py', []),
            ("perl -pi -e 's/a/b/' tests/x.py", ['tests/x.py']),
            ('rm -rf tests', ['tests']),
            ('cp tests/a.py /tmp/a.py', []),
            ('cp src/a.py ./tests/a.py', ['./tests/a.py']),
            ('mv tests/a.py tests/b.py', ['tests/a.py', 'tests/b.py']),
            ('git checkout -- tests/', ['tests/']),
            ('git diff tests/', []),
            ("bash -lc 'rm tests/q.py'", ['tests/q.py']),
            ('cat tests/a.py | tee /tmp/out', []),
            ('echo x # > tests/c.py', []),
            ('FOO=1 sudo tee -a /repo/tests/a.py < in.txt', ['/repo/tests/a.py']),
            ("find tests -name '*.pyc' -delete", ['tests']),
            ('echo "tests/a.py" > notes.md', []),
            ('cat tests/a.py > out.txt 2>&1', []),
            (f"echo y > 'tests/{WIDE}.py'", [f'tests/{WIDE}.py']),
            ('echo y > "tests/a\\"b.py"', ['"tests/a\\"b.py"']),
            ('cd src && touch a.py; ln -sf ../evil tests/conftest.py', ['tests/conftest.py']),
            ('cat > notes.md <<EOF\nrm -rf tests\nEOF\nrm tests/x', ['tests/x']),
            ("cat <<-'EOF' > tests/a.py\n\tx > tests/b.py\n\tEOF", ['tests/a.py']),
            ('cat <<<word\nrm tests/x\nword', ['tests/x']),
            ('if true; then rm tests/a; else rm tests/b; fi', ['tests/a', 'tests/b']),
            ('while x; do rm tests/c; done; until y; do rm tests/d; done', ['tests/c', 'tests/d']),
            ('! rm tests/e; { rm tests/f; }', ['tests/e', 'tests/f']),
            ('sudo -u root rm tests/g', ['tests/g']),
            ('sudo --user root -E rm tests/h', ['tests/h']),
            ('nice -n 5 rm tests/i', ['tests/i']),
            ('env -u HOME X=1 rm tests/j', ['tests/j']),
            ('timeout 10 rm tests/k', ['tests/k']),
            ('timeout -s KILL 5s sudo -- rm tests/l', ['tests/l']),
            ('sudo -u tests ls', []),
            ("sed -ri 's/a/b/' tests/x.py", ['tests/x.py']),
            ("sed -n -e 's/i/x/p' tests/x.py", []),
            ("perl -Mstrict -e 'print' tests/x.py", []),
            ("perl -i.bak -pe 's/a/b/' tests/x.py", ['tests/x.py']),
            ("ruby -rjson -e 'p 1' tests/x.py", []),
            ('git -C repo rm tests/a.py', ['tests/a.py']),
            ('git checkout -b tests main', []),
            ('git clean -fd -e tests src/', []),
            ('git clean -fdx', []),
            ('rm tests/*', ['tests/*']),
            ("sed -i 's/tests/specs/' README.md", []),
            ("sed -i -e 's/a/b/' -f prog.sed tests/x.py", ['tests/x.py']),
            ("sed --expression='s/tests/x/' -i tests/x.py", ['tests/x.py']),
            ("sed -ie 's/a/b/' tests/x.py", ['tests/x.py']),
            ("perl -i -pe 's/tests/x/' docs.md", []),
            ('cp -t tests/ a.py b.py', ['tests/', 'tests/']),
            ('cp --target-directory=tests a.py', ['tests']),
            ('install -ttests a.py', ['tests']),
            ('mv -t /tmp tests/a.py', ['tests/a.py']),
            ('cp -rt tests/ x.py', ['tests/']),
            ('cp -rttests x.py', ['tests']),
            ('install -Dt tests/ f', ['tests/']),
            ('cp --target-directory tests x.py y.py', ['tests', 'tests']),
            ('cp -v --target-directory=tests/sub x.py', ['tests/sub']),
            ('install -m 755 tests/run.sh /usr/local/bin/', []),
            ('install -d tests/sub out', ['tests/sub']),
            ('cp a.py tests/b.py --suffix .bak', ['tests/b.py']),
            ('cp tests/a.py -t', ['tests/a.py']),  # no DIR: every operand
            ('cp -t a -t b tests/x.py', ['tests/x.py']),  # two DIRs: every operand
            ('cp tests/a.py', ['tests/a.py']),  # one operand: every operand
            ('rm -rf .', ['.']),
            ('rm -rf ../..', ['../..']),
            ('git checkout .', ['.']),
        ],
    )
    def test_bash(self, command: str, quotes: list[str]) -> None:
        found = _scan_one(PROTECTED, _bash(command))

        assert [f.quote for f in found] == quotes
        assert all(command[slice(*f.span)] == f.quote for f in found)
        assert {f.message for f in found} <= {'changes a path matching tests/**'}
        assert {f.severity for f in found} <= {'high'}

    def test_target_directories_name_each_source_inside(self) -> None:
        cases = {
            'cp -rt tests/ x.py src/y.py': ['tests/x.py', 'tests/y.py'],
            'install -Dttests f': ['tests/f'],
            'mv --target-directory tests a b/': ['tests/a', 'tests/b', 'a', 'b/'],
            'mv src/a tests/b': ['src/a', 'tests/b'],
            'cp -t a -t b x': ['x', 'a', 'b'],
        }

        assert {command: [w.path for w in shell_writes(command)] for command in cases} == cases

    @pytest.mark.parametrize(
        ('globs', 'command', 'flagged'),
        [
            (('/repo/tests/**',), 'rm tests/a.py', False),
            (('/repo/tests/**',), 'rm /repo/tests/a.py', True),
            (('/repo/tests/**',), 'rm -r /repo', True),
            (('conftest.py',), 'touch src/pkg/conftest.py', True),
            (('tests/test_*.py',), 'rm tests/sub/test_a.py', False),
            (('tests/test_*.py',), 'rm tests/test_a.py', True),
            (('tests/test_*.py',), 'rm -r tests', True),
            (('**/fixtures/*.json',), 'rm a/b/fixtures/x.json', True),
            (('tests/**',), 'rm tests/../src/a.py', False),
            (('tests/**',), 'rm -rf .', True),
            (('/repo/tests/**',), 'rm -rf .', False),
            (('tests',), 'rm tests/test_x.py', True),
            (('tests/',), 'rm tests/test_x.py', True),
            (('./tests/**',), 'rm tests/test_x.py', True),
            (('./tests/**',), 'rm ./tests/test_x.py', True),
            (('tests',), 'rm src/pkg/tests/sub/x.py', True),
            (('tests',), 'rm src/tests_old/x.py', False),
            (('tests/conftest.py',), 'rm tests/other.py', False),
            (('tests/conftest.py',), 'rm -r tests', True),
            (('tests/[ab].py',), 'rm tests/a.py', False),
            (('tests/[ab].py',), 'rm "tests/[ab].py"', True),
            (('tests/{a,b}.py',), 'rm tests/a.py', False),
        ],
    )
    def test_globs(self, globs: tuple[str, ...], command: str, flagged: bool) -> None:
        scanner = ProtectedPathScanner(name='protected', globs=globs)

        assert bool(_scan_one(scanner, _bash(command))) is flagged

    def test_python_code(self) -> None:
        code = (
            'import os, shutil, subprocess\n'
            'from pathlib import Path\n'
            f"note = '{WIDE}'\n"
            "open('tests/a.py', 'w').write(note)\n"
            "open('tests/read.py').read()\n"
            "Path('tests/b.py').write_text('tests/content.py')\n"
            'os.remove("tests/c.py")\n'
            "shutil.copy('tests/src.py', 'tests/d.py')\n"
            "note.replace('tests/e.py', 'x')\n"
            "os.system('rm tests/f.py')\n"
            "subprocess.run(['rm', '-f', 'tests/g.py'])\n"
            "Path('tests/h.py').open(mode='a')\n"
            "str('tests/i.py').replace('x', 'y')\n"
            "os.remove(os.path.join('tests', 'j.py'))\n"
            "pathlib.PurePosixPath('tests/k.py').unlink()\n"
            "Template('tests/l.py').touch()\n"
        )

        found = _scan_one(
            PROTECTED, _trace('t1', _step(1, Action(action_type='code', content=code)))
        )

        quotes = [f'tests/{c}.py' for c in 'abcdfgh'] + ["os.path.join('tests', 'j.py')"]
        assert [f.quote for f in found] == [*quotes, 'tests/k.py']
        assert all(code[slice(*f.span)] == f.quote for f in found)
        assert found[0].span[0] == code.index("'tests/a.py'") + 1

    def test_python_imports_and_path_wrappers(self) -> None:
        code = (
            'import os, os.path as osp, pathlib as pl, shutil as sh, subprocess as sp\n'
            'from os import fspath, remove as rm_file\n'
            'from pathlib import Path as P\n'
            'from subprocess import run as sh_run\n'
            "P('tests/a.py').write_text('x')\n"
            "pl.Path('tests/b.py').unlink()\n"
            "sh.rmtree('tests/c')\n"
            "rm_file(osp.abspath('tests/d.py'))\n"
            "open(str(P('tests/e.py')), 'w')\n"
            "sh.move('src/x', fspath('tests/f.py'))\n"
            "rm_file(os.path.realpath(osp.normpath('tests/g.py')))\n"
            "(P('tests') / 'h.py').touch()\n"
            "P('tests').joinpath('sub', 'i.py').unlink()\n"
            "sp.run(['rm', 'tests/j.py'])\n"
            "sh_run('rm tests/k.py', shell=True)\n"
            "rm_file(osp.join('tests', 'l.py'))\n"
            "P('tests/m.py').resolve().unlink()\n"
            "P(P('tests/n.py')).chmod(0o644)\n"
            "str(P('tests/o.py')).replace('a', 'b')\n"
            "osp.exists('tests/p.py')\n"
            "name = 'tests/q.py'\n"
            'rm_file(name)\n'
        )

        writes = python_writes(code)

        assert [(w.path, code[slice(*w.span)]) for w in writes] == [
            ('tests/a.py', 'tests/a.py'),
            ('tests/b.py', 'tests/b.py'),
            ('tests/c', 'tests/c'),
            ('tests/d.py', 'tests/d.py'),
            ('tests/e.py', 'tests/e.py'),
            ('src/x', 'src/x'),  # a move changes both paths
            ('tests/f.py', 'tests/f.py'),
            ('tests/g.py', 'tests/g.py'),
            ('tests/h.py', "P('tests') / 'h.py'"),
            ('tests/sub/i.py', "P('tests').joinpath('sub', 'i.py')"),
            ('tests/j.py', 'tests/j.py'),
            ('tests/k.py', 'tests/k.py'),
            ('tests/l.py', "osp.join('tests', 'l.py')"),
            ('tests/m.py', 'tests/m.py'),
            ('tests/n.py', 'tests/n.py'),
        ]

    def test_python_keywords_and_path_parts(self) -> None:
        code = (
            'import os\nfrom pathlib import Path\n'
            "open(file='tests/a.py', mode='w')\n"
            "open(mode='a', file='tests/b.py')\n"
            "open(file='tests/c.py')\n"
            "os.remove(path='tests/d.py')\n"
            "os.rename(src='src/x', dst='tests/e.py')\n"
            "Path('tests', 'f.py').unlink()\n"
        )

        writes = python_writes(code)

        assert [(w.path, code[slice(*w.span)]) for w in writes] == [
            ('tests/a.py', 'tests/a.py'),
            ('tests/b.py', 'tests/b.py'),
            ('tests/d.py', 'tests/d.py'),
            ('src/x', 'src/x'),
            ('tests/e.py', 'tests/e.py'),
            ('tests/f.py', "Path('tests', 'f.py')"),
        ]

    @pytest.mark.parametrize(
        ('code', 'paths'),
        [
            (  # a conditional rebinding can't hide the module
                "import os\nif flag:\n    import foo as os\nos.remove('tests/a.py')\n",
                ['tests/a.py'],
            ),
            (  # a later or nested rebinding neither
                "import os\ndef f():\n    import mylib as os\n    os.remove('tests/b.py')\n",
                ['tests/b.py'],
            ),
            (  # a function-local import
                "def clean():\n    import shutil as s\n    s.rmtree('tests/c')\nclean()\n",
                ['tests/c'],
            ),
            (  # one name bound to a module and to something unrelated
                'try:\n    import shutil as sh\nexcept ImportError:\n'
                "    from mylib import helper as sh\nsh.move('src/x', 'tests/d.py')\n",
                ['src/x', 'tests/d.py'],
            ),
            (  # a builtin shadowed by an import somewhere keeps its own meaning
                "if flag:\n    from mylib import open\nopen('tests/e.py', 'w')\n",
                ['tests/e.py'],
            ),
            (  # a name bound only to something unrelated is not a file call
                "from mylib import move\nmove('tests/f.py')\n",
                [],
            ),
        ],
    )
    def test_python_names_take_every_meaning_they_may_have(
        self, code: str, paths: list[str]
    ) -> None:
        assert [write.path for write in python_writes(code)] == paths

    def test_a_very_deep_path_matches_in_linear_time(self) -> None:
        deep = 'd/' * 200_000
        scanner = ProtectedPathScanner(
            name='protected',
            globs=('tests/**', '**/conftest.py', 'tests', '/repo/tests/**', 'tests/test_*.py'),
        )
        trace = _trace(
            't1',
            _step(1, Action(action_type='write_file', path=f'{deep}x.py', content='')),
            _step(2, Action(action_type='write_file', path=f'{deep}tests/x.py', content='')),
            _step(3, Action(action_type='write_file', path=f'{deep}conftest.py', content='')),
        )

        started = time.perf_counter()
        found = _scan_one(scanner, trace)

        assert time.perf_counter() - started < 5
        assert [(f.step, f.message.rpartition(' ')[2]) for f in found] == [
            (2, 'tests/**'), (3, '**/conftest.py')
        ]  # fmt: skip

    def test_code_that_python_refused_changed_nothing(self) -> None:
        broken = Action(action_type='code', content="open('tests/a.py', 'w'")
        refused = _step(1, broken, 'SyntaxError: unexpected EOF', exit_code=1)
        unknown = _step(1, broken, 'done', exit_code=0)

        assert _scan_one(PROTECTED, _trace('t1', refused)) == []
        report = scan_traces([_trace('t2', unknown)], [PROTECTED])
        assert report.findings == ()
        assert [(e.scanner, e.trace_id) for e in report.errors] == [('protected', 't2')]
        assert 'does not parse as Python here' in report.errors[0].error

    def test_cli_events(self) -> None:
        edit = {'file_path': '/w/tests/e.py', 'old_string': 'a', 'new_string': 'b'}
        claude = [
            {'type': 'text', 'text': 'Editing'},
            {'type': 'tool_use', 'id': 'a', 'name': 'Read', 'input': {'file_path': '/w/r.py'}},
            {'type': 'tool_use', 'id': 'b', 'name': 'Edit', 'input': edit},
            {'type': 'tool_use', 'id': 'c', 'name': 'Bash', 'input': {'command': 'rm tests/x.py'}},
            {'type': 'tool_use', 'id': 'd', 'name': 'Write', 'input': {'file_path': 'tests/n.py'}},
        ]
        refused = {'type': 'tool_result', 'tool_use_id': 'd', 'is_error': True, 'content': 'no'}
        changes = [{'path': '/w/tests/f.py', 'kind': 'update'}, {'path': '/w/ok.py', 'kind': 'add'}]
        command = "bash -lc 'echo > tests/g.py'"
        failed = {'type': 'file_change', 'status': 'failed', 'changes': [{'path': 'tests/m.py'}]}
        run = {'id': 'c1', 'type': 'command_execution', 'command': command}
        cut = {'id': 'c2', 'type': 'command_execution', 'command': 'rm tests/s.py'}
        events = (
            {'type': 'assistant', 'message': {'content': claude}},
            {'type': 'user', 'message': {'content': [refused]}},
            {'type': 'item.completed', 'item': {'type': 'file_change', 'changes': changes}},
            {'type': 'item.completed', 'item': failed},
            {'type': 'item.started', 'item': run},
            {'type': 'item.completed', 'item': run},
            {'type': 'item.started', 'item': cut},
            {'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}},
        )
        action = Action(action_type='cli', content='codex exec --json task', events=events)

        found = _scan_one(PROTECTED, _trace('t1', _step(1, action)))

        assert [(w.path, w.failed) for w in cli_writes(events)] == [
            ('/w/tests/e.py', False),
            ('tests/x.py', False),
            ('tests/n.py', True),
            ('/w/tests/f.py', False),
            ('/w/ok.py', False),
            ('tests/m.py', True),
            ('tests/g.py', False),
            ('tests/s.py', False),
        ]
        failure = 'matching tests/**; the tool reported failure'
        assert [f.message for f in found] == [
            'changes /w/tests/e.py matching tests/**',
            'changes tests/x.py matching tests/**',
            f'tried to change tests/n.py {failure}',
            'changes /w/tests/f.py matching tests/**',
            f'tried to change tests/m.py {failure}',
            'changes tests/g.py matching tests/**',
            'changes tests/s.py matching tests/**',
        ]
        assert {(f.span, f.quote, f.field, f.severity) for f in found} == {
            (None, None, 'action', 'high')
        }

    def test_other_action_kinds(self) -> None:
        steps = (
            _step(1, Action(action_type='write_file', path='tests/a.py', content='x')),
            _step(2, Action(action_type='write_file', path='src/a.py', content='tests/a.py')),
            _step(3, Action(action_type='bash', content='rm tests/b.py'), applied=False),
            _step(4, Action(action_type='type', content='rm tests/c.py')),
            _step(5, Action(action_type='tool', tool='files', arguments={'path': 'tests/d.py'})),
            _step(6, Action(action_type='done')),
        )

        found = _scan_one(PROTECTED, _trace('t1', *steps))

        assert [(f.step, f.message, f.span) for f in found] == [
            (1, 'changes tests/a.py matching tests/**', None)
        ]
        absolute = ProtectedPathScanner(name='protected', globs=('/repo/tests/**',))
        assert _scan_one(absolute, _trace('t1', *steps)) == []


class FakeJudge:
    """A judge client that replays canned replies (or raises them) and records each call."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, system: str, message: str) -> str:
        self.calls.append((system, json.loads(message)))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _reply(*findings: dict) -> str:
    return json.dumps({'findings': list(findings)})


def _cited(step: int | None, field: str | None, quote: str | None, **rest: str) -> dict:
    return {'step': step, 'field': field, 'quote': quote, 'message': 'Edits the tests',
            'severity': 'high'} | rest  # fmt: skip


LONG = 'x' * 200_000
LLM_TRACE = _trace(
    't1',
    _step(1, Action(action_type='bash', content=f'echo {WIDE} > tests/a.py # {LONG}'), None),
    _step(2, Action(action_type='bash', content='pytest'), f'{WIDE} 3 passed', exit_code=0),
).model_copy(update={'error': 'budget ran out'})


class TestLLMScanner:
    def test_reads_cited_findings_and_sends_the_whole_trace(self) -> None:
        reply = _reply(
            _cited(1, 'action', 'tests/a.py'),
            _cited(2, 'observation', 'passed', severity='low'),
            _cited(None, None, None, message='Never ran the full suite'),
        )
        judge = FakeJudge(f'```json\n{reply}\n```')
        scanner = LLMScanner('tamper', 'Find edits to tests.', judge)

        found = _scan_one(scanner, LLM_TRACE)

        content = LLM_TRACE.steps[0].action.content
        assert found[0].span == (content.index('tests/a.py'), content.index('tests/a.py') + 10)
        assert found[0].span == (9, 19)
        assert found[1].span == (4, 10) and found[1].severity == 'low'
        assert (found[2].step, found[2].field, found[2].message) == (
            None, None, 'Never ran the full suite'
        )  # fmt: skip
        ((system, sent),) = judge.calls
        assert system == SCAN_PROMPT and SCAN_PROMPT_VERSION == 'trace-scan/2'
        assert sent['instruction'] == 'Find edits to tests.' and sent['task'] == LLM_TRACE.task
        assert sent['initial_observation'] == '$ ' and sent['episode_error'] == 'budget ran out'
        assert sent['scope'] == 'trace'
        assert sent['steps'][0] == {
            'step': 1,
            'action': {'action_type': 'bash', 'content': content},
            'applied': True,
            'error': None,
            'exit_code': None,
            'observation': None,
            'messages': [],
        }
        assert sent['steps'][1]['observation'] == f'{WIDE} 3 passed'
        assert sent['steps'][1]['exit_code'] == 0
        assert (scanner.version, scanner.instruction_sha256) == (
            'trace-scan/2', sha256_text('Find edits to tests.')
        )  # fmt: skip

    def test_step_scope_sends_one_step_per_call(self) -> None:
        judge = FakeJudge(_reply(), _reply(_cited(2, 'observation', 'passed')))
        scanner = LLMScanner('tamper', 'Find edits to tests.', judge, scope='step')

        found = _scan_one(scanner, LLM_TRACE)

        assert [[s['step'] for s in sent['steps']] for _, sent in judge.calls] == [[1], [2]]
        assert {sent['scope'] for _, sent in judge.calls} == {'step'}
        assert [f.step for f in found] == [2]

    def test_step_scope_refuses_a_whole_episode_finding(self) -> None:
        judge = FakeJudge(_reply(_cited(None, None, None)))
        scanner = LLMScanner('tamper', 'Find edits.', judge, scope='step')

        (error,) = scan_traces([LLM_TRACE], [scanner]).errors

        assert 'cites the whole episode, but was sent one step' in error.error

    @pytest.mark.parametrize('scope', ['trace', 'step'])
    def test_a_trace_without_steps_gets_one_call(self, scope: str) -> None:
        judge = FakeJudge(_reply(_cited(None, None, None, message='Gave up at once')))
        scanner = LLMScanner('tamper', 'Find edits.', judge, scope=scope)

        (found,) = _scan_one(scanner, _trace('t1'))

        ((_, sent),) = judge.calls
        assert (sent['scope'], sent['steps']) == ('trace', [])
        assert found.message == 'Gave up at once'

    @pytest.mark.parametrize(
        'reply',
        [
            'No findings.',
            '[]',
            '{"findings": [], "findings": []}',
            '{"findings": [], "notes": "x"}',
            _reply({'step': 1, 'field': 'action', 'quote': 'tests'}),
            _reply(_cited(1, 'action', 'tests', extra='x')),
            _reply(_cited(True, 'action', 'tests')),
            _reply(_cited(1, 'answer', 'tests')),
            _reply(_cited(1, 'action', 'tests', severity='critical')),
            _reply(_cited(1, 'action', 'tests', message='two\nlines')),
            _reply(_cited(1, 'action', 'not in the text')),
            _reply(_cited(1, 'observation', 'anything')),
            _reply(_cited(2, 'message:0', 'anything')),
            _reply(_cited(3, 'action', 'pytest')),
            _reply(_cited(None, None, 'pytest')),
            _reply(_cited(1, 'action', '')),
        ],
    )
    def test_a_reply_not_as_asked_is_an_error(self, reply: str) -> None:
        scanner = LLMScanner('tamper', 'Find edits to tests.', FakeJudge(reply))

        report = scan_traces([LLM_TRACE], [scanner])

        assert report.findings == ()
        assert [(e.scanner, e.trace_id) for e in report.errors] == [('tamper', 't1')]

    def test_a_step_outside_the_call_is_an_error(self) -> None:
        judge = FakeJudge(_reply(_cited(2, 'action', 'pytest')))
        scanner = LLMScanner('tamper', 'Find edits.', judge, scope='step')

        report = scan_traces([LLM_TRACE], [scanner])

        assert 'cites step 2, which was not sent' in report.errors[0].error

    def test_judge_failure_is_an_error(self) -> None:
        scanner = LLMScanner('tamper', 'Find edits.', FakeJudge(RuntimeError('provider down')))

        report = scan_traces([LLM_TRACE], [scanner])

        assert report.errors == (
            ScanError(scanner='tamper', trace_id='t1', error='RuntimeError: provider down'),
        )

    def test_overflow_says_how_to_split(self) -> None:
        judge = FakeJudge(ContextOverflowError('prompt is too long'))
        scanner = LLMScanner('tamper', 'Find edits.', judge)

        (error,) = scan_traces([LLM_TRACE], [scanner]).errors

        assert "the trace does not fit the judge's context; scope='step'" in error.error

    def test_name_must_be_one_line_and_scope_trace_or_step(self) -> None:
        with pytest.raises(ValidationError):
            LLMScanner('two\nlines', 'Find edits.', FakeJudge())
        with pytest.raises(ValueError, match="scope must be 'trace' or 'step'"):
            LLMScanner('tamper', 'Find edits.', FakeJudge(), scope='steps')


class Broken:
    name = 'broken'
    source = 'verifier'
    version = 'broken/1'
    instruction_sha256 = None
    judge_id = None

    def scan(self, trace: Trace) -> list[Finding]:
        raise RuntimeError(f'cannot read {trace.trace_id}')


class Fixed:
    """Return the findings it was given, whatever the trace."""

    source = 'verifier'
    version = 'fixed/1'
    instruction_sha256 = None
    judge_id = None

    def __init__(self, name: str, *findings: object, kind: type = list) -> None:
        self.name = name
        self.findings = findings
        self.kind = kind

    def scan(self, trace: Trace) -> object:
        return None if self.kind is type(None) else self.kind(iter(self.findings))


class Returns:
    """Return the value it was given from every scan."""

    name = 'returns'
    source = 'verifier'
    version = 'returns/1'
    instruction_sha256 = None
    judge_id = None

    def __init__(self, value: object) -> None:
        self.value = value

    def scan(self, trace: Trace) -> object:
        return self.value


def _finding(**changes: object) -> Finding:
    fields = {'scanner': 'fixed', 'trace_id': 't1', 'message': 'Found it'}
    return Finding.model_validate(fields | changes)


class TestScanTraces:
    def test_a_scanner_that_raises_is_an_error_not_clean(self) -> None:
        traces = [_bash('rm tests/a.py'), _bash('ls').model_copy(update={'trace_id': 't2'})]

        report = scan_traces(traces, [Broken(), PROTECTED], max_parallel=4)

        assert report.traces == ('t1', 't2')
        assert report.scanners == {
            'broken': ScannerInfo(source='verifier', version='broken/1'),
            'protected': ScannerInfo(source='verifier', version=PROTECTED.version),
        }
        assert [(f.scanner, f.trace_id) for f in report.findings] == [('protected', 't1')]
        assert [e.error for e in report.errors] == [
            'RuntimeError: cannot read t1', 'RuntimeError: cannot read t2'
        ]  # fmt: skip
        assert report.failed('broken') == {'t1', 't2'} and report.failed('protected') == set()

    @pytest.mark.parametrize(
        ('finding', 'error'),
        [
            (_finding(trace_id='t9'), 'came back from scanner fixed on trace t1'),
            (_finding(step=5), 'has no step 5'),
            (_finding(step=1, field='message:0'), 'has no message 0'),
            (_finding(step=1, field='action', span=(0, 2), quote='xx'), "is 'rm', not the quote"),
        ],
    )
    def test_a_citation_the_trace_lacks_is_an_error(self, finding: Finding, error: str) -> None:
        report = scan_traces([_bash('rm a')], [Fixed('fixed', finding)])

        assert report.findings == () and error in report.errors[0].error

    @pytest.mark.parametrize('kind', [tuple, lambda found: (each for each in found)])
    def test_any_iterable_of_findings_counts(self, kind: type) -> None:
        found = _finding(step=1, field='action', span=(0, 2), quote='rm')

        report = scan_traces([_bash('rm a')], [Fixed('fixed', found, kind=kind)])

        assert (report.findings, report.errors) == ((found,), ())

    @pytest.mark.parametrize(
        ('scanner', 'error'),
        [
            (Fixed('fixed', {'step': 1}), 'TypeError: scan returned a dict, not a Finding'),
            (Fixed('fixed', kind=type(None)), "TypeError: 'NoneType' object is not iterable"),
        ],
    )
    def test_anything_but_findings_is_an_error(self, scanner: Fixed, error: str) -> None:
        report = scan_traces([_bash('rm a')], [scanner])

        assert report.findings == () and report.errors[0].error == error

    @pytest.mark.parametrize('value', ['', 'findings', b'', {}, {'step': 1}])
    def test_a_string_bytes_or_mapping_result_is_an_error(self, value: object) -> None:
        report = scan_traces([_bash('rm a')], [Returns(value)])

        assert report.errors[0].error == (
            f'TypeError: scan returned a {type(value).__name__}, not findings'
        )

    def test_rule_versions_name_their_reading_and_hash_their_config(self) -> None:
        other = ProtectedPathScanner(name='protected', globs=('tests/**', 'conftest.py'))
        plain = PatternScanner(name='p', patterns=[{'regex': 'rm', 'message': 'Rm'}])
        folded = PatternScanner(
            name='p', patterns=[{'regex': re.compile('rm', re.I), 'message': 'Rm'}]
        )

        config = sha256_text(json_text(PROTECTED.model_dump(mode='json')))
        assert PROTECTED.version == f'protected-paths/2+config-sha256:{config}'
        assert other.version != PROTECTED.version
        assert plain.version.startswith('pattern/1+config-sha256:')
        assert plain.model_dump_json() == folded.model_dump_json()
        assert plain.version != folded.version  # the flags count, though the JSON drops them
        assert (PROTECTED.instruction_sha256, PROTECTED.judge_id) == (None, None)

    def test_needs_a_scanner(self) -> None:
        with pytest.raises(ValueError, match='at least one scanner'):
            scan_traces([_bash('ls')], [])
        with pytest.raises(ValidationError):
            load_rule_scanners('[]')

    def test_refuses_shared_names_verdict_names_and_shared_trace_ids(self) -> None:
        with pytest.raises(ValueError, match='share names'):
            scan_traces([_bash('ls')], [PROTECTED, PROTECTED])
        with pytest.raises(ValueError, match='verdict names'):
            scan_traces([_bash('ls')], [Fixed('passed')])
        with pytest.raises(ValueError, match='share ids'):
            scan_traces([_bash('ls'), _bash('ls')], [PROTECTED])

    def test_a_finding_follows_the_label_target_rules(self) -> None:
        for bad in (
            {'field': 'action'},
            {'step': 1, 'field': 'answer'},
            {'step': 1, 'field': 'action', 'span': (2, 2)},
            {'step': 1, 'field': 'action', 'span': (0, 3), 'quote': 'ab'},
            {'step': 0},
            {'message': ''},
        ):
            with pytest.raises(ValidationError):
                _finding(**bad)


class TestGroups:
    def test_groups_by_scanner_normalised_message_and_key(self) -> None:
        findings = [
            _finding(scanner='a', message='Deletes Tests', step=1),
            _finding(scanner='a', message='deletes   tests', trace_id='t2', severity='high'),
            _finding(scanner='a', message='Other'),
            _finding(scanner='b', message='deletes tests'),
            _finding(scanner='a', message='DELETES TESTS'),
        ]

        groups = group_findings(findings)
        by_severity = group_findings(findings, key=lambda f: f.severity)

        assert [(g.scanner, g.message, len(g.findings), g.traces) for g in groups] == [
            ('a', 'deletes tests', 3, 2), ('a', 'other', 1, 1), ('b', 'deletes tests', 1, 1)
        ]  # fmt: skip
        assert groups[0].findings == (findings[0], findings[1], findings[4])
        assert [(g.message, g.key, len(g.findings)) for g in by_severity[:2]] == [
            ('deletes tests', 'medium', 2), ('deletes tests', 'high', 1)
        ]  # fmt: skip


def _scan_for_labels() -> tuple[Trace, ScanReport]:
    command = f'echo {WIDE} > tests/a.py'
    trace = _bash(command)
    judge = FakeJudge(_reply(_cited(1, 'action', WIDE, severity='low')))
    scanner = LLMScanner('tamper', 'Find edits.', judge, judge_id='judge-model, effort high')
    report = scan_traces([trace], [PROTECTED, scanner])
    return trace, report


class TestLabels:
    def test_findings_round_trip_through_the_label_store(self, tmp_path: Path) -> None:
        _, report = _scan_for_labels()
        store = LabelStore(tmp_path / 'console.db')

        store.add_many(finding_labels(report))
        store.add_many(finding_labels(report))  # the same report again is a retry
        stored = store.current('trace', 't1')

        assert [(label.id, label.target, label.name, label.value, label.source, label.author,
                 label.comment, label.group_id, label.rubric_version) for label in stored] == [
            (f'{report.run_id}-{n}', f.target, f.scanner, f.severity,
             'judge' if f.scanner == 'tamper' else 'verifier', f.scanner, f.message,
             report.run_id, report.scanners[f.scanner].version)
            for n, f in enumerate(report.findings)
        ]  # fmt: skip
        assert stored[1].target.quote == WIDE and stored[1].target.span == (5, 6)
        assert stored[1].rubric_version == 'trace-scan/2'
        assert [label.rubric_sha256 for label in stored] == [None, sha256_text('Find edits.')]
        assert all(label.type == 'label' for label in stored)
        assert report.scanners['tamper'] == ScannerInfo(
            source='judge',
            version='trace-scan/2',
            instruction_sha256=sha256_text('Find edits.'),
            judge_id='judge-model, effort high',
        )

    def test_a_scan_error_is_a_whole_trace_mark(self, tmp_path: Path) -> None:
        trace = _bash('rm tests/a.py')
        report = scan_traces([trace], [PROTECTED, Broken()])
        store = LabelStore(tmp_path / 'console.db')

        store.add_many(finding_labels(report))
        labels = read_labels(tmp_path / 'console.db')

        error = next(label for label in labels if label.name == 'broken')
        assert error.target == LabelTarget(kind='trace', id='t1')
        assert (error.value, error.comment) == (SCAN_ERROR, 'RuntimeError: cannot read t1')
        assert (error.source, error.author, error.id) == (
            'verifier',
            'broken',
            f'{report.run_id}-1',
        )
        spec = DatasetSpec(forbid_labels=(LabelMatch(name='broken'),), test_ratio=0, val_ratio=0)
        assert list(build_dataset([trace], spec, labels).entries) == []

    def test_scanner_forbids_drop_errored_traces_under_a_value_filter(self, tmp_path: Path) -> None:
        trace = _bash('ls')
        report = scan_traces([trace], [Broken()])
        LabelStore(tmp_path / 'console.db').add_many(finding_labels(report))
        labels = read_labels(tmp_path / 'console.db')

        def kept(forbid: tuple[LabelMatch, ...]) -> list[str]:
            spec = DatasetSpec(forbid_labels=forbid, test_ratio=0, val_ratio=0)
            return list(build_dataset([trace], spec, labels).entries)

        forbids = scanner_forbids('broken', source='verifier', values=('high',))
        assert forbids == (
            LabelMatch(name='broken', source='verifier', value='high'),
            LabelMatch(name='broken', source='verifier', value=SCAN_ERROR),
        )
        assert kept((LabelMatch(name='broken', source='verifier', value='high'),)) == ['t1']
        assert kept(forbids) == []
        assert scanner_forbids('broken') == (
            LabelMatch(name='broken'),
            LabelMatch(name='broken', value=SCAN_ERROR),
        )

    def test_retracting_an_old_scan_error_mark_lifts_it(self, tmp_path: Path) -> None:
        trace = _bash('ls')
        report = scan_traces([trace], [Broken()])
        store = LabelStore(tmp_path / 'console.db')
        (mark,) = store.add_many(finding_labels(report))
        spec = DatasetSpec(forbid_labels=scanner_forbids('broken'), test_ratio=0, val_ratio=0)

        assert (
            list(build_dataset([trace], spec, read_labels(tmp_path / 'console.db')).entries) == []
        )
        store.add(
            LabelCreate(
                type='retract',
                target={'kind': 'trace', 'id': 't1'},
                source='verifier',
                author='broken',
                supersedes=mark.id,
                comment='Rescanned cleanly.',
            )
        )
        labels = read_labels(tmp_path / 'console.db')
        assert list(build_dataset([trace], spec, labels).entries) == ['t1']

    def test_dataset_forbid_labels_drop_the_flagged_trace(self, tmp_path: Path) -> None:
        trace, report = _scan_for_labels()
        store = LabelStore(tmp_path / 'console.db')
        store.add_many(finding_labels(report))
        labels = read_labels(tmp_path / 'console.db')

        def kept(*forbid: LabelMatch) -> list[str]:
            spec = DatasetSpec(forbid_labels=forbid, test_ratio=0, val_ratio=0)
            return list(build_dataset([trace], spec, labels).entries)

        assert kept(LabelMatch(name='protected', source='verifier')) == []
        assert kept(LabelMatch(name='tamper', value='low')) == []
        assert kept(LabelMatch(name='tamper', value='high')) == ['t1']


def _mark(trace_id: str, value: bool, *, step: int | None = None, **changes: object) -> LabelCreate:
    fields = {
        'target': {'kind': 'trace', 'id': trace_id, 'step': step},
        'name': 'hack',
        'value': value,
        'source': 'human',
        'author': 'ana',
    }
    return LabelCreate.model_validate(fields | changes)


class TestAgreement:
    def test_compares_each_scanner_with_the_human_labels(self, tmp_path: Path) -> None:
        report = ScanReport(
            scanners={
                'a': ScannerInfo(source='verifier', version='a/1'),
                'b': ScannerInfo(source='judge', version='b/1'),
            },
            traces=tuple(f't{n}' for n in range(1, 10)),
            findings=(
                _finding(scanner='a', trace_id='t1', step=1),
                _finding(scanner='a', trace_id='t2'),
                _finding(scanner='a', trace_id='t5', step=2),
                _finding(scanner='b', trace_id='t1'),
            ),
            errors=(ScanError(scanner='a', trace_id='t4', error='boom'),),
        )
        store = LabelStore(tmp_path / 'console.db')
        store.add_many([
            _mark('t1', True), _mark('t2', False), _mark('t3', True),
            _mark('t3', False, author='bo'), _mark('t4', True), _mark('t6', True),
            _mark('t7', False), _mark('t7', True, source='judge', author='llm'), _mark('t9', True),
            _mark('t1', True, step=1), _mark('t1', True, step=2), _mark('t5', False, step=2),
            _mark('t4', True, step=1), _mark('t10', True),
            LabelCreate(type='invalidate', target={'kind': 'trace', 'id': 't9'}, source='human',
                        author='ana', comment='The sandbox broke.'),
            _mark('t1', False, name='passed', comment='Wrong.'),
            _mark('t7', True, name='passed', comment='Right.'),
        ])  # fmt: skip
        marks = {'a': HumanMark(name='hack'), 'b': HumanMark(name='passed', flagged=False)}

        agreement = scanner_agreement(report, read_labels(tmp_path / 'console.db'), marks)

        a, b = agreement['a'], agreement['b']
        assert a.traces.model_dump(include={'tp', 'fn', 'tn', 'fp'}) == {
            'tp': 1, 'fn': 1, 'tn': 1, 'fp': 1
        }  # fmt: skip
        assert a.steps.model_dump(include={'tp', 'fn', 'tn', 'fp'}) == {
            'tp': 1, 'fn': 1, 'tn': 0, 'fp': 1
        }  # fmt: skip
        assert (a.contested, a.scan_errors) == (1, 2)
        assert a.traces.kappa == 0.0 and a.traces.f1 == 0.5
        assert b.traces.model_dump(include={'tp', 'fn', 'tn', 'fp'}) == {
            'tp': 1, 'fn': 0, 'tn': 1, 'fp': 0
        }  # fmt: skip
        assert b.traces.kappa == 1.0 and b.steps.agreement is None
        with pytest.raises(ValueError, match="no scanners \\['c'\\]"):
            scanner_agreement(report, [], {'c': HumanMark(name='hack')})


class TestCli:
    @pytest.fixture(autouse=True)
    def default_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / 'default.db'
        monkeypatch.setattr('deepgym.console.db.DEFAULT_DB_PATH', path)
        return path

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
        monkeypatch.setattr(sys, 'argv', ['deepgym', 'traces', 'scan', *args])
        main()

    @staticmethod
    def _sources(tmp_path: Path, *traces: Trace) -> tuple[str, str]:
        config = tmp_path / 'scanners.json'
        config.write_text(json.dumps([
            {'kind': 'protected_paths', 'name': 'protected', 'globs': ['tests/**']},
            {'kind': 'pattern', 'name': 'danger',
             'patterns': [{'regex': 'rm -rf', 'message': 'Deletes recursively'}]},
        ]))  # fmt: skip
        folder = tmp_path / 'traces'
        folder.mkdir()
        for trace in traces:
            (folder / f'{trace.trace_id}.jsonl').write_text(_event_log(trace))
        return str(config), str(folder)

    def test_prints_groups_writes_the_report_and_labels(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
        default_db: Path,
    ) -> None:
        first = _bash('rm -rf tests', 'ls')
        second = _bash(f'echo {WIDE} > tests/a.py').model_copy(update={'trace_id': 't2'})
        config, folder = self._sources(tmp_path, first, second)
        out = tmp_path / 'report.json'

        self._run(monkeypatch, '--config', config, '--traces', folder, '--out', str(out),
                  '--write-labels')  # fmt: skip

        printed = capsys.readouterr().out
        assert 'Scanned 2 traces with 2 scanners: 3 findings in 2 groups, 0 errors' in printed
        assert 'protected: changes a path matching tests/** (2 in 2 traces)' in printed
        assert "t1 step 1 action [7:12] 'tests' (high)" in printed
        assert "t2 step 1 action [9:19] 'tests/a.py' (high)" in printed
        assert 'danger: Deletes recursively (1 in 1 traces)' in printed
        report = ScanReport.model_validate_json(out.read_text())
        assert len(report.findings) == 3
        assert (
            f'stored 3 labels (3 findings, 0 scan errors) in {default_db} '
            f'(group_id {report.run_id})'
        ) in printed
        assert {label.name for label in read_labels(default_db)} == {'protected', 'danger'}

    def test_exits_two_when_a_scanner_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        unparsed = _trace('t1', _step(1, Action(action_type='code', content='def f(:'), 'done'))
        config, folder = self._sources(tmp_path, unparsed)
        db = tmp_path / 'labels.db'
        LabelStore(db)

        with pytest.raises(SystemExit) as exit_info:
            self._run(monkeypatch, '--config', config, '--traces', folder, '--db', str(db),
                      '--write-labels')  # fmt: skip

        assert exit_info.value.code == 3
        captured = capsys.readouterr()
        assert '0 findings in 0 groups, 1 errors' in captured.out
        assert 'stored 1 labels (0 findings, 1 scan errors)' in captured.out
        assert 'error: protected on t1: ValueError: step 1: the code does not parse' in captured.err
        assert [(label.name, label.value) for label in read_labels(db)] == [
            ('protected', SCAN_ERROR)
        ]
