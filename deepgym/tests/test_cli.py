"""Tests for deepgym.cli -- CLI argument parsing and basic execution."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import deepgym.qa
from deepgym.cli import (
    _build_parser,
    _print_batch_result,
    _print_eval_result,
    _print_run_result,
    _read_file_or_string,
    main,
)
from deepgym.console.app import create_app
from deepgym.core import build_batch_result, build_eval_result
from deepgym.exceptions import SandboxError
from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import RunResult
from deepgym.qa import QACheck, QAReport
from deepgym.sandbox import error_result

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


class TestCLIHelp:
    """Verify that --help exits 0 for all subcommands."""

    @pytest.mark.parametrize(
        'args',
        [
            ['deepgym', '--help'],
            ['deepgym', 'audit', '--help'],
            ['deepgym', 'benchmark-audit', '--help'],
            ['deepgym', 'run', '--help'],
            ['deepgym', 'serve', '--help'],
            ['deepgym', 'console', '--help'],
            ['deepgym', 'web', '--help'],
            ['deepgym', 'eval', '--help'],
            ['deepgym', 'create', '--help'],
            ['deepgym', 'run-batch', '--help'],
            ['deepgym', 'qa', '--help'],
            ['deepgym', 'harbor', 'import', '--help'],
            ['deepgym', 'harbor', 'export', '--help'],
        ],
    )
    def test_help_exits_zero(self, args: list[str], capsys: pytest.CaptureFixture) -> None:
        with patch.object(sys, 'argv', args), pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'deepgym' in captured.out.lower() or 'usage' in captured.out.lower()


class TestBuildParser:
    """Verify argument parser construction."""

    def test_parser_requires_subcommand(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_run_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                'run',
                '--task',
                'test',
                '--verifier',
                'v.py',
                '--solution',
                's.py',
            ]
        )
        assert args.command == 'run'
        assert args.task == 'test'
        assert args.verifier == 'v.py'
        assert args.solution == 's.py'
        assert args.timeout == 30  # default

    def test_audit_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['audit', '--task', 't', '--verifier', 'v.py', '--persist'])
        assert args.command == 'audit'
        assert args.persist is True

    def test_benchmark_audit_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['benchmark-audit', '--env-dir', '/tmp', '--seed', '7'])
        assert args.command == 'benchmark-audit'
        assert args.seed == 7

    def test_serve_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['serve', '--host', '0.0.0.0', '--port', '9000'])
        assert args.command == 'serve'
        assert args.host == '0.0.0.0'
        assert args.port == 9000

    def test_eval_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['eval', '--suite', 'easy', '--solutions-dir', '/tmp'])
        assert args.command == 'eval'
        assert args.suite == 'easy'

    def test_create_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                'create',
                '--name',
                'test_env',
                '--task',
                'task',
                '--verifier',
                'v.py',
            ]
        )
        assert args.command == 'create'
        assert args.name == 'test_env'
        assert args.difficulty == 'medium'  # default

    def test_run_batch_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                'run-batch',
                '--task',
                't',
                '--verifier',
                'v.py',
                '--solutions-dir',
                '/tmp',
            ]
        )
        assert args.command == 'run-batch'
        assert args.max_parallel == 10  # default

    @pytest.mark.parametrize('command', ['console', 'web'])
    def test_console_subcommand_parses(self, command: str) -> None:
        parser = _build_parser()
        args = parser.parse_args([command, '--port', '9090'])
        assert args.command == command
        assert args.port == 9090
        assert args.allow_local_exec is False


class TestConsoleCommand:
    """Verify `deepgym console` (and its `web` alias) builds the app and serves it."""

    @pytest.mark.parametrize('command', ['console', 'web'])
    def test_runs_uvicorn_with_console_app(
        self, command: str, capsys: pytest.CaptureFixture
    ) -> None:
        argv = ['deepgym', command, '--port', '9191', '--allow-local-exec']
        with (
            patch.object(sys, 'argv', argv),
            patch('deepgym.console.app.create_app') as create_app,
            patch('uvicorn.run') as run,
        ):
            main()
        create_app.assert_called_once_with(
            allow_local_exec=True, allowed_hosts=['127.0.0.1', '[::1]', 'localhost']
        )
        run.assert_called_once_with(create_app.return_value, host='127.0.0.1', port=9191)
        assert 'http://127.0.0.1:9191' in capsys.readouterr().out

    @pytest.mark.parametrize(
        ('host', 'netloc', 'browser'),
        [
            ('localhost', 'localhost', 'localhost'),
            ('::1', '[::1]', '[::1]'),
            ('127.0.0.2', '127.0.0.2', '127.0.0.2'),
            ('127.2', '127.2', '127.0.0.2'),
            ('::ffff:127.0.0.1', '[::ffff:127.0.0.1]', '[::ffff:7f00:1]'),
        ],
    )
    def test_guards_the_host_header_on_every_loopback_address(
        self, host: str, netloc: str, browser: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The served HTML holds the launch token, so on any loopback bind a page on another
        name (DNS rebinding) must be refused, while the printed URL still works, as typed
        and as a browser normalizes it."""

        def isolated(**kwargs: object) -> object:
            return create_app(tmp_path / 'c.db', tmp_path / 'qa', **kwargs)

        with (
            patch.object(sys, 'argv', ['deepgym', 'console', '--host', host, '--port', '9191']),
            patch('deepgym.console.app.create_app', side_effect=isolated),
            patch('uvicorn.run') as run,
        ):
            main()
        app = run.call_args.args[0]
        assert TestClient(app, base_url='http://evil.example:9191').get('/').status_code == 400
        for name in (netloc, browser):
            assert TestClient(app, base_url=f'http://{name}:9191').get('/').status_code != 400
        assert f'http://{netloc}:9191' in capsys.readouterr().out

    def test_warns_when_ui_is_not_built(self, capsys: pytest.CaptureFixture) -> None:
        with (
            patch.object(sys, 'argv', ['deepgym', 'console', '--host', '0.0.0.0']),
            patch('deepgym.console.app.ui_built', return_value=False),
            patch('deepgym.console.app.create_app') as create_app,
            patch('uvicorn.run'),
        ):
            main()
        assert 'bun run build' in capsys.readouterr().err
        assert create_app.call_args.kwargs['allowed_hosts'] is None


class TestReadFileOrString:
    """Verify _read_file_or_string helper."""

    def test_returns_file_contents_when_path_exists(self, tmp_path: Path) -> None:
        f = tmp_path / 'test.txt'
        f.write_text('file content here', encoding='utf-8')
        assert _read_file_or_string(str(f)) == 'file content here'

    def test_returns_string_when_not_a_path(self) -> None:
        result = _read_file_or_string('just a plain string')
        assert result == 'just a plain string'


class TestCLIRunSubcommand:
    """Verify the run subcommand executes with example files."""

    def test_run_with_example_files(self, capsys: pytest.CaptureFixture) -> None:
        sorting_dir = ENVS_DIR / 'python_sorting'
        if not sorting_dir.exists():
            pytest.skip('envs/python_sorting not found')

        args = [
            'deepgym',
            'run',
            '--task',
            str(sorting_dir / 'task.md'),
            '--verifier',
            str(sorting_dir / 'verifier.py'),
            '--solution',
            str(sorting_dir / 'reference_solution.py'),
        ]
        with patch.object(sys, 'argv', args):
            main()

        captured = capsys.readouterr()
        assert 'PASS' in captured.out


class TestCLICreateSubcommand:
    """Verify the create subcommand prints the named environment and saves nothing."""

    def test_create_prints_the_named_environment_and_says_it_is_not_saved(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        sorting_dir = ENVS_DIR / 'python_sorting'
        args = [
            'deepgym',
            'create',
            '--name',
            'test_env',
            '--task',
            str(sorting_dir / 'task.md'),
            '--verifier',
            str(sorting_dir / 'verifier.py'),
        ]
        with patch.object(sys, 'argv', args):
            main()

        out, err = capsys.readouterr()
        env = json.loads(out)
        assert env['name'] == 'test_env'
        assert env['verifier_code'] == (sorting_dir / 'verifier.py').read_text(encoding='utf-8')
        assert 'Not saved' in err


class TestCLIAuditSubcommand:
    """Verify the verifier audit and benchmark audit CLI flows."""

    def test_audit_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        task_path = tmp_path / 'task.md'
        verifier_path = tmp_path / 'verifier.py'
        task_path.write_text('Do anything', encoding='utf-8')
        verifier_path.write_text(
            'import sys, json\n'
            'if __name__ == "__main__":\n'
            '    print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,'
            '"details":None,"truncated":False}))\n'
            '    sys.exit(0)\n',
            encoding='utf-8',
        )

        args = [
            'deepgym',
            'audit',
            '--task',
            str(task_path),
            '--verifier',
            str(verifier_path),
            '--verifier-id',
            'weak-cli',
            '--strategies',
            'empty',
            '--json',
        ]
        with patch.object(sys, 'argv', args):
            main()

        data = json.loads(capsys.readouterr().out)
        assert data['verifier_id'] == 'weak-cli'
        assert data['exploitable'] is True

    def test_benchmark_audit_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        env_a = tmp_path / 'env_a'
        env_b = tmp_path / 'env_b'
        env_a.mkdir()
        env_b.mkdir()
        for env_dir in (env_a, env_b):
            (env_dir / 'task.md').write_text('Shared task', encoding='utf-8')
            (env_dir / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')

        args = [
            'deepgym',
            'benchmark-audit',
            '--env-dir',
            str(tmp_path),
            '--split',
            'env_a=public_train',
            '--split',
            'env_b=private_holdout',
            '--json',
        ]
        with patch.object(sys, 'argv', args):
            main()

        data = json.loads(capsys.readouterr().out)
        assert data['contamination_risk'] is True
        assert len(data['leaks']) >= 1


class TestCLIQASubcommand:
    """Verify `deepgym qa` writes reports and sets the exit code."""

    def test_qa_json_output_and_report_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path))
        args = ['deepgym', 'qa', 'reverse_string', 'signup_form', '--reruns', '1', '--json']

        with patch.object(sys, 'argv', args):
            main()

        data = json.loads(capsys.readouterr().out)
        assert [(r['environment'], r['status'], r['passed']) for r in data] == [
            ('reverse_string', 'pass', True),
            ('signup_form', 'skip', False),
        ]
        saved = json.loads((tmp_path / 'reverse_string.json').read_text(encoding='utf-8'))
        assert saved['checks'] == data[0]['checks']

    @pytest.mark.parametrize(
        ('envs', 'status'),
        [(['signup_form'], 'skip'), (['reverse_string', 'odd'], 'inconclusive')],
    )
    def test_qa_exits_two_without_a_verdict(
        self,
        envs: list[str],
        status: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Nothing failed, but nothing (skip) or not everything (inconclusive) was verified."""
        monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path))
        run_qa = deepgym.qa.run_qa

        def fake_run_qa(name: str, *args: object, **kwargs: object) -> QAReport:
            if name != 'odd':
                return run_qa(name, *args, **kwargs)
            check = QACheck(name='hack_resistant', status='inconclusive', details='crashed')
            return QAReport(environment=name, checks=[check])

        monkeypatch.setattr('deepgym.qa.run_qa', fake_run_qa)

        with (
            patch.object(sys, 'argv', ['deepgym', 'qa', *envs, '--reruns', '0', '--json']),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 2
        assert json.loads(capsys.readouterr().out)[-1]['status'] == status

    def test_qa_exits_one_when_an_env_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path))

        with (
            patch.object(sys, 'argv', ['deepgym', 'qa', 'no_such_env']),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        assert 'no_such_env: load:' in capsys.readouterr().out
        assert (tmp_path / 'no_such_env.json').exists()

    def test_qa_requires_an_env(self) -> None:
        with patch.object(sys, 'argv', ['deepgym', 'qa']), pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1


class TestCLIQALocalExecGuard:
    """`deepgym qa --model` refuses to run model code locally without --allow-local-exec."""

    @pytest.mark.parametrize('mode', ['local', 'auto'])
    def test_model_in_local_mode_needs_the_flag(
        self, mode: str, capsys: pytest.CaptureFixture
    ) -> None:
        args = ['deepgym', 'qa', 'reverse_string', '--model', 'm', '--mode', mode]

        with patch.object(sys, 'argv', args), pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert '--allow-local-exec' in err
        assert '--mode docker' in err

    @pytest.mark.parametrize('isolation', ['bwrap', 'none'])
    def test_model_with_the_flag_proceeds(
        self,
        isolation: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            answer = '```python\ndef reverse_string(s):\n    return s[::-1]\n```'
            return httpx.Response(200, json={'choices': [{'message': {'content': answer}}]})

        client = ZaiChatClient('key', model='m', transport=httpx.MockTransport(handler))
        monkeypatch.setattr('deepgym.qa.client_from_env', lambda model: client)
        monkeypatch.setattr('deepgym.cli.local_isolation_kind', lambda: isolation)
        monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path))
        args = ['deepgym', 'qa', 'reverse_string', '--model', 'm', '--samples', '2']
        args += ['--reruns', '0', '--allow-local-exec']

        with patch.object(sys, 'argv', args):
            main()  # every sample solved: reported as too_easy, never a failed gate

        captured = capsys.readouterr()
        assert 'difficulty: too_easy, no_variance: pass@1=1.00 pass@2=1.00' in captured.out
        assert ('WARNING: no sandbox-exec or bwrap' in captured.err) is (isolation == 'none')


_OK = RunResult(
    score=1.0, passed=True, output='', stderr='', exit_code=0, execution_time_ms=1.0,
    sandbox_id='local',
)  # fmt: skip
_FAILED = _OK.model_copy(update={'score': 0.5, 'passed': False, 'exit_code': 1})


class TestResultOutput:
    """Runs without a verdict print as errors with their cause; totals count them as 0
    and show the judged figures and the share without a verdict beside them."""

    def test_run_without_a_verdict_is_an_error(self, capsys: pytest.CaptureFixture) -> None:
        _print_run_result(error_result(SandboxError('sandbox unreachable')), 1.0)
        assert 'Result:  ERROR (sandbox_error)' in capsys.readouterr().out

    def test_batch_shows_judged_and_no_verdict_beside_the_totals(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        outage = error_result(SandboxError('sandbox unreachable'))
        batch = build_batch_result([_OK, _FAILED, outage], 1.0)

        _print_batch_result(batch, [Path('a.py'), Path('b.py'), Path('c.py')], 1.0)

        out = capsys.readouterr().out
        assert 'Failed:  2' in out
        assert 'Avg:     0.5000  (judged 0.7500, 33.33% no verdict)' in out
        assert '[ERROR (sandbox_error)] c.py' in out and 'sandbox unreachable' in out

    def test_eval_shows_judged_and_no_verdict_beside_the_totals(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        outcomes = [_OK, _FAILED, SandboxError('sandbox down')]
        result = build_eval_result('suite', 'model', ['a', 'b', 'c'], outcomes)

        _print_eval_result(result, 1.0)

        out = capsys.readouterr().out
        assert 'Pass Rate:   33.33%  (judged 50.00%, 33.33% no verdict)' in out
        assert 'Avg Score:   0.5000  (judged 0.7500)' in out
        assert '[ERROR (sandbox_error)] c: sandbox down' in out
