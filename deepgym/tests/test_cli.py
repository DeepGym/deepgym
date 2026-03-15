"""Tests for deepgym.cli -- CLI argument parsing and basic execution."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from deepgym.cli import _build_parser, _read_file_or_string, main

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


class TestCLIHelp:
    """Verify that --help exits 0 for all subcommands."""

    @pytest.mark.parametrize(
        'args',
        [
            ['deepgym', '--help'],
            ['deepgym', 'run', '--help'],
            ['deepgym', 'serve', '--help'],
            ['deepgym', 'web', '--help'],
            ['deepgym', 'eval', '--help'],
            ['deepgym', 'create', '--help'],
            ['deepgym', 'run-batch', '--help'],
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

    def test_web_subcommand_parses(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(['web', '--port', '9090'])
        assert args.command == 'web'
        assert args.port == 9090


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
    """Verify the create subcommand prints environment JSON."""

    def test_create_prints_json(self, capsys: pytest.CaptureFixture) -> None:
        sorting_dir = ENVS_DIR / 'python_sorting'
        if not sorting_dir.exists():
            pytest.skip('envs/python_sorting not found')

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

        captured = capsys.readouterr()
        assert 'Environment created' in captured.out
