"""Command-line interface for DeepGym."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='deepgym',
        description='DeepGym — managed RL training and evaluation infrastructure',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── run ────────────────────────────────────────────────────────────
    run_p = sub.add_parser('run', help='Run a single episode')
    run_p.add_argument('--task', required=True, help='Task description / prompt')
    run_p.add_argument('--verifier', required=True, help='Path to verifier Python file')
    run_p.add_argument('--solution', required=True, help='Path to solution Python file')
    run_p.add_argument('--timeout', type=int, default=30, help='Execution timeout in seconds')
    run_p.add_argument('--snapshot', default=None, help='Pre-built sandbox snapshot name')

    # ── run-batch ─────────────────────────────────────────────────────
    batch_p = sub.add_parser('run-batch', help='Run a batch of solutions')
    batch_p.add_argument('--task', required=True, help='Task description / prompt')
    batch_p.add_argument('--verifier', required=True, help='Path to verifier Python file')
    batch_p.add_argument(
        '--solutions-dir', required=True, help='Directory containing solution files'
    )
    batch_p.add_argument('--timeout', type=int, default=30, help='Execution timeout in seconds')
    batch_p.add_argument('--max-parallel', type=int, default=10, help='Max concurrent runs')
    batch_p.add_argument('--snapshot', default=None, help='Pre-built sandbox snapshot name')

    # ── eval ──────────────────────────────────────────────────────────
    eval_p = sub.add_parser('eval', help='Run an evaluation suite')
    eval_p.add_argument(
        '--suite',
        required=True,
        help=(
            'Evaluation suite name: easy, medium, hard (difficulty), '
            'coding, computer-use, tool-use (type), all, '
            'or a family name like dynamic-programming'
        ),
    )
    eval_p.add_argument(
        '--solutions-dir', required=True, help='Directory containing solution files'
    )
    eval_p.add_argument('--max-parallel', type=int, default=100, help='Max concurrent runs')

    # ── create ────────────────────────────────────────────────────────
    create_p = sub.add_parser('create', help='Create/register an environment')
    create_p.add_argument('--name', required=True, help='Environment name')
    create_p.add_argument('--task', required=True, help='Task description / prompt')
    create_p.add_argument('--verifier', required=True, help='Path to verifier Python file')
    create_p.add_argument('--difficulty', default='medium', choices=['easy', 'medium', 'hard'])
    create_p.add_argument('--domain', default='coding', help='Task domain')
    create_p.add_argument('--tags', nargs='*', default=[], help='Freeform tags')

    # ── serve ─────────────────────────────────────────────────────────
    serve_p = sub.add_parser('serve', help='Start the DeepGym API server')
    serve_p.add_argument('--host', default='127.0.0.1', help='Bind address')
    serve_p.add_argument('--port', type=int, default=8000, help='Port number')
    serve_p.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    serve_p.add_argument(
        '--allow-local-exec',
        action='store_true',
        default=False,
        help='Allow local code execution without Daytona sandboxes (dangerous)',
    )
    serve_p.add_argument(
        '--no-auth',
        action='store_true',
        default=False,
        help='Disable API key authentication (development only)',
    )

    # ── web ───────────────────────────────────────────────────────────
    web_p = sub.add_parser('web', help='Launch web debugging UI')
    web_p.add_argument('--host', default='127.0.0.1', help='Bind address')
    web_p.add_argument('--port', type=int, default=8080, help='Port number')
    web_p.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    web_p.add_argument(
        '--allow-local-exec',
        action='store_true',
        default=False,
        help='Allow local code execution without Daytona sandboxes (dangerous)',
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _read_file_or_string(value: str) -> str:
    """Read value as a file path if it exists, otherwise return as-is.

    Args:
        value: A file path or literal string.

    Returns:
        File contents if the path exists, otherwise the original string.
    """
    path = Path(value)
    if path.exists():
        return path.read_text(encoding='utf-8')
    return value


def _cmd_run(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.models import Environment

    task_value = _read_file_or_string(args.task)
    verifier_code = _read_file_or_string(args.verifier)
    solution_code = _read_file_or_string(args.solution)

    env = Environment(
        task=task_value,
        verifier_code=verifier_code,
        difficulty='medium',
        timeout=args.timeout,
        snapshot=args.snapshot,
    )

    dg = DeepGym()
    start = time.perf_counter()
    result = dg.run(env, solution_code)
    elapsed = time.perf_counter() - start

    _print_run_result(result, elapsed)


def _cmd_run_batch(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.models import Environment

    verifier_code = _read_file_or_string(args.verifier)
    solutions_dir = Path(args.solutions_dir)

    if not solutions_dir.is_dir():
        print(f'Error: {solutions_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    solution_files = sorted(solutions_dir.glob('*.py'))
    if not solution_files:
        print(f'Error: no .py files found in {solutions_dir}', file=sys.stderr)
        sys.exit(1)

    outputs = [f.read_text(encoding='utf-8') for f in solution_files]

    task_value = _read_file_or_string(args.task)

    env = Environment(
        task=task_value,
        verifier_code=verifier_code,
        difficulty='medium',
        timeout=args.timeout,
        snapshot=args.snapshot,
    )

    dg = DeepGym()
    start = time.perf_counter()
    batch = dg.run_batch(env, outputs, max_parallel=args.max_parallel)
    elapsed = time.perf_counter() - start

    _print_batch_result(batch, solution_files, elapsed)


def _cmd_eval(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym

    solutions_dir = Path(args.solutions_dir)
    if not solutions_dir.is_dir():
        print(f'Error: {solutions_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    model_outputs: dict[str, str] = {}
    for f in sorted(solutions_dir.glob('*.py')):
        model_outputs[f.stem] = f.read_text(encoding='utf-8')

    if not model_outputs:
        print(f'Error: no .py files found in {solutions_dir}', file=sys.stderr)
        sys.exit(1)

    dg = DeepGym()
    start = time.perf_counter()
    result = dg.eval(args.suite, model_outputs, max_parallel=args.max_parallel)
    elapsed = time.perf_counter() - start

    _print_eval_result(result, elapsed)


def _cmd_create(args: argparse.Namespace) -> None:
    from deepgym.models import Environment

    task_value = _read_file_or_string(args.task)
    verifier_code = _read_file_or_string(args.verifier)

    env = Environment(
        task=task_value,
        verifier_code=verifier_code,
        difficulty=args.difficulty,
        domain=args.domain,
        tags=args.tags,
    )

    # Pretty-print the environment as JSON.
    print('Environment created:')
    print(json.dumps(env.model_dump(), indent=2, default=str))


def _cmd_web(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            'Error: uvicorn is required to run the web UI. '
            'Install it with: pip install uvicorn[standard]',
            file=sys.stderr,
        )
        sys.exit(1)

    # Propagate the allow-local-exec flag via environment variable so the
    # web app can read it at startup.
    allow_local = args.allow_local_exec or os.environ.get(
        'DEEPGYM_ALLOW_LOCAL_EXEC', ''
    ).lower() in ('true', '1', 'yes')
    os.environ['DEEPGYM_ALLOW_LOCAL_EXEC'] = str(allow_local).lower()

    from deepgym.web import create_web_app

    app = create_web_app()
    print(f'DeepGym Web UI: http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            'Error: uvicorn is required to run the server. '
            'Install it with: pip install uvicorn[standard]',
            file=sys.stderr,
        )
        sys.exit(1)

    # Propagate the allow-local-exec flag via environment variable so the
    # FastAPI app can read it at startup.
    allow_local = args.allow_local_exec or os.environ.get(
        'DEEPGYM_ALLOW_LOCAL_EXEC', ''
    ).lower() in ('true', '1', 'yes')
    os.environ['DEEPGYM_ALLOW_LOCAL_EXEC'] = str(allow_local).lower()

    # Propagate the no-auth flag via environment variable.
    no_auth = args.no_auth or os.environ.get('DEEPGYM_NO_AUTH', '').lower() in (
        'true',
        '1',
        'yes',
    )
    if no_auth:
        os.environ['DEEPGYM_NO_AUTH'] = 'true'

    uvicorn.run(
        'deepgym.api.app:app',
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _print_run_result(result, elapsed: float) -> None:
    status = 'PASS' if result.passed else 'FAIL'
    print(f'\n{"=" * 50}')
    print(f'  Result:  {status}')
    print(f'  Score:   {result.score:.4f}')
    print(f'  Time:    {elapsed:.2f}s')
    print(f'{"=" * 50}')
    if result.output:
        print(f'\n  Output:\n{_indent(result.output)}')
    if result.stderr:
        print(f'\n  Stderr:\n{_indent(result.stderr)}')


def _print_batch_result(batch, solution_files, elapsed: float) -> None:
    print(f'\n{"=" * 50}')
    print('  Batch Results')
    print(f'  Total:   {batch.total}')
    print(f'  Passed:  {batch.passed}')
    print(f'  Failed:  {batch.failed}')
    print(f'  Avg:     {batch.avg_score:.4f}')
    print(f'  Time:    {elapsed:.2f}s')
    print(f'{"=" * 50}')

    for i, result in enumerate(batch.results):
        name = solution_files[i].name if i < len(solution_files) else f'solution-{i}'
        status = 'PASS' if result.passed else 'FAIL'
        print(f'  [{status}] {name:30s}  score={result.score:.4f}')


def _print_eval_result(result, elapsed: float) -> None:
    print(f'\n{"=" * 50}')
    print(f'  Eval Suite:  {result.suite}')
    print(f'  Model:       {result.model_name}')
    print(f'  Pass Rate:   {result.pass_rate:.2%}')
    print(f'  Total:       {result.total}')
    print(f'  Passed:      {result.passed}')
    print(f'  Avg Score:   {result.avg_score:.4f}')
    print(f'  Time:        {elapsed:.2f}s')
    print(f'{"=" * 50}')


def _indent(text: str, prefix: str = '    ') -> str:
    return '\n'.join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    'run': _cmd_run,
    'run-batch': _cmd_run_batch,
    'eval': _cmd_eval,
    'create': _cmd_create,
    'serve': _cmd_serve,
    'web': _cmd_web,
}


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    try:
        handler(args)
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
