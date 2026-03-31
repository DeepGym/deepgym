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

    # ── audit ─────────────────────────────────────────────────────────
    audit_p = sub.add_parser('audit', help='Audit a verifier for reward hacking risk')
    audit_p.add_argument('--task', required=True, help='Task description / prompt')
    audit_p.add_argument('--verifier', required=True, help='Path to verifier Python file')
    audit_p.add_argument('--benchmark', default='custom', help='Benchmark label for the audit')
    audit_p.add_argument('--verifier-id', default=None, help='Optional stable verifier identifier')
    audit_p.add_argument(
        '--strategies',
        nargs='*',
        default=None,
        help='Subset of adversarial strategies to run',
    )
    audit_p.add_argument(
        '--persist',
        action='store_true',
        help='Persist the audit result to the exploit database',
    )
    audit_p.add_argument('--db-path', default=None, help='Optional SQLite path for persisted audits')
    audit_p.add_argument('--json', action='store_true', help='Print machine-readable JSON only')

    # ── benchmark-audit ───────────────────────────────────────────────
    bench_p = sub.add_parser('benchmark-audit', help='Audit benchmark split leakage and holdouts')
    bench_p.add_argument('--env-dir', required=True, help='Root directory containing task/verifier envs')
    bench_p.add_argument('--benchmark', default=None, help='Benchmark label (defaults to dir name)')
    bench_p.add_argument('--seed', type=int, default=0, help='Deterministic split seed')
    bench_p.add_argument(
        '--public-eval-ratio',
        type=float,
        default=0.2,
        help='Fraction of environments assigned to public eval',
    )
    bench_p.add_argument(
        '--holdout-ratio',
        type=float,
        default=0.1,
        help='Fraction of environments assigned to private holdout',
    )
    bench_p.add_argument(
        '--canary-ratio',
        type=float,
        default=0.05,
        help='Fraction of environments assigned to canary',
    )
    bench_p.add_argument(
        '--split',
        action='append',
        default=[],
        help='Explicit split override in the form env_id=public_train|public_eval|private_holdout|canary',
    )
    bench_p.add_argument('--json', action='store_true', help='Print machine-readable JSON only')

    # ── generate-prm ───────────────────────────────────────────────────
    prm_p = sub.add_parser(
        'generate-prm',
        help='Generate PRM stepwise-supervision dataset from DeepGym environments',
    )
    prm_p.add_argument(
        '--env', required=True, help='Built-in environment name (e.g. coin_change)'
    )
    prm_p.add_argument(
        '--solutions-dir',
        required=True,
        help='Directory containing .py solution files to score',
    )
    prm_p.add_argument(
        '-o', '--output', required=True, help='Output path for .jsonl dataset file'
    )
    prm_p.add_argument(
        '--step-separator', default='\n\n', help='Step separator for PRM records'
    )
    prm_p.add_argument(
        '--max-parallel', type=int, default=32, help='Max concurrent sandbox runs'
    )
    prm_p.add_argument(
        '--include-metadata', action='store_true', help='Include solution metadata in output'
    )
    prm_p.add_argument(
        '--axolotl-config',
        default=None,
        help='Also generate an Axolotl PRM YAML config at this path',
    )
    prm_p.add_argument(
        '--base-model',
        default='Qwen/Qwen3.5-Coder-7B',
        help='Base model for generated Axolotl config',
    )

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


def _cmd_audit(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.models import Environment
    from deepgym.reward_qa import RewardAuditor

    task_value = _read_file_or_string(args.task)
    verifier_code = _read_file_or_string(args.verifier)
    env = Environment(task=task_value, verifier_code=verifier_code)

    auditor = RewardAuditor(DeepGym(mode='local'))
    report = auditor.audit(
        env,
        verifier_id=args.verifier_id or '',
        benchmark=args.benchmark,
        strategies=args.strategies,
        persist=args.persist,
        db_path=Path(args.db_path) if args.db_path else None,
    )

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return
    _print_verifier_audit(report)


def _cmd_benchmark_audit(args: argparse.Namespace) -> None:
    from deepgym.benchmark_ops import build_benchmark_audit, load_environments_from_dir

    root = Path(args.env_dir)
    if not root.is_dir():
        print(f'Error: {root} is not a directory', file=sys.stderr)
        sys.exit(1)

    environments = load_environments_from_dir(root)
    if not environments:
        print(f'Error: no environments found under {root}', file=sys.stderr)
        sys.exit(1)

    report = build_benchmark_audit(
        environments,
        benchmark=args.benchmark or root.name,
        provenance='filesystem',
        seed=args.seed,
        public_eval_ratio=args.public_eval_ratio,
        holdout_ratio=args.holdout_ratio,
        canary_ratio=args.canary_ratio,
        split_overrides=_parse_split_overrides(args.split),
    )

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return
    _print_benchmark_audit(report)


def _cmd_generate_prm(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.integrations.axolotl import (
        generate_axolotl_config,
        generate_prm_dataset,
        write_prm_dataset,
    )
    from deepgym.registry import load_environment

    env = load_environment(args.env)

    solutions_dir = Path(args.solutions_dir)
    if not solutions_dir.is_dir():
        print(f'Error: {solutions_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    solution_files = sorted(solutions_dir.glob('*.py'))
    if not solution_files:
        print(f'Error: no .py files found in {solutions_dir}', file=sys.stderr)
        sys.exit(1)

    solutions = [f.read_text(encoding='utf-8') for f in solution_files]
    print(f'Loaded {len(solutions)} solutions from {solutions_dir}')

    dg = DeepGym()
    start = time.perf_counter()
    records = generate_prm_dataset(
        env=env,
        solutions=solutions,
        dg=dg,
        max_parallel=args.max_parallel,
        step_separator=args.step_separator,
    )
    elapsed = time.perf_counter() - start

    output_path = Path(args.output)
    written = write_prm_dataset(
        records, output_path, include_metadata=args.include_metadata
    )

    print(f'\n{"=" * 50}')
    print('  PRM Dataset Generated')
    print(f'  Solutions scored:  {len(solutions)}')
    print(f'  Records written:   {written}')
    print(f'  Skipped:           {len(solutions) - written}')
    print(f'  Output:            {output_path}')
    print(f'  Time:              {elapsed:.2f}s')
    print(f'{"=" * 50}')

    if args.axolotl_config:
        config_path = Path(args.axolotl_config)
        config = generate_axolotl_config(
            base_model=args.base_model,
            method='prm',
            dataset_path=str(output_path),
            step_separator=args.step_separator.encode('unicode_escape').decode('ascii'),
            config_filename=config_path.name,
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config, encoding='utf-8')
        print(f'\n  Axolotl config:    {config_path}')
        print(f'  Train with:        axolotl train {config_path}')


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


def _print_verifier_audit(report) -> None:
    print(f'\n{"=" * 50}')
    print(f'  Verifier Audit: {report.verifier_id}')
    print(f'  Benchmark:      {report.benchmark}')
    print(f'  Risk:           {report.risk_level.upper()} ({report.risk_score:.2f})')
    print(f'  Exploitable:    {report.exploitable}')
    print(f'  Attacks:        {report.exploits_found}/{report.attacks_run}')
    print(f'  Stored:         {report.stored}')
    print(f'{"=" * 50}')
    if report.patterns:
        print('\n  Patterns:')
        for pattern in report.patterns:
            print(f'    - {pattern}')
    if report.recommendations:
        print('\n  Recommendations:')
        for recommendation in report.recommendations:
            print(f'    - {recommendation}')


def _print_benchmark_audit(report) -> None:
    print(f'\n{"=" * 60}')
    print(f'  Benchmark Audit: {report.benchmark}')
    print(f'  Total envs:      {report.total_environments}')
    print(f'  Contamination:   {report.contamination_risk}')
    print(f'{"=" * 60}')
    print('  Split counts:')
    for split, count in report.split_counts.items():
        print(f'    - {split}: {count}')
    print(f'  Duplicate task groups: {len(report.duplicate_task_groups)}')
    print(f'  Duplicate verifier groups: {len(report.duplicate_verifier_groups)}')
    print(f'  Leak findings: {len(report.leaks)}')
    if report.recommendations:
        print('\n  Recommendations:')
        for recommendation in report.recommendations:
            print(f'    - {recommendation}')


def _indent(text: str, prefix: str = '    ') -> str:
    return '\n'.join(prefix + line for line in text.splitlines())


def _parse_split_overrides(values: list[str]) -> dict[str, str]:
    allowed = {'public_train', 'public_eval', 'private_holdout', 'canary'}
    overrides: dict[str, str] = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'Invalid --split value {value!r}; expected env_id=split')
        env_id, split = value.split('=', 1)
        split = split.strip()
        if split not in allowed:
            raise ValueError(f'Invalid split {split!r}; expected one of {sorted(allowed)}')
        overrides[env_id.strip()] = split
    return overrides


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMANDS = {
    'audit': _cmd_audit,
    'benchmark-audit': _cmd_benchmark_audit,
    'generate-prm': _cmd_generate_prm,
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
