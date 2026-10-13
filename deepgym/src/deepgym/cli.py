"""Command-line interface for DeepGym."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from deepgym.factory.scaffold import KINDS
from deepgym.judge import local_isolation_kind
from deepgym.sandboxes import PROVIDERS

if TYPE_CHECKING:
    from deepgym.qa.report import QAStatus


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
    create_p = sub.add_parser(
        'create', help='Print an environment as JSON, ready to POST to /v1/environments'
    )
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
    audit_p.add_argument(
        '--db-path',
        default=None,
        help='Optional SQLite path for persisted audits',
    )
    audit_p.add_argument('--json', action='store_true', help='Print machine-readable JSON only')

    # ── qa ────────────────────────────────────────────────────────────
    qa_p = sub.add_parser(
        'qa',
        help='Quality-gate environments before training on them',
        description='Exits 0 when the checked environments pass (skipped ones are neutral), '
        '1 if any fails, 2 if none fails but some are inconclusive or none could be checked.',
    )
    qa_p.add_argument(
        'envs',
        nargs='*',
        metavar='ENV',
        help='Registered environment names, or environment folders (any existing directory)',
    )
    qa_p.add_argument('--all', action='store_true', help='Check every registered environment')
    qa_p.add_argument(
        '--mode',
        default='local',
        choices=['local', 'daytona', 'auto', *PROVIDERS],
        help='Where verifiers and solutions run (default: local)',
    )
    qa_p.add_argument(
        '--reruns', type=int, default=3, help='Extra reference runs for the determinism check'
    )
    qa_p.add_argument(
        '--model',
        default=None,
        help='Probe difficulty with this model via OPENAI_BASE_URL / OPENAI_API_KEY',
    )
    qa_p.add_argument(
        '--samples', type=int, default=8, help='Completions per environment for --model'
    )
    qa_p.add_argument(
        '--allow-local-exec',
        action='store_true',
        help='Let --model run model-generated code on this machine (--mode local or auto)',
    )
    qa_p.add_argument('--json', action='store_true', help='Print the reports as JSON')
    qa_p.add_argument(
        '--mutants',
        action='store_true',
        help='Also run AST mutants of the reference; any mutant that passes fails the env',
    )
    qa_p.add_argument(
        '--approve',
        action='store_true',
        help='If one environment folder passes, write its approval.json (needs --by)',
    )
    qa_p.add_argument('--by', default=None, metavar='NAME', help='Who approves, for --approve')

    # ── new ───────────────────────────────────────────────────────────
    new_p = sub.add_parser(
        'new',
        help='Scaffold a new environment folder',
        description='Create a working example environment to edit: task.md, verifier.py, '
        'a reference solution, a seeded case generator (variants.py) and metadata.json.',
    )
    new_p.add_argument('name', help='Environment name, lowercase snake_case')
    new_p.add_argument(
        '--kind',
        required=True,
        choices=KINDS,
        help='function: call a function; script: run a stdin/stdout script; repo-patch: '
        'apply a diff to fixture files; multi-turn: score an agent episode',
    )
    new_p.add_argument('--dir', default='.', help='Parent directory (default: current directory)')

    # ── check ─────────────────────────────────────────────────────────
    check_p = sub.add_parser(
        'check',
        help='Validate an environment folder and run its reference, empty and stub solutions',
        description='Exits 0 when every check passes, 1 if any fails, 2 if some are inconclusive.',
    )
    check_p.add_argument('path', help='Environment folder')
    check_p.add_argument(
        '--mode',
        default='local',
        choices=['local', 'daytona', 'auto', *PROVIDERS],
        help='Where verifiers and solutions run (default: local)',
    )

    # ── harbor ────────────────────────────────────────────────────────
    harbor_p = sub.add_parser('harbor', help='Import Harbor tasks, or export envs as Harbor tasks')
    harbor_sub = harbor_p.add_subparsers(dest='harbor_command', required=True)
    import_p = harbor_sub.add_parser('import', help='List and validate Harbor tasks')
    import_p.add_argument(
        'source', help='Task or dataset folder, git URL, or Hugging Face dataset (org/name)'
    )
    import_p.add_argument('--limit', type=int, default=None, help='Read at most N tasks')
    import_p.add_argument(
        '--register', metavar='NAME', default=None, help='Make valid tasks loadable as NAME/TASK'
    )
    harbor_export_p = harbor_sub.add_parser(
        'export', help='Write environments as Harbor task folders'
    )
    harbor_export_p.add_argument(
        'envs', nargs='+', metavar='ENV', help='Registered environment names'
    )
    harbor_export_p.add_argument(
        '--out', required=True, help='Folder to write the task folders into'
    )

    # ── benchmark-audit ───────────────────────────────────────────────
    bench_p = sub.add_parser(
        'benchmark-audit',
        help='Audit benchmark split leakage and holdouts',
    )
    bench_p.add_argument(
        '--env-dir',
        required=True,
        help='Root directory containing task/verifier envs',
    )
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
        help=(
            'Explicit split override in the form '
            'env_id=public_train|public_eval|private_holdout|canary'
        ),
    )
    bench_p.add_argument('--json', action='store_true', help='Print machine-readable JSON only')

    # ── traces ────────────────────────────────────────────────────────
    traces_p = sub.add_parser('traces', help='Export recorded agent traces as training data')
    traces_sub = traces_p.add_subparsers(dest='traces_command', required=True)
    export_p = traces_sub.add_parser(
        'export',
        help='Write traces as JSONL, Parquet or ATIF',
        description='Reads the console database and the traces directory unless --db or '
        '--traces names the sources. Traces without a verdict are exported flagged '
        'has_verdict=false, or left out with --only-valid.',
    )
    export_p.add_argument('--format', required=True, choices=['jsonl', 'parquet', 'atif'])
    export_p.add_argument('--out', required=True, help='Output file (a directory for atif)')
    export_p.add_argument(
        '--rows',
        default='trace',
        choices=['trace', 'rl'],
        help="'trace': whole traces (default); 'rl': one SFT/RL conversation per trace",
    )
    export_p.add_argument(
        '--only-valid', action='store_true', help='Leave out traces without a verdict'
    )
    export_p.add_argument('--db', default=None, help='Console database (~/.deepgym/console.db)')
    export_p.add_argument(
        '--traces',
        nargs='+',
        default=None,
        metavar='PATH',
        help='Trace files or directories (~/.deepgym/traces or $DEEPGYM_TRACES_DIR)',
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

    # ── console ───────────────────────────────────────────────────────
    console_p = sub.add_parser(
        'console', aliases=['web'], help='Launch the management console (web UI + API)'
    )
    console_p.add_argument('--host', default='127.0.0.1', help='Bind address')
    console_p.add_argument('--port', type=int, default=8080, help='Port number')
    console_p.add_argument(
        '--allow-local-exec',
        action='store_true',
        default=False,
        help="Allow runs that execute solutions on this machine (mode 'local')",
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

    env = Environment(
        name=args.name,
        task=_read_file_or_string(args.task),
        verifier_code=_read_file_or_string(args.verifier),
        difficulty=args.difficulty,
        domain=args.domain,
        tags=args.tags,
    )
    print(json.dumps(env.model_dump(mode='json'), indent=2))
    print(
        'Not saved anywhere: POST it to /v1/environments to register it on a server.',
        file=sys.stderr,
    )


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


def _cmd_qa(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.exceptions import DeepGymError
    from deepgym.factory.approval import approve, content_hashes, missing_evidence, snapshot
    from deepgym.factory.check import lint_env
    from deepgym.qa import QAOptions, client_from_env, format_table, run_qa, write_report
    from deepgym.qa.report import overall
    from deepgym.registry import entry_label, list_environments

    if args.all:
        targets: list[str | Path] = [entry_label(entry) for entry in list_environments()]
    else:
        targets = [Path(name) if Path(name).is_dir() else name for name in args.envs]
    if not targets:
        print('Error: name at least one environment or pass --all', file=sys.stderr)
        sys.exit(1)
    if args.reruns < 0 or args.samples < 2:
        print('Error: --reruns must be >= 0 and --samples >= 2', file=sys.stderr)
        sys.exit(1)
    if args.approve and not (args.by and len(targets) == 1 and isinstance(targets[0], Path)):
        print(
            'Error: --approve needs --by NAME and exactly one environment folder', file=sys.stderr
        )
        sys.exit(1)
    if args.approve and args.reruns < 1:
        print('Error: --approve needs --reruns 1 or more to check determinism', file=sys.stderr)
        sys.exit(1)
    if args.model and args.mode in ('local', 'auto'):
        _guard_local_model_code(args.mode, args.allow_local_exec)

    client = client_from_env(args.model) if args.model else None
    dg = DeepGym(mode=args.mode)
    options = QAOptions(
        reruns=args.reruns, client=client, samples=args.samples, mutants=args.mutants
    )
    with contextlib.ExitStack() as stack:
        folder = targets[0]
        if args.approve:
            # QA runs on a private copy, so the approval names exactly the files tested.
            targets = [stack.enter_context(snapshot(folder))]
            tested = content_hashes(targets[0])
        reports = []
        for target in targets:
            report = run_qa(target, dg, options)
            write_report(report)
            reports.append(report)
        unproven = missing_evidence(reports[0])
        approval = None
        if args.approve and reports[0].passed and not unproven:
            if content_hashes(targets[0]) != tested:
                raise DeepGymError('the copy QA ran on changed during QA; nothing approved')
            approval = approve(folder, reports[0], args.by, tested, lint_env(targets[0]))

    if args.json:
        print(json.dumps([report.model_dump(mode='json') for report in reports], indent=2))
    else:
        print(format_table(reports))
    if approval is not None:
        print(f'Approved: {approval}', file=sys.stderr)
    elif args.approve and reports[0].passed:
        print(f'Not approved: needed checks did not run: {", ".join(unproven)}', file=sys.stderr)
        sys.exit(1)
    elif args.approve:
        print(f'Not approved: QA result is {reports[0].status}', file=sys.stderr)
    _exit_for(overall(report.status for report in reports))


def _exit_for(status: QAStatus) -> None:
    """Exit 1 on 'fail', 2 on 'inconclusive' or 'skip' (not everything was verified)."""
    if status != 'pass':
        sys.exit(1 if status == 'fail' else 2)


def _cmd_new(args: argparse.Namespace) -> None:
    from deepgym.factory.scaffold import scaffold

    path = scaffold(args.name, args.kind, Path(args.dir))
    print(f'Created {path}:')
    for file in sorted(p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()):
        print(f'  {file}')
    print(
        '\nIt is a working example. Rewrite task.md, the reference, expected() in '
        f'verifier.py and variants.py for your task, then run:\n'
        f'  deepgym check {path}\n'
        f'  deepgym qa {path} --mutants'
    )


def _cmd_check(args: argparse.Namespace) -> None:
    from deepgym.core import DeepGym
    from deepgym.factory.check import check_env, lint_env
    from deepgym.qa.report import format_checks, overall
    from deepgym.verifier_lint import ADVISORY_NOTE

    path = Path(args.path)
    if not path.is_dir():
        print(f'Error: {path} is not a directory', file=sys.stderr)
        sys.exit(1)
    checks = check_env(path, DeepGym(mode=args.mode))
    print(format_checks(checks))
    if warnings := lint_env(path):
        print(f'\nwarnings ({ADVISORY_NOTE})')
        print('\n'.join(f'  warning: {warning}' for warning in warnings))
    _exit_for(overall(check.status for check in checks))


def _guard_local_model_code(mode: str, allowed: bool) -> None:
    """Exit unless the user opted in to running model-generated code on this machine.

    'auto' counts as local because it silently falls back to local execution.
    With the opt-in but no sandbox-exec or bwrap, print a warning and continue.
    """
    if not allowed:
        print(
            f'Error: --model runs model-generated code, and --mode {mode} can run it on this '
            'machine. Use a sandboxed mode (--mode docker, apple, daytona, modal or e2b), '
            'or pass --allow-local-exec to accept the risk.',
            file=sys.stderr,
        )
        sys.exit(1)
    if local_isolation_kind() == 'none':
        print(
            'WARNING: no sandbox-exec or bwrap on this machine. Model-generated code will run '
            'with only process separation and can read your files.',
            file=sys.stderr,
        )


def _cmd_harbor(args: argparse.Namespace) -> None:
    if args.harbor_command == 'import':
        _harbor_import(args)
    else:
        _harbor_export(args)


def _harbor_import(args: argparse.Namespace) -> None:
    """Print one line per task folder; register the valid ones if asked; exit 1 if any fail."""
    from deepgym.harbor import HarborTask, dataset_root, task_dirs
    from deepgym.registry import register_benchmark

    if args.limit is not None and args.limit < 1:
        print('Error: --limit must be >= 1', file=sys.stderr)
        sys.exit(1)
    dirs = task_dirs(dataset_root(args.source))[: args.limit]
    if not dirs:
        print(f'Error: no Harbor tasks (task.toml folders) in {args.source}', file=sys.stderr)
        sys.exit(1)
    width = max(len(task_dir.name) for task_dir in dirs)
    valid: list[Path] = []
    for task_dir in dirs:
        try:
            task = HarborTask.from_dir(task_dir)
        except ValueError as exc:
            print(f'  FAIL  {task_dir.name:{width}}  {exc}')
            continue
        valid.append(task_dir)
        image = task.docker_image or ('environment/Dockerfile' if task.dockerfile else 'none')
        solution = 'yes' if task.solution else 'no'
        print(f'  ok    {task.name:{width}}  image={image}  solution={solution}')
    print(f'\n{len(valid)} of {len(dirs)} tasks valid')
    if args.register and valid:
        metadata = {'type': 'tool-use', 'domain': 'terminal', 'tags': ['harbor']}
        registry = register_benchmark(args.register, valid, metadata)
        print(f'Registered in {registry}: load_environment("{args.register}/{valid[0].name}")')
    if len(valid) < len(dirs):
        sys.exit(1)


def _harbor_export(args: argparse.Namespace) -> None:
    from deepgym.harbor import export_environment
    from deepgym.registry import load_environment

    for name in args.envs:
        print(export_environment(load_environment(name), Path(args.out)))


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


def _cmd_traces(args: argparse.Namespace) -> None:
    from deepgym.console.db import DEFAULT_DB_PATH
    from deepgym.console.episode_store import EpisodeStore
    from deepgym.traces import Trace, export_traces, is_trace_file, load_trace, traces_dir

    if args.db is None and args.traces is None:
        db = DEFAULT_DB_PATH if DEFAULT_DB_PATH.exists() else None
        paths = [path for path in [traces_dir()] if path.exists()]
    else:
        db = Path(args.db) if args.db else None
        paths = [Path(path) for path in args.traces or []]
        missing = [str(path) for path in [db, *paths] if path and not path.exists()]
        if missing:
            print(f'Error: not found: {", ".join(missing)}', file=sys.stderr)
            sys.exit(1)

    # In a directory, read only trace files: an export written there must not break the next.
    found = [sorted(path.glob('*.jsonl')) if path.is_dir() else [path] for path in paths]
    files = [file for path, group in zip(paths, found) for file in group
             if not path.is_dir() or is_trace_file(file)]  # fmt: skip
    others = sum(map(len, found)) - len(files)
    out = Path(args.out)
    if out.exists() and any(out.samefile(source) for source in [db, *files] if source):
        print(f'Error: {out} is one of the inputs; write the export elsewhere', file=sys.stderr)
        sys.exit(1)

    traces, untraced = EpisodeStore(db).traces() if db else ([], 0)
    traces += [load_trace(file) for file in files]
    # A trace can be both on disk and in the console: keep the copy that got furthest.
    unique: dict[str, Trace] = {}
    for trace in sorted(traces, key=lambda t: (t.result is not None, len(t.steps))):
        unique[trace.trace_id] = trace
    if not unique:
        print('Error: no traces found', file=sys.stderr)
        sys.exit(1)

    report = export_traces(
        sorted(unique.values(), key=lambda trace: trace.provenance.started_at),
        args.format,
        out,
        rows=args.rows,
        only_valid=args.only_valid,
    )
    print(f'Wrote {report.written} of {len(unique)} traces to {args.out}')
    for reason, count in report.skipped.items():
        print(f'  skipped {count}: {reason}')
    if untraced:
        print(f'  {untraced} console episodes were recorded without a trace and not read')
    if others:
        print(f'  {others} .jsonl files in the traces directories are not traces and were not read')


def _cmd_console(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(
            'Error: uvicorn is required to run the console. '
            'Install it with: pip install "deepgym[server]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from deepgym.console.app import MISSING_UI_MESSAGE, create_app, ui_built

    if not ui_built():
        print(MISSING_UI_MESSAGE, file=sys.stderr)
    allow_local = args.allow_local_exec or os.environ.get(
        'DEEPGYM_ALLOW_LOCAL_EXEC', ''
    ).lower() in ('true', '1', 'yes')
    loopback = args.host in ('127.0.0.1', 'localhost')
    app = create_app(
        allow_local_exec=allow_local,
        allowed_hosts=['127.0.0.1', 'localhost'] if loopback else None,
    )
    print(f'DeepGym Console: http://{args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)


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


def _verdict(result) -> str:
    """PASS or FAIL, or ERROR and the error type when the run got no verdict."""
    if not result.has_verdict:
        return f'ERROR ({result.error_type or "verifier_error"})'
    return 'PASS' if result.passed else 'FAIL'


def _cause(result) -> str:
    """First line of why a run without a verdict failed."""
    return result.output.strip().partition('\n')[0]


def _print_run_result(result, elapsed: float) -> None:
    print(f'\n{"=" * 50}')
    print(f'  Result:  {_verdict(result)}')
    print(f'  Score:   {result.score:.4f}')
    print(f'  Time:    {elapsed:.2f}s')
    print(f'{"=" * 50}')
    if result.output:
        print(f'\n  Output:\n{_indent(result.output)}')
    if result.stderr:
        print(f'\n  Stderr:\n{_indent(result.stderr)}')


def _judged(totals, value: str) -> str:
    """*value*, a judged-only figure, or '-' when no run got a verdict."""
    return value if totals.judged_count else '-'


def _no_verdict_share(totals) -> str:
    """How many of the runs got no verdict, e.g. '12.00% no verdict'."""
    return f'{totals.no_verdict_count / max(totals.total, 1):.2%} no verdict'


def _print_batch_result(batch, solution_files, elapsed: float) -> None:
    judged_avg = _judged(batch, f'{batch.judged_mean_score:.4f}')
    print(f'\n{"=" * 50}')
    print('  Batch Results')
    print(f'  Total:   {batch.total}')
    print(f'  Passed:  {batch.passed}')
    print(f'  Failed:  {batch.failed}')
    print(f'  Avg:     {batch.avg_score:.4f}  (judged {judged_avg}, {_no_verdict_share(batch)})')
    print(f'  Time:    {elapsed:.2f}s')
    print(f'{"=" * 50}')

    for i, result in enumerate(batch.results):
        name = solution_files[i].name if i < len(solution_files) else f'solution-{i}'
        detail = f'score={result.score:.4f}' if result.has_verdict else _cause(result)
        print(f'  [{_verdict(result)}] {name:30s}  {detail}')


def _print_eval_result(result, elapsed: float) -> None:
    judged_rate = _judged(result, f'{result.judged_pass_rate:.2%}')
    judged_avg = _judged(result, f'{result.judged_mean_score:.4f}')
    no_verdict = _no_verdict_share(result)
    print(f'\n{"=" * 50}')
    print(f'  Eval Suite:  {result.suite}')
    print(f'  Model:       {result.model_name}')
    print(f'  Pass Rate:   {result.pass_rate:.2%}  (judged {judged_rate}, {no_verdict})')
    print(f'  Total:       {result.total}')
    print(f'  Passed:      {result.passed}')
    print(f'  Avg Score:   {result.avg_score:.4f}  (judged {judged_avg})')
    print(f'  Time:        {elapsed:.2f}s')
    print(f'{"=" * 50}')
    for name, run in zip(result.env_names or [], result.results):
        if not run.has_verdict:
            print(f'  [{_verdict(run)}] {name}: {_cause(run)}')


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
    'check': _cmd_check,
    'harbor': _cmd_harbor,
    'new': _cmd_new,
    'qa': _cmd_qa,
    'run': _cmd_run,
    'run-batch': _cmd_run_batch,
    'eval': _cmd_eval,
    'create': _cmd_create,
    'serve': _cmd_serve,
    'console': _cmd_console,
    'web': _cmd_console,
    'traces': _cmd_traces,
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
