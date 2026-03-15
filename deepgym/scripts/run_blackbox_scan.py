"""Run black-box adversarial scanning against environments.

Black-box attacks simulate realistic RL training conditions: the attacker
can only see the task description and pass/fail feedback, NOT the verifier
source code. This is Tier 3 of the verification funnel.

Usage:
    python scripts/run_blackbox_scan.py --benchmark humaneval --limit 10
    python scripts/run_blackbox_scan.py --examples
    python scripts/run_blackbox_scan.py --all --limit 10
    python scripts/run_blackbox_scan.py --compare  # compare white-box vs black-box
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from deepgym import DeepGym, Environment
from deepgym.adversarial import BlackBoxAdversarialTester
from deepgym.exploit_db import ExploitDB, ExploitRecord

logger = logging.getLogger(__name__)

ENV_ROOT = Path(__file__).resolve().parent.parent / 'environments'
EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / 'examples'

EXAMPLE_ENVS = ['python_sorting', 'two_sum', 'string_manipulation']


def _load_example_env(name: str) -> Environment | None:
    """Load one of the hardened example environments.

    Args:
        name: Directory name under examples/ (e.g. 'python_sorting').

    Returns:
        Environment, or None if the directory is missing required files.
    """
    d = EXAMPLES_ROOT / name
    verifier_path = d / 'verifier.py'
    task_path = d / 'task.md'
    if not verifier_path.exists() or not task_path.exists():
        return None
    return Environment(
        task=task_path.read_text(encoding='utf-8'),
        verifier_code=verifier_path.read_text(encoding='utf-8'),
    )


def scan_examples(
    *,
    api_key: str | None = None,
    model: str = 'claude-sonnet-4-6',
    max_rounds: int = 5,
    attempts_per_round: int = 3,
) -> None:
    """Scan the 3 hardened example verifiers with black-box attacks.

    Args:
        api_key: Anthropic API key.
        model: Anthropic model ID.
        max_rounds: Max feedback rounds per environment.
        attempts_per_round: Attempts generated per round.
    """
    dg = DeepGym(mode='local')
    tester = BlackBoxAdversarialTester(
        dg=dg,
        api_key=api_key,
        model=model,
        max_rounds=max_rounds,
        attempts_per_round=attempts_per_round,
    )
    db = ExploitDB()

    print(f'Black-box scanning {len(EXAMPLE_ENVS)} hardened example environments...\n')

    for name in EXAMPLE_ENVS:
        env = _load_example_env(name)
        if env is None:
            print(f'  {name}: missing files, skipping')
            continue

        existing = db.get(name, attack_type='black-box')
        if existing:
            print(f'  {name}: already tested (black-box), skipping')
            continue

        print(f'  {name}...', end=' ', flush=True)
        start = time.time()

        try:
            report = tester.test(env)
            verifier_code = (EXAMPLES_ROOT / name / 'verifier.py').read_text(encoding='utf-8')

            record = ExploitRecord(
                verifier_id=name,
                benchmark='examples',
                exploitable=report.exploitable,
                patterns=report.strategies_tried,
                num_attacks=report.total_attempts,
                num_exploits=sum(1 for r in report.rounds for a in r.attempts if a['score'] >= 0.5),
                max_exploit_score=report.best_exploit_score,
                attacks=[
                    {
                        'round': r.round_num,
                        'strategy': a['strategy'],
                        'score': a['score'],
                    }
                    for r in report.rounds
                    for a in r.attempts
                ],
                verifier_hash=hashlib.sha256(verifier_code.encode()).hexdigest()[:16],
                attack_type='black-box',
            )
            db.save(record)

            elapsed = time.time() - start
            status = 'EXPLOITABLE' if report.exploitable else 'ROBUST'
            print(
                f'{status} (best={report.best_exploit_score:.2f}, '
                f'{report.total_rounds} rounds, {report.total_attempts} attempts, '
                f'{elapsed:.1f}s)'
            )
            _print_round_detail(report)

        except KeyboardInterrupt:
            print('\nInterrupted by user.')
            break
        except Exception as exc:
            logger.error('Error scanning %s: %s', name, exc)
            print(f'ERROR: {exc}')
            continue

    db.close()


def scan_benchmark(
    benchmark: str,
    limit: int | None = None,
    *,
    api_key: str | None = None,
    model: str = 'claude-sonnet-4-6',
    max_rounds: int = 5,
    attempts_per_round: int = 3,
) -> None:
    """Scan environments in a benchmark with black-box attacks.

    Args:
        benchmark: Name of the benchmark directory (e.g. humaneval).
        limit: Maximum number of environments to test. None means all.
        api_key: Anthropic API key.
        model: Anthropic model ID.
        max_rounds: Max feedback rounds per environment.
        attempts_per_round: Attempts generated per round.
    """
    dg = DeepGym(mode='local')
    tester = BlackBoxAdversarialTester(
        dg=dg,
        api_key=api_key,
        model=model,
        max_rounds=max_rounds,
        attempts_per_round=attempts_per_round,
    )
    db = ExploitDB()

    env_dir = ENV_ROOT / benchmark
    if not env_dir.is_dir():
        print(f'Benchmark directory not found: {env_dir}')
        return

    env_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])
    if limit:
        env_dirs = env_dirs[:limit]

    print(f'Black-box scanning {len(env_dirs)} {benchmark} environments...\n')

    for i, d in enumerate(env_dirs):
        verifier_path = d / 'verifier.py'
        task_path = d / 'task.md'

        if not verifier_path.exists():
            continue

        existing = db.get(d.name, attack_type='black-box')
        if existing:
            print(f'  [{i + 1}/{len(env_dirs)}] {d.name}: already tested, skipping')
            continue

        verifier_code = verifier_path.read_text(encoding='utf-8')
        task = task_path.read_text(encoding='utf-8') if task_path.exists() else d.name

        env = Environment(task=task, verifier_code=verifier_code)

        print(f'  [{i + 1}/{len(env_dirs)}] {d.name}...', end=' ', flush=True)
        start = time.time()

        try:
            report = tester.test(env)

            record = ExploitRecord(
                verifier_id=d.name,
                benchmark=benchmark,
                exploitable=report.exploitable,
                patterns=report.strategies_tried,
                num_attacks=report.total_attempts,
                num_exploits=sum(1 for r in report.rounds for a in r.attempts if a['score'] >= 0.5),
                max_exploit_score=report.best_exploit_score,
                attacks=[
                    {
                        'round': r.round_num,
                        'strategy': a['strategy'],
                        'score': a['score'],
                    }
                    for r in report.rounds
                    for a in r.attempts
                ],
                verifier_hash=hashlib.sha256(verifier_code.encode()).hexdigest()[:16],
                attack_type='black-box',
            )
            db.save(record)

            elapsed = time.time() - start
            status = 'EXPLOITABLE' if report.exploitable else 'ROBUST'
            print(
                f'{status} (best={report.best_exploit_score:.2f}, '
                f'{report.total_rounds} rounds, {report.total_attempts} attempts, '
                f'{elapsed:.1f}s)'
            )

        except KeyboardInterrupt:
            print('\nInterrupted by user.')
            break
        except Exception as exc:
            logger.error('Error scanning %s: %s', d.name, exc)
            print(f'ERROR: {exc}')
            continue

    _print_summary(benchmark, db)
    db.close()


def compare_results() -> None:
    """Compare white-box vs black-box exploit rates from the database."""
    db = ExploitDB()

    white = db.get_all(attack_type='white-box')
    black = db.get_all(attack_type='black-box')

    print('\n=== WHITE-BOX vs BLACK-BOX COMPARISON ===\n')

    if not white and not black:
        print('No results found. Run scans first.')
        db.close()
        return

    white_ids = {r.verifier_id for r in white}
    black_ids = {r.verifier_id for r in black}
    common = white_ids & black_ids

    print(f'White-box results: {len(white)}')
    print(f'Black-box results: {len(black)}')
    print(f'Tested with both:  {len(common)}\n')

    if white:
        w_exploitable = sum(1 for r in white if r.exploitable)
        print(
            f'White-box exploit rate: {w_exploitable}/{len(white)} '
            f'({w_exploitable / len(white):.1%})'
        )

    if black:
        b_exploitable = sum(1 for r in black if r.exploitable)
        print(
            f'Black-box exploit rate: {b_exploitable}/{len(black)} '
            f'({b_exploitable / len(black):.1%})'
        )

    if common:
        print('\n--- Verifiers tested with both approaches ---')
        white_lookup = {r.verifier_id: r for r in white}
        black_lookup = {r.verifier_id: r for r in black}
        both_exploitable = 0
        white_only = 0
        black_only = 0
        neither = 0

        for vid in sorted(common):
            w = white_lookup[vid]
            b = black_lookup[vid]
            if w.exploitable and b.exploitable:
                both_exploitable += 1
                label = 'BOTH'
            elif w.exploitable:
                white_only += 1
                label = 'WHITE-ONLY'
            elif b.exploitable:
                black_only += 1
                label = 'BLACK-ONLY'
            else:
                neither += 1
                label = 'ROBUST'
            print(
                f'  {vid}: {label} '
                f'(white={w.max_exploit_score:.2f}, black={b.max_exploit_score:.2f})'
            )

        print(f'\nBoth exploitable:  {both_exploitable}')
        print(f'White-box only:    {white_only}')
        print(f'Black-box only:    {black_only}')
        print(f'Neither (robust):  {neither}')

    db.close()


def _print_round_detail(report) -> None:
    """Print per-round detail for a black-box report.

    Args:
        report: A BlackBoxReport instance.
    """
    for r in report.rounds:
        scores = [f'{a["score"]:.2f}' for a in r.attempts]
        strategies = [a['strategy'] for a in r.attempts]
        print(f'    Round {r.round_num}: scores=[{", ".join(scores)}] strategies={strategies}')


def _print_summary(benchmark: str, db: ExploitDB) -> None:
    """Print scan summary statistics.

    Args:
        benchmark: Benchmark name for the header.
        db: ExploitDB with results.
    """
    results = db.get_all(benchmark=benchmark, attack_type='black-box')
    total = len(results)
    exploitable = sum(1 for r in results if r.exploitable)
    rate = exploitable / total if total > 0 else 0.0

    print(f'\n=== {benchmark.upper()} BLACK-BOX SCAN RESULTS ===')
    print(f'Tested:      {total}')
    print(f'Exploitable: {exploitable} ({rate:.1%})')

    if results:
        scores = [r.max_exploit_score for r in results]
        print(f'Avg best score: {sum(scores) / len(scores):.3f}')
        print(f'Max score:      {max(scores):.3f}')

    # Strategy frequency
    strategy_counts: dict[str, int] = {}
    for r in results:
        for p in r.patterns:
            strategy_counts[p] = strategy_counts.get(p, 0) + 1
    if strategy_counts:
        print('Strategy frequency:')
        for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f'  {s}: {c}')


def main() -> None:
    """Parse arguments and run the black-box exploit scanner."""
    parser = argparse.ArgumentParser(description='Run black-box adversarial exploit scanning.')
    parser.add_argument(
        '--benchmark',
        type=str,
        help='Benchmark to scan (humaneval, mbpp, etc.)',
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Scan the 3 hardened example environments.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Scan humaneval + examples.',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare white-box vs black-box results.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max environments to scan per benchmark.',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Anthropic API key (falls back to ANTHROPIC_API_KEY env var).',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-6',
        help='Anthropic model to use for black-box attacks.',
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=5,
        help='Maximum feedback rounds per environment.',
    )
    parser.add_argument(
        '--attempts-per-round',
        type=int,
        default=3,
        help='Number of exploit attempts per round.',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    if args.compare:
        compare_results()
        return

    common_kwargs = {
        'api_key': args.api_key,
        'model': args.model,
        'max_rounds': args.max_rounds,
        'attempts_per_round': args.attempts_per_round,
    }

    if args.all:
        scan_benchmark('humaneval', args.limit, **common_kwargs)
        scan_examples(**common_kwargs)
    elif args.examples:
        scan_examples(**common_kwargs)
    elif args.benchmark:
        scan_benchmark(args.benchmark, args.limit, **common_kwargs)
    else:
        parser.error('Specify --benchmark, --examples, --all, or --compare')


if __name__ == '__main__':
    main()
