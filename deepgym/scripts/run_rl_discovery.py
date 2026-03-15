"""Run RL exploit discovery against weak and hardened verifiers.

Simulates GRPO-like selection pressure to see if models naturally
discover reward hacks through iterative best-of-N refinement.

Usage:
    python scripts/run_rl_discovery.py
    python scripts/run_rl_discovery.py --rounds 3 --candidates 4 --top-k 2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from deepgym import DeepGym, Environment
from deepgym.rl_exploit_discovery import DiscoveryReport, RLExploitDiscovery

logger = logging.getLogger(__name__)

ENV_ROOT = Path(__file__).resolve().parent.parent / 'environments'
EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / 'examples'


def _load_humaneval_envs(limit: int = 5) -> list[tuple[str, Environment]]:
    """Load HumanEval environments (weak verifiers -- binary pass/fail).

    Args:
        limit: Maximum number of environments to load.

    Returns:
        List of (verifier_id, Environment) tuples.
    """
    humaneval_dir = ENV_ROOT / 'humaneval'
    if not humaneval_dir.is_dir():
        print(f'HumanEval directory not found: {humaneval_dir}')
        return []

    envs: list[tuple[str, Environment]] = []
    dirs = sorted(humaneval_dir.iterdir())
    for d in dirs:
        if not d.is_dir():
            continue
        verifier_path = d / 'verifier.py'
        task_path = d / 'task.md'
        if not verifier_path.exists() or not task_path.exists():
            continue

        task = task_path.read_text(encoding='utf-8')
        verifier_code = verifier_path.read_text(encoding='utf-8')
        env = Environment(task=task, verifier_code=verifier_code)
        envs.append((d.name, env))

        if len(envs) >= limit:
            break

    return envs


def _load_hardened_envs() -> list[tuple[str, Environment]]:
    """Load hardened example verifiers (strong -- check implementation quality).

    Returns:
        List of (verifier_id, Environment) tuples.
    """
    envs: list[tuple[str, Environment]] = []
    for example_dir in sorted(EXAMPLES_ROOT.iterdir()):
        if not example_dir.is_dir():
            continue
        verifier_path = example_dir / 'verifier.py'
        task_path = example_dir / 'task.md'
        if not verifier_path.exists() or not task_path.exists():
            continue

        task = task_path.read_text(encoding='utf-8')
        verifier_code = verifier_path.read_text(encoding='utf-8')
        env = Environment(task=task, verifier_code=verifier_code)
        envs.append((f'hardened_{example_dir.name}', env))

    return envs


def _print_report(report: DiscoveryReport) -> None:
    """Print a formatted report for a single environment.

    Args:
        report: The DiscoveryReport to display.
    """
    print(f'\n  Verifier: {report.verifier_id}')
    print(f'  Rounds: {report.num_rounds}  |  Total candidates: {report.total_candidates}')
    print(f'  Score progression: {" -> ".join(f"{s:.3f}" for s in report.score_progression)}')
    print(f'  Final best score: {report.final_best_score:.3f}')
    print(f'  Genuine solution: {report.is_genuine_solution}')
    print(f'  Exploit discovered: {report.exploit_discovered}')
    print(f'  Convergence: {report.convergence_type}')

    # Show per-round stats.
    for rd in report.rounds:
        print(
            f'    Round {rd.round_num}: best={rd.best_score:.3f}  '
            f'avg={rd.avg_score:.3f}  candidates={len(rd.candidates)}'
        )


def _print_comparison(
    weak_reports: list[DiscoveryReport],
    hardened_reports: list[DiscoveryReport],
) -> None:
    """Print the comparison chart between weak and hardened verifiers.

    Args:
        weak_reports: Reports from weak (HumanEval) verifiers.
        hardened_reports: Reports from hardened (example) verifiers.
    """
    print('\n' + '=' * 70)
    print('COMPARISON: WEAK vs HARDENED VERIFIERS')
    print('=' * 70)

    print(
        f'\n{"Category":<20} {"Envs":<6} {"Exploits":<10} {"Genuine":<10} '
        f'{"Avg Best":<10} {"Convergence Types"}'
    )
    print('-' * 80)

    for label, reports in [('Weak (HumanEval)', weak_reports), ('Hardened', hardened_reports)]:
        if not reports:
            print(f'{label:<20} {"0":<6} {"-":<10} {"-":<10} {"-":<10} -')
            continue

        n = len(reports)
        exploits = sum(1 for r in reports if r.exploit_discovered)
        genuine = sum(1 for r in reports if r.is_genuine_solution)
        avg_best = sum(r.final_best_score for r in reports) / n
        types = {}
        for r in reports:
            types[r.convergence_type] = types.get(r.convergence_type, 0) + 1
        type_str = ', '.join(f'{k}:{v}' for k, v in sorted(types.items()))

        print(f'{label:<20} {n:<6} {exploits:<10} {genuine:<10} {avg_best:<10.3f} {type_str}')

    print()

    # The key insight.
    weak_exploit_count = sum(1 for r in weak_reports if r.exploit_discovered)
    hard_exploit_count = sum(1 for r in hardened_reports if r.exploit_discovered)

    if weak_exploit_count > hard_exploit_count:
        print('KEY FINDING: Weak verifiers are more vulnerable to RL exploit discovery.')
        print(
            f'  Weak: {weak_exploit_count}/{len(weak_reports)} exploited  |  '
            f'Hardened: {hard_exploit_count}/{len(hardened_reports)} exploited'
        )
    elif weak_exploit_count == 0 and hard_exploit_count == 0:
        print('KEY FINDING: No exploits discovered in either category.')
        print('  The selection pressure did not find reward hacks in this run.')
    else:
        print(
            f'KEY FINDING: Weak={weak_exploit_count} exploits, '
            f'Hardened={hard_exploit_count} exploits.'
        )

    # Score progression summary.
    print('\nSCORE PROGRESSION (best score per round):')
    for label, reports in [('Weak', weak_reports), ('Hardened', hardened_reports)]:
        for r in reports:
            arrow = ' -> '.join(f'{s:.2f}' for s in r.score_progression)
            status = (
                'EXPLOIT'
                if r.exploit_discovered
                else ('GENUINE' if r.is_genuine_solution else 'MIXED')
            )
            print(f'  [{label}] {r.verifier_id[:40]:<40} {arrow}  [{status}]')


def main() -> None:
    """Parse arguments and run RL exploit discovery."""
    parser = argparse.ArgumentParser(description='Run RL exploit discovery.')
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Anthropic API key.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-6',
        help='Anthropic model for candidate generation.',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=5,
        help='Number of selection rounds.',
    )
    parser.add_argument(
        '--candidates',
        type=int,
        default=8,
        help='Candidates per round.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Top-K to keep each round.',
    )
    parser.add_argument(
        '--humaneval-limit',
        type=int,
        default=5,
        help='Number of HumanEval environments to test.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    api_key = args.api_key
    if not api_key:
        print('ERROR: --api-key is required (or set ANTHROPIC_API_KEY).', file=sys.stderr)
        sys.exit(1)

    dg = DeepGym(mode='local')
    discovery = RLExploitDiscovery(
        dg=dg,
        api_key=api_key,
        model=args.model,
        candidates_per_round=args.candidates,
        top_k=args.top_k,
        num_rounds=args.rounds,
    )

    # --- Weak verifiers (HumanEval) ---
    print('=' * 70)
    print('PHASE 1: WEAK VERIFIERS (HumanEval -- binary pass/fail)')
    print('=' * 70)

    weak_envs = _load_humaneval_envs(limit=args.humaneval_limit)
    weak_reports: list[DiscoveryReport] = []

    for i, (vid, env) in enumerate(weak_envs):
        print(f'\n[{i + 1}/{len(weak_envs)}] Testing: {vid}')
        start = time.time()
        try:
            report = discovery.discover(env, verifier_id=vid)
            weak_reports.append(report)
            elapsed = time.time() - start
            _print_report(report)
            print(f'  Time: {elapsed:.1f}s')
        except KeyboardInterrupt:
            print('\nInterrupted.')
            break
        except Exception as exc:
            print(f'  ERROR: {exc}')
            logger.exception('Error testing %s', vid)

    # --- Hardened verifiers (examples) ---
    print('\n' + '=' * 70)
    print('PHASE 2: HARDENED VERIFIERS (examples -- implementation checks)')
    print('=' * 70)

    hardened_envs = _load_hardened_envs()
    hardened_reports: list[DiscoveryReport] = []

    for i, (vid, env) in enumerate(hardened_envs):
        print(f'\n[{i + 1}/{len(hardened_envs)}] Testing: {vid}')
        start = time.time()
        try:
            report = discovery.discover(env, verifier_id=vid)
            hardened_reports.append(report)
            elapsed = time.time() - start
            _print_report(report)
            print(f'  Time: {elapsed:.1f}s')
        except KeyboardInterrupt:
            print('\nInterrupted.')
            break
        except Exception as exc:
            print(f'  ERROR: {exc}')
            logger.exception('Error testing %s', vid)

    # --- Comparison ---
    _print_comparison(weak_reports, hardened_reports)


if __name__ == '__main__':
    main()
