#!/usr/bin/env python
"""Run artifact-backed CyberGym patch tasks through DeepGym.

Default mode uses reference patches to validate importer/verifier/Daytona
throughput. Use --answer-source glm to ask Z.ai for patches, with fallback to
reference patches when the model API is rate-limited.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepgym import DeepGym  # noqa: E402
from deepgym.cybergym_artifacts import (  # noqa: E402
    CyberGymPatchEnvironment,
    download_cybergym_artifacts,
    load_cybergym_rows,
)
from deepgym.integrations.zai import ZaiChatClient  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _safe_task_id(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in value)


def _glm_patch(
    client: ZaiChatClient,
    env: CyberGymPatchEnvironment,
    cache_path: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    if cache_path.exists() and not args.refresh_cache:
        return cache_path.read_text(encoding='utf-8'), 'cache'
    prompt = env.task + '\n\nReturn only a unified diff patch.'
    answer = client.complete_prompt(
        prompt,
        system='You repair local synthetic vulnerable repositories. Return a unified diff only.',
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(answer, encoding='utf-8')
    return answer, 'glm'


def _run_one(index: int, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    task_id = str(row.get('task_id') or f'row-{index}')
    started = time.perf_counter()
    artifacts = download_cybergym_artifacts(row, repo_id=args.repo_id, level=args.level)
    env = CyberGymPatchEnvironment.from_row(
        row,
        repo_id=args.repo_id,
        artifacts=artifacts,
        level=args.level,
        timeout=args.timeout,
    )

    answer_source = args.answer_source
    error = ''
    if args.answer_source == 'reference':
        if artifacts.patch is None:
            answer = ''
            error = 'missing reference patch'
        else:
            answer = artifacts.patch.read_text(encoding='utf-8', errors='replace')
    else:
        try:
            client = ZaiChatClient.from_env()
            client.timeout = args.zai_timeout
            answer, answer_source = _glm_patch(
                client,
                env,
                Path(args.cache_dir) / f'{_safe_task_id(task_id)}.diff',
                args,
            )
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            if args.fallback_to_reference and artifacts.patch is not None:
                answer = artifacts.patch.read_text(encoding='utf-8', errors='replace')
                answer_source = 'reference_fallback'
            else:
                answer = ''

    dg = DeepGym(mode=args.mode)
    result = dg.run(env, answer, timeout=args.timeout)
    return {
        'index': index,
        'task_id': task_id,
        'project': row.get('project_name', ''),
        'language': row.get('project_language', ''),
        'answer_source': answer_source,
        'answer_error': error,
        'score': result.score,
        'passed': result.passed,
        'details': result.output,
        'reward_components': result.reward_components,
        'metrics': result.metrics,
        'cases': [case.model_dump() for case in result.cases] if result.cases else [],
        'sandbox_id': result.sandbox_id,
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate artifact-backed CyberGym patch tasks.')
    parser.add_argument('--repo-id', default='sunblaze-ucb/cybergym')
    parser.add_argument('--split', default='tasks')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--level', default='level3')
    parser.add_argument('--mode', choices=['local', 'auto', 'daytona'], default='daytona')
    parser.add_argument('--max-parallel', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--answer-source', choices=['reference', 'glm'], default='reference')
    parser.add_argument('--fallback-to-reference', action='store_true')
    parser.add_argument('--output', default='data/cyberbench/top100_artifact_eval.jsonl')
    parser.add_argument('--cache-dir', default='data/cyberbench/glm_patch_cache')
    parser.add_argument('--refresh-cache', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--thinking', choices=['enabled', 'disabled'], default='enabled')
    parser.add_argument('--zai-timeout', type=float, default=60.0)
    args = parser.parse_args()

    _load_dotenv(ROOT / '.env')
    rows = load_cybergym_rows(
        args.repo_id,
        split=args.split,
        count=args.count,
        start_index=args.start_index,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(_run_one, args.start_index + local_index, row, args): local_index
            for local_index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records.sort(key=lambda item: item['index'])
    output_path.write_text(
        ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records),
        encoding='utf-8',
    )
    summary = {
        'count': len(records),
        'mode': args.mode,
        'answer_source': args.answer_source,
        'passed': sum(bool(record['passed']) for record in records),
        'avg_score': round(sum(float(record['score']) for record in records) / len(records), 4)
        if records
        else 0.0,
        'output': str(output_path),
    }
    summary_path = output_path.with_suffix('.summary.json')
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + '\n'
    summary_path.write_text(summary_text, encoding='utf-8')
    print(json.dumps({'summary': summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
