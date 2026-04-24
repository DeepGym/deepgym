#!/usr/bin/env python
"""Generate safe DeepGym CyberBench-RL seed specs from CyberGym HF metadata.

This is the bridge between a CyberGym-style HF dataset and DeepGym RL tasks:
1. sample vulnerability metadata rows;
2. optionally ask Z.ai/GLM to synthesize local-only task specs;
3. validate safety/reward shape;
4. write JSONL seed specs for later env materialization.

The output is not a weaponized exploit corpus. It is a set of synthetic,
local-only RL task plans with deterministic verifier requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepgym.cyberbench import (  # noqa: E402
    cyber_seed_from_hf_row,
    glm_seed_prompt,
    spec_from_glm_json,
    validate_seed_spec,
    write_seed_specs,
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


def _load_hf_rows(repo_id: str, split: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install 'deepgym[hf]'") from exc

    rows: list[dict[str, Any]] = []
    dataset = load_dataset(repo_id, split=split, streaming=True)
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate CyberBench-RL seed specs.')
    parser.add_argument('--repo-id', default='sunblaze-ucb/cybergym')
    parser.add_argument('--split', default='tasks')
    parser.add_argument('--count', type=int, default=25)
    parser.add_argument('--output', default='data/cyberbench/seed_specs.jsonl')
    parser.add_argument('--use-zai', action='store_true', help='Use Z.ai/GLM to enrich specs')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-tokens', type=int, default=1600)
    parser.add_argument('--zai-timeout', type=float, default=30.0)
    parser.add_argument('--fail-on-invalid', action='store_true')
    args = parser.parse_args()

    _load_dotenv(ROOT / '.env')
    rows = _load_hf_rows(args.repo_id, args.split, args.count)
    client = ZaiChatClient.from_env() if args.use_zai else None
    if client is not None:
        client.timeout = args.zai_timeout

    specs = []
    invalid: list[tuple[str, list[str]]] = []
    zai_errors: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if client is None:
            spec = cyber_seed_from_hf_row(row, index=index)
        else:
            try:
                response = client.complete_prompt(
                    glm_seed_prompt(row),
                    system=(
                        'You create safe, local-only cyber RL benchmark seed specs. '
                        'Return strict JSON only.'
                    ),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    thinking='enabled',
                )
                spec = spec_from_glm_json(response, row=row, index=index)
            except Exception as exc:
                spec = cyber_seed_from_hf_row(row, index=index)
                zai_errors.append(
                    {
                        'task_id': str(row.get('task_id') or index),
                        'error': f'{type(exc).__name__}: {exc}',
                    }
                )

        errors = validate_seed_spec(spec)
        if errors:
            invalid.append((spec.seed_id, errors))
            if args.fail_on_invalid:
                continue
        specs.append(spec)

    output = ROOT / args.output
    write_seed_specs(specs, output)
    print(json.dumps({
        'repo_id': args.repo_id,
        'rows_read': len(rows),
        'specs_written': len(specs),
        'invalid': invalid[:10],
        'zai_errors': zai_errors[:10],
        'output': str(output),
        'used_zai': client is not None,
        'zai_successes': len(rows) - len(zai_errors) if client is not None else 0,
    }, indent=2, ensure_ascii=False))
    return 1 if invalid and args.fail_on_invalid else 0


if __name__ == '__main__':
    raise SystemExit(main())
