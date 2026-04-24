#!/usr/bin/env python
"""Generate a GLM/Z.ai baseline answer for a DeepGym environment and score it.

Usage:
    cp .env.example .env
    # Fill rotated ZAI_API_KEY and optional DAYTONA_API_KEY in .env
    python scripts/smoke_zai_deepgym.py --env cyber/cve_2021_44228_log_triage

This script never prints API keys.  Rewards still come from deterministic
DeepGym verifiers; Z.ai is only the candidate policy/model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepgym import DeepGym, load_environment  # noqa: E402
from deepgym.integrations.zai import ZaiChatClient  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def main() -> int:
    parser = argparse.ArgumentParser(description='Smoke-test a DeepGym env with Z.ai/GLM.')
    parser.add_argument('--env', default='cyber/cve_2021_44228_log_triage')
    parser.add_argument('--mode', choices=['local', 'auto', 'daytona'], default='local')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--thinking', choices=['enabled', 'disabled'], default='enabled')
    args = parser.parse_args()

    _load_dotenv(ROOT / '.env')
    env = load_environment(args.env)
    client = ZaiChatClient.from_env()
    prompt = (
        'You are solving a sandboxed DeepGym cybersecurity benchmark task. '\
        'Only operate on the provided synthetic/local artifacts. '\
        'Return exactly the requested answer format, with no markdown.\n\n' + env.task
    )
    completion = client.complete_prompt(
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )

    dg = DeepGym(mode=args.mode)
    result = dg.run(env, completion)
    print(f'env={args.env}')
    print(f'model={client.model}')
    print(f'score={result.score:.3f} passed={result.passed}')
    print(f'details={result.output}')
    if result.reward_components:
        print(f'reward_components={result.reward_components}')
    return 0 if result.score > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
