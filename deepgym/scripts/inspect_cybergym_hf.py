#!/usr/bin/env python
"""Inspect a CyberGym-style Hugging Face dataset for DeepGym curriculum planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepgym.cyberbench import summarize_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Inspect CyberGym HF dataset rows.')
    parser.add_argument('--repo-id', default='sunblaze-ucb/cybergym')
    parser.add_argument('--split', default='tasks')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets first: pip install 'deepgym[hf]'", file=sys.stderr)
        return 2

    dataset = load_dataset(args.repo_id, split=args.split, streaming=True)
    rows = []
    for row in dataset:
        rows.append(dict(row))
        if len(rows) >= args.limit:
            break

    print(json.dumps(summarize_rows(rows), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
