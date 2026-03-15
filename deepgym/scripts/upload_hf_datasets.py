"""Upload DeepGym environments to HuggingFace Hub as datasets.

This makes environments browsable on HuggingFace and downloadable via:
    from datasets import load_dataset
    ds = load_dataset('deepgym/environments', 'humaneval')

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_hf_datasets.py

Requirements:
    pip install datasets huggingface_hub
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = 'deepgym/environments'
ENVIRONMENTS_DIR = Path(__file__).resolve().parent.parent / 'environments'


def collect_environment_records(env_dir: Path) -> list[dict]:
    """Collect all environment records from the registry and task files.

    Args:
        env_dir: Path to the environments directory.

    Returns:
        List of dicts, each representing one environment with task, verifier, etc.
    """
    registry_path = env_dir / 'registry.json'
    if not registry_path.exists():
        logger.error('registry.json not found at %s', registry_path)
        sys.exit(1)

    with registry_path.open() as f:
        registry = json.load(f)

    records: list[dict] = []
    for entry in registry:
        name = entry.get('name', '')
        env_path = env_dir / name
        if not env_path.is_dir():
            logger.warning('Skipping %s: directory not found', name)
            continue

        task_file = env_path / 'task.md'
        verifier_file = env_path / 'verifier.py'
        solution_file = env_path / 'solution.py'

        record = {
            'name': name,
            'suite': entry.get('suite', ''),
            'difficulty': entry.get('difficulty', ''),
            'task': task_file.read_text() if task_file.exists() else '',
            'verifier': verifier_file.read_text() if verifier_file.exists() else '',
            'solution': solution_file.read_text() if solution_file.exists() else '',
        }
        records.append(record)

    return records


def upload(records: list[dict], token: str) -> None:
    """Upload records as a HuggingFace dataset.

    Args:
        records: List of environment record dicts.
        token: HuggingFace API token.
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        logger.error('Install dependencies: pip install datasets huggingface_hub')
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type='dataset', exist_ok=True)

    ds = Dataset.from_list(records)
    ds.push_to_hub(REPO_ID, token=token)
    logger.info('Uploaded %d environments to https://huggingface.co/datasets/%s', len(records), REPO_ID)


def main() -> None:
    """Entry point for HuggingFace dataset upload."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    token = os.environ.get('HF_TOKEN', '')
    if not token:
        logger.error('Set HF_TOKEN environment variable with your HuggingFace token')
        sys.exit(1)

    records = collect_environment_records(ENVIRONMENTS_DIR)
    logger.info('Collected %d environment records', len(records))
    upload(records, token)


if __name__ == '__main__':
    main()
