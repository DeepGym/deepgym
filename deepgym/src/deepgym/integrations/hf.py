"""HuggingFace Hub integration for DeepGym environments.

Push and pull DeepGym environments as HuggingFace datasets, enabling community
sharing, versioning, and discoverability through the HF ecosystem.

Usage::

    from deepgym.integrations.hf import push_environment_to_hub, load_environment_from_hub

    # Push a built-in environment to your HF org
    env = load_environment('coin_change')
    push_environment_to_hub(env, repo_id='your-org/deepgym-coin-change', env_name='coin_change')

    # Load it back on any machine
    env = load_environment_from_hub('your-org/deepgym-coin-change')

    # Push eval results as a leaderboard dataset
    push_results_to_hub(results_dict, repo_id='your-org/deepgym-leaderboard')

Dataset schema (each row is one environment)::

    {
        "env_name": "coin_change",
        "task": "Write a function coin_change(coins, amount)...",
        "verifier_code": "...",
        "test_cases": "[{...}]",   # JSON string, empty string if None
        "timeout": 30,
        "schema_version": "1.0"
    }
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from deepgym.models import Environment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = '1.0'
_DATASET_FEATURES_SCHEMA = {
    'env_name': 'string',
    'task': 'string',
    'verifier_code': 'string',
    'test_cases': 'string',
    'timeout': 'int32',
    'schema_version': 'string',
}


def _require_datasets() -> object:
    """Import datasets library or raise a helpful error."""
    try:
        import datasets  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub and datasets are required for HF Hub integration. '
            "Install with: pip install 'deepgym[hf]'"
        ) from exc
    return datasets


def environment_to_dict(env: Environment, env_name: str = '') -> dict:
    """Serialise an Environment to a flat dict for HF dataset rows.

    Args:
        env: Environment to serialise.
        env_name: Friendly name for the dataset row (optional).

    Returns:
        Dict with keys: env_name, task, verifier_code, test_cases, timeout,
        schema_version.
    """
    if env.verifier_path is not None:
        verifier_code = env.verifier_path.read_text(encoding='utf-8')
    else:
        verifier_code = env.verifier_code

    test_cases_str = json.dumps(env.test_cases, ensure_ascii=False) if env.test_cases else ''

    return {
        'env_name': env_name or '',
        'task': env.task,
        'verifier_code': verifier_code,
        'test_cases': test_cases_str,
        'timeout': env.timeout,
        'schema_version': _SCHEMA_VERSION,
    }


def environment_from_dict(row: dict) -> Environment:
    """Deserialise an Environment from a HF dataset row.

    Args:
        row: Dict produced by :func:`environment_to_dict` or loaded from HF.

    Returns:
        Reconstructed Environment.
    """
    test_cases = None
    raw_tc = row.get('test_cases', '')
    if raw_tc:
        try:
            test_cases = json.loads(raw_tc)
        except json.JSONDecodeError as exc:
            logger.warning('Failed to parse test_cases from HF row: %s', exc)

    return Environment(
        task=row['task'],
        verifier_code=row['verifier_code'],
        test_cases=test_cases,
        timeout=int(row.get('timeout', 30)),
    )


def push_environment_to_hub(
    env: Environment,
    repo_id: str,
    env_name: str = '',
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    """Push a single DeepGym environment to HuggingFace Hub as a dataset.

    Creates or updates the dataset at ``repo_id`` with one row containing
    the environment's task, verifier code, and test cases.

    Args:
        env: Environment to push.
        repo_id: HF dataset repo ID, e.g. ``'your-org/deepgym-coin-change'``.
        env_name: Friendly name stored in the dataset row.
        private: Whether to create the dataset as private.
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Raises:
        ImportError: If ``datasets`` package is not installed.
    """
    datasets = _require_datasets()
    row = environment_to_dict(env, env_name=env_name)
    dataset = datasets.Dataset.from_list([row])
    dataset.push_to_hub(repo_id, private=private, token=token)
    logger.info('Pushed environment %r to %s', env_name or 'unnamed', repo_id)


def push_environments_to_hub(
    envs: dict[str, Environment],
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    """Push multiple environments to HuggingFace Hub as a single dataset.

    Args:
        envs: Mapping of ``{env_name: Environment}``.
        repo_id: HF dataset repo ID, e.g. ``'your-org/deepgym-environments'``.
        private: Whether to create the dataset as private.
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Raises:
        ImportError: If ``datasets`` package is not installed.
    """
    datasets = _require_datasets()
    rows = [environment_to_dict(env, env_name=name) for name, env in envs.items()]
    dataset = datasets.Dataset.from_list(rows)
    dataset.push_to_hub(repo_id, private=private, token=token)
    logger.info('Pushed %d environments to %s', len(rows), repo_id)


def load_environment_from_hub(
    repo_id: str,
    env_name: str | None = None,
    *,
    split: str = 'train',
    token: str | None = None,
) -> Environment:
    """Load a DeepGym environment from a HuggingFace Hub dataset.

    Args:
        repo_id: HF dataset repo ID, e.g. ``'deepgym/coin-change'``.
        env_name: If the dataset contains multiple environments, filter by
            this name. If None, loads the first (or only) row.
        split: Dataset split to load from (default: ``'train'``).
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Returns:
        Reconstructed Environment.

    Raises:
        ImportError: If ``datasets`` package is not installed.
        ValueError: If ``env_name`` is specified but not found in the dataset.
    """
    datasets = _require_datasets()
    dataset = datasets.load_dataset(repo_id, split=split, token=token)

    if env_name is not None:
        matches = [row for row in dataset if row.get('env_name') == env_name]
        if not matches:
            available = [row.get('env_name', '') for row in dataset]
            raise ValueError(
                f'Environment {env_name!r} not found in {repo_id}. '
                f'Available: {available}'
            )
        row = matches[0]
    else:
        row = dataset[0]

    return environment_from_dict(row)


def load_all_environments_from_hub(
    repo_id: str,
    *,
    split: str = 'train',
    token: str | None = None,
) -> dict[str, Environment]:
    """Load all environments from a HuggingFace Hub dataset.

    Args:
        repo_id: HF dataset repo ID containing multiple environment rows.
        split: Dataset split to load from (default: ``'train'``).
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Returns:
        Dict mapping ``env_name`` to ``Environment``. Rows without an
        ``env_name`` are keyed by their zero-based index.

    Raises:
        ImportError: If ``datasets`` package is not installed.
    """
    datasets = _require_datasets()
    dataset = datasets.load_dataset(repo_id, split=split, token=token)
    result: dict[str, Environment] = {}
    for i, row in enumerate(dataset):
        key = row.get('env_name') or str(i)
        result[key] = environment_from_dict(row)
    return result


def push_results_to_hub(
    results: dict[str, float],
    repo_id: str,
    model_name: str = '',
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    """Push model evaluation results to HuggingFace Hub as a leaderboard dataset.

    Args:
        results: Mapping of ``{env_name: score}`` from a DeepGym eval run.
        repo_id: HF dataset repo ID for the leaderboard.
        model_name: Model identifier (e.g. ``'Qwen/Qwen2-0.5B-Instruct'``).
        private: Whether to create the dataset as private.
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Raises:
        ImportError: If ``datasets`` package is not installed.
    """
    datasets = _require_datasets()
    from datetime import datetime, timezone

    rows = [
        {
            'model_name': model_name,
            'env_name': env_name,
            'score': score,
            'evaluated_at': datetime.now(timezone.utc).isoformat(),
        }
        for env_name, score in results.items()
    ]
    dataset = datasets.Dataset.from_list(rows)
    dataset.push_to_hub(repo_id, private=private, token=token)
    avg = sum(results.values()) / len(results) if results else 0.0
    logger.info('Pushed results for %s (avg=%.3f) to %s', model_name, avg, repo_id)
