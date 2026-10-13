"""HuggingFace Hub integration for DeepGym environments.

Push and pull DeepGym environments as HuggingFace datasets, enabling community
sharing, versioning, and discoverability through the HF ecosystem.

Usage::

    from deepgym.integrations.hf import push_environment_to_hub, load_environment_from_hub

    # Push a built-in environment to your HF org; the push returns its commit
    env = load_environment('coin_change')
    revision = push_environment_to_hub(
        env, repo_id='your-org/deepgym-coin-change', env_name='coin_change'
    )

    # Load exactly that commit on any machine
    env = load_environment_from_hub('your-org/deepgym-coin-change', revision=revision)

    # Push eval results as a leaderboard dataset
    push_results_to_hub(results_dict, repo_id='your-org/deepgym-leaderboard')

Dataset schema (each row is one environment)::

    {
        "env_name": "coin_change",
        "environment": "{...}",   # JSON: every Environment field, verifier inlined
        "schema_version": "2.0"
    }

A round trip gives back the environment that was pushed: verifier source, support
files, test cases, timeouts, verifier mode, image and the rest. Only a plain
``Environment`` round-trips; subclasses (repo-patch, benchmark, mixed environments) are
refused. Rows in the 1.0 schema (task, verifier_code, test_cases, timeout) still load.

Every environment needs an ``env_name`` of its own: pushes refuse a blank name, and
loading by name, or loading all, refuses a dataset with a blank or repeated one rather
than drop an environment.

A Hub dataset is untrusted input, so pushing and loading run the same checks (see
:func:`environment_from_dict`) and data can't change how the sandbox runs it: no
``env_vars``, no host paths, no verifier mode the verifier doesn't read as, plain
relative support file names, and a custom ``image`` or ``snapshot`` only with
``trust_image=True``.

Without a ``revision``, ``datasets`` loads the repo's current ``main``, which changes
with every push; pass the commit a push returned to pin it.
"""

from __future__ import annotations

import json
import logging
import posixpath
from collections import Counter
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

from deepgym.models import Environment
from deepgym.verifier_template import detect_verifier_mode

if TYPE_CHECKING:
    from datasets import Dataset, Features  # type: ignore[import-untyped]
    from huggingface_hub import CommitInfo

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = '2.0'


def _require_datasets() -> ModuleType:
    """Import datasets library or raise a helpful error."""
    try:
        import datasets  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'huggingface_hub and datasets are required for HF Hub integration. '
            "Install with: pip install 'deepgym[hf]'"
        ) from exc
    return datasets


def _hub_environment(manifest: Mapping[str, Any], *, trust_image: bool) -> Environment:
    """Validate an environment manifest on its way to or from the Hub.

    Pushes and loads run these same checks, so a row that loads is one a push could have
    written, and a row can't change how the sandbox runs its environment.

    Args:
        manifest: Every Environment field, with the verifier source inlined.
        trust_image: Whether a custom ``image`` or ``snapshot`` may be used.

    Returns:
        The validated Environment.

    Raises:
        ValueError: If a field is unknown or of the wrong type (there is no coercion);
            the row names a ``verifier_path`` (that file would be read on the loading
            machine); it sets ``env_vars`` (they reach the verifier's and the solution's
            processes, and may hold credentials); ``verifier_mode`` is neither ``auto``
            nor the mode the verifier reads as; a support file name isn't a plain
            relative path; or it names an image or snapshot and *trust_image* is False.
    """
    unknown = manifest.keys() - Environment.model_fields.keys()
    if unknown:
        raise ValueError(f'Unknown environment fields {sorted(unknown)}; upgrade deepgym')
    env = Environment.model_validate(manifest, strict=True)
    if env.verifier_path is not None:
        raise ValueError('A Hub environment must inline its verifier, not name a verifier_path')
    if env.env_vars:
        raise ValueError(
            'env_vars cannot go through the Hub: they reach the verifier and the solution, and '
            "may hold credentials. Clear them (env.model_copy(update={'env_vars': None})) and "
            'set them where the environment runs'
        )
    detected = detect_verifier_mode(env.verifier_code)
    if env.verifier_mode not in ('auto', detected):
        raise ValueError(
            f'verifier_mode {env.verifier_mode!r} does not match the verifier, which reads as '
            f'{detected!r}'
        )
    # The model already refuses absolute paths and ..; canonical form also rules out
    # ., empty parts and trailing slashes.
    bad = [
        name
        for name in env.support_files
        if '\\' in name or '\0' in name or posixpath.normpath(name) != name
    ]
    if bad:
        raise ValueError(f'Support file names must be plain relative paths: {bad}')
    if (env.image or env.snapshot) and not trust_image:
        raise ValueError(
            f'The environment runs in {env.image or env.snapshot!r}; pass trust_image=True '
            'to run an image a Hub dataset names'
        )
    return env


def _check_names(names: Sequence[object], source: str) -> None:
    """Refuse missing, blank or repeated environment names, so no environment is lost.

    Environments are keyed by name, so a repeated or blank one would replace another.

    Args:
        names: Every environment's name, in order.
        source: What holds them (a push, a repo id), for the error.

    Raises:
        ValueError: Naming the positions without a name, or the names given twice.
    """
    blank = [i for i, name in enumerate(names) if not isinstance(name, str) or not name.strip()]
    if blank:
        raise ValueError(
            f'{source}: environments at {blank} have no env_name; each needs a name of its own'
        )
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(
            f'{source}: env_name {repeated} names more than one environment; '
            'each needs a name of its own'
        )


def environment_to_dict(env: Environment, env_name: str = '') -> dict[str, str]:
    """Serialise an Environment to a flat dict for HF dataset rows.

    Args:
        env: Environment to serialise.
        env_name: Friendly name for the dataset row (optional).

    Returns:
        Dict with keys env_name, environment (every field as JSON, with the verifier
        source inlined) and schema_version.

    Raises:
        TypeError: If *env* is an Environment subclass, whose extra fields and runner
            the row can't carry.
        ValueError: If loading the row would refuse it (see :func:`_hub_environment`);
            your own image is fine to push.
    """
    if type(env) is not Environment:
        raise TypeError(
            f'{type(env).__name__} cannot be pushed to the Hub: only a plain Environment '
            'round-trips'
        )
    manifest = env.model_dump(mode='json', exclude={'verifier_path'})
    manifest['verifier_code'] = env.verifier_source
    _hub_environment(manifest, trust_image=True)
    return {
        'env_name': env_name,
        'environment': json.dumps(manifest, ensure_ascii=False),
        'schema_version': _SCHEMA_VERSION,
    }


def environment_from_dict(row: Mapping[str, Any], *, trust_image: bool = False) -> Environment:
    """Deserialise an Environment from a HF dataset row.

    Args:
        row: Dict produced by :func:`environment_to_dict` or loaded from HF.
        trust_image: Allow the row's ``image`` or ``snapshot``; refused otherwise.

    Returns:
        Reconstructed Environment.

    Raises:
        ValueError: If the row is in an unknown schema version, holds invalid JSON, or
            fails the checks every Hub environment goes through (see
            :func:`_hub_environment`).
    """
    version = row.get('schema_version', '1.0')
    if version == '1.0':
        manifest = {
            'task': row['task'],
            'verifier_code': row['verifier_code'],
            'timeout': int(row.get('timeout', 30)),
        }
        if row.get('test_cases'):
            manifest['test_cases'] = json.loads(row['test_cases'])
        return _hub_environment(manifest, trust_image=trust_image)
    if version != _SCHEMA_VERSION:
        raise ValueError(f'Unknown environment schema_version {version!r}; upgrade deepgym')
    return _hub_environment(json.loads(row['environment']), trust_image=trust_image)


def push_environment_to_hub(
    env: Environment,
    repo_id: str,
    env_name: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Push a single DeepGym environment to HuggingFace Hub as a dataset.

    Creates or updates the dataset at ``repo_id`` with one row holding the
    whole environment.

    Args:
        env: Environment to push.
        repo_id: HF dataset repo ID, e.g. ``'your-org/deepgym-coin-change'``.
        env_name: Name stored in the dataset row; it must not be blank.
        private: Whether to create the dataset as private.
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Returns:
        The commit the push made: pass it as ``revision`` to load exactly this version.

    Raises:
        ImportError: If ``datasets`` package is not installed.
        TypeError, ValueError: If *env* can't round-trip (see :func:`environment_to_dict`).
        ValueError: If *env_name* is blank.
    """
    return push_environments_to_hub({env_name: env}, repo_id, private=private, token=token)


def push_environments_to_hub(
    envs: dict[str, Environment],
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Push multiple environments to HuggingFace Hub as a single dataset.

    Args:
        envs: Mapping of ``{env_name: Environment}``; no name may be blank.
        repo_id: HF dataset repo ID, e.g. ``'your-org/deepgym-environments'``.
        private: Whether to create the dataset as private.
        token: HF API token (uses HF_TOKEN env var / cached login if None).

    Returns:
        The commit the push made: pass it as ``revision`` to load exactly this version.

    Raises:
        ImportError: If ``datasets`` package is not installed.
        TypeError, ValueError: If an environment can't round-trip (see
            :func:`environment_to_dict`).
        ValueError: If a name is blank.
    """
    _check_names(list(envs), f'Push to {repo_id}')
    datasets = _require_datasets()
    rows = [environment_to_dict(env, env_name=name) for name, env in envs.items()]
    dataset = datasets.Dataset.from_list(rows)
    commit = dataset.push_to_hub(repo_id, private=private, token=token)
    revision = _replace_stale_features(commit, dataset.features, token) or commit.oid
    logger.info('Pushed %d environments to %s at %s', len(rows), repo_id, revision)
    return revision


def _replace_stale_features(
    commit: CommitInfo, features: Features, token: str | None
) -> str | None:
    """Rewrite the dataset card's features if an earlier push left different ones there.

    ``push_to_hub`` keeps the features a repo's card already lists for the config, so
    after a schema 1.0 push the 2.0 rows wouldn't load. The card is read at the push's
    *commit* and fixed on top of it, in the repository the push resolved.

    Returns:
        The commit that fixed the card, or None if it was already right.

    Raises:
        huggingface_hub.errors.HfHubHTTPError: If another push landed after *commit*, so
            the fix can't build on it; push again.
    """
    from datasets.info import DatasetInfosDict  # type: ignore[import-untyped]
    from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

    repo_id = commit.repo_url.repo_id
    readme = hf_hub_download(
        repo_id, 'README.md', repo_type='dataset', revision=commit.oid, token=token
    )
    card = DatasetCard.load(readme)
    infos = DatasetInfosDict.from_dataset_card_data(card.data)
    info = infos.get('default')
    if info is None or info.features == features:
        return None
    info.features = features
    infos.to_dataset_card_data(card.data)
    fixed = HfApi(token=token).create_commit(
        repo_id,
        operations=[
            CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(card).encode())
        ],
        commit_message=f'Update the dataset card to DeepGym environment schema {_SCHEMA_VERSION}',
        repo_type='dataset',
        parent_commit=commit.oid,
    )
    return fixed.oid


def _load_rows(repo_id: str, split: str, revision: str | None, token: str | None) -> Dataset:
    """Load the dataset at *revision*, warning when it isn't pinned."""
    if revision is None:
        logger.warning(
            'Loading %s at its current main, which changes with every push; pass revision= '
            '(the commit a push returned) to load the same environments every time',
            repo_id,
        )
    return _require_datasets().load_dataset(repo_id, split=split, revision=revision, token=token)


def load_environment_from_hub(
    repo_id: str,
    env_name: str | None = None,
    *,
    split: str = 'train',
    revision: str | None = None,
    token: str | None = None,
    trust_image: bool = False,
) -> Environment:
    """Load a DeepGym environment from a HuggingFace Hub dataset.

    Args:
        repo_id: HF dataset repo ID, e.g. ``'deepgym/coin-change'``.
        env_name: If the dataset contains multiple environments, filter by
            this name. If None, loads the first (or only) row.
        split: Dataset split to load from (default: ``'train'``).
        revision: Commit (or branch or tag) to load; None loads the current ``main``.
        token: HF API token (uses HF_TOKEN env var / cached login if None).
        trust_image: Run the image or snapshot the row names; refused otherwise.

    Returns:
        Reconstructed Environment.

    Raises:
        ImportError: If ``datasets`` package is not installed.
        ValueError: If ``env_name`` is specified but not found in the dataset, or the
            dataset's names aren't all non-blank and distinct (as
            :func:`load_all_environments_from_hub` requires), or the row can't be read
            (see :func:`environment_from_dict`).
    """
    dataset = _load_rows(repo_id, split, revision, token)

    if env_name is not None:
        names = [row.get('env_name') for row in dataset]
        _check_names(names, repo_id)
        if env_name not in names:
            raise ValueError(f'Environment {env_name!r} not found in {repo_id}. Available: {names}')
        row = dataset[names.index(env_name)]
    else:
        row = dataset[0]

    return environment_from_dict(row, trust_image=trust_image)


def load_all_environments_from_hub(
    repo_id: str,
    *,
    split: str = 'train',
    revision: str | None = None,
    token: str | None = None,
    trust_image: bool = False,
) -> dict[str, Environment]:
    """Load all environments from a HuggingFace Hub dataset.

    Args:
        repo_id: HF dataset repo ID containing multiple environment rows.
        split: Dataset split to load from (default: ``'train'``).
        revision: Commit (or branch or tag) to load; None loads the current ``main``.
        token: HF API token (uses HF_TOKEN env var / cached login if None).
        trust_image: Run the images or snapshots the rows name; refused otherwise.

    Returns:
        Dict mapping ``env_name`` to ``Environment``.

    Raises:
        ImportError: If ``datasets`` package is not installed.
        ValueError: If a row has no ``env_name`` or shares one with another row, so one
            of them would be lost, or a row can't be read (see :func:`environment_from_dict`).
    """
    dataset = _load_rows(repo_id, split, revision, token)
    names = [row.get('env_name') for row in dataset]
    _check_names(names, repo_id)
    return {
        name: environment_from_dict(row, trust_image=trust_image)
        for name, row in zip(names, dataset, strict=True)
    }


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
