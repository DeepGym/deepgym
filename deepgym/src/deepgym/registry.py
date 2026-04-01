"""Environment registry -- load environments by name from the built-in library."""

from __future__ import annotations

import importlib.resources
import json
import logging
from pathlib import Path

from deepgym.models import Environment

logger = logging.getLogger(__name__)

_SPECIAL_ENV_FACTORIES = {
    'swebench_pro': 'SWEBenchProEnvironment',
    'terminal_bench_2': 'TerminalBenchEnvironment',
}

_DIFFICULTY_LEVELS = frozenset({'easy', 'medium', 'hard'})
_TYPE_SUITES = frozenset({'coding', 'computer-use', 'tool-use'})

# Large benchmarks that are NOT shipped with the wheel.
_BENCHMARK_NAMES = frozenset(
    {
        'bigcodebench',
        'humaneval',
        'humaneval_plus',
        'mbpp',
        'mbpp_plus',
    }
)


def _get_builtin_envs_path() -> Path:
    """Get path to built-in environments shipped with the package."""
    return Path(importlib.resources.files('deepgym')) / 'envs'


def _get_cache_dir() -> Path:
    """Get cache directory for downloaded benchmarks."""
    cache = Path.home() / '.deepgym' / 'environments'
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _iter_registry_json_paths() -> list[Path]:
    """Locate every known registry.json file.

    Returns:
        Existing registry paths in deterministic order.
    """
    builtin = _get_builtin_envs_path() / 'registry.json'
    cache_root = _get_cache_dir()
    candidates = [builtin, cache_root / 'registry.json']
    candidates.extend(
        sorted(
            path
            for path in cache_root.rglob('registry.json')
            if path != cache_root / 'registry.json'
        )
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _normalize_registry_entries(data: object, registry_path: Path) -> list[dict]:
    """Normalize builtin and imported registry formats to one entry list."""
    if isinstance(data, dict):
        entries = data.get('environments')
    elif isinstance(data, list):
        entries = data
    else:
        entries = None

    if not isinstance(entries, list):
        raise KeyError('environments')

    normalized: list[dict] = []
    inferred_benchmark = (
        registry_path.parent.name if registry_path.parent != _get_cache_dir() else None
    )
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        if inferred_benchmark is not None:
            entry.setdefault('benchmark', inferred_benchmark.lower())
        normalized.append(entry)
    return normalized


def _read_registry() -> list[dict]:
    """Read and return the registry.json entries.

    Returns:
        List of environment metadata dicts from the registry.

    Raises:
        DeepGymError: If registry.json cannot be read or parsed.
    """
    from deepgym.exceptions import DeepGymError

    registry_paths = _iter_registry_json_paths()
    if not registry_paths:
        raise DeepGymError('Registry not found. Checked built-in package and cache dir.')

    entries: list[dict] = []
    for registry_path in registry_paths:
        try:
            data = json.loads(registry_path.read_text(encoding='utf-8'))
            entries.extend(_normalize_registry_entries(data, registry_path))
        except (json.JSONDecodeError, KeyError) as exc:
            raise DeepGymError(f'Failed to parse registry {registry_path}: {exc}') from exc
    return entries


def list_environments() -> list[dict]:
    """List all available environments from the registry.

    Returns:
        List of environment metadata dicts with keys: id, name, path,
        difficulty, domain, family, tags.
    """
    return _read_registry()


_NESTED_SUBDIRS = ('computer_use', 'tool_use', 'multi_turn')

# Environments that exist on disk but are not loadable via load_environment().
# These require special runtime support (multi-turn state, web servers, etc.)
# and will crash during dg.run().
_BLOCKED_ENVS = frozenset({'debug_fix', 'navigate_and_check'})


def _find_env_dir(name: str) -> Path | None:
    """Locate an environment directory by name.

    Search order:
        1. Built-in envs shipped with the package (src/deepgym/envs/)
        2. Nested subdirectories (computer_use/, tool_use/, multi_turn/)
        3. Downloaded benchmarks in cache (~/.deepgym/environments/)

    Args:
        name: Environment directory name (e.g. 'coin_change', 'file_organizer').

    Returns:
        Path to the environment directory, or None if not found.
    """
    builtin_root = _get_builtin_envs_path()
    cache_root = _get_cache_dir()
    normalized_name = name.replace('\\', '/')
    rel_path = Path(normalized_name)

    if normalized_name and not rel_path.is_absolute() and '..' not in rel_path.parts:
        direct_candidates = [
            builtin_root / rel_path,
            cache_root / rel_path,
        ]
        if rel_path.parts and rel_path.parts[0] == 'environments':
            direct_candidates.extend(
                [
                    builtin_root.parent / rel_path,
                    cache_root.parent / rel_path,
                    cache_root.joinpath(*rel_path.parts[1:]),
                ]
            )
        for candidate in direct_candidates:
            if candidate.is_dir():
                return candidate

    # 1. Built-in package envs (top-level).
    builtin = builtin_root / normalized_name
    if builtin.is_dir():
        return builtin

    # 2. Check nested subdirectories within built-in envs.
    for subdir in _NESTED_SUBDIRS:
        nested = builtin_root / subdir / normalized_name
        if nested.is_dir():
            return nested

    # 3. Cache directory.
    cached = cache_root / normalized_name
    if cached.is_dir():
        return cached

    # 4. Benchmark cache directories one level down.
    nested_cached = sorted(cache_root.glob(f'*/{normalized_name}'))
    for candidate in nested_cached:
        if candidate.is_dir():
            return candidate

    safe_name = normalized_name.replace('/', '_')
    if safe_name != normalized_name:
        safe_cached = sorted(cache_root.glob(f'*/{safe_name}'))
        for candidate in safe_cached:
            if candidate.is_dir():
                return candidate

    return None


def _name_tokens(name: str) -> set[str]:
    """Generate normalized lookup tokens for an environment name."""
    normalized = name.replace('\\', '/').strip()
    tokens = {normalized, normalized.lower()}
    if '/' in normalized:
        safe = normalized.replace('/', '_')
        tokens.update({safe, safe.lower(), Path(normalized).name, Path(normalized).name.lower()})
    return {token for token in tokens if token}


def _entry_tokens(entry: dict) -> set[str]:
    """Generate normalized lookup tokens for a registry entry."""
    tokens: set[str] = set()
    path_value = str(entry.get('path', '')).strip()
    if path_value:
        path_obj = Path(path_value)
        tokens.update(_name_tokens(path_value))
        tokens.update(_name_tokens(path_obj.name))
        if path_value.startswith('environments/'):
            tokens.update(_name_tokens('/'.join(path_obj.parts[1:])))
    for key in ('id', 'name', 'benchmark'):
        raw = entry.get(key)
        if raw:
            tokens.update(_name_tokens(str(raw)))
    return tokens


def _benchmark_entries(registry: list[dict], benchmark_name: str) -> list[dict]:
    """Return registry entries belonging to a benchmark alias."""
    result = [
        entry
        for entry in registry
        if str(entry.get('benchmark', '')).lower() == benchmark_name
        or f'/{benchmark_name}/' in f'/{str(entry.get("path", "")).lower()}/'
    ]
    return sorted(result, key=lambda entry: str(entry.get('path', '')))


def _load_env_from_dir(env_dir: Path, metadata: dict | None = None) -> Environment:
    """Build an Environment from an environment directory.

    Args:
        env_dir: Path to the environment directory containing task.md and verifier.py.
        metadata: Optional registry metadata dict to populate difficulty/domain/tags.

    Returns:
        Configured Environment ready for dg.run().

    Raises:
        ValueError: If required files are missing.
    """
    task_path = env_dir / 'task.md'
    verifier_path = env_dir / 'verifier.py'
    metadata_path = env_dir / 'metadata.json'

    if not task_path.exists():
        raise ValueError(f'task.md not found in {env_dir}')
    if not verifier_path.exists():
        raise ValueError(f'verifier.py not found in {env_dir}')

    task = task_path.read_text(encoding='utf-8')

    # Merge metadata from file if no registry metadata provided.
    if metadata is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    difficulty = 'medium'
    domain = 'coding'
    tags: list[str] = []
    env_type = 'coding'

    if metadata is not None:
        difficulty = metadata.get('difficulty', difficulty)
        domain = metadata.get('domain', domain)
        tags = metadata.get('tags', tags)
        env_type = metadata.get('type', env_type)

    return Environment(
        name=env_dir.name,
        task=task,
        type=env_type,
        verifier_path=verifier_path,
        difficulty=difficulty,
        domain=domain,
        tags=tags,
    )


def load_environment(name: str) -> Environment:
    """Load an environment by name from the built-in library.

    Search order: registry.json entries, then built-in envs, then cache.

    Args:
        name: Registered environment name (e.g. 'coin_change', 'two_sum').
              Must not contain path separators or start with '.'.

    Returns:
        Configured Environment ready for dg.run().

    Raises:
        ValueError: If environment not found or name looks like a path.
    """
    normalized_name = name.replace('\\', '/')

    # Reject paths that escape the registry namespace.
    if (
        normalized_name.startswith('.')
        or normalized_name.startswith('/')
        or '..' in Path(normalized_name).parts
    ):
        raise ValueError(
            f'Invalid environment name {name!r}. '
            'Use a registered name like "coin_change", not a path.'
        )

    # Reject environments that exist on disk but are not runnable.
    if name in _BLOCKED_ENVS:
        raise ValueError(
            f'Environment {name!r} is not loadable. '
            'It requires special runtime support and cannot be used with dg.run().'
        )

    if name in _SPECIAL_ENV_FACTORIES:
        from deepgym.benchmark_envs import SWEBenchProEnvironment, TerminalBenchEnvironment

        factory_map = {
            'swebench_pro': SWEBenchProEnvironment,
            'terminal_bench_2': TerminalBenchEnvironment,
        }
        return factory_map[name]()

    # 1. Check registry.json for a matching path.
    try:
        registry = _read_registry()
    except Exception:
        registry = []

    benchmark_name = normalized_name.lower()
    if benchmark_name in _BENCHMARK_NAMES:
        benchmark_entries = _benchmark_entries(registry, benchmark_name)
        if benchmark_entries:
            entry = benchmark_entries[0]
            env_dir = _find_env_dir(entry['path'])
            if env_dir is not None:
                return _load_env_from_dir(env_dir, metadata=entry)

    lookup_tokens = _name_tokens(normalized_name)
    for entry in registry:
        if lookup_tokens & _entry_tokens(entry):
            env_dir = _find_env_dir(entry['path'])
            if env_dir is not None:
                return _load_env_from_dir(env_dir, metadata=entry)

    # 2. Direct directory lookup across all sources.
    env_dir = _find_env_dir(normalized_name)
    if env_dir is not None:
        return _load_env_from_dir(env_dir)

    available = [e.get('path', e.get('name')) for e in registry]
    raise ValueError(f"Environment '{name}' not found. Available: {available}")


def load_suite(suite_name: str) -> list[Environment]:
    """Load a suite of environments by difficulty, type, family, or 'all'.

    Args:
        suite_name: One of 'easy', 'medium', 'hard' (difficulty filter),
                    'coding', 'computer-use', 'tool-use' (type filter),
                    'all' (everything), or a family name like 'array-string'.

    Returns:
        List of Environment objects matching the suite criteria.

    Raises:
        ValueError: If no environments match the suite criteria.
    """
    registry = _read_registry()

    if suite_name == 'all':
        entries = registry
    elif suite_name in _DIFFICULTY_LEVELS:
        entries = [e for e in registry if e.get('difficulty') == suite_name]
    elif suite_name in _TYPE_SUITES:
        entries = [e for e in registry if e.get('type', 'coding') == suite_name]
    else:
        # Treat as family name.
        entries = [e for e in registry if e.get('family') == suite_name]

    if not entries:
        families = sorted({e.get('family', '') for e in registry if e.get('family')})
        raise ValueError(
            f"No environments match suite '{suite_name}'. "
            f'Valid suites: easy, medium, hard, coding, computer-use, tool-use, all, '
            f'or families: {families}'
        )

    environments: list[Environment] = []
    for entry in entries:
        env_dir = _find_env_dir(entry['path'])
        if env_dir is not None:
            environments.append(_load_env_from_dir(env_dir, metadata=entry))
        else:
            logger.warning('Environment directory not found: %s (skipped)', entry['path'])

    return environments


def download_benchmark(name: str) -> Path:
    """Download a benchmark dataset. Currently not supported -- use import scripts instead.

    Args:
        name: Benchmark name.

    Returns:
        Path to the cached benchmark directory (if already present).

    Raises:
        ValueError: If benchmark name is not recognized or download is unavailable.
    """
    if name not in _BENCHMARK_NAMES:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {sorted(_BENCHMARK_NAMES)}")

    cache = _get_cache_dir() / name
    if cache.is_dir():
        logger.info('Benchmark %r already cached at %s', name, cache)
        return cache

    raise ValueError(
        f'Automatic download for {name!r} is not yet available. '
        f'Use the import scripts instead: python scripts/import_{name}.py'
    )
