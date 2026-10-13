"""Environment registry -- load environments by name from the built-in library."""

from __future__ import annotations

import importlib.resources
import json
import logging
from pathlib import Path
from string import Template

from deepgym.computer_use import ComputerUseEnvironment
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


def entry_label(entry: dict) -> str:
    """Registry path without the 'environments/' prefix, e.g. 'humaneval_plus/HumanEval_0'."""
    return str(entry.get('path', '')).removeprefix('environments/')


def _entry_keys(entry: dict) -> set[str]:
    """Fully qualified names that pick exactly this entry (slash/underscore-insensitive)."""
    names = {str(entry.get('path', '')), entry_label(entry)}
    if entry.get('benchmark') and entry.get('id'):
        names.add(f'{entry["benchmark"]}/{entry["id"]}')
    return {name.replace('/', '_').lower() for name in names if name}


def _benchmark_entries(registry: list[dict], benchmark_name: str) -> list[dict]:
    """Return registry entries belonging to a benchmark alias."""
    result = [
        entry
        for entry in registry
        if str(entry.get('benchmark', '')).lower() == benchmark_name
        or f'/{benchmark_name}/' in f'/{str(entry.get("path", "")).lower()}/'
    ]
    return sorted(result, key=lambda entry: str(entry.get('path', '')))


_COMPUTER_USE_SPEC_FILES = frozenset(
    {'task.md', 'verifier.py', 'metadata.json', 'setup.sh', 'variants.json'}
)


def _read_text_files(root: Path, prefix: str = '') -> dict[str, str]:
    """Map every text file under *root* (skipping __pycache__) to its contents."""
    return {
        f'{prefix}{path.relative_to(root).as_posix()}': path.read_text(encoding='utf-8')
        for path in sorted(root.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts
    }


def _load_computer_use_env(env_dir: Path, metadata: dict, variant: int) -> ComputerUseEnvironment:
    """Build a ComputerUseEnvironment from task.md, verifier.py, setup.sh, and app files.

    Every other file in the directory, plus the sibling ``_shared/`` runtime files,
    is written into the session work directory before setup runs. The verifier gets
    ``_checks.py`` and the variant's ``PARAMS`` prepended rather than importing
    anything from disk, so nothing the agent can reach is executed at scoring time.
    ``$name`` placeholders in task.md are filled from the same params.

    Raises:
        ValueError: If *variant* is out of range.
    """
    variants_path = env_dir / 'variants.json'
    variants = (
        json.loads(variants_path.read_text(encoding='utf-8')) if variants_path.exists() else [{}]
    )
    if not 0 <= variant < len(variants):
        raise ValueError(f'{env_dir.name} has {len(variants)} variants, got variant={variant}')
    params = variants[variant]
    files = _read_text_files(env_dir.parent / '_shared', prefix='_shared/')
    files |= {
        rel: text
        for rel, text in _read_text_files(env_dir).items()
        if rel not in _COMPUTER_USE_SPEC_FILES
    }
    setup_path = env_dir / 'setup.sh'
    return ComputerUseEnvironment(
        name=env_dir.name,
        task=Template((env_dir / 'task.md').read_text(encoding='utf-8')).substitute(params),
        final_verifier_code=(env_dir.parent / '_checks.py').read_text(encoding='utf-8')
        + f'\n\nPARAMS = {params!r}\n\n'
        + (env_dir / 'verifier.py').read_text(encoding='utf-8'),
        setup_script=setup_path.read_text(encoding='utf-8') if setup_path.exists() else '',
        files=files,
        **{
            key: metadata[key]
            for key in ('difficulty', 'domain', 'tags', 'max_steps', 'timeout_per_step', 'image')
            if key in metadata
        },
    )


def _load_env_from_dir(
    env_dir: Path, metadata: dict | None = None, variant: int = 0
) -> Environment | ComputerUseEnvironment:
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
    test_patch_path = env_dir / 'test_patch.diff'
    if test_patch_path.exists():
        return _load_repo_patch_env(env_dir, test_patch_path, metadata)
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

    if env_type == 'computer-use':
        file_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        return _load_computer_use_env(env_dir, {**file_metadata, **(metadata or {})}, variant)
    if variant:
        raise ValueError(f'{env_dir.name} has no variants; only computer-use envs do')

    file_metadata = (
        json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    )
    settings = {**file_metadata, **(metadata or {})}
    return Environment(
        name=env_dir.name,
        task=task,
        type=env_type,
        verifier_path=verifier_path,
        difficulty=difficulty,
        domain=domain,
        tags=tags,
        support_files=_read_support_files(env_dir, settings.get('support_files', [])),
        **{
            key: settings[key]
            for key in ('verifier_mode', 'timeout', 'case_timeout', 'image')
            if key in settings
        },
    )


_REPO_PATCH_KEYS = ('repo_dir', 'test_command', 'timeout', 'image', 'difficulty', 'domain', 'tags')


def _load_repo_patch_env(
    env_dir: Path, test_patch_path: Path, metadata: dict | None
) -> Environment:
    """Build a RepoPatchEnvironment from task.md, test_patch.diff and metadata.json.

    The repo and its test command live in the image the metadata names; see
    :class:`~deepgym.benchmark_envs.RepoPatchEnvironment`.
    """
    from deepgym.benchmark_envs import RepoPatchEnvironment

    file_metadata = json.loads((env_dir / 'metadata.json').read_text(encoding='utf-8'))
    settings = {**file_metadata, **(metadata or {})}
    return RepoPatchEnvironment(
        name=env_dir.name,
        task=(env_dir / 'task.md').read_text(encoding='utf-8'),
        # Bytes, not text mode: a patch to a CRLF file must keep its \r characters.
        test_patch=test_patch_path.read_bytes().decode('utf-8'),
        **{key: settings[key] for key in _REPO_PATCH_KEYS if key in settings},
    )


def _read_support_files(env_dir: Path, paths: list[str]) -> dict[str, str]:
    """Load helper modules listed in metadata, keyed by file name.

    Paths are relative to the env directory and may point at shared code
    elsewhere in the same environments root (e.g. ``../_shared/scoring.py``).

    Raises:
        ValueError: If a path escapes the environments root or two names collide.
    """
    roots = [_get_builtin_envs_path().resolve(), _get_cache_dir().resolve()]
    root = next((r for r in roots if env_dir.resolve().is_relative_to(r)), env_dir.resolve())
    files: dict[str, str] = {}
    for rel in paths:
        path = (env_dir / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'support file {rel!r} for {env_dir.name} is not a file under {root}')
        if path.name in files:
            raise ValueError(f'two support files for {env_dir.name} are named {path.name!r}')
        files[path.name] = path.read_text(encoding='utf-8')
    return files


def load_environment(name: str, variant: int = 0) -> Environment | ComputerUseEnvironment:
    """Load an environment by name from the built-in library.

    Search order: registry.json entries, then built-in envs, then cache.

    Args:
        name: Registered environment name (e.g. 'coin_change', 'two_sum').
              Must not contain path separators or start with '.'.
        variant: Which parameter set of a computer-use env to load (see the
            'variants' count in list_environments()). Other envs only have 0.

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
    known_benchmarks = _BENCHMARK_NAMES | {str(e.get('benchmark', '')).lower() for e in registry}
    if benchmark_name in known_benchmarks:
        benchmark_entries = _benchmark_entries(registry, benchmark_name)
        if benchmark_entries:
            entry = benchmark_entries[0]
            env_dir = _find_env_dir(entry['path'])
            if env_dir is not None:
                return _load_env_from_dir(env_dir, metadata=entry, variant=variant)

    # Benchmarks share task ids (humaneval vs humaneval_plus both have HumanEval_0),
    # so a fully qualified name beats loose token matches, and a name that still
    # fits several environments is an error rather than a silent first pick.
    lookup_tokens = _name_tokens(normalized_name)
    matches = [entry for entry in registry if lookup_tokens & _entry_tokens(entry)]
    qualified = normalized_name.replace('/', '_').lower()
    exact = [entry for entry in matches if qualified in _entry_keys(entry)]
    candidates: dict[Path, tuple[Path, dict]] = {}
    for entry in exact or matches:
        env_dir = _find_env_dir(entry['path'])
        if env_dir is not None:
            candidates.setdefault(env_dir.resolve(), (env_dir, entry))
    if len(candidates) > 1:
        options = sorted(entry_label(entry) for _, entry in candidates.values())
        raise ValueError(f'Environment name {name!r} is ambiguous; use one of: {options}')
    if candidates:
        env_dir, entry = next(iter(candidates.values()))
        return _load_env_from_dir(env_dir, metadata=entry, variant=variant)

    # 2. Direct directory lookup across all sources.
    env_dir = _find_env_dir(normalized_name)
    if env_dir is not None:
        return _load_env_from_dir(env_dir, variant=variant)

    available = [e.get('path', e.get('name')) for e in registry]
    raise ValueError(f"Environment '{name}' not found. Available: {available}")


def load_suite(suite_name: str) -> list[Environment | ComputerUseEnvironment]:
    """Load a suite of environments by difficulty, type, family, or 'all'.

    Computer-use environments need an agent rather than a single output, so
    they only appear in the 'computer-use' suite.

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

    if suite_name != 'computer-use':
        entries = [e for e in entries if e.get('type') != 'computer-use']

    if not entries:
        families = sorted({e.get('family', '') for e in registry if e.get('family')})
        raise ValueError(
            f"No environments match suite '{suite_name}'. "
            f'Valid suites: easy, medium, hard, coding, computer-use, tool-use, all, '
            f'or families: {families}'
        )

    environments: list[Environment | ComputerUseEnvironment] = []
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
