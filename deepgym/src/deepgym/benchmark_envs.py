# ruff: noqa: E501, Q001
"""Benchmark-backed environments and custom execution helpers."""

from __future__ import annotations

import json
import random
import re
import textwrap
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

from deepgym.exceptions import DeepGymError
from deepgym.harbor import HarborEnvironment, HarborTask, load_tasks
from deepgym.integrations.hf import _require_datasets
from deepgym.models import Environment, RunResult, VerifierResult
from deepgym.sandbox import _extract_json_line, build_run_result

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_RUNNER_PATH = '/home/user/deepgym_runner.py'
_TASK_PATH = '/home/user/deepgym_task.json'
_PATCH_PATH = '/home/user/deepgym_candidate.diff'
_ARCHIVE_PATH = '/home/user/deepgym_task.tar.gz'

_PATCH_BLOCK_RE = re.compile(r'```(?:diff|patch)?\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)
_JSON_LIST_RE = re.compile(r'^\s*\[', re.DOTALL)
_TASK_ROUTE_KEYS = (
    'task',
    'task_spec',
    'instance',
    'instance_spec',
    'task_row',
    'row',
)
_MIXED_ROUTE_KEYS = (
    'environment_name',
    'env_name',
    'task_type',
    'benchmark',
    'dataset_name',
    'suite',
    'env',
)


@dataclass(slots=True)
class ResolvedRun:
    """A single expanded run request."""

    env: Environment
    output: str
    kwargs: dict[str, Any]


@dataclass(slots=True)
class SWEBenchTask:
    """Normalized SWE-bench Pro task metadata."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    selected_test_files_to_run: list[str]
    before_repo_set_cmd: str = ''
    repo_language: str = ''
    test_commands: list[str] | None = None
    repo_source: str | None = None


def _coerce_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and _JSON_LIST_RE.match(value.strip()):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _split_batch_kwargs(batch_size: int, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    per_item = [dict() for _ in range(batch_size)]
    for key, value in kwargs.items():
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            if len(value) != batch_size:
                raise ValueError(
                    f'Keyword argument {key!r} has {len(value)} items, '
                    f'but batch size is {batch_size}.'
                )
            for index, item in enumerate(value):
                per_item[index][key] = item
            continue
        for item_kwargs in per_item:
            item_kwargs[key] = value
    return per_item


def _build_plain_requests(
    env: Environment,
    outputs: Sequence[str],
    kwargs: dict[str, Any],
) -> list[ResolvedRun]:
    per_item_kwargs = _split_batch_kwargs(len(outputs), kwargs)
    return [
        ResolvedRun(env=env, output=output, kwargs=per_item_kwargs[i])
        for i, output in enumerate(outputs)
    ]


def build_run_requests(
    env: Environment,
    outputs: Sequence[str],
    kwargs: dict[str, Any],
) -> list[ResolvedRun]:
    """Expand a batch request into per-sample environment runs."""
    if hasattr(env, 'prepare_batch_requests'):
        return env.prepare_batch_requests(outputs, **kwargs)  # type: ignore[return-value]
    return _build_plain_requests(env, outputs, kwargs)


def _infer_environment_name(env: Environment) -> str:
    if env.name:
        return env.name
    if env.verifier_path is not None:
        return env.verifier_path.parent.name
    return env.domain or env.type


def _environment_aliases(env: Environment) -> set[str]:
    aliases = {_infer_environment_name(env)}
    aliases.update({env.type, env.domain})
    aliases.update(env.tags)
    if isinstance(env, SWEBenchProEnvironment):
        aliases.update({'swebench_pro', 'swebench', env.dataset_id})
    if isinstance(env, TerminalBenchEnvironment):
        aliases.update({'terminal_bench_2', 'terminal-bench-2', env.dataset_id})
    return {alias for alias in aliases if alias}


def _parse_patch_from_model_output(model_output: str) -> str:
    match = _PATCH_BLOCK_RE.search(model_output)
    # Trim only blank lines at the end: a diff's last line may be a context line that
    # is a single space, and stripping it corrupts the hunk.
    patch = (match.group(1) if match else model_output).lstrip().rstrip('\n')
    if not match and not patch.startswith(('diff --git', '--- ', '*** ')):
        return ''
    return patch + '\n' if patch else ''


def _shell_snippet(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _verifier_result_from_output(
    stdout: str,
    stderr: str,
    *,
    exit_code: int,
) -> VerifierResult:
    json_str = _extract_json_line(stdout)
    if json_str is None:
        raise ValueError(
            f'Custom benchmark runner did not emit JSON. exit_code={exit_code}, '
            f'stdout={stdout!r}, stderr={stderr!r}'
        )
    return VerifierResult.model_validate_json(json_str)


def _run_sandbox_runner(
    env: Environment,
    dg: DeepGym,
    script: str,
    payload: dict[str, Any],
    *,
    patch_text: str | None = None,
    archive_path: Path | None = None,
    timeout: int,
) -> tuple[VerifierResult, int, str]:
    """Run *script* as root in a fresh sandbox from *dg*'s provider, booted from *env*."""
    sandbox = dg.open_sandbox(env)
    try:
        sandbox.upload(script.encode('utf-8'), _RUNNER_PATH)
        sandbox.upload(json.dumps(payload, ensure_ascii=False).encode('utf-8'), _TASK_PATH)

        command = f'python3 {_RUNNER_PATH} {_TASK_PATH}'
        if patch_text is not None:
            sandbox.upload(patch_text.encode('utf-8'), _PATCH_PATH)
            command += f' {_PATCH_PATH}'

        if archive_path is not None:
            sandbox.upload(archive_path.read_bytes(), _ARCHIVE_PATH)
            command += f' {_ARCHIVE_PATH}'

        response = sandbox.exec(command, timeout=timeout)
        result = _verifier_result_from_output(response.output, '', exit_code=response.exit_code)
        return result, response.exit_code, sandbox.id
    finally:
        sandbox.delete()


def _execute_runner(
    env: Environment,
    dg: DeepGym,
    script: str,
    payload: dict[str, Any],
    *,
    patch_text: str | None = None,
    archive_path: Path | None = None,
    timeout: int,
) -> RunResult:
    """Run a self-contained benchmark runner in a fresh sandbox and time it.

    Raises:
        SandboxError: If *dg* runs locally. These runners apply model-written
            patches and run commands and tests outside any judge isolation, so they
            only ever run inside a container or VM.
    """
    start = time.perf_counter()
    result, exit_code, sandbox_id = _run_sandbox_runner(
        env,
        dg,
        script,
        payload,
        patch_text=patch_text,
        archive_path=archive_path,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return build_run_result(result, elapsed_ms, sandbox_id, exit_code=exit_code)


class PatchVerifier:
    """Reusable repo-level patch verifier."""

    @staticmethod
    def extract_patch(model_output: str) -> str:
        """Extract a unified diff from fenced or raw model output."""
        return _parse_patch_from_model_output(model_output)

    @staticmethod
    def build_runner_script() -> str:
        """Return the self-contained patch runner script."""
        return textwrap.dedent(
            """
            import json
            import os
            import re
            import shutil
            import subprocess
            import sys
            import tempfile
            from pathlib import Path


            def _emit(result: dict, exit_code: int) -> int:
                print(json.dumps(result))
                return exit_code


            def _tail(text: str, limit: int = 400) -> str:
                text = text.strip()
                if len(text) <= limit:
                    return text
                return text[-limit:]


            def _run(command: str, *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    command,
                    shell=True,
                    cwd=str(cwd) if cwd is not None else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )


            # Test infrastructure a model's diff must not touch outside the hidden test
            # patch's own paths (those are put back anyway): the tests, their helpers,
            # data and snapshots, everything a test runner or interpreter loads by name
            # or from its config, and the installed test harness and dependencies.
            #
            # This is a stopgap denylist and can never be complete: there will always be
            # another file a runner autoloads, and it cannot stop a diff to the code under
            # test that forces an exit (os._exit / atexit at import), which the exit-code
            # reward still trusts. Do not keep growing it indefinitely. The real defence is
            # the per-test report gate (verdict from the runner's own report, fail closed),
            # which supersedes this; add new frameworks there, not here.
            # Directories whose *executable* contents a runner collects or loads. A data
            # fixture here (.yml, .json, .golden) is not refused -- real fixes edit those,
            # and weakening a data oracle is a separate problem the report gate addresses.
            # 'testing' is left out: it is as often a shipped package (django.test, @proton/
            # testing) as a test dir, and refusing it gives correct fixes a false zero.
            _TEST_DIRS = {
                'test', 'tests', '__tests__', 'testdata', 'spec', 'specs',
                '__snapshots__', '__mocks__',
            }
            _HARNESS_DIRS = {
                'node_modules', 'site-packages', 'dist-packages', 'vendor', '.pytest_cache',
                '.tox', '.nox', '_pytest', '.venv', 'venv', 'virtualenv', '.cargo',
            }
            # Build scripts and runner config that execute code: `make test`, a gradle/maven
            # test task or a build.rs can be rewritten to exit 0 without running the tests, so
            # any edit is refused. (A lockfile or go.sum cannot forge a pass on its own -- the
            # stub package a repoint needs is caught by its own runner `bin` -- so those stay
            # allowed, as a real dependency fix updates them.)
            _INFRA_FILES = {
                'conftest.py', 'pytest.py', 'pytest.ini', '.pytest.ini', 'pytest.toml',
                '.pytest.toml', 'tox.ini', 'noxfile.py', 'tests.py', 'sitecustomize.py',
                'usercustomize.py', '.pythonrc', '.pythonrc.py', '.pythonstartup', '.rspec',
                'phpunit.xml', 'phpunit.xml.dist', 'phpunit.dist.xml', 'pom.xml', 'build.gradle',
                'build.gradle.kts', 'settings.gradle', 'settings.gradle.kts', 'build.rs',
                'rakefile', 'makefile', 'gnumakefile', '.npmrc', '.yarnrc', '.yarnrc.yml',
                'setup.py',
            }
            # Executable extensions: a code file inside a test directory can run at import or
            # be imported by a test, so it is refused there; a data file is not.
            _CODE_EXTS = {
                'py', 'pyi', 'js', 'jsx', 'ts', 'tsx', 'mjs', 'cjs', 'mts', 'cts', 'go', 'rb',
                'rs', 'java', 'kt', 'kts', 'php', 'sh', 'bash', 'c', 'cc', 'cpp', 'h', 'hpp',
            }
            _JS_TOOLS = {
                'jest', 'vitest', 'vite', 'karma', 'babel', 'ava', 'playwright', 'cypress',
                'mocha', 'nyc', 'wdio',
            }
            _JS_EXTS = {'js', 'jsx', 'ts', 'tsx', 'mjs', 'cjs', 'mts', 'cts'}
            # package.json keys that run or expose a test runner; the rest of the file, including
            # the dependency ranges a real fix updates, is left alone. Repointing a runner
            # dependency to a stub still needs that stub package to declare a 'bin', which this
            # catches on the stub's own package.json, and the report gate is the backstop.
            _PACKAGE_KEYS = (
                'scripts', 'bin', 'jest', 'mocha', 'ava', 'babel', 'nyc', 'c8', 'vitest', 'tap',
                'wdio',
            )
            # Config files that are test infrastructure only in part; _config_tampered decides.
            _CONFIG_FILES = {'pyproject.toml', 'setup.cfg', 'package.json', 'cargo.toml'}
            _SYMLINK = b'\\0deepgym-symlink'


            def _components(rel: str) -> list[str]:
                # Lowercased path components, dropping '' and '.' (git -z paths are raw, normalized).
                return [part for part in rel.lower().split('/') if part not in ('', '.')]


            def _is_test_file(name: str) -> bool:
                # Whether *name* (lowercased basename) is itself a test, a test runner config,
                # or an import hook a runner loads by name -- refused wherever it sits.
                segments = name.split('.')
                return (
                    name in _INFRA_FILES
                    or name.endswith(('_test.py', '_test.go', '_spec.rb', '.pth'))
                    or (name.startswith('test_') and name.endswith('.py'))
                    or (segments[0] in ('test', 'tests', 'spec', 'suite_test') and segments[-1] == 'go')
                    or (len(segments) > 2 and segments[-2] in ('test', 'spec', 'e2e', 'setup') and segments[-1] in _JS_EXTS)
                    or (len(segments) > 1 and segments[0] in _JS_TOOLS and segments[1] in ('config', 'conf', 'setup', 'workspace', 'preset'))
                    or segments[0] in ('setuptests', 'jest', 'vitest')
                    or name.startswith(('.babelrc', '.mocharc', '.nycrc', '.c8rc'))
                )


            def _infra_reason(repo_root: Path, rel: str) -> str | None:
                # Why *rel* is test infrastructure judging by its path alone, or None.
                parts = _components(rel)
                if not parts:
                    return None
                name, dirs = parts[-1], parts[:-1]
                if any(
                    part in _HARNESS_DIRS or part.endswith(('.egg-info', '.dist-info', '.egg'))
                    for part in parts
                ):
                    return 'test harness or installed dependency'
                raw = rel.split('/')
                if any((repo_root / Path(*raw[:depth]) / 'pyvenv.cfg').is_file() for depth in range(len(raw))):
                    return 'inside a virtualenv'
                if _is_test_file(name):
                    return 'test file or test configuration'
                # In a test directory, refuse executable code (a helper a test imports, or a
                # module that runs at import) and the package marker, but allow data fixtures.
                if any(part in _TEST_DIRS for part in dirs):
                    if name == '__init__.py':
                        return 'test package marker'
                    if name.rsplit('.', 1)[-1] in _CODE_EXTS and '.' in name:
                        return 'code in a test directory'
                return None


            def _toml_header_path(header: str) -> tuple[str, ...]:
                # The normalized dotted path of a TOML table header, with quotes stripped, so
                # [tool."pytest".ini_options] and [tool.pytest.ini_options] compare equal.
                segments, current, quote = [], '', ''
                for char in header:
                    if quote:
                        if char == quote:
                            quote = ''
                        else:
                            current += char
                    elif char in '"\\'':
                        quote = char
                    elif char == '.':
                        segments.append(current.strip())
                        current = ''
                    else:
                        current += char
                segments.append(current.strip())
                return tuple(seg for seg in segments if seg)


            def _toml_section_lines(text: str, prefixes: tuple[tuple[str, ...], ...]) -> list:
                # The content lines grouped under any table whose normalized path starts with a
                # prefix. A pure-text TOML reader, so it needs no tomllib (absent before Python
                # 3.11, and the runner runs in the image's interpreter). Package-level keys
                # (prefix of length 1 matching the table exactly) keep only auto*/build lines.
                out, table = [], ()
                for raw in text.splitlines():
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('[['):
                        table = _toml_header_path(line[2 : line.find(']]')])
                    elif line.startswith('['):
                        table = _toml_header_path(line[1 : line.find(']')])
                    elif any(table[: len(prefix)] == prefix for prefix in prefixes):
                        key = line.split('=', 1)[0].strip().strip('"\\'')
                        if table == ('package',) and not key.startswith(('auto', 'build')):
                            continue
                        out.append((table, line))
                return sorted(out)


            def _test_part_changed(name: str, old_text: str, new_text: str) -> bool:
                # Whether the test-relevant part of a config file changed. Raises on anything it
                # cannot parse or compare; the caller turns that into tampering (fails closed).
                if name == 'package.json':
                    before = json.loads(old_text or '{}')
                    after = json.loads(new_text or '{}')
                    before = before if isinstance(before, dict) else {}
                    after = after if isinstance(after, dict) else {}
                    return any(before.get(key) != after.get(key) for key in _PACKAGE_KEYS)
                if name == 'setup.cfg':
                    import configparser

                    # interpolation=None: a '%(x)s' value must not raise; optionxform=str: pytest
                    # reads option names case-sensitively, so a case-only change still counts.
                    before, after = (configparser.ConfigParser(interpolation=None) for _ in range(2))
                    before.optionxform = after.optionxform = str
                    before.read_string(old_text)
                    after.read_string(new_text)
                    # [options] holds metadata and dependencies a real fix may change; only its
                    # entry-points subsection registers pytest plugins, so that is what counts.
                    keys = ('tool:pytest', 'options.entry_points', 'nosetests', 'tox:tox')

                    def view(cfg):
                        return {key: dict(cfg[key]) for key in cfg.sections() if key in keys}

                    return view(before) != view(after)
                if name == 'cargo.toml':
                    # Target tables and workspace selection decide which tests cargo runs, and
                    # [package] auto-discovery/build toggles do too.
                    prefixes = (('lib',), ('bin',), ('test',), ('bench',), ('workspace',), ('package',))
                else:
                    # pyproject.toml: the pytest/coverage/tox/nose tool tables and the build system.
                    prefixes = (
                        ('tool', 'pytest'), ('tool', 'coverage'), ('tool', 'tox'),
                        ('tool', 'nose'), ('build-system',),
                    )
                return _toml_section_lines(old_text, prefixes) != _toml_section_lines(new_text, prefixes)


            def _config_tampered(name: str, old: bytes | None, new: bytes | None) -> bool:
                # Whether turning *old* into *new* changes the test-relevant part of a config file.
                # Fails closed: a symlink, or anything that cannot be parsed and shown unchanged,
                # counts as tampering, so a parse error becomes a trained zero, never a masked run.
                if _SYMLINK in (old, new):
                    return True
                old_text = (old or b'').decode('utf-8', 'replace')
                new_text = (new or b'').decode('utf-8', 'replace')
                if old_text == new_text:
                    return False
                try:
                    return _test_part_changed(name, old_text, new_text)
                except Exception:
                    return True


            def _read_config(path: Path) -> bytes | None:
                # A config file's bytes; a symlink reads as a sentinel no real file has.
                if path.is_symlink():
                    return _SYMLINK
                return path.read_bytes() if path.is_file() else None


            _MODE_RE = re.compile(r'^(old mode|new mode|new file mode|deleted file mode)\\s+([0-7]+)\\s*$')


            def _mode_change_paths(patch_text: str) -> list[str]:
                # The files whose mode changes or that become symlinks. A real code fix edits
                # content; a mode flip or a symlink (which can aim a protected path at attacker
                # content) is refused outright. The numeric mode is parsed so non-canonical git
                # spellings (0120000, trailing space) cannot slip a symlink past an exact match.
                flagged, current = [], '(file)'
                for line in patch_text.splitlines():
                    if line.startswith('diff --git '):
                        current = line[len('diff --git ') :]
                        continue
                    match = _MODE_RE.match(line)
                    if match is None:
                        continue
                    kind, mode = match.group(1), int(match.group(2), 8)
                    if kind in ('old mode', 'new mode') or (mode & 0o170000) == 0o120000:
                        flagged.append(current)
                return list(dict.fromkeys(flagged))


            def _git_root(repo_dir: Path, timeout: int) -> Path:
                # The repository top-level: numstat reports paths relative to it, not to repo_dir.
                # git appends exactly one newline, so strip only that -- a path may legitimately
                # end in a space or '\\r', and either .strip() or text-mode newline translation
                # would then name a different, missing directory. Read bytes and trim one '\\n'.
                proc = subprocess.run(
                    ['git', '-C', str(repo_dir), 'rev-parse', '--show-toplevel'],
                    capture_output=True,
                    timeout=timeout,
                )
                raw = proc.stdout[:-1] if proc.stdout.endswith(b'\\n') else proc.stdout
                top = os.fsdecode(raw)
                return Path(top) if proc.returncode == 0 and top else repo_dir


            def _patch_paths(repo_dir: Path, patch_file: Path, timeout: int) -> list[str] | None:
                # Every path a patch reads or writes, as git's own parser sees it: --numstat
                # names each post-image path, -R --numstat each pre-image one, and -z emits raw
                # (unquoted) bytes so non-ASCII names are not mis-parsed.
                paths = []
                for reverse in ([], ['-R']):
                    proc = subprocess.run(
                        ['git', '-C', str(repo_dir), 'apply', *reverse, '--numstat', '-z', str(patch_file)],
                        capture_output=True,
                        timeout=timeout,
                    )
                    if proc.returncode != 0:
                        return None
                    records = proc.stdout.decode('utf-8', 'surrogateescape').split('\\0')
                    for record in records:
                        fields = record.split('\\t', 2)
                        # A path field present (3 fields, non-empty) is the changed path; a bare
                        # record is a rename endpoint from the -z old/new form.
                        if len(fields) == 3 and fields[2]:
                            paths.append(fields[2])
                        elif len(fields) == 1 and record:
                            paths.append(record)
                return list(dict.fromkeys(paths))


            def _snapshot(repo_dir: Path, paths: list[str]) -> dict:
                saved = {}
                for rel in paths:
                    path = repo_dir / rel
                    if path.is_symlink():
                        saved[rel] = ('link', os.readlink(path))
                    elif path.is_file():
                        saved[rel] = ('file', path.read_bytes(), path.stat().st_mode)
                    elif not path.exists():
                        saved[rel] = None
                return saved


            def _restore(repo_dir: Path, saved: dict) -> str | None:
                # Put each hidden path back as it was before the model's diff, whatever the
                # diff did to it (edit, rename, delete, replace with a link or a directory).
                try:
                    for rel, state in saved.items():
                        path = repo_dir / rel
                        for ancestor in Path(rel).parents[:-1]:
                            folder = repo_dir / ancestor
                            if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
                                return f'{ancestor} is no longer a directory'
                        if path.is_symlink() or path.is_file():
                            path.unlink()
                        elif path.exists():
                            shutil.rmtree(path)
                        if state is None:
                            continue
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if state[0] == 'link':
                            os.symlink(state[1], path)
                        else:
                            path.write_bytes(state[1])
                            os.chmod(path, state[2])
                except OSError as exc:
                    return str(exc)
                return None


            def main() -> int:
                task = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
                patch_path = Path(sys.argv[2])
                patch_text = patch_path.read_text(encoding='utf-8').strip()

                if not task.get('test_commands'):
                    # No real test runner: no verdict, so trainers mask the sample
                    # instead of rewarding a patch nothing checked.
                    return _emit(
                        {
                            'schema_version': '1.0',
                            'score': 0.0,
                            'passed': False,
                            'details': 'No test runner for this task, so it was not scored.',
                            'reward_components': {'apply': 0.0, 'tests': 0.0},
                            'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                            'truncated': False,
                            'error_type': 'unsupported_runner',
                            'cases': [],
                        },
                        1,
                    )

                if not patch_text:
                    return _emit(
                        {
                            'schema_version': '1.0',
                            'score': 0.0,
                            'passed': False,
                            'details': 'Model output did not contain a unified diff patch.',
                            'reward_components': {'apply': 0.0, 'tests': 0.0},
                            'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                            'truncated': False,
                            'error_type': 'invalid_patch',
                            'cases': [],
                        },
                        1,
                    )

                timeout = int(task.get('timeout', 1800))
                test_commands = task['test_commands']
                setup_commands = task.get('setup_commands', '')
                test_patch = task.get('test_patch', '')
                apply_credit = float(task.get('apply_credit', 0.3))

                with tempfile.TemporaryDirectory(prefix='deepgym_patch_repo_') as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    # A repo_dir is already checked out (e.g. baked into the task's image).
                    repo_dir = Path(task.get('repo_dir') or temp_dir / 'repo')

                    if not task.get('repo_dir'):
                        clone = subprocess.run(
                            ['git', 'clone', task['repo_source'], str(repo_dir)],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        if clone.returncode != 0:
                            return _emit(
                                {
                                    'schema_version': '1.0',
                                    'score': 0.0,
                                    'passed': False,
                                    'details': f'git clone failed: {_tail(clone.stderr or clone.stdout)}',
                                    'reward_components': {'apply': 0.0, 'tests': 0.0},
                                    'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                    'truncated': False,
                                    'error_type': 'clone_failed',
                                    'cases': [],
                                },
                                1,
                            )

                        checkout = subprocess.run(
                            ['git', '-C', str(repo_dir), 'checkout', task['base_commit']],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        if checkout.returncode != 0:
                            return _emit(
                                {
                                    'schema_version': '1.0',
                                    'score': 0.0,
                                    'passed': False,
                                    'details': f'git checkout failed: {_tail(checkout.stderr or checkout.stdout)}',
                                    'reward_components': {'apply': 0.0, 'tests': 0.0},
                                    'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                    'truncated': False,
                                    'error_type': 'checkout_failed',
                                    'cases': [],
                                },
                                1,
                            )

                    if setup_commands:
                        setup = _run(setup_commands, cwd=repo_dir, timeout=timeout)
                        if setup.returncode != 0:
                            return _emit(
                                {
                                    'schema_version': '1.0',
                                    'score': 0.0,
                                    'passed': False,
                                    'details': f'setup command failed: {_tail(setup.stderr or setup.stdout)}',
                                    'reward_components': {'apply': 0.0, 'tests': 0.0},
                                    'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                    'truncated': False,
                                    'error_type': 'setup_failed',
                                    'cases': [],
                                },
                                1,
                            )

                    # The model's diff goes first. Every path the hidden test patch touches
                    # is then put back to its state before that diff, and the test patch
                    # goes last, so no spelling of a path in the diff can change the tests.
                    test_patch_file = temp_dir / 'test_patch.diff'
                    test_patch_file.write_text(test_patch, encoding='utf-8')
                    hidden = _patch_paths(repo_dir, test_patch_file, timeout) if test_patch else []
                    if hidden is None:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': 'git cannot parse the test patch.',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'test_patch_failed',
                                'cases': [],
                            },
                            1,
                        )
                    saved = _snapshot(repo_dir, hidden)

                    check = subprocess.run(
                        ['git', '-C', str(repo_dir), 'apply', '--check', str(patch_path)],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if check.returncode != 0:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': f'Patch does not apply cleanly: {_tail(check.stderr or check.stdout)}',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'patch_apply_failed',
                                'cases': [
                                    {
                                        'id': 'git_apply_check',
                                        'passed': False,
                                        'score': 0.0,
                                        'input_summary': 'git apply --check',
                                        'actual_summary': _tail(check.stdout),
                                        'error': _tail(check.stderr),
                                        'execution_time_ms': 0.0,
                                    }
                                ],
                            },
                            1,
                        )

                    # Only the hidden test patch's own paths are put back below, so a diff
                    # that touches any other test infrastructure is refused outright: a new
                    # conftest.py, plugin, TestMain or runner config could make the tests
                    # report a pass without running them.
                    model_paths = _patch_paths(repo_dir, patch_path, timeout)
                    if model_paths is None:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': 'git cannot list the paths the patch touches.',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'invalid_patch',
                                'cases': [],
                            },
                            1,
                        )
                    repo_root = _git_root(repo_dir, timeout)
                    hidden_set = set(hidden)
                    exposed = [rel for rel in model_paths if rel not in hidden_set]
                    tampered = {rel: _infra_reason(repo_root, rel) for rel in exposed}
                    for flagged in _mode_change_paths(patch_text):
                        tampered[flagged] = 'changes a file mode or creates a symlink'
                    configs = {
                        rel: _read_config(repo_root / rel)
                        for rel in exposed
                        if _components(rel) and _components(rel)[-1] in _CONFIG_FILES
                    }

                    apply_patch = subprocess.run(
                        ['git', '-C', str(repo_dir), 'apply', str(patch_path)],
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    if apply_patch.returncode != 0:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': f'Patch apply failed: {_tail(apply_patch.stderr or apply_patch.stdout)}',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'patch_apply_failed',
                                'cases': [],
                            },
                            1,
                        )

                    for rel, old in configs.items():
                        name = _components(rel)[-1]
                        if _config_tampered(name, old, _read_config(repo_root / rel)):
                            tampered[rel] = tampered[rel] or 'changes test configuration'
                    tampered = {rel: reason for rel, reason in sorted(tampered.items()) if reason}
                    if tampered:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': 'Patch changes test infrastructure, so it was not tested: '
                                + _tail(', '.join(tampered)),
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'test_infra_tampering',
                                'cases': [
                                    {
                                        'id': rel,
                                        'passed': False,
                                        'score': 0.0,
                                        'input_summary': 'patched path',
                                        'actual_summary': reason,
                                        'error': f'{rel}: {reason}',
                                        'execution_time_ms': 0.0,
                                    }
                                    for rel, reason in tampered.items()
                                ],
                            },
                            1,
                        )

                    blocked = _restore(repo_dir, saved)
                    if blocked is not None:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': f'Patch blocks restoring the hidden tests: {_tail(blocked)}',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'cases': [],
                            },
                            1,
                        )

                    if test_patch:
                        test_patch_proc = subprocess.run(
                            ['git', '-C', str(repo_dir), 'apply', str(test_patch_file)],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        if test_patch_proc.returncode != 0:
                            return _emit(
                                {
                                    'schema_version': '1.0',
                                    'score': 0.0,
                                    'passed': False,
                                    'details': f'test patch failed: {_tail(test_patch_proc.stderr or test_patch_proc.stdout)}',
                                    'reward_components': {'apply': 0.0, 'tests': 0.0},
                                    'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                    'truncated': False,
                                    'error_type': 'test_patch_failed',
                                    'cases': [],
                                },
                                1,
                            )

                    per_command_timeout = max(1, timeout // max(len(test_commands), 1))
                    passed_count = 0
                    cases = []
                    for index, command in enumerate(test_commands):
                        proc = _run(command, cwd=repo_dir, timeout=per_command_timeout)
                        passed = proc.returncode == 0
                        if passed:
                            passed_count += 1
                        cases.append(
                            {
                                'id': f'test_{index}',
                                'passed': passed,
                                'score': 1.0 if passed else 0.0,
                                'input_summary': command,
                                'actual_summary': _tail(proc.stdout),
                                'error': _tail(proc.stderr) or None,
                                'execution_time_ms': 0.0,
                            }
                        )

                    pass_fraction = passed_count / len(test_commands)
                    score = apply_credit + (1 - apply_credit) * pass_fraction
                    return _emit(
                        {
                            'schema_version': '1.0',
                            'score': round(max(0.0, min(1.0, score)), 6),
                            'passed': passed_count == len(test_commands),
                            'details': (
                                f'Patch applies cleanly; '
                                f'{passed_count}/{len(test_commands)} test commands passed.'
                            ),
                            'reward_components': {
                                'apply': apply_credit,
                                'tests': round((1 - apply_credit) * pass_fraction, 6),
                            },
                            'metrics': {
                                'pass_fraction': pass_fraction,
                                'tests_run': len(test_commands),
                                'tests_passed': passed_count,
                            },
                            'truncated': False,
                            'cases': cases,
                        },
                        0 if passed_count == len(test_commands) else 1,
                    )


            if __name__ == '__main__':
                sys.exit(main())
            """
        )


class SWEBenchProEnvironment(Environment):
    """SWE-bench Pro benchmark adapter."""

    name: str | None = 'swebench_pro'
    task: str = (
        'Repair the repository by returning a unified diff patch. Change only the code: '
        'a patch that touches tests or test configuration scores 0.'
    )
    verifier_code: str = 'return 0.0'
    domain: str = 'software-engineering'
    tags: list[str] = Field(default_factory=lambda: ['benchmark', 'swebench-pro', 'patch'])
    timeout: int = 1800
    dataset_id: str = 'ScaleAI/SWE-bench_Pro'
    split: str = 'test'
    repo_url_template: str = 'https://github.com/{repo}.git'
    default_instance_id: str | None = None

    _instances: dict[str, SWEBenchTask] = PrivateAttr(default_factory=dict)

    def _ensure_loaded(self) -> None:
        if self._instances:
            return
        datasets = _require_datasets()
        dataset = datasets.load_dataset(self.dataset_id, split=self.split)
        for row in dataset:
            task = self._task_from_row(dict(row))
            self._instances[task.instance_id] = task
        if self.default_instance_id is None and self._instances:
            self.default_instance_id = next(iter(self._instances))

    def _task_from_row(self, row: dict[str, Any]) -> SWEBenchTask:
        instance_id = str(row.get('instance_id') or f'{row["repo"]}@{row["base_commit"][:12]}')
        repo = str(row['repo'])
        return SWEBenchTask(
            instance_id=instance_id,
            repo=repo,
            base_commit=str(row['base_commit']),
            problem_statement=str(row.get('problem_statement', '')),
            test_patch=str(row.get('test_patch', '')),
            fail_to_pass=_coerce_json_list(row.get('fail_to_pass')),
            pass_to_pass=_coerce_json_list(row.get('pass_to_pass')),
            selected_test_files_to_run=_coerce_json_list(row.get('selected_test_files_to_run')),
            before_repo_set_cmd=str(row.get('before_repo_set_cmd', '')),
            repo_language=str(row.get('repo_language', '')),
            test_commands=_coerce_json_list(row.get('test_commands'))
            if row.get('test_commands')
            else None,
            repo_source=str(row.get('repo_source'))
            if row.get('repo_source')
            else self.repo_url_template.format(repo=repo),
        )

    def list_instance_ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._instances)

    def get_task(self, instance_id: str) -> SWEBenchTask:
        self._ensure_loaded()
        try:
            return self._instances[instance_id]
        except KeyError as exc:
            raise ValueError(f'Unknown SWE-bench Pro instance: {instance_id!r}') from exc

    def _resolve_task(self, **kwargs: Any) -> SWEBenchTask:
        for key in _TASK_ROUTE_KEYS:
            if key in kwargs and kwargs[key]:
                raw = kwargs[key]
                if isinstance(raw, SWEBenchTask):
                    return raw
                if isinstance(raw, dict):
                    return self._task_from_row(raw)

        instance_id = kwargs.get('instance_id')
        if instance_id:
            return self.get_task(str(instance_id))

        if kwargs.get('repo') and kwargs.get('base_commit'):
            row = dict(kwargs)
            row.setdefault('instance_id', kwargs.get('instance_id') or 'inline')
            return self._task_from_row(row)

        if self.default_instance_id is not None:
            return self.get_task(self.default_instance_id)

        raise ValueError(
            'SWE-bench Pro runs require an instance context. '
            'Pass instance_id=... or the raw dataset row fields.'
        )

    def _build_test_commands(self, task: SWEBenchTask) -> list[str]:
        """Return real test commands, or [] when there is no runner for the task.

        Every command counts toward the score, so setup steps never stand alone and
        an unknown language gets no stand-in command; the runner then reports
        ``unsupported_runner`` instead of a score.
        """
        if task.test_commands:
            return task.test_commands

        selected = task.selected_test_files_to_run
        language = task.repo_language.lower()
        if not selected:
            return []

        joined = ' '.join(_shell_snippet(path) for path in selected)
        if language in {'python', 'py'}:
            return [f'PYTHONPATH=. pytest -q {joined}']

        if language in {'js', 'javascript', 'ts', 'typescript'}:
            install = 'if [ -f package-lock.json ]; then npm install --silent; fi'
            return [f'{install} && npm test -- {joined}']

        return []

    def prepare_batch_requests(
        self,
        outputs: Sequence[str],
        **kwargs: Any,
    ) -> list[ResolvedRun]:
        per_item_kwargs = _split_batch_kwargs(len(outputs), kwargs)
        runs: list[ResolvedRun] = []
        for index, output in enumerate(outputs):
            task = self._resolve_task(**per_item_kwargs[index])
            runs.append(ResolvedRun(env=self, output=output, kwargs={'task': task}))
        return runs

    def run_with_deepgym(
        self,
        dg: DeepGym,
        model_output: str,
        **kwargs: Any,
    ) -> RunResult:
        task = kwargs.get('task')
        if not isinstance(task, SWEBenchTask):
            task = self._resolve_task(**kwargs)

        patch_text = PatchVerifier.extract_patch(model_output)
        payload = {
            'repo_source': task.repo_source or self.repo_url_template.format(repo=task.repo),
            'base_commit': task.base_commit,
            'test_patch': task.test_patch,
            'setup_commands': task.before_repo_set_cmd,
            'test_commands': self._build_test_commands(task),
            'timeout': self.timeout,
        }
        return _execute_runner(
            self,
            dg,
            PatchVerifier.build_runner_script(),
            payload,
            patch_text=patch_text,
            timeout=self.timeout,
        )


class RepoPatchEnvironment(Environment):
    """A repo-level patch task whose repository, toolchain and tests ship in its image.

    For datasets that publish one image per task with the repo already at
    ``repo_dir``, such as the MiMo-V2.6-RL code subset. The model returns a unified
    diff; the SWE-bench Pro patch runner applies it, puts every file the hidden
    ``test_patch`` touches back to its prior state, applies the test patch, runs
    ``test_command`` and scores 1.0 if it exits 0, else 0.0. A diff that touches any
    other test infrastructure (tests and their helpers or data, conftest.py, runner
    config, installed dependencies) scores 0.0 with ``error_type='test_infra_tampering'``.

    The runner and the tests run as root in the sandbox booted from ``image``:
    these test scripts expect root's toolchains and caches (``/root/go``,
    ``$HOME/.pyenv``), so the sandbox, not ``deepgym_solution``, is the isolation
    boundary. As with any repo patch task, a patch can also make the tests exit 0 by
    editing the code they run, so the reward is only as strong as the tests.
    """

    verifier_code: str = 'return 0.0'
    domain: str = 'software-engineering'
    tags: list[str] = Field(default_factory=lambda: ['benchmark', 'patch'])
    timeout: int = 1800
    repo_dir: str
    test_command: str
    test_patch: str = ''

    def run_with_deepgym(self, dg: DeepGym, model_output: str, **kwargs: Any) -> RunResult:
        """Score *model_output*'s diff against the hidden tests; *kwargs* are ignored."""
        payload = {
            'repo_dir': self.repo_dir,
            'test_patch': self.test_patch,
            'test_commands': [self.test_command],
            'timeout': self.timeout,
            'apply_credit': 0.0,
        }
        return _execute_runner(
            self,
            dg,
            PatchVerifier.build_runner_script(),
            payload,
            patch_text=PatchVerifier.extract_patch(model_output),
            timeout=self.timeout,
        )


class TerminalBenchEnvironment(Environment):
    """Terminal-Bench 2.0, a Harbor dataset: each run scores one task as a HarborEnvironment.

    Tasks load from *task_root* if given, else from Hugging Face, falling back to GitHub.
    Runs pick a task by ``task_id`` (or ``instance_id``), ``task_dir``, a task dict with a
    ``task_dir``, or a :class:`~deepgym.harbor.HarborTask`, else ``default_task_id``.
    """

    name: str | None = 'terminal_bench_2'
    task: str = 'Complete the terminal task by executing shell commands in the sandbox.'
    verifier_code: str = 'return 0.0'
    type: str = 'tool-use'
    domain: str = 'terminal'
    tags: list[str] = Field(default_factory=lambda: ['benchmark', 'terminal-bench', 'shell'])
    dataset_id: str = 'harborframework/terminal-bench-2.0'
    source_url: str = 'https://huggingface.co/datasets/harborframework/terminal-bench-2.0'
    fallback_source_url: str = 'https://github.com/laude-institute/terminal-bench-2.git'
    task_root: Path | None = None
    default_task_id: str | None = None

    _tasks: dict[str, HarborTask] = PrivateAttr(default_factory=dict)

    def _ensure_loaded(self) -> None:
        if self._tasks:
            return
        if self.task_root is not None:
            tasks = load_tasks(self.task_root)
        else:
            try:
                tasks = load_tasks(self.source_url)
            except DeepGymError:
                tasks = load_tasks(self.fallback_source_url)
        self._tasks = {task.name: task for task in tasks}
        if self.default_task_id is None:
            self.default_task_id = next(iter(self._tasks))

    def list_task_ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._tasks)

    def get_task(self, task_id: str) -> HarborTask:
        self._ensure_loaded()
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f'Unknown Terminal-Bench task: {task_id!r}') from exc

    def _resolve_task(self, **kwargs: Any) -> HarborTask:
        for key in _TASK_ROUTE_KEYS:
            raw = kwargs.get(key)
            if isinstance(raw, HarborTask):
                return raw
            if isinstance(raw, dict) and raw.get('task_dir'):
                return HarborTask.from_dir(Path(raw['task_dir']))

        task_id = kwargs.get('task_id') or kwargs.get('instance_id')
        if task_id:
            return self.get_task(str(task_id))

        if kwargs.get('task_dir'):
            return HarborTask.from_dir(Path(kwargs['task_dir']))

        if self.default_task_id is not None:
            return self.get_task(self.default_task_id)

        raise ValueError(
            'Terminal-Bench runs require task_id=..., task_dir=..., or a task payload dict.'
        )

    def prepare_batch_requests(
        self,
        outputs: Sequence[str],
        **kwargs: Any,
    ) -> list[ResolvedRun]:
        per_item_kwargs = _split_batch_kwargs(len(outputs), kwargs)
        return [
            ResolvedRun(
                env=HarborEnvironment.from_task(self._resolve_task(**per_item_kwargs[index])),
                output=output,
                kwargs={},
            )
            for index, output in enumerate(outputs)
        ]

    def run_with_deepgym(
        self,
        dg: DeepGym,
        model_output: str,
        **kwargs: Any,
    ) -> RunResult:
        """Score *model_output*, a bash script, on the task the kwargs pick."""
        env = HarborEnvironment.from_task(self._resolve_task(**kwargs))
        return env.run_with_deepgym(dg, model_output)


class MixedEnvironment(Environment):
    """Ratio-weighted environment mixer with reward routing."""

    name: str | None = 'mixed'
    task: str = 'Route each sample to the correct underlying DeepGym environment.'
    verifier_code: str = 'return 0.0'
    domain: str = 'mixed'
    tags: list[str] = Field(default_factory=lambda: ['mixed', 'benchmark'])
    timeout: int = 1800
    environments: list[tuple[Environment, float]] = Field(default_factory=list)
    seed: int | None = None

    _rng: random.Random = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._rng = random.Random(self.seed)
        if not self.environments:
            raise ValueError('MixedEnvironment requires at least one child environment')
        if any(weight <= 0 for _, weight in self.environments):
            raise ValueError('MixedEnvironment weights must be positive')

    def sample_environment(self) -> Environment:
        envs = [env for env, _ in self.environments]
        weights = [weight for _, weight in self.environments]
        return self._rng.choices(envs, weights=weights, k=1)[0]

    def sample_batch(self, batch_size: int) -> list[Environment]:
        return [self.sample_environment() for _ in range(batch_size)]

    def _direct_environment_match(self, item_kwargs: dict[str, Any]) -> Environment | None:
        for route_key in ('environment', 'env'):
            route_value = item_kwargs.get(route_key)
            if not isinstance(route_value, Environment):
                continue
            for env, _ in self.environments:
                if route_value is env or _infer_environment_name(
                    route_value
                ) == _infer_environment_name(env):
                    return env
            raise ValueError(
                f'MixedEnvironment received {route_key}=... that is not one of its child environments.'
            )
        return None

    def _route_matches(self, route_key: str, route_value: str) -> list[Environment]:
        matches: list[Environment] = []
        for env, _ in self.environments:
            if route_key in {'environment_name', 'env_name'}:
                candidates = {_infer_environment_name(env), env.name or ''}
            elif route_key == 'task_type':
                candidates = {env.type}
            else:
                candidates = _environment_aliases(env)
            if route_value in {candidate for candidate in candidates if candidate}:
                matches.append(env)
        return matches

    def _route_env(self, item_kwargs: dict[str, Any]) -> Environment:
        direct_match = self._direct_environment_match(item_kwargs)
        if direct_match is not None:
            return direct_match

        for route_key in _MIXED_ROUTE_KEYS:
            route_value = item_kwargs.get(route_key)
            if not route_value:
                continue
            matches = self._route_matches(route_key, str(route_value))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1 and route_key in {
                'environment_name',
                'env_name',
                'benchmark',
                'dataset_name',
            }:
                raise ValueError(
                    f'MixedEnvironment routing for {route_key}={route_value!r} is ambiguous.'
                )

        candidates: list[Environment] = []
        for env, _ in self.environments:
            try:
                if isinstance(env, SWEBenchProEnvironment):
                    env._resolve_task(**item_kwargs)
                    candidates.append(env)
                    continue
                if isinstance(env, TerminalBenchEnvironment):
                    env._resolve_task(**item_kwargs)
                    candidates.append(env)
                    continue
            except Exception:
                continue

        if len(candidates) == 1:
            return candidates[0]

        raise ValueError(
            'MixedEnvironment could not route the sample. '
            'Pass environment_name=.../env_name=..., an exact child environment, '
            'or benchmark-specific task metadata.'
        )

    def prepare_batch_requests(
        self,
        outputs: Sequence[str],
        **kwargs: Any,
    ) -> list[ResolvedRun]:
        per_item_kwargs = _split_batch_kwargs(len(outputs), kwargs)
        resolved: list[ResolvedRun] = []
        for index, output in enumerate(outputs):
            routed_env = self._route_env(per_item_kwargs[index])
            child_requests = build_run_requests(routed_env, [output], per_item_kwargs[index])
            resolved.extend(child_requests)
        return resolved


def route_single_run(env: Environment, kwargs: dict[str, Any]) -> Environment:
    """Return the environment that should score one output.

    A MixedEnvironment has only a placeholder verifier, so it must hand the
    output to the child that the routing kwargs select; other envs score it
    themselves.
    """
    return env._route_env(kwargs) if isinstance(env, MixedEnvironment) else env
