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

from deepgym.exceptions import DeepGymError, VerifierError
from deepgym.harbor import HarborEnvironment, HarborTask, load_tasks
from deepgym.integrations.hf import _require_datasets
from deepgym.judge import DEADLINE_GRACE_S
from deepgym.models import Environment, RunResult, VerifierResult
from deepgym.sandbox import (
    ExecutionResult,
    build_run_result,
    check_exit,
    parse_wrapped_output,
    split_wrapped_output,
    wrap_stderr,
)

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_RUNNER_PATH = '/home/user/deepgym_runner.py'
_TASK_PATH = '/home/user/deepgym_task.json'
_PATCH_PATH = '/home/user/deepgym_candidate.diff'
_ARCHIVE_PATH = '/home/user/deepgym_task.tar.gz'
# The patch runner prints this line, flushed, just before it runs the tests: the first time
# anything the patch wrote executes, and in the runner's process group.
_TESTS_STARTED = 'deepgym-runner: running the tests'

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


def _parse_runner_output(output: str, marker: str, exit_code: int) -> ExecutionResult:
    """Parse a runner's verdict from its stdout alone, as the judges read theirs.

    The provider merges the streams, and a JSON line on stderr is no verdict. A runner that
    printed :data:`_TESTS_STARTED` and then no valid verdict its exit code (0 or 1) backs was
    stopped by what the patch ran: the environment had run every step before, so that is a
    judged 0, ``policy_sandbox_failure``.

    Raises:
        VerifierError: If the runner stopped without a verdict before its tests ran.
    """
    stdout, stderr = split_wrapped_output(output, marker)
    if _TESTS_STARTED not in stdout.splitlines():
        return parse_wrapped_output(output, marker, exit_code)
    if exit_code in (0, 1):
        try:
            executed = parse_wrapped_output(output, marker, exit_code)
            check_exit(exit_code, executed.verifier_result.score, executed.verifier_result.passed)
        except VerifierError:
            pass
        else:
            return executed
    result = VerifierResult(
        score=0.0,
        passed=False,
        details=f'The runner stopped without a valid verdict (exit {exit_code}) while running '
        'the tests, so the patch scores 0.',
        error_type='policy_sandbox_failure',
    )
    return ExecutionResult(result, stderr=stderr, exit_code=1)


def _run_sandbox_runner(
    env: Environment,
    dg: DeepGym,
    script: str,
    payload: dict[str, Any],
    *,
    patch_text: str | None = None,
    archive_path: Path | None = None,
    timeout: int,
) -> tuple[ExecutionResult, str]:
    """Run *script* as root in a fresh sandbox from *dg*'s provider, booted from *env*.

    The runner keeps to *timeout* itself; the provider allows a grace period on top, so a
    test the runner kills at its limit is still scored by the runner.

    Returns:
        The parsed run (see :func:`_parse_runner_output`) and the sandbox's id.

    Raises:
        VerifierError: If the runner stopped without a verdict before its tests ran.
    """
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

        command, marker = wrap_stderr(command)
        response = sandbox.exec(command, timeout=timeout + DEADLINE_GRACE_S)
        return _parse_runner_output(response.output, marker, response.exit_code), sandbox.id
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
        VerifierError: If the runner stopped without a verdict before its tests ran.
    """
    start = time.perf_counter()
    executed, sandbox_id = _run_sandbox_runner(
        env,
        dg,
        script,
        payload,
        patch_text=patch_text,
        archive_path=archive_path,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return build_run_result(
        executed.verifier_result,
        elapsed_ms,
        sandbox_id,
        stderr=executed.stderr,
        exit_code=executed.exit_code,
    )


def _vendored_toml_source() -> str:
    """Return a self-contained TOML parser to inline in the runner.

    The runner executes in the task image's interpreter, where ``tomllib`` (Python 3.11+)
    may be absent, so a config comparison that imported it would fail closed on every
    older image. ``vendored_toml.txt`` is CPython's ``tomllib`` (the ``tomli`` package
    verbatim) concatenated into one module, so the runner parses ``pyproject.toml``/
    ``Cargo.toml`` exactly as the tools that consume them do -- dotted keys, inline tables,
    duplicate-key and table-redefinition rejection -- independent of the image's Python.
    It ships as package data and is read as text, not rebuilt from this host's modules
    with ``inspect.getsource``, which fails when the host's ``tomllib``/``tomli`` is
    compiled or installed without its ``.py`` source.

    Returns:
        Python source defining ``loads`` and ``TOMLDecodeError`` at module scope.
    """
    return Path(__file__).with_name('vendored_toml.txt').read_text(encoding='utf-8')


class PatchVerifier:
    """Reusable repo-level patch verifier."""

    @staticmethod
    def extract_patch(model_output: str) -> str:
        """Extract a unified diff from fenced or raw model output."""
        return _parse_patch_from_model_output(model_output)

    @staticmethod
    def build_runner_script() -> str:
        """Return the self-contained patch runner script.

        It runs in the task image's ``python3``, which can be as old as 3.8 (some SWE-bench
        Pro images), so its annotations are postponed and it uses nothing newer. A vendored
        TOML parser is inlined as ``_VENDORED_TOML`` and exec'd only when the image's Python
        lacks ``tomllib`` (before 3.11), so the config comparison matches the tools either
        way; see :func:`_vendored_toml_source`.
        """
        toml = _vendored_toml_source()
        header = (
            'from __future__ import annotations\n\n'
            f'_TESTS_STARTED = {_TESTS_STARTED!r}\n_VENDORED_TOML = {toml!r}\n\n'
        )
        return header + textwrap.dedent(
            """
            import fnmatch
            import json
            import os
            import posixpath
            import re
            import shlex
            import shutil
            import signal
            import subprocess
            import sys
            import tempfile
            import time
            from pathlib import Path


            def _emit(result: dict, exit_code: int) -> int:
                print(json.dumps(result))
                return exit_code


            def _tail(text: str, limit: int = 400) -> str:
                text = text.strip()
                if len(text) <= limit:
                    return text
                return text[-limit:]


            def _path_verdict(error_type: str, summary: str, flagged: dict) -> int:
                # Score 0 with one case per flagged path and its reason.
                return _emit(
                    {
                        'schema_version': '1.0',
                        'score': 0.0,
                        'passed': False,
                        'details': summary + _tail(', '.join(flagged)),
                        'reward_components': {'apply': 0.0, 'tests': 0.0},
                        'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                        'truncated': False,
                        'error_type': error_type,
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
                            for rel, reason in flagged.items()
                        ],
                    },
                    1,
                )


            def _read_end(file, limit: int = 65536) -> str:
                # The last *limit* bytes written to *file*, decoded as UTF-8 with replacements.
                file.seek(max(0, file.seek(0, os.SEEK_END) - limit))
                return file.read().decode('utf-8', 'replace')


            def _run(command: str, *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
                # Run *command* in its own session, its output in files that nothing it leaves running
                # can hold open. One still running after *timeout* seconds is killed with everything
                # it started (SIGKILL) and reported as exit -9: a setup step then fails, and a test
                # the patch made hang scores 0, without touching the next command.
                with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                    proc = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=str(cwd) if cwd is not None else None,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                    try:
                        returncode, note = proc.wait(timeout=timeout), ''
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                        returncode, note = -9, f'\\ntimed out after {timeout}s'
                    return subprocess.CompletedProcess(command, returncode, _read_end(out), _read_end(err) + note)


            # Rustup's layout in the official rust images. An image flattened without its config
            # (MiMo's on Docker Hub) keeps the toolchain there but loses the ENV that put it on
            # PATH, so cargo would be "command not found" for every answer, right or wrong.
            _CARGO_HOME = '/usr/local/cargo'
            _RUSTUP_HOME = '/usr/local/rustup'


            def _restore_rust_env() -> None:
                # Put that toolchain back when cargo is missing from PATH but installed there.
                if shutil.which('cargo') or not os.path.isfile(_CARGO_HOME + '/bin/cargo'):
                    return
                os.environ.setdefault('CARGO_HOME', _CARGO_HOME)
                if os.path.isdir(_RUSTUP_HOME):
                    os.environ.setdefault('RUSTUP_HOME', _RUSTUP_HOME)
                os.environ['PATH'] = _CARGO_HOME + '/bin' + os.pathsep + (os.environ.get('PATH') or os.defpath)


            # Reserved words that open and close a compound command. What runs inside one may never
            # run at all (a branch not taken, a loop over nothing), so it is not looked up.
            _OPENERS = {'if', 'while', 'until', 'for', '{'}
            _CLOSERS = {'fi', 'done', '}'}
            _IMPORTABLE = 'import importlib.util, sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)'


            def _top_level_commands(command: str) -> list:
                # The simple commands of shell *command* that must run for it to pass, each as its
                # words without redirections: none inside a compound command or subshell, in an
                # and-or list with `||` (which handles a missing command), or calling a function
                # *command* defines. It ends at a here-document, `case` (whose patterns look like
                # commands) or `function`, and is empty if shlex, stricter than sh, can't split it.
                lexer = shlex.shlex(command, posix=True, punctuation_chars='();<>|&\\n')
                lexer.whitespace, lexer.whitespace_split = ' \\t\\r', True
                try:
                    tokens = iter(list(lexer) + [';'])
                except ValueError:
                    return []
                found, listed, words, depth, has_or, defined = [], [], [], 0, False, set()
                for token in tokens:
                    if token.startswith('<<') or (not words and token in ('case', 'function')):
                        break
                    if set(token) <= set('<>&|') and set(token) & set('<>'):
                        if words and words[-1].isdigit():
                            words.pop()  # the redirected descriptor, as in 2>/dev/null
                        next(tokens, None)  # and the redirection's target
                    elif not set(token) <= set('();|&\\n'):
                        if words or token not in _OPENERS | _CLOSERS:
                            words.append(token)
                        else:
                            depth += 1 if token in _OPENERS else -1
                    else:
                        if token == '()' and len(words) == 1:
                            defined.add(words[0])
                        elif words and not depth and words[0] not in defined:
                            listed.append(words)
                        words, has_or = [], has_or or token == '||'
                        if token not in ('&&', '||', '|', '|&'):  # the and-or list ends
                            found += [] if has_or else listed
                            listed, has_or = [], False
                        depth += token.count('(') - token.count(')')
                return found


            def _python_module(program: str, args: list):
                # (interpreter flags, module) when *program* is a python run as `-m module`, else None.
                if not re.fullmatch(r'python[0-9.]*', posixpath.basename(program)):
                    return None
                flags = []
                while args and args[0].startswith('-') and args[0] not in ('-', '-c', '-m'):
                    take = 2 if args[0] in ('-W', '-X') else 1
                    flags, args = flags + args[:take], args[take:]
                return (flags, args[1]) if args[:1] == ['-m'] and args[1:2] else None


            def _preflight(command: str, cwd: Path, repo_root: Path) -> str | None:
                # Why this environment cannot run test *command*, checked before the patch, or None:
                # a program that always runs (see _top_level_commands) is neither on PATH nor an
                # executable outside the repo, or the package a `python -m` names can't be found
                # (its top level only, found without running repo code). It never runs the command
                # (dash 0.5.10's `sh -n -c` would). A path in the repo is the repo's own (the hidden
                # tests may add it); nothing is looked up after a PATH change, `source` or `eval`,
                # nor a module after `cd` or `export` or with a computed assignment. Best effort: what
                # a wrapper script runs is not seen, so its failure after the patch is scored.
                repo_dirs = {str(repo_root), str(cwd)}
                names, modules, plain = [], [], True
                for words in _top_level_commands(command):
                    if any(word.startswith('PATH=') for word in words):
                        break
                    words = words[1:] if words[0] == '!' else words
                    assigned = {}
                    while words and re.match(r'[A-Za-z_][A-Za-z0-9_]*=', words[0]):
                        name, _, value = words.pop(0).partition('=')
                        assigned[name] = value
                    if not words or re.search(r'[$`*?~\\[]', words[0]):
                        continue
                    program = words[0]
                    if program in ('.', 'source', 'eval', 'alias'):
                        break
                    if program.startswith('/'):
                        if any(program == base or program.startswith(base + '/') for base in repo_dirs):
                            continue
                        if not (os.path.isfile(program) and os.access(program, os.X_OK)):
                            return f'{program} is not an executable file'
                    elif '/' in program:
                        continue
                    else:
                        names.append(program)
                    module = _python_module(program, words[1:])
                    if module and plain and not any(re.search('[$`]', value) for value in assigned.values()):
                        modules.append((program, *module, assigned))
                    plain = plain and program not in ('cd', 'pushd', 'popd', 'export', 'unset')
                if names:
                    lookup = 'for w; do command -v "$w" >/dev/null 2>&1 || echo "$w"; done'
                    found = subprocess.run(['/bin/sh', '-c', lookup, 'sh', *names], cwd=str(cwd), capture_output=True, text=True)
                    missing = found.stdout.split()
                    if missing:
                        return 'not found: ' + ', '.join(dict.fromkeys(missing))
                for python, flags, module, assigned in modules:
                    probe = [python, *flags, '-c', _IMPORTABLE, module.split('.')[0]]
                    if subprocess.run(probe, cwd=str(cwd), env=dict(os.environ, **assigned), capture_output=True).returncode:
                        return f'{python} cannot import {module}'
                return None


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
            # any edit is refused. tox.ini/tox.toml (whose [testenv] commands are what `tox`
            # runs) and go.work (which modules `go test` sees) are refused the same way. (A
            # lockfile or go.sum cannot forge a pass on its own -- the stub package a repoint
            # needs is caught by its own runner `bin` -- so those stay allowed, as a real
            # dependency fix updates them.)
            _INFRA_FILES = {
                'conftest.py', 'pytest.py', 'pytest.ini', '.pytest.ini', 'pytest.toml',
                '.pytest.toml', 'tox.ini', 'noxfile.py', 'tests.py', 'sitecustomize.py',
                'usercustomize.py', '.pythonrc', '.pythonrc.py', '.pythonstartup', '.rspec',
                'phpunit.xml', 'phpunit.xml.dist', 'phpunit.dist.xml', 'pom.xml', 'build.gradle',
                'build.gradle.kts', 'settings.gradle', 'settings.gradle.kts', 'build.rs',
                'rakefile', 'makefile', 'gnumakefile', '.npmrc', '.yarnrc', '.yarnrc.yml',
                'setup.py', 'tox.toml', 'go.work',
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
            # package.json keys that configure a test runner, compared whole; scripts and bin are
            # judged in _test_part_changed. The rest of the file, including the dependency ranges
            # a real fix updates, is left alone.
            _PACKAGE_KEYS = (
                'jest', 'mocha', 'ava', 'babel', 'nyc', 'c8', 'vitest', 'tap', 'wdio', 'workspaces',
            )
            # Config files that are test infrastructure only in part; _config_tampered decides.
            # The keys it compares are knowingly incomplete (a runner reads more than any list
            # here); the per-test report gate supersedes this comparison.
            _CONFIG_FILES = {'pyproject.toml', 'setup.cfg', 'package.json', 'cargo.toml', 'go.mod'}
            _SYMLINK = b'\\0deepgym-symlink'
            # An edit the guard can't classify with certainty is uncertain: the tests still run, a
            # failure scores the normal 0, and only a pass (which can't be trusted) is masked.
            # [workspace.package] keys known to be plain metadata members inherit; a change to any
            # other key there is uncertain.
            _WORKSPACE_METADATA = {
                'version', 'description', 'authors', 'license', 'repository', 'homepage',
                'documentation', 'readme', 'keywords', 'categories', 'rust-version',
            }


            def _components(rel: str) -> list[str]:
                # Lowercased path components, dropping '' and '.' (git -z paths are raw, normalized).
                return [part for part in rel.lower().split('/') if part not in ('', '.')]


            def _is_test_file(name: str) -> bool:
                # Whether *name* (lowercased basename) is itself a test, a test runner config,
                # or an import hook a runner loads by name -- refused wherever it sits. A name
                # that only shares a word with one is ordinary code: in Go only *_test.go is a
                # test (testdata/ is in _TEST_DIRS), so cmd/test.go is not, nor is a jest.md doc.
                # Whether a *.setup.* file is a test setup file depends on where it sits; see
                # _infra_reason.
                segments = name.split('.')
                return (
                    name in _INFRA_FILES
                    or name.endswith(('_test.py', '_test.go', '_spec.rb', '.pth'))
                    or (name.startswith('test_') and name.endswith('.py'))
                    or (len(segments) > 2 and segments[-2] in ('test', 'spec', 'e2e') and segments[-1] in _JS_EXTS)
                    or (len(segments) > 1 and segments[0] in _JS_TOOLS and segments[1] in ('config', 'conf', 'setup', 'workspace', 'preset'))
                    or segments[0] == 'setuptests'
                    or (len(segments) > 2 and segments[0] in ('jest', 'vitest') and segments[-1] in _JS_EXTS)
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
                # A test setup file runs inside the test process, but ordinary modules are named
                # *.setup.* too (src/i18n.setup.ts): refuse one named for a test tool or kept in
                # a test directory.
                segments = name.split('.')
                if len(segments) > 2 and segments[-2] == 'setup' and segments[-1] in _JS_EXTS and (
                    name.startswith(('test', 'jest', 'vitest', 'playwright'))
                    or any(part in ('test', 'tests', '__tests__', 'e2e', 'spec') for part in dirs)
                ):
                    return 'test file or test configuration'
                # In a test directory, refuse executable code (a helper a test imports, or a
                # module that runs at import) and the package marker, but allow data fixtures.
                if any(part in _TEST_DIRS for part in dirs):
                    if name == '__init__.py':
                        return 'test package marker'
                    if name.rsplit('.', 1)[-1] in _CODE_EXTS and '.' in name:
                        return 'code in a test directory'
                return None


            def _toml_table(value) -> dict:
                # A parsed TOML value as a dict, or {} when it is some other shape.
                return value if isinstance(value, dict) else {}


            _TOML_LOADS = []


            def _toml_loads(text: str) -> dict:
                # Parse TOML with the image's stdlib tomllib if present, else the inlined copy.
                # Raises on malformed TOML, which fails the check closed.
                if not _TOML_LOADS:
                    try:
                        import tomllib

                        _TOML_LOADS.append(tomllib.loads)
                    except ModuleNotFoundError:
                        namespace: dict = {}
                        exec(compile(_VENDORED_TOML, 'vendored_toml', 'exec'), namespace)
                        _TOML_LOADS.append(namespace['loads'])
                return _TOML_LOADS[0](text)


            def _test_part_changed(rel: str, old_text: str, new_text: str) -> bool | None:
                # Whether the test-relevant part of config file *rel* changed, or None for a change
                # it can't classify. Raises on anything it cannot parse or compare; the caller turns
                # that into tampering (fails closed).
                parts = _components(rel)
                name = parts[-1]
                if name == 'package.json':
                    before = json.loads(old_text or '{}')
                    after = json.loads(new_text or '{}')
                    before = before if isinstance(before, dict) else {}
                    after = after if isinstance(after, dict) else {}
                    if any(before.get(key) != after.get(key) for key in _PACKAGE_KEYS):
                        return True
                    # A script named for tests (test, pretest, test:unit) is refused. One such a
                    # script runs by name (npm run unit) may be the runner, and so may a nested
                    # package's bin once a dependency is repointed to that package: uncertain. The
                    # root package's own bin and any other script are a fix's business.
                    scripts = [doc.get('scripts') if isinstance(doc.get('scripts'), dict) else {} for doc in (before, after)]
                    keys = scripts[0].keys() | scripts[1].keys()
                    changed = {key for key in keys if scripts[0].get(key) != scripts[1].get(key)}
                    tests = {key for key in keys if 'test' in key}
                    if changed & tests:
                        return True
                    related, frontier = set(), tests
                    while frontier:
                        text = ' '.join(str(script.get(key, '')) for script in scripts for key in frontier)
                        frontier = {key for key in keys - tests - related if re.search(rf'(?<![\\w:.-]){re.escape(key)}(?![\\w:.-])', text)}
                        related |= frontier
                    related |= {key for key in keys if key[:3] == 'pre' and key[3:] in related or key[:4] == 'post' and key[4:] in related}
                    nested_bin = len(parts) > 1 and before.get('bin') != after.get('bin')
                    return None if changed & related or nested_bin else False
                if name == 'setup.cfg':
                    import configparser

                    # interpolation=None: a '%(x)s' value must not raise; optionxform=str: pytest
                    # reads option names case-sensitively, so a case-only change still counts.
                    before, after = (configparser.ConfigParser(interpolation=None) for _ in range(2))
                    before.optionxform = after.optionxform = str
                    before.read_string(old_text)
                    after.read_string(new_text)
                    # [options] holds metadata, dependencies and entry points (console_scripts) a real
                    # fix may change; only the pytest11 entry-point group registers pytest plugins.
                    keys = ('tool:pytest', 'nosetests', 'tox:tox')

                    def view(cfg):
                        plugins = cfg.get('options.entry_points', 'pytest11', fallback=None)
                        return {key: dict(cfg[key]) for key in cfg.sections() if key in keys}, plugins

                    return view(before) != view(after)
                # cargo.toml / pyproject.toml: _toml_loads parses exactly like the tools (dotted
                # keys and inline tables normalize into these tables) and raises on a duplicate
                # key, a redefined table or any syntax error, which fails closed above.
                before, after = _toml_loads(old_text), _toml_loads(new_text)
                if name == 'cargo.toml':
                    # Every target table (with its `test` flag) and what only tests or build.rs
                    # build with. [dependencies] is left out: a version bump there is a real fix.
                    keys = ('lib', 'bin', 'test', 'bench', 'example', 'features',
                            'dev-dependencies', 'build-dependencies')
                    if any(before.get(key) != after.get(key) for key in keys):
                        return True
                    # [workspace] picks the crates that build and test (members, default-members,
                    # exclude) and their shared dependencies; [workspace.package] is checked last.
                    ws_before, ws_after = _toml_table(before.get('workspace')), _toml_table(after.get('workspace'))
                    if dict(ws_before, package=None) != dict(ws_after, package=None):
                        return True
                    # Package-level auto-discovery and build script decide which tests cargo runs.
                    auto = ('autotests', 'autobenches', 'autobins', 'autoexamples', 'autolib', 'build')
                    pkg_before, pkg_after = _toml_table(before.get('package')), _toml_table(after.get('package'))
                    if any(pkg_before.get(key) != pkg_after.get(key) for key in auto):
                        return True
                    # [workspace.package] is what members inherit (version.workspace = true): a listed
                    # metadata key is a release bump; a change to any other key is uncertain.
                    meta_before, meta_after = _toml_table(ws_before.get('package')), _toml_table(ws_after.get('package'))
                    changed = {key for key in meta_before.keys() | meta_after.keys() if meta_before.get(key) != meta_after.get(key)}
                    return None if changed - _WORKSPACE_METADATA else False
                # [tool.tox] is compared whole, so its [testenv] commands count whether they sit
                # in a legacy_tox_ini string or in env_run_base/env tables. Of the entry points, only
                # the pytest11 group registers pytest plugins; console scripts are a fix's business.
                tool_keys = ('pytest', 'coverage', 'tox', 'nose')
                before_tool, after_tool = _toml_table(before.get('tool')), _toml_table(after.get('tool'))
                plugins = [_toml_table(_toml_table(doc.get('project')).get('entry-points')).get('pytest11') for doc in (before, after)]
                return (
                    any(before_tool.get(key) != after_tool.get(key) for key in tool_keys)
                    or before.get('build-system') != after.get('build-system')
                    or plugins[0] != plugins[1]
                )


            def _config_tampered(rel: str, old: bytes | None, new: bytes | None) -> bool | None:
                # Whether turning *old* into *new* changes the test-relevant part of config file
                # *rel*, or None for a change it can't classify. Fails closed: a symlink, or anything
                # that cannot be parsed and shown unchanged, counts as tampering, so a parse error
                # becomes a trained zero, never a masked run.
                parts = _components(rel)
                name = parts[-1]
                if _SYMLINK in (old, new):
                    return True
                if name == 'go.mod':
                    # Adding or removing a go.mod below the root makes or merges a nested module,
                    # which `go test ./...` skips or newly runs; editing one (a version bump) is a fix.
                    return len(parts) > 1 and (old is None) != (new is None)
                old_text = (old or b'').decode('utf-8', 'replace')
                new_text = (new or b'').decode('utf-8', 'replace')
                if old_text == new_text:
                    return False
                try:
                    return _test_part_changed(rel, old_text, new_text)
                except Exception:
                    return True


            def _mentioned(rel: str, text: str) -> bool:
                # Whether the test command text may name *rel*. A wrong yes only masks a passing run,
                # so a word naming rel's basename counts unless it plainly names another folder.
                # Words are split at whitespace, quotes and shell punctuation (no shell parsing) and
                # normalized like the patch paths; a glob matches as a glob. A relative word counts if
                # rel may end with it (./ci.ini, sub/./ci.ini, sub/ci.ini after a cd), as does one
                # whose folder is unknown (../, ~, $VAR, `cmd`); an absolute one only if it ends in
                # /rel. So test/posts.js names another file than src/socket.io/posts.js, and
                # test_calc.py is not calc.py. A Python file is also named by its module (-p plugin),
                # under the same suffix rule. A value attached to a short flag (-cci.ini) is tried
                # without the flag too; --opt=value is split at the '='.
                base = rel.rsplit('/', 1)[-1]
                stem = rel[:-12] if rel.endswith('/__init__.py') else rel[:-3] if rel.endswith('.py') else None
                words = []
                for word in filter(None, re.split(r'[\\s\\'"=:;,|&()<>]+', text)):
                    words += [word, word[2:]] if re.match(r'-[A-Za-z].', word) else [word]
                for word in words:
                    if stem and re.fullmatch(r'[A-Za-z_][\\w.]*', word):
                        module = word.replace('.', '/')
                        if stem == module or stem.endswith('/' + module):
                            return True
                    norm = posixpath.normpath(word)
                    if not fnmatch.fnmatchcase(base, norm.rsplit('/', 1)[-1]):
                        continue
                    if norm.startswith(('../', '~')) or '$' in norm or '`' in norm:
                        return True
                    if norm.startswith('/'):
                        if re.search(r'[*?\\[]', norm) or norm.endswith('/' + rel):
                            return True
                    elif fnmatch.fnmatchcase(rel, norm) or fnmatch.fnmatchcase(rel, '*/' + norm):
                        return True
                return False


            def _read_config(path: Path) -> bytes | None:
                # A config file's bytes; a symlink reads as a sentinel no real file has.
                if path.is_symlink():
                    return _SYMLINK
                return path.read_bytes() if path.is_file() else None


            _MODE_RE = re.compile(r'^(old mode|new mode|new file mode|deleted file mode)\\s+([0-7]+)\\s*$')


            def _mode_change_paths(patch_text: str) -> list[str]:
                # The files that become or stop being symlinks, or change to a mode other than a
                # plain file's 100644/100755. A symlink can aim a protected path at attacker content,
                # so it is refused outright; an exec-bit flip (restoring bin/calc to 100755) is an
                # ordinary fix unless the path itself is refused, which _infra_reason decides (git
                # lists a mode-only change). The numeric mode is parsed so non-canonical git
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
                    plain = mode in (0o100644, 0o100755)
                    if (mode & 0o170000) == 0o120000 or (kind in ('old mode', 'new mode') and not plain):
                        flagged.append(current)
                return list(dict.fromkeys(flagged))


            def _git_root(repo_dir: Path, timeout: int) -> Path | None:
                # The repository top-level: numstat reports paths relative to it, not to repo_dir.
                # None when git cannot tell: guessing repo_dir (maybe a subdirectory) would check
                # the wrong, missing files and grade a patch that edits the real ones.
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
                return Path(top) if proc.returncode == 0 and top else None


            def _normalize(path: str) -> str | None:
                # The one spelling every comparison uses: git's raw -z path through normpath, case
                # as-is, so ./x, a/../x and x/ are all x. None when that is no plain path inside
                # the repo (absolute, '..', empty); the caller treats such a path as uncertain.
                norm = posixpath.normpath(path)
                return None if norm.startswith(('/', '../')) or norm in ('.', '..') else norm


            def _patch_paths(repo_dir: Path, patch_file: Path, timeout: int) -> list[str] | None:
                # Every path a patch reads or writes, as git's own parser sees it: --numstat
                # names each post-image path, -R --numstat each pre-image one, and -z emits raw
                # (unquoted) bytes so non-ASCII names are not mis-parsed. Each is returned in its
                # _normalize form, or raw if it has none.
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
                return list(dict.fromkeys(_normalize(path) or path for path in paths))


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
                        for ancestor in list(Path(rel).parents)[:-1]:
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
                started = time.monotonic()
                _restore_rust_env()
                task = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
                patch_path = Path(sys.argv[2])
                patch_text = patch_path.read_text(encoding='utf-8').strip()

                test_commands = task.get('test_commands') or []
                blank = any(not command.strip() for command in test_commands)
                if blank or not test_commands:
                    # No real test runner, or a broken task whose blank command would "pass"
                    # without running a test: no verdict, so trainers mask the sample instead
                    # of rewarding a patch nothing checked.
                    return _emit(
                        {
                            'schema_version': '1.0',
                            'score': 0.0,
                            'passed': False,
                            'details': (
                                'The task has a blank test command, so it was not run or scored.'
                                if blank
                                else 'No test runner for this task, so it was not scored.'
                            ),
                            'reward_components': {'apply': 0.0, 'tests': 0.0},
                            'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                            'truncated': False,
                            'error_type': 'invalid_task' if blank else 'unsupported_runner',
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
                # The whole run keeps to the timeout, so the runner, not the provider, stops a
                # test that hangs (the provider allows a few seconds more).
                deadline = started + timeout
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
                    # Listing paths only parses the patches. If git stalls, the runner failed,
                    # not the model, so there is no verdict and trainers mask the sample.
                    try:
                        hidden = _patch_paths(repo_dir, test_patch_file, timeout) if test_patch else []
                        model_paths = _patch_paths(repo_dir, patch_path, timeout)
                        repo_root = _git_root(repo_dir, timeout)
                    except subprocess.TimeoutExpired:
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': 'Listing the paths the patches touch timed out, so the patch was not scored.',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'validation_timeout',
                                'cases': [],
                            },
                            1,
                        )
                    if repo_root is None:
                        # A missing or broken repo (the image lacks repo_dir, or it is no checkout)
                        # is the environment's fault, never the patch's, and without the top-level
                        # the paths can't be checked: no verdict, so trainers mask the sample.
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': f'git cannot find the repository at {repo_dir}, so the patch was not scored.',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'validation_error',
                                'cases': [],
                            },
                            1,
                        )
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
                    for index, command in enumerate(test_commands):
                        broken = _preflight(command, repo_dir, repo_root)
                        if broken:
                            # Found before the patch is applied, so the environment's fault, never
                            # the model's: no verdict. After the patch, every outcome is scored.
                            return _emit(
                                {
                                    'schema_version': '1.0',
                                    'score': 0.0,
                                    'passed': False,
                                    'details': f'Test command {index} cannot run in this environment ({broken}), so the patch was not scored.',
                                    'reward_components': {'apply': 0.0, 'tests': 0.0},
                                    'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                    'truncated': False,
                                    'error_type': 'preflight_failed',
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
                    hidden_set = set(hidden)
                    exposed = [rel for rel in model_paths if rel not in hidden_set]
                    tampered = {rel: _infra_reason(repo_root, rel) for rel in exposed}
                    for flagged in _mode_change_paths(patch_text):
                        tampered[flagged] = 'creates a symlink or a mode other than 100644/100755'
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

                    uncertain = {}
                    for rel, old in configs.items():
                        verdict = _config_tampered(rel, old, _read_config(repo_root / rel))
                        if verdict:
                            tampered[rel] = tampered[rel] or 'changes test configuration'
                        elif verdict is None:
                            uncertain[rel] = 'changes a config key that may affect the tests'
                    tampered = {rel: reason for rel, reason in sorted(tampered.items()) if reason}
                    if tampered:
                        return _path_verdict(
                            'test_infra_tampering',
                            'Patch changes test infrastructure, so it was not tested: ',
                            tampered,
                        )
                    # Not clearly tampering, not clearly safe: a path with no clean form, or a file
                    # the (trusted) test command mentions, which it may load. The tests still run;
                    # see below.
                    command_text = '\\n'.join(test_commands)
                    for rel in exposed:
                        if _normalize(rel) is None:
                            uncertain[rel] = 'the path does not normalize to one inside the repo'
                        elif rel not in uncertain and _mentioned(rel, command_text):
                            uncertain[rel] = 'the test command mentions it'

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

                    if deadline - time.monotonic() < 1:
                        # The steps before the tests used up the time: the harness's doing.
                        return _emit(
                            {
                                'schema_version': '1.0',
                                'score': 0.0,
                                'passed': False,
                                'details': 'The time limit ran out before the tests started, so the patch was not scored.',
                                'reward_components': {'apply': 0.0, 'tests': 0.0},
                                'metrics': {'pass_fraction': 0.0, 'tests_run': 0},
                                'truncated': False,
                                'error_type': 'validation_timeout',
                                'cases': [],
                            },
                            1,
                        )
                    # From here on, code the patch wrote runs, in this process group: if it stops the
                    # runner, this line tells the host the tests had started (a scored 0).
                    print(_TESTS_STARTED, flush=True)
                    passed_count = 0
                    cases = []
                    for index, command in enumerate(test_commands):
                        # The preflight found the environment able to run this command before the
                        # patch, so whatever it does now (no tests collected, a crash, a kill) is
                        # the patch's to own: every exit is scored.
                        budget = max(1, int(deadline - time.monotonic()) // (len(test_commands) - index))
                        proc = _run(command, cwd=repo_dir, timeout=budget)
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

                    # An uncertain edit gains nothing: failing tests score the normal 0 below, and a
                    # pass that edit may have caused can't be trusted, so it gets no verdict (masked).
                    if uncertain and passed_count:
                        return _path_verdict(
                            'validation_uncertain',
                            'Tests passed, but the patch edits paths the guard cannot classify, so the pass was not scored: ',
                            dict(sorted(uncertain.items())),
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
    ``test_command`` and scores 1.0 if it exits 0, else 0.0. An environment that can't run
    the tests, found before the patch is applied (the image lacks ``repo_dir`` or the
    command's runner, toolchain or module), has no verdict, nor does a blank
    ``test_command``; after the patch, every exit is scored. A diff that touches any
    other test infrastructure (tests and their helper code, conftest.py, runner config,
    installed dependencies) scores 0.0 with ``error_type='test_infra_tampering'``; data
    fixtures the tests read may change.

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
