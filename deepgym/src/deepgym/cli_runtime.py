"""Supervise opt-in CLI agents inside an isolated sandbox network namespace.

Phase 1 exposes the real provider API key to the model and its subprocesses. The
CONNECT proxy limits destination host and port, not HTTPS routes, account spending,
or request contents. Provider-reported usage is untrusted and delayed; the later
key-injecting relay can replace this transport without changing Session.run_cli.
"""

from __future__ import annotations

import base64
import codecs
import json
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from deepgym.exceptions import SandboxError

if TYPE_CHECKING:
    from deepgym.session import SandboxBackend

EventObserver = Callable[[dict[str, JsonValue]], str | None]
_WORKER = Path(__file__).with_name('_cli_worker.py')


class CliRequest(BaseModel):
    """Describe one explicitly authorized, unprivileged CLI process.

    Only the selected provider key is accepted in env. The opt-in acknowledges that
    the model can read and reuse it. argv must contain no credentials.
    """

    model_config = ConfigDict(frozen=True)
    argv: tuple[str, ...] = Field(min_length=1)
    env: dict[str, str] = Field(repr=False)
    provider: Literal['anthropic', 'openai']
    timeout: float = Field(gt=0, allow_inf_nan=False)
    allow_model_credentials: bool = False
    supported_version: str | None = None
    """Exact supported release for agent adapters; None for direct runtime callers."""

    @model_validator(mode='after')
    def check_credentials(self) -> CliRequest:
        """Accept exactly the chosen provider key; never arbitrary startup variables."""
        name = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'CODEX_API_KEY'}[self.provider]
        if set(self.env) != {name} or not self.env[name]:
            raise ValueError(f'CLI env must contain only a non-empty {name}')
        if any('\x00' in arg for arg in self.argv):
            raise ValueError('CLI argv cannot contain NUL')
        return self


class CliExecution(BaseModel):
    """Return protected capture of a CLI process after all agent processes stop."""

    model_config = ConfigDict(frozen=True)
    events: tuple[dict[str, JsonValue], ...]
    output: str
    exit_code: int
    stop_reason: str | None = None
    cli_version: str | None = None
    version_error: str | None = None


class CliExecutionError(SandboxError):
    """Preserve an observed CLI version when later supervision fails."""

    def __init__(self, message: str, cli_version: str | None) -> None:
        super().__init__(message)
        self.cli_version = cli_version


def execute_cli(
    backend: SandboxBackend, request: CliRequest, on_event: EventObserver
) -> CliExecution:
    """Run a sandbox CLI, poll protected output, and stop on observer or deadline limits.

    Args:
        backend: An isolated SandboxBackend with its exclusive agent UID.
        request: The process and explicit credential-exposure opt-in.
        on_event: Return a stop reason when a parsed JSON stdout event exceeds a budget.

    Returns:
        Captured output and nested events; no parsed command is executed again.

    Raises:
        SandboxError: If opt-in, namespace support, supervision, or cleanup is unavailable.
    """
    if not request.allow_model_credentials:
        raise SandboxError(
            'CLI agents require allow_model_credentials=True: the key is model-readable'
        )
    if not backend._offline or not backend._agent_namespace:
        raise SandboxError('CLI egress needs a retained isolated namespace and provider egress')
    deadline = time.monotonic() + request.timeout
    directory = f'/opt/deepgym-staging/cli-{uuid4().hex}'
    worker = f'{directory}/worker.py'
    config = {
        'argv': request.argv,
        'env': {**backend._agent_env, **request.env},
        'provider': request.provider,
        'timeout': request.timeout,
        'namespace': shlex.split(backend._offline),
        'prefix': [
            *shlex.split(backend._offline + backend._agent_namespace),
            '/usr/bin/unshare',
            '--pid',
            '--fork',
            '--mount-proc',
            '--kill-child=KILL',
            '--',
            *shlex.split(backend._agent_drop),
        ],
        'uid': backend._agent_uid,
        'cwd': backend.workdir,
        'supported_version': request.supported_version,
        'version_env': backend._agent_env,
    }
    events: list[dict[str, JsonValue]] = []
    chunks: list[str] = []
    pending = ''
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    offset = 0
    reason = None
    status = None
    launch_attempted = False
    cleanup_deadline = deadline + 20
    try:
        _checked(backend, f'mkdir -m 700 {directory}')
        backend._sandbox.upload(_WORKER.read_bytes(), worker)
        config['timeout'] = deadline - time.monotonic()
        if config['timeout'] <= 0:
            return CliExecution(events=(), output='', exit_code=-1, stop_reason='time_budget')
        backend._sandbox.upload(json.dumps(config).encode(), f'{directory}/request.json')
        command = f'python3 -I {worker} supervise {directory}'
        launch_attempted = True
        _checked(backend, f'setsid {command} </dev/null >/dev/null 2>&1 &')
        while status is None:
            if reason is None and time.monotonic() >= deadline:
                reason = 'time_budget'
                _request_stop(backend, directory, reason)
            packet = _poll(backend, directory, offset)
            raw = base64.b64decode(packet['data'])
            offset += len(raw)
            text = decoder.decode(raw, final=packet['status'] is not None)
            chunks.append(text)
            pending += text
            status = packet['status']
            lines = pending.split('\n')
            pending = lines.pop()
            if status is not None and pending:
                lines.append(pending)
                pending = ''
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    event = _redact_event(event, request)
                    events.append(event)
                    observed_reason = on_event(event)
                    if reason is None and observed_reason:
                        reason = observed_reason
                        _request_stop(backend, directory, reason)
            if time.monotonic() > cleanup_deadline:
                raise SandboxError('CLI supervisor did not finish before its cleanup deadline')
            if status is None:
                time.sleep(0.1)
        if status.get('error'):
            raise CliExecutionError(
                f'CLI supervision failed: {status["error"]}', status.get('cli_version')
            )
        return CliExecution(
            events=tuple(events),
            output=_redact_text(''.join(chunks), request),
            exit_code=status['exit_code'],
            stop_reason=reason or status['stop_reason'],
            cli_version=status.get('cli_version'),
            version_error=status.get('version_error'),
        )
    finally:
        failures = []
        try:
            # Setup descendants also need teardown when staging exhausts the deadline.
            backend.stop_agent()
        except Exception as exc:
            failures.append(f'Agent containment cleanup failed: {type(exc).__name__}')
        if launch_attempted and status is None:
            try:
                _request_stop(backend, directory, 'cancelled')
                for _ in range(100):
                    if _poll(backend, directory, offset)['status'] is not None:
                        break
                    time.sleep(0.1)
                else:
                    raise SandboxError('CLI supervisor cleanup was not acknowledged')
            except Exception as exc:
                failures.append(f'CLI transport cleanup failed: {type(exc).__name__}')
        try:
            _checked(backend, f'rm -rf -- {directory}')
        except Exception as exc:
            failures.append(f'CLI credential staging cleanup failed: {type(exc).__name__}')
        if failures:
            raise CliExecutionError('; '.join(failures), (status or {}).get('cli_version'))


def _checked(backend: SandboxBackend, command: str) -> str:
    result = backend.exec_trusted(command, 30)
    if result.exit_code:
        raise SandboxError(f'CLI supervisor operation failed: {result.output}')
    return result.output


def _redact_text(text: str, request: CliRequest) -> str:
    for value in request.env.values():
        text = text.replace(value, '[REDACTED]')
    return text


def _redact_event(event: JsonValue, request: CliRequest) -> JsonValue:
    if isinstance(event, str):
        return _redact_text(event, request)
    if isinstance(event, list):
        return [_redact_event(item, request) for item in event]
    if isinstance(event, dict):
        return {
            _redact_text(key, request): _redact_event(value, request)
            for key, value in event.items()
        }
    return event


def _request_stop(backend: SandboxBackend, directory: str, reason: str) -> None:
    script = f'from pathlib import Path; Path({directory + "/stop"!r}).write_text({reason!r})'
    _checked(backend, f'python3 -I -c {shlex.quote(script)}')


def _poll(backend: SandboxBackend, directory: str, offset: int) -> dict:
    script = (
        'import base64,json; from pathlib import Path; '
        f'p=Path({directory!r}); f=p/"output"; s=p/"status.json"; '
        'status=json.loads(s.read_text()) if s.exists() else None; '
        f'h=f.open("rb") if f.exists() else None; '
        f'h.seek({offset}) if h else None; d=h.read() if h else b""; '
        'print(json.dumps({"data":base64.b64encode(d).decode(),'
        '"status":status}))'
    )
    return json.loads(_checked(backend, f'python3 -I -c {shlex.quote(script)}'))
