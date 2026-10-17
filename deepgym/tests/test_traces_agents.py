"""Offline checks for capturing supervised CLI launches and agent session access."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deepgym.agents import Budget, ClaudeCodeAgent
from deepgym.agents.versions import SUPPORTED_CLIS
from deepgym.cli_runtime import CliExecution, CliExecutionError, CliRequest, EventObserver
from deepgym.exceptions import DeepGymError, SandboxError
from deepgym.models import Action, MultiTurnEnvironment
from deepgym.sandboxes import ExecResult
from deepgym.session import SandboxBackend, Session
from deepgym.traces import (
    AgentMeta,
    EnvFingerprint,
    Provenance,
    TracedSession,
    TraceMessage,
    export_traces,
    load_trace,
)
from deepgym.traces.atif import to_atif


class RecordedBackend(SandboxBackend):
    """Return recorded output without launching a process, sandbox, or provider call."""

    sandbox_id = 'recorded'

    def __init__(self) -> None:
        self.executions = 0
        self.shell_calls: list[tuple[str, float]] = []
        self.failure: Exception | None = None
        self.result = CliExecution(
            events=(
                {
                    'type': 'item.completed',
                    'item': {
                        'id': 'cmd-1',
                        'type': 'command_execution',
                        'command': 'touch already_executed',
                        'aggregated_output': '',
                    },
                },
                {'type': 'turn.completed', 'usage': {'input_tokens': 12, 'output_tokens': 3}},
            ),
            output='recorded CLI stdout',
            exit_code=0,
            cli_version=SUPPORTED_CLIS['openai'].version,
        )

    def run_cli(self, request: CliRequest, on_event: EventObserver) -> CliExecution:
        self.executions += 1
        for event in self.result.events:
            on_event(event)
        if self.failure:
            raise self.failure
        return self.result

    def exec(self, command: str, timeout: float, env: dict[str, str] | None = None) -> ExecResult:
        self.shell_calls.append((command, timeout))
        return ExecResult(0, 'shell output')

    def close(self) -> None:
        pass


def make_trace(path: Path, backend: RecordedBackend) -> TracedSession:
    env = MultiTurnEnvironment(task='Complete the fixture task', cli_provider_egress=True)
    provenance = Provenance(
        env=EnvFingerprint.of(env),
        agent=AgentMeta(name='codex'),
        provider='recorded',
        deepgym_version='test',
        started_at=datetime.now(timezone.utc),
    )
    return TracedSession(Session(env, backend), env, provenance, path)


def request() -> CliRequest:
    return CliRequest(
        argv=('codex', 'exec', '--json', 'Complete the fixture task'),
        env={'CODEX_API_KEY': 'fixture-key-not-a-secret'},
        provider='openai',
        timeout=30,
        allow_model_credentials=True,
    )


def test_cli_capture_round_trips_nested_events_without_replaying(tmp_path: Path) -> None:
    backend = RecordedBackend()
    received = []
    with make_trace(tmp_path / 'episode.jsonl', backend) as traced:
        traced.start()
        result = traced.run_cli(request(), lambda event: received.append(event))
        assert result == backend.result
        assert traced.done
        assert len(traced.trajectory.steps) == 1
        assert len(traced.trace.steps) == 1
        assert traced.trace.provenance.agent.version == SUPPORTED_CLIS['openai'].version
        assert received == list(result.events)
        assert backend.executions == 1
        assert backend.shell_calls == []
        # Neither the returned process result nor an agent's trajectory can alter the trace.
        result.events[0]['type'] = 'mutated'
        traced.trajectory.steps.clear()
        assert traced.trace.steps[0].action.events[0]['type'] == 'item.completed'
        assert len(traced.trajectory.steps) == 1

    restored = load_trace(traced.path)
    assert restored.provenance.agent.version == SUPPORTED_CLIS['openai'].version
    action = restored.steps[0].action
    assert action.action_type == 'cli'
    assert action.events[0]['item']['command'] == 'touch already_executed'
    assert 'fixture-key-not-a-secret' not in traced.path.read_text()
    document, _ = to_atif(restored)
    agent_steps = [step for step in document['steps'] if step['source'] == 'agent']
    assert len(agent_steps) == 1
    step = agent_steps[0]
    assert len(step['tool_calls']) == 1
    assert step['tool_calls'][0]['function_name'] == 'cli'
    assert 'events' not in step['tool_calls'][0]['arguments']
    assert step['observation']['results'][0]['extra']['cli_events'] == list(action.events)


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_no_launch_cli_finalization_is_forwarded_without_a_trace_action(
    monkeypatch, tmp_path: Path, cleanup_fails: bool
) -> None:
    backend = RecordedBackend()
    stops = []

    def stop() -> None:
        stops.append('stop')
        if cleanup_fails:
            raise SandboxError('setup population still active')

    monkeypatch.setattr(backend, 'stop_agent', stop)
    with make_trace(tmp_path / 'no-launch.jsonl', backend) as traced:
        initial = traced.start()
        agent = ClaudeCodeAgent(model='m')
        if cleanup_fails:
            with pytest.raises(SandboxError, match='setup population'):
                agent.run(traced, initial, task='task', budget=Budget(max_cost_usd=0))
            assert not traced.score().has_verdict
        else:
            result = agent.run(traced, initial, task='task', budget=Budget(max_cost_usd=0))
            assert result.termination_reason == 'cost_budget'
            assert result.trajectory.final_observation.done
            traced.finalize_cli()
        assert traced.done
        assert traced.trace.steps == ()
        assert backend.executions == 0
    assert stops == ['stop']
    if cleanup_fails:
        restored = load_trace(traced.path)
        assert 'setup population' in restored.error
        assert not restored.has_verdict


def test_cli_failure_preserves_observed_version_in_persisted_trace(tmp_path: Path) -> None:
    backend = RecordedBackend()
    backend.failure = CliExecutionError('supervision failed', SUPPORTED_CLIS['openai'].version)
    with make_trace(tmp_path / 'failed-version.jsonl', backend) as traced:
        traced.start()
        with pytest.raises(CliExecutionError):
            traced.run_cli(request(), lambda event: None)
    restored = load_trace(traced.path)
    assert restored.provenance.agent.version == SUPPORTED_CLIS['openai'].version


def test_cli_failure_records_the_single_applied_action(tmp_path: Path) -> None:
    backend = RecordedBackend()
    backend.failure = SandboxError('fixture supervision failure')
    with make_trace(tmp_path / 'failed.jsonl', backend) as traced:
        traced.start()
        with pytest.raises(SandboxError, match='fixture supervision failure'):
            traced.run_cli(request(), lambda _: None)
        assert traced.done
    restored = load_trace(traced.path)
    assert len(restored.steps) == 1
    assert restored.steps[0].applied
    assert restored.steps[0].action.action_type == 'cli'
    assert restored.steps[0].action.events == backend.result.events
    assert restored.steps[0].error == 'SandboxError: fixture supervision failure'


def test_cli_capture_rejects_unstarted_and_scored_sessions(tmp_path: Path) -> None:
    backend = RecordedBackend()
    with make_trace(tmp_path / 'blocked.jsonl', backend) as traced:
        with pytest.raises(DeepGymError, match='not running'):
            traced.run_cli(request(), lambda _: None)
        traced.start()
        traced.score()
        with pytest.raises(DeepGymError, match='already scored'):
            traced.run_cli(request(), lambda _: None)
    assert backend.executions == 0


def test_traced_step_forwards_deadline_and_copies_trajectory(tmp_path: Path) -> None:
    backend = RecordedBackend()
    with make_trace(tmp_path / 'step.jsonl', backend) as traced:
        traced.start()
        result = traced.step(Action(action_type='bash', content='echo fixture'), timeout=0.25)
        assert not result.done
        command, timeout = backend.shell_calls[0]
        assert command == 'echo fixture'
        assert 0 < timeout <= 0.25
        assert traced.trace.steps[0].observation.content == 'shell output'
        trajectory = traced.trajectory
        trajectory.steps.clear()
        assert len(traced.trajectory.steps) == 1


def test_cli_events_survive_parquet_exports(tmp_path: Path) -> None:
    parquet = pytest.importorskip('pyarrow.parquet')
    backend = RecordedBackend()
    with make_trace(tmp_path / 'episode.jsonl', backend) as traced:
        traced.start()
        traced.run_cli(request(), lambda _: None)
    original = traced.trace
    export_traces([original], 'parquet', tmp_path / 'trace.parquet')
    (summary,) = parquet.read_table(tmp_path / 'trace.parquet').to_pylist()
    assert json.loads(summary['trace'])['steps'][0]['action']['events'] == list(
        backend.result.events
    )

    # Explicitly supplied transcript messages make this trace eligible for RL export.
    step = original.steps[0].model_copy(
        update={
            'messages': (TraceMessage(role='assistant', content='Run the CLI'),),
        }
    )
    annotated = original.model_copy(update={'steps': (step,)})
    report = export_traces([annotated], 'parquet', tmp_path / 'rl.parquet', rows='rl')
    assert report.written == 1
    (row,) = parquet.read_table(tmp_path / 'rl.parquet').to_pylist()
    events = [json.loads(event) for event in row['steps'][0]['action']['events']]
    assert events == list(backend.result.events)
    assert annotated.steps[0].action.events == backend.result.events
