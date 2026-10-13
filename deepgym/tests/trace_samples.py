"""Deterministic traces for the trace schema and exporter tests."""

import base64
from datetime import datetime, timezone

from deepgym.models import Action, MultiTurnEnvironment, Observation, RunResult
from deepgym.traces import AgentMeta, EnvFingerprint, Provenance, Trace, TraceMessage, TraceStep

PNG = b'\x89PNG\r\n\x1a\n sample screenshot'
STARTED = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

ENV = MultiTurnEnvironment(
    name='write_notes',
    task='Write hello to notes/out.txt, then finish.',
    step_verifier_code='print(0.5)\n',
    final_verifier_code='import json\nprint(json.dumps({"score": 1.0, "passed": True}))\n',
    image='python:3.12-slim@sha256:' + 'ab' * 32,
)


def _at(seconds: int) -> datetime:
    return STARTED.replace(second=seconds)


def _provenance() -> Provenance:
    return Provenance(
        env=EnvFingerprint.of(ENV, variant=2, seed=7, reference='print("ref")\n'),
        agent=AgentMeta(
            name='react-shell',
            version='1.2.0',
            model='Qwen/Qwen3-8B',
            sampling={'temperature': 0.7, 'top_p': 0.95},
        ),
        provider='local',
        deepgym_version='0.4.1',
        started_at=STARTED,
    )


def _result(**overrides: object) -> RunResult:
    fields = {
        'score': 1.0,
        'passed': True,
        'output': 'notes/out.txt says hello',
        'stderr': '',
        'exit_code': 0,
        'execution_time_ms': 12.5,
        'sandbox_id': 'local-session',
    }
    return RunResult.model_validate(fields | overrides)


def graded_trace() -> Trace:
    """A scored three-step episode: a command, a rejected action, then done."""
    shot = base64.b64encode(PNG).decode()
    return Trace(
        trace_id='trace0001',
        task=ENV.task,
        provenance=_provenance(),
        initial_observation=Observation(content='$ ', step=0, screenshot=shot),
        steps=(
            TraceStep(
                index=1,
                action=Action(
                    action_type='bash', content='mkdir -p notes && echo hello > notes/out.txt'
                ),
                messages=(
                    TraceMessage(role='system', content='You are a shell agent.'),
                    TraceMessage(role='user', content=f'{ENV.task}\n$ '),
                    TraceMessage(
                        role='assistant',
                        content='mkdir -p notes && echo hello > notes/out.txt',
                        token_ids=(11, 12, 13),
                        logprobs=(-0.25, -0.5, -0.125),
                    ),
                ),
                observation=Observation(content='', step=1, metadata={'exit_code': 0}),
                reward=0.5,
                step_verifier={'score': 0.5},
                timestamp=_at(1),
                duration_ms=40.0,
            ),
            TraceStep(
                index=2,
                action=Action(action_type='write_file', path='notes/extra.txt', content='x'),
                messages=(
                    TraceMessage(role='tool', content='exit 0'),
                    TraceMessage(role='assistant', content='write notes/extra.txt'),
                ),
                error="ActionError: Action 'write_file' not allowed here.",
                timestamp=_at(2),
                duration_ms=1.0,
            ),
            TraceStep(
                index=3,
                action=Action(action_type='done'),
                messages=(
                    TraceMessage(role='user', content='write_file is not allowed.'),
                    TraceMessage(role='assistant', content='done'),
                ),
                observation=Observation(content='', step=2, done=True),
                reward=0.5,
                step_verifier={'score': 0.5},
                timestamp=_at(3),
                duration_ms=30.0,
            ),
        ),
        result=_result(),
        finished_at=_at(4),
    )


def crashed_trace() -> Trace:
    """An episode cut off after one step, before scoring; no model messages recorded."""
    first = graded_trace().steps[0]
    return graded_trace().model_copy(
        update={
            'trace_id': 'trace0002',
            'steps': (first.model_copy(update={'messages': ()}),),
            'result': None,
            'error': 'SandboxError: sandbox vanished',
            'finished_at': None,
        }
    )


def unjudged_trace() -> Trace:
    """A finished episode whose final verifier failed to judge it."""
    return graded_trace().model_copy(
        update={
            'trace_id': 'trace0003',
            'result': _result(score=0.0, passed=False, error_type='sandbox_error', exit_code=2),
        }
    )
