"""Replay schema-derived CLI samples offline, without provider or sandbox calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, SecretStr

from deepgym import DeepGym
from deepgym.agents.base import Budget, BudgetTracker, TokenPrices
from deepgym.agents.cli import ClaudeCodeAgent, CodexAgent
from deepgym.agents.cli_events import CliEvents
from deepgym.agents.versions import SUPPORTED_CLIS
from deepgym.cli_runtime import CliExecution, CliRequest
from deepgym.exceptions import ActionError, SandboxError
from deepgym.models import Action, MultiTurnEnvironment, Observation, Trajectory
from deepgym.sandboxes import ExecResult
from deepgym.session import SandboxBackend, Session, VerifierOutput

FIXTURES = Path(__file__).parent / 'fixtures'
PRICES = TokenPrices(
    input_per_million=2,
    output_per_million=4,
    cache_read_per_million=1,
    cache_write_per_million=3,
    source='offline test rates, not real model prices',
)


def stream(name: str) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in (FIXTURES / f'{name}.jsonl').read_text().splitlines()]


def parser(provider: str, **budget: float | int | None) -> CliEvents:
    return CliEvents(provider, BudgetTracker(Budget(**budget)), PRICES)


def test_claude_aggregate_usage_replaces_message_sum() -> None:
    events = parser('anthropic')
    for event in stream('claude'):
        assert events.consume(event) is None
    assert events.finish(0, None) == 'completed'
    assert events.tracker.turns == 2
    assert events.tracker.usage.model_dump() == {
        'input_tokens': 30,
        'output_tokens': 13,
        'cache_read_tokens': 10,
        'cache_write_tokens': 2,
    }
    assert events.tracker.cost.usd == 0.0012
    assert 'CLI-reported' in events.tracker.cost.provenance


def test_codex_usage_subtracts_cached_input_and_estimates_cost() -> None:
    events = parser('openai')
    for event in stream('codex'):
        assert events.consume(event) is None
    assert events.finish(0, None) == 'completed'
    assert events.tracker.turns == 2
    assert events.tracker.usage.input_tokens == 30
    assert events.tracker.usage.cache_read_tokens == 10
    assert events.tracker.cost.usd == pytest.approx(0.000118)
    assert PRICES.source in events.tracker.cost.provenance


@pytest.mark.parametrize('provider,name', [('anthropic', 'claude'), ('openai', 'codex')])
def test_activity_budget_stops_stream_before_next_item(provider: str, name: str) -> None:
    events = parser(provider, max_turns=1)
    consumed = 0
    for event in stream(name):
        consumed += 1
        if events.consume(event):
            break
    assert consumed < len(stream(name))
    assert events.finish(-9, 'turn_budget') == 'turn_budget'
    assert events.tracker.turns == 1
    assert 'incomplete' in events.tracker.cost.provenance


def test_duplicate_assistant_message_does_not_double_count() -> None:
    events = parser('anthropic')
    event = stream('claude')[1]
    events.consume(event)
    events.consume(event)
    assert events.tracker.turns == 1
    assert events.tracker.usage.input_tokens == 20


def test_same_message_growing_usage_crosses_live_cost_cap() -> None:
    events = parser('anthropic', max_cost_usd=0.0001)
    event = stream('claude')[1]
    assert events.consume(event) is None
    event['message']['usage']['output_tokens'] = 100
    assert events.consume(event) == 'cost_budget'
    assert events.tracker.turns == 1
    assert events.tracker.usage.output_tokens == 100
    assert events.tracker.usage.input_tokens == 20


def test_same_message_usage_regression_fails_closed() -> None:
    events = parser('anthropic')
    event = stream('claude')[1]
    assert events.consume(event) is None
    event['message']['usage']['output_tokens'] = 1
    assert events.consume(event) == 'error'
    assert 'regressed' in events.error


def test_cumulative_result_cannot_erase_observed_usage() -> None:
    events = parser('anthropic')
    for event in stream('claude')[:-1]:
        assert events.consume(event) is None
    result = stream('claude')[-1]
    result['usage']['input_tokens'] = 1
    assert events.consume(result) == 'error'
    assert events.tracker.usage.input_tokens == 30


def test_duplicate_codex_item_does_not_double_count() -> None:
    events = parser('openai')
    event = stream('codex')[3]
    events.consume(event)
    events.consume(event)
    assert events.tracker.turns == 1


def test_usage_after_activity_stop_is_accounted_without_changing_reason() -> None:
    events = parser('openai', max_turns=1)
    assert events.consume(stream('codex')[3]) == 'turn_budget'
    assert events.consume(stream('codex')[-1]) == 'turn_budget'
    assert events.tracker.usage.input_tokens == 30
    assert events.tracker.usage.output_tokens == 12
    assert events.tracker.cost.usd == pytest.approx(0.000118)
    assert events.finish(-9, 'turn_budget') == 'turn_budget'


def test_claude_estimated_spend_stops_on_observed_usage() -> None:
    events = parser('anthropic', max_cost_usd=0.000001)
    assert events.consume(stream('claude')[1]) == 'cost_budget'
    assert events.finish(-9, None) == 'cost_budget'


def test_deadline_is_checked_while_events_arrive() -> None:
    events = parser('openai')
    events.tracker.started -= 301
    assert events.consume({'type': 'thread.started'}) == 'time_budget'


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_zero_exit_requires_terminal_event(provider: str) -> None:
    events = parser(provider)
    assert events.finish(0, None) == 'error'
    assert 'without a terminal' in events.error


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_nonzero_exit_is_error(provider: str) -> None:
    events = parser(provider)
    assert events.finish(127, None) == 'error'
    assert '127' in events.error


@pytest.mark.parametrize('exit_code', [0, 1])
def test_crash_or_nonterminal_exit_labels_observed_cost_incomplete(exit_code: int) -> None:
    events = parser('anthropic')
    assert events.consume(stream('claude')[1]) is None
    observed_usage = events.tracker.usage
    observed_cost = events.tracker.cost

    assert events.finish(exit_code, None) == 'error'
    assert events.tracker.usage == observed_usage
    assert events.tracker.cost.usd == observed_cost.usd
    assert observed_cost.provenance in events.tracker.cost.provenance
    assert 'incomplete observed CLI subtotal' in events.tracker.cost.provenance
    assert 'additional usage may be unreported' in events.tracker.cost.provenance


def test_successful_terminal_accounting_is_not_labeled_incomplete() -> None:
    events = parser('anthropic')
    for event in stream('claude'):
        assert events.consume(event) is None
    assert events.finish(0, None) == 'completed'
    assert 'incomplete' not in events.tracker.cost.provenance


@pytest.mark.parametrize(
    'subtype,reason',
    [
        ('error_max_turns', 'turn_budget'),
        ('error_max_budget_usd', 'cost_budget'),
        ('error_during_execution', 'error'),
    ],
)
def test_claude_terminal_errors(subtype: str, reason: str) -> None:
    events = parser('anthropic')
    events.consume({'type': 'result', 'subtype': subtype, 'is_error': True})
    assert events.finish(1, None) == reason


@pytest.mark.parametrize(
    'event,reason',
    [
        ({'type': 'turn.failed', 'error': {'message': 'provider refused'}}, 'error'),
        ({'type': 'error', 'message': 'provider refused'}, None),
    ],
)
def test_codex_error_events(event: dict[str, JsonValue], reason: str | None) -> None:
    events = parser('openai')
    assert events.consume(event) == reason
    if reason:
        assert events.error == 'provider refused'
    else:
        assert events.error is None
        assert events.finish(1, None) == 'error'


@pytest.mark.parametrize('retry_flag', [True, False, None])
def test_codex_standalone_error_waits_for_terminal_evidence(retry_flag: bool | None) -> None:
    events = parser('openai')
    error = {'type': 'error', 'message': 'No prose classification'}
    if retry_flag is not None:
        error['willRetry'] = retry_flag
    assert events.consume(error) is None
    for event in stream('codex'):
        assert events.consume(event) is None
    assert events.finish(0, None) == 'completed'


@pytest.mark.parametrize(
    'filename,reason',
    [('retry-then-success', 'completed'), ('terminal-provider-error', 'provider_error')],
)
def test_codex_error_scenario_fixtures(filename: str, reason: str) -> None:
    release = SUPPORTED_CLIS['openai']
    path = FIXTURES / release.command / release.version / f'{filename}.jsonl'
    events = parser('openai')
    for event in map(json.loads, path.read_text().splitlines()):
        observed = events.consume(event)
        assert observed == ('provider_error' if event['type'] == 'turn.failed' else None)
    assert events.finish(0 if reason == 'completed' else 1, None) == reason
    if reason == 'provider_error':
        assert 'incomplete observed CLI subtotal' in events.tracker.cost.provenance


@pytest.mark.parametrize(
    'code,reason',
    [
        ('rateLimitExceeded', 'provider_error'),
        ('unauthorized', 'provider_error'),
        ('serverOverloaded', 'provider_error'),
        ({'responseStreamDisconnected': {}}, 'provider_error'),
        ({'httpConnectionFailed': {'httpStatusCode': 503}}, 'provider_error'),
        ({'httpConnectionFailed': {'httpStatusCode': 400}}, 'error'),
        ({'httpConnectionFailed': {'httpStatusCode': '503'}}, 'error'),
        ('contextWindowExceeded', 'error'),
        ('sandboxError', 'error'),
        ('unknownFutureCode', 'error'),
        (None, 'error'),
    ],
)
def test_codex_provider_classification_uses_only_structured_fields(code, reason: str) -> None:
    events = parser('openai')
    event = {
        'type': 'turn.failed',
        'error': {'message': 'HTTP 503 rate limit unauthorized', 'codexErrorInfo': code},
    }
    assert events.consume(event) == reason
    assert events.finish(1, None) == reason


@pytest.mark.parametrize('provider_failure', [True, False])
def test_first_terminal_error_reason_survives_later_events(provider_failure: bool) -> None:
    events = parser('openai')
    reason = 'provider_error' if provider_failure else 'error'
    code = 'rateLimitExceeded' if provider_failure else 'sandboxError'
    assert (
        events.consume(
            {'type': 'turn.failed', 'error': {'message': 'first', 'codexErrorInfo': code}}
        )
        == reason
    )
    assert events.consume({}) == reason
    assert (
        events.consume(
            {
                'type': 'turn.failed',
                'error': {'message': 'later', 'codexErrorInfo': 'serverOverloaded'},
            }
        )
        == reason
    )
    assert events.finish(1, None) == reason


@pytest.mark.parametrize(
    'event',
    [
        {},
        {'type': 'item.completed', 'item': []},
        {'type': 'item.completed', 'item': {'id': 'a', 'type': 'agent_message', 'text': 2}},
        {'type': 'turn.completed', 'usage': {'input_tokens': -1, 'output_tokens': 1}},
        {'type': 'turn.completed', 'usage': {'input_tokens': True, 'output_tokens': 1}},
        {
            'type': 'turn.completed',
            'usage': {'input_tokens': 1, 'output_tokens': 1, 'cached_input_tokens': 2},
        },
    ],
)
def test_malformed_events_fail_closed(event: dict[str, JsonValue]) -> None:
    events = parser('openai')
    assert events.consume(event) == 'error'
    assert 'Invalid openai CLI event' in events.error


def test_unknown_event_is_preserved_by_caller_without_inventing_actions() -> None:
    events = parser('openai')
    assert events.consume({'type': 'future.event', 'payload': {'anything': True}}) is None
    assert events.tracker.turns == 0


class RecordingSession:
    """Model the supervised Session boundary, not Daytona or a provider SDK."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.done = False
        self.trajectory = Trajectory()
        self.requests: list[CliRequest] = []
        self.finalizations = 0
        self.abandoned: list[tuple[str, str]] = []

    def abandon(self, error_type: str, detail: str) -> None:
        self.abandoned.append((error_type, detail))
        self.done = True

    def finalize_cli(self) -> None:
        self.finalizations += 1
        self.done = True
        self.trajectory.final_observation = Observation(content='', step=0, done=True)

    def run_cli(
        self,
        request: CliRequest,
        consume: Callable[[dict[str, JsonValue]], str | None],
    ) -> CliExecution:
        self.requests.append(request)
        events = []
        reason = None
        for event in stream(self.name):
            events.append(event)
            if reason := consume(event):
                break
        self.trajectory.steps.append(
            (
                Observation(content='', step=0),
                Action(action_type='cli', content='installed CLI', events=events),
            )
        )
        return CliExecution(
            events=tuple(events),
            output='',
            exit_code=0,
            stop_reason=reason,
            cli_version=request.supported_version,
        )

    def step(self, action: Action) -> None:
        pytest.fail(f'Parsed CLI events must never be replayed: {action}')


@pytest.mark.parametrize('agent_type,name', [(ClaudeCodeAgent, 'claude'), (CodexAgent, 'codex')])
def test_run_once_records_nested_events_without_replay(agent_type: type, name: str) -> None:
    session = RecordingSession(name)
    agent = agent_type(
        model='fixture-model',
        api_key='secret-fixture',
        allow_model_credentials=True,
        prices=PRICES,
    )
    result = agent.run(
        cast(Session, session),
        Observation(content='', step=0),
        task='literal $(task)',
        budget=Budget(),
    )
    assert result.termination_reason == 'completed'
    assert result.cli_version == SUPPORTED_CLIS[agent.provider].version
    assert len(result.trajectory.steps) == len(session.requests) == 1
    assert result.trajectory.steps[0][1].events == tuple(stream(name))
    assert session.requests[0].argv[-1] == 'literal $(task)'
    assert 'secret-fixture' not in repr(agent)
    assert 'secret-fixture' not in result.model_dump_json()
    assert not any('secret-fixture' in arg for arg in session.requests[0].argv)
    assert session.finalizations == 1
    assert result.trajectory.final_observation.done


@pytest.mark.parametrize('agent_type', [ClaudeCodeAgent, CodexAgent])
def test_credentials_require_explicit_opt_in(agent_type: type) -> None:
    session = RecordingSession('claude')
    with pytest.raises(ValueError, match='allow_model_credentials=True'):
        agent_type(model='m', api_key='k').run(
            cast(Session, session),
            Observation(content='', step=0),
            task='task',
            budget=Budget(),
        )
    assert not session.requests
    assert session.finalizations == 1


@pytest.mark.parametrize('cap', [0, 1])
def test_codex_finite_usd_budget_rejected_before_launch(cap: float) -> None:
    session = RecordingSession('codex')
    agent = CodexAgent(model='m', api_key='k', allow_model_credentials=True, prices=PRICES)
    with pytest.raises(ValueError, match='finite USD budget'):
        agent.run(
            cast(Session, session),
            Observation(content='', step=0),
            task='task',
            budget=Budget(max_cost_usd=cap),
        )
    assert not session.requests


def test_claude_native_dollar_budget() -> None:
    session = RecordingSession('claude')
    agent = ClaudeCodeAgent(model='m', api_key='k', allow_model_credentials=True, prices=PRICES)
    agent.run(
        cast(Session, session),
        Observation(content='', step=0),
        task='task',
        budget=Budget(
            max_turns=8,
            max_seconds=22,
            max_cost_usd=3,
        ),
    )
    request = session.requests[0]
    assert request.argv[request.argv.index('--max-budget-usd') + 1] == '3.0'
    assert 0 < request.timeout <= 22


def test_claude_finite_budget_requires_prices_before_launch() -> None:
    session = RecordingSession('claude')
    agent = ClaudeCodeAgent(model='m', api_key='k', allow_model_credentials=True)
    with pytest.raises(ValueError, match='TokenPrices'):
        agent.run(
            cast(Session, session),
            Observation(content='', step=0),
            task='task',
            budget=Budget(max_cost_usd=1),
        )
    assert not session.requests


def test_claude_missing_per_message_usage_fails_closed() -> None:
    events = parser('anthropic', max_cost_usd=1)
    assert (
        events.consume(
            {
                'type': 'assistant',
                'message': {'id': 'missing-usage', 'content': []},
            }
        )
        == 'error'
    )
    assert 'assistant usage' in events.error


def test_claude_without_usd_budget_accepts_final_only_usage() -> None:
    events = parser('anthropic')
    event = stream('claude')[1]
    del event['message']['usage']
    assert events.consume(event) is None
    assert events.tracker.cost.usd is None
    assert events.consume(stream('claude')[-1]) is None
    assert events.finish(0, None) == 'completed'
    assert events.tracker.usage.input_tokens == 30
    assert events.tracker.cost.usd == 0.0012


def test_claude_missing_message_and_final_usage_is_error() -> None:
    events = parser('anthropic')
    event = stream('claude')[1]
    del event['message']['usage']
    assert events.consume(event) is None
    result = stream('claude')[-1]
    del result['usage']
    assert events.consume(result) is None
    assert events.finish(0, None) == 'error'
    assert 'without token usage' in events.error


def test_claude_zero_dollar_budget_does_not_launch() -> None:
    session = RecordingSession('claude')
    result = ClaudeCodeAgent(model='m').run(
        cast(Session, session),
        Observation(content='', step=0),
        task='task',
        budget=Budget(max_cost_usd=0),
    )
    assert result.termination_reason == 'cost_budget'
    assert not session.requests


def test_done_session_does_not_launch() -> None:
    session = RecordingSession('claude')
    session.done = True
    result = ClaudeCodeAgent(model='m').run(
        session,
        Observation(content='', step=0),
        task='task',
        budget=Budget(),
    )
    assert result.termination_reason == 'session_done'
    assert session.finalizations == 1


def test_exhausted_wall_budget_does_not_launch() -> None:
    session = RecordingSession('claude')
    agent = ClaudeCodeAgent(model='m', api_key='k', allow_model_credentials=True)
    result = agent.run(
        cast(Session, session),
        Observation(content='', step=0),
        task='task',
        budget=Budget(max_seconds=1e-100),
    )
    assert result.termination_reason == 'time_budget'
    assert not session.requests


@pytest.mark.parametrize('expired,reason', [(False, 'error'), (True, 'time_budget')])
def test_failure_retains_partial_accounting_and_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    expired: bool,
    reason: str,
) -> None:
    now = [0.0]
    monkeypatch.setattr('deepgym.agents.base.time.monotonic', lambda: now[0])

    class FailingSession(RecordingSession):
        def run_cli(
            self,
            request: CliRequest,
            consume: Callable[[dict[str, JsonValue]], str | None],
        ) -> CliExecution:
            event = stream('claude')[1]
            consume(event)
            self.trajectory.steps.append(
                (
                    Observation(content='', step=0),
                    Action(action_type='cli', content='installed CLI', events=(event,)),
                )
            )
            if expired:
                now[0] = 301.0
            raise SandboxError('provider failure includes secret-fixture')

    session = FailingSession('claude')
    agent = ClaudeCodeAgent(
        model='m',
        api_key='secret-fixture',
        allow_model_credentials=True,
        prices=PRICES,
    )
    result = agent.run(
        cast(Session, session),
        Observation(content='', step=0),
        task='task',
        budget=Budget(),
    )
    assert result.termination_reason == reason
    assert result.error == 'SandboxError'
    assert result.usage.input_tokens == 20
    assert result.cost.usd > 0
    assert 'incomplete' in result.cost.provenance
    assert len(result.trajectory.steps) == 1
    assert 'secret-fixture' not in result.model_dump_json()


class SetupWriterBackend(SandboxBackend):
    """Track setup population and grading order without creating processes or a sandbox."""

    def __init__(self, *, cleanup_fails: bool = False) -> None:
        self.sandbox_id = 'setup-writer-fixture'
        self.events: list[str] = []
        self.writer_alive = False
        self.cleanup_fails = cleanup_fails

    def exec(
        self,
        command: str,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self.events.append('setup')
        self.writer_alive = True
        return ExecResult(0, 'setup writer started')

    def stop_agent(self) -> None:
        self.events.append('stop')
        if self.cleanup_fails:
            raise SandboxError('fixture cleanup failed')
        self.writer_alive = False

    def verify(
        self,
        code: str,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> VerifierOutput:
        self.events.append('verify')
        assert not self.writer_alive, 'grading ran while setup writer was alive'
        return VerifierOutput(0, '{"score":1.0,"passed":true}', '')

    def run_cli(
        self,
        request: CliRequest,
        consume: Callable[[dict[str, JsonValue]], str | None],
    ) -> CliExecution:
        pytest.fail('A preflight return must not launch the CLI')

    def close(self) -> None:
        self.events.append('close')


@pytest.mark.parametrize('agent_type', [ClaudeCodeAgent, CodexAgent])
def test_cli_agent_requires_explicit_environment_egress(agent_type) -> None:
    backend = SetupWriterBackend()
    session = Session(MultiTurnEnvironment(task='task', setup_script='start writer'), backend)
    initial = session.start()
    agent = agent_type(model='m', api_key='fixture-key', allow_model_credentials=True)
    with pytest.raises(ActionError, match='set cli_provider_egress=True.*before creating'):
        agent.run(session, initial, task='task', budget=Budget())
    assert backend.events == ['setup', 'stop']
    assert session.trajectory.steps == []


@pytest.mark.parametrize('agent_type,name', [(ClaudeCodeAgent, 'claude'), (CodexAgent, 'codex')])
def test_finalization_preserves_unknown_state_after_partial_cli_failure(
    agent_type: type[ClaudeCodeAgent] | type[CodexAgent], name: str
) -> None:
    partial = next(
        event for event in stream(name) if event['type'] in ('assistant', 'item.completed')
    )

    class MutatingBackend(SetupWriterBackend):
        def run_cli(
            self, request: CliRequest, consume: Callable[[dict[str, JsonValue]], str | None]
        ) -> CliExecution:
            self.events.append('mutated')
            consume(partial)
            raise SandboxError('transport failed after possible side effects')

    backend = MutatingBackend()
    session = Session(
        MultiTurnEnvironment(task='task', setup_script='start writer', cli_provider_egress=True),
        backend,
    )
    initial = session.start()
    agent = agent_type(model='m', api_key='fixture-key', allow_model_credentials=True)
    result = agent.run(session, initial, task='task', budget=Budget())

    assert result.termination_reason == 'error'
    assert session.done
    assert result.trajectory.final_observation is None
    assert session.trajectory.final_observation is None
    assert len(result.trajectory.steps) == 1
    before, action = result.trajectory.steps[0]
    assert before.content == initial.content
    assert action.action_type == 'cli'
    assert action.events == (partial,)
    assert backend.events == ['setup', 'mutated', 'stop']
    assert not backend.writer_alive


@pytest.mark.parametrize('reason', ['cost_budget', 'time_budget'])
@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_no_launch_episode_cleans_setup_before_grading(
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    cleanup_fails: bool,
) -> None:
    now = [0.0]
    monkeypatch.setattr('deepgym.agents.base.time.monotonic', lambda: now[0])
    environment = MultiTurnEnvironment(
        task='fixture task',
        setup_script='start fixture writer',
        final_verifier_code='return 1.0',
    )
    backend = SetupWriterBackend(cleanup_fails=cleanup_fails)
    session = Session(environment, backend)
    client = DeepGym(mode='local')
    monkeypatch.setattr(client, 'session', lambda env: session)
    agent = ClaudeCodeAgent(model='m', api_key='fixture-key', allow_model_credentials=True)
    budget = Budget(max_cost_usd=0) if reason == 'cost_budget' else Budget(max_seconds=1)

    if reason == 'time_budget':
        original = SecretStr.get_secret_value

        def expire_preflight(key: SecretStr) -> str:
            assert backend.writer_alive
            now[0] = 2.0
            return original(key)

        monkeypatch.setattr(SecretStr, 'get_secret_value', expire_preflight)

    if cleanup_fails:
        with pytest.raises(SandboxError, match='fixture cleanup failed'):
            client.run_episode(environment, agent, budget=budget)
        assert backend.events == ['setup', 'stop', 'close']
        assert backend.writer_alive
        assert session.failed_result is not None
        assert not session.score().has_verdict
        assert 'verify' not in backend.events
    else:
        result, grade = client.run_episode(environment, agent, budget=budget)
        assert result.termination_reason == reason
        assert result.trajectory.final_observation.done
        assert result.trajectory.steps == []
        assert grade.has_verdict and grade.score == 1.0
        assert backend.events == ['setup', 'stop', 'verify', 'close']
        assert not backend.writer_alive
    assert session.done
    assert session.trajectory.steps == []


def test_already_done_session_still_cleans_setup_population() -> None:
    environment = MultiTurnEnvironment(task='fixture task', setup_script='start fixture writer')
    backend = SetupWriterBackend()
    session = Session(environment, backend)
    initial = session.start()
    session.step(Action(action_type='done'))
    assert backend.writer_alive

    result = ClaudeCodeAgent(model='m').run(session, initial, task='task', budget=Budget())
    session.finalize_cli()

    assert result.termination_reason == 'session_done'
    assert result.trajectory.final_observation.done
    assert len(result.trajectory.steps) == 1
    assert result.trajectory.steps[0][1].action_type == 'done'
    assert backend.events == ['setup', 'stop']
    assert not backend.writer_alive


def test_gui_scoring_does_not_finalize_cli_containment(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SetupWriterBackend()
    monkeypatch.setattr(backend, 'start_desktop', lambda: None)
    monkeypatch.setattr(backend, 'screenshot', lambda: 'fixture-png')
    monkeypatch.setattr(backend, 'stop_agent', lambda: pytest.fail('GUI desktop was stopped'))
    monkeypatch.setattr(
        backend,
        'verify',
        lambda *args: VerifierOutput(0, '{"score":1.0,"passed":true}', ''),
    )
    session = Session(
        MultiTurnEnvironment(
            task='GUI task',
            desktop=True,
            allowed_actions={'screenshot'},
            setup_script='start fixture desktop',
            final_verifier_code='return 1.0',
        ),
        backend,
    )
    session.start()
    assert session.score().has_verdict
    assert backend.writer_alive
    assert backend.events == ['setup']


@pytest.mark.parametrize('preflight', ['credentials', 'prices', 'codex_budget'])
def test_preflight_exception_cleans_setup_population(
    monkeypatch: pytest.MonkeyPatch,
    preflight: str,
) -> None:
    environment = MultiTurnEnvironment(
        task='fixture task',
        setup_script='start fixture writer',
        final_verifier_code='return 1.0',
    )
    backend = SetupWriterBackend()
    session = Session(environment, backend)
    client = DeepGym(mode='local')
    monkeypatch.setattr(client, 'session', lambda env: session)
    agent_type = CodexAgent if preflight == 'codex_budget' else ClaudeCodeAgent
    agent = agent_type(
        model='m',
        api_key='fixture-key',
        allow_model_credentials=preflight != 'credentials',
    )
    budget = Budget() if preflight == 'credentials' else Budget(max_cost_usd=1)

    with pytest.raises(ValueError):
        client.run_episode(environment, agent, budget=budget)

    assert backend.events == ['setup', 'stop', 'close']
    assert not backend.writer_alive
    assert session.done
    assert session.trajectory.steps == []


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_a_cli_stopped_at_the_output_bound_says_so(provider: str) -> None:
    """Stopped for printing more than the host keeps: shown as cut short, never an error."""
    assert parser(provider).finish(-9, 'output_limit') == 'output_limit'
