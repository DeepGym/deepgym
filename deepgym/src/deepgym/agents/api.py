"""Host-side provider loops that act only through an environment's declared API tools.

The model gets the environment's tools (HTTP calls to the app under test, sent from inside
the sandbox's offline namespace; see :mod:`deepgym.tools`) and a ``done`` tool, nothing
else: no hosted or server-side tool (web search, code execution, browsing) is ever
requested, so the model cannot read a live public copy of the data it is graded on.
Provider keys stay in this process; the sandbox only ever sees the app's requests.

Each tool call is one Session ``tool`` action. Its observation, which the model gets back
whole, is the runner's record of the call: the full request, the full response (status,
headers, body), timing, and whether a write whose response was lost may have happened.
A call is never retried. Only the done tool ends an episode; a reply without tool calls
gets a reminder instead. Streaming, accounting and incomplete-response handling are the
computer agents' (see :mod:`deepgym.agents.providers`). Every model message goes to
``AgentResult.messages``, a response that never completed too (what arrived of it, marked
``status: incomplete``; its tool calls never run); under a TracedSession, each step also
carries the messages that produced it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from deepgym.agents.base import (
    AgentResult,
    Budget,
    BudgetExceeded,
    BudgetTracker,
    Cost,
    TerminationReason,
    TokenPrices,
    Usage,
)
from deepgym.agents.providers import (
    ANTHROPIC_MAX_TOKENS,
    SDK_RETRIES,
    account,
    anthropic_incomplete,
    anthropic_message,
    api_error,
    openai_incomplete,
    validate_budget,
)
from deepgym.exceptions import ActionError, DeepGymError
from deepgym.models import Action, Observation
from deepgym.session import Session
from deepgym.tools import DONE_TOOL
from deepgym.traces.capture import TracedSession
from deepgym.traces.schema import TraceMessage, json_text

_SYSTEM = (
    "Complete the user task using only the provided tools, which call the application's HTTP "
    'API. Each tool result is a JSON record of the request and the full response. A result '
    'with "uncertain": true means a write may or may not have been applied: check the '
    "application's state before repeating it. Call done once the task is finished, or when "
    'it cannot be finished.'
)
_NO_CALL = 'No tool was called. Continue with the tools, or call done when the task is finished.'
_SKIPPED = 'Not executed: an earlier call in this batch failed'
_DONE_DESCRIPTION = 'End the episode. Call it once the task is finished, or when it cannot be.'
_DONE_SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}


def _specs(session: Session) -> list[tuple[str, str, dict[str, Any]]]:
    """Name, description and input schema of each tool the model gets: the declared ones
    and done.

    Raises:
        ValueError: If the environment declares no tools.
    """
    if not session.tools:
        raise ValueError('API tool agents need an environment that declares tools')
    declared = [(tool.name, tool.description, tool.input_schema) for tool in session.tools]
    return [*declared, (DONE_TOOL, _DONE_DESCRIPTION, _DONE_SCHEMA)]


def _prompt(task: str, initial: Observation) -> str:
    return f'{task}\n\n{initial.content}' if initial.content else task


class _Episode:
    """One run's session, accounting and model messages.

    Every message goes to the result. Under a TracedSession the messages not yet on a step
    also go with the next action, so each step carries the messages that produced it.
    """

    def __init__(self, session: Session, budget: Budget, prompt: str) -> None:
        self.session = session
        self.tracker = BudgetTracker(budget)
        self.messages = [
            TraceMessage(role='system', content=_SYSTEM),
            TraceMessage(role='user', content=prompt),
        ]
        self._sent = 0

    def say(self, role: Literal['user', 'assistant', 'tool'], content: str | Mapping) -> None:
        """Record a message: text as it is, a provider payload as its full JSON."""
        text = content if isinstance(content, str) else json_text(content)
        self.messages.append(TraceMessage(role=role, content=text))

    def act(self, action: Action) -> Observation:
        """Apply *action* within budget, with the messages since the last recorded step."""
        self.tracker.check()
        if self.session.done:
            raise BudgetExceeded('session_done')
        new, self._sent = self.messages[self._sent :], len(self.messages)
        timeout = self.tracker.request_timeout
        if isinstance(self.session, TracedSession):
            result = self.session.step(action, new, timeout=timeout)
        else:
            result = self.session.step(action, timeout=timeout)
        self.tracker.check()
        return result.observation

    def call(self, name: str, arguments: object) -> tuple[str, bool]:
        """Run one tool call; return its full result and whether it failed.

        A call fails when the session rejects it (unknown tool, arguments its routes don't
        allow) or no response came back.
        """
        try:
            if not isinstance(arguments, dict):
                raise ValueError('arguments must be a JSON object')
            action = Action(action_type='tool', tool=name, arguments=arguments)
        except ValueError as exc:
            return f'Tool call rejected: {exc}', True
        try:
            observation = self.act(action)
        except ActionError as exc:
            return f'Tool call rejected: {exc}', True
        return observation.content, (observation.metadata or {}).get('exit_code') != 0

    def execute(self, calls: Sequence[tuple[str, object]]) -> list[tuple[str, bool]] | None:
        """Run a reply's calls in order; None once its done call ended the episode.

        After a call that failed, the rest are not executed, since they may rest on it:
        each gets an error result instead.
        """
        outcomes: list[tuple[str, bool]] = []
        for name, arguments in calls:
            if outcomes and outcomes[-1][1]:
                outcomes.append((_SKIPPED, True))
            elif name == DONE_TOOL:
                self.act(Action(action_type='done'))
                return None
            else:
                outcomes.append(self.call(name, arguments))
        return outcomes

    def result(
        self, reason: TerminationReason, final_text: str = '', error: str | None = None
    ) -> AgentResult:
        """Snapshot accounting, the trajectory and every message, without grading."""
        result = self.tracker.result(self.session, reason, final_text, error)
        return result.model_copy(update={'messages': tuple(self.messages)})


class AnthropicToolAgent:
    """Drive Session through its declared API tools with Anthropic; import the SDK on run.

    Args:
        model: Provider model with client tool use.
        prices: Explicit model-specific rates, required for a USD budget.
        effort: Adaptive-thinking effort; None sends no thinking or effort settings.
        max_tokens: Output limit per response; defaults to Opus 5.5's maximum so the
            harness never caps the model below what it can produce.
    """

    def __init__(
        self,
        model: str,
        prices: TokenPrices | None = None,
        effort: Literal['low', 'medium', 'high', 'xhigh', 'max'] | None = 'max',
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.prices = prices
        self.effort = effort
        self.max_tokens = max_tokens

    def run(
        self, session: Session, initial: Observation, *, task: str, budget: Budget
    ) -> AgentResult:
        """Run the task until the model calls done, and return its trajectory without grading.

        Raises:
            ImportError: If the optional anthropic extra is absent.
            ValueError: If required prices are missing or the environment declares no tools.
        """
        validate_budget(budget, self.prices)
        tools = [
            {'name': name, 'description': description, 'input_schema': schema}
            for name, description, schema in _specs(session)
        ]
        import anthropic

        prompt = _prompt(task, initial)
        episode = _Episode(session, budget, prompt)
        tracker = episode.tracker
        messages: list[dict[str, Any]] = [{'role': 'user', 'content': prompt}]
        thinking = (
            {
                'thinking': {'type': 'adaptive', 'display': 'summarized'},
                'output_config': {'effort': self.effort},
            }
            if self.effort
            else {}
        )
        client = anthropic.Anthropic(max_retries=SDK_RETRIES)
        try:
            while True:
                if session.done:
                    return episode.result('session_done')
                tracker.before_turn()
                response = anthropic_message(
                    client,
                    tracker,
                    self.prices,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    cache_control={'type': 'ephemeral'},
                    **thinking,
                    system=_SYSTEM,
                    tools=tools,
                    messages=messages,
                    timeout=tracker.request_timeout,
                )
                episode.say('assistant', response)
                if error := anthropic_incomplete(response):
                    return episode.result('error', error=error)
                account(tracker, response.get('usage'), self.prices, 'anthropic')
                content = response['content']
                messages.append({'role': 'assistant', 'content': content})
                text = ' '.join(part['text'] for part in content if part['type'] == 'text')
                stop = response.get('stop_reason')
                if stop == 'refusal':
                    return episode.result('refusal', text)
                calls = [part for part in content if part['type'] == 'tool_use']
                if not calls:
                    if stop in {'max_tokens', 'pause_turn'}:
                        note = 'Continue from the output limit.'
                    elif stop in {'end_turn', 'stop_sequence'}:
                        note = _NO_CALL
                    else:
                        return episode.result('error', error='Response did not complete')
                    messages.append({'role': 'user', 'content': note})
                    episode.say('user', note)
                    continue
                outcomes = episode.execute([(call['name'], call.get('input')) for call in calls])
                if outcomes is None:
                    return episode.result('completed', text)
                results = [
                    {
                        'type': 'tool_result',
                        'tool_use_id': call['id'],
                        'content': output,
                        **({'is_error': True} if failed else {}),
                    }
                    for call, (output, failed) in zip(calls, outcomes)
                ]
                for result in results:
                    episode.say('tool', result)
                messages.append({'role': 'user', 'content': results})
        except BudgetExceeded as exc:
            return episode.result(exc.reason)
        except DeepGymError as exc:
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return episode.result(reason, error=f'Session step failed: {type(exc).__name__}')
        except anthropic.APIError as exc:
            tracker.add_usage(Usage(), Cost(provenance='Anthropic failed request; billing unknown'))
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return episode.result(reason, error=f'Anthropic {api_error(exc)}')
        except (ValueError, KeyError, TypeError) as exc:
            return episode.result('error', error=f'Invalid Anthropic response: {exc}')
        finally:
            client.close()


def _arguments(text: str) -> object:
    """A function call's arguments, or its raw text when that isn't JSON (then rejected)."""
    try:
        return json.loads(text)
    except ValueError:
        return text


class OpenAIToolAgent:
    """Drive Session through its declared API tools with OpenAI Responses function calls.

    Args:
        model: Provider model with function calling.
        prices: Explicit model-specific rates, required for a USD budget.
        effort: Reasoning effort; None sends no reasoning setting. No output limit is sent,
            so the model may use its full output allowance.
    """

    def __init__(
        self,
        model: str,
        prices: TokenPrices | None = None,
        effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = 'max',
    ) -> None:
        self.model = model
        self.prices = prices
        self.effort = effort

    def run(
        self, session: Session, initial: Observation, *, task: str, budget: Budget
    ) -> AgentResult:
        """Run the task until the model calls done, and return its trajectory without grading.

        Raises:
            ImportError: If the optional openai extra is absent.
            ValueError: If required prices are missing or the environment declares no tools.
        """
        validate_budget(budget, self.prices)
        tools = [
            {
                'type': 'function',
                'name': name,
                'description': description,
                'parameters': schema,
                'strict': False,  # strict mode can't express a free-form JSON body
            }
            for name, description, schema in _specs(session)
        ]
        import openai

        prompt = _prompt(task, initial)
        episode = _Episode(session, budget, prompt)
        tracker = episode.tracker
        items: list[dict[str, Any]] = [
            {'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}
        ]
        previous: str | None = None
        client = openai.OpenAI(max_retries=SDK_RETRIES)
        try:
            while True:
                if session.done:
                    return episode.result('session_done')
                tracker.before_turn()
                response = client.responses.create(
                    model=self.model,
                    instructions=_SYSTEM,
                    tools=tools,
                    input=items,
                    previous_response_id=previous,
                    **({'reasoning': {'effort': self.effort}} if self.effort else {}),
                    timeout=tracker.request_timeout,
                ).model_dump(exclude_none=True)
                episode.say('assistant', response)
                account(tracker, response.get('usage'), self.prices, 'openai')
                previous = response['id']
                output = response['output']
                parts = [
                    part
                    for item in output
                    if item['type'] == 'message'
                    for part in item.get('content', [])
                ]
                refusal = ' '.join(part['refusal'] for part in parts if part['type'] == 'refusal')
                if refusal:
                    return episode.result('refusal', refusal)
                text = ' '.join(part['text'] for part in parts if part['type'] == 'output_text')
                if error := openai_incomplete(response):
                    return episode.result('error', error=error)
                calls = [item for item in output if item['type'] == 'function_call']
                if not calls:
                    items = [{'role': 'user', 'content': _NO_CALL}]
                    episode.say('user', _NO_CALL)
                    continue
                outcomes = episode.execute(
                    [(call['name'], _arguments(call['arguments'])) for call in calls]
                )
                if outcomes is None:
                    return episode.result('completed', text)
                items = [
                    {'type': 'function_call_output', 'call_id': call['call_id'], 'output': result}
                    for call, (result, _) in zip(calls, outcomes)
                ]
                for item in items:
                    episode.say('tool', item)
        except BudgetExceeded as exc:
            return episode.result(exc.reason)
        except DeepGymError as exc:
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return episode.result(reason, error=f'Session step failed: {type(exc).__name__}')
        except openai.APIError as exc:
            tracker.add_usage(Usage(), Cost(provenance='OpenAI failed request; billing unknown'))
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return episode.result(reason, error=f'OpenAI {api_error(exc)}')
        except (ValueError, KeyError, TypeError) as exc:
            return episode.result('error', error=f'Invalid OpenAI response: {exc}')
        finally:
            client.close()
