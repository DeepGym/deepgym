"""Host-side provider loops: credentials never enter the desktop sandbox.

USD enforcement uses returned usage and caller-supplied prices, so one response
can overshoot the cap. SDK retries are disabled and request timeouts use remaining
wall time. Session propagates remaining deadlines through actions and screenshots;
SDK transport timeouts do not guarantee an absolute response deadline. These agents
do not score, close, or acknowledge provider safety checks automatically.
"""

from __future__ import annotations

from typing import Any, Literal

from deepgym.agents.actions import anthropic_actions, openai_actions
from deepgym.agents.base import (
    AgentResult,
    Budget,
    BudgetExceeded,
    BudgetTracker,
    Cost,
    TokenPrices,
    Usage,
)
from deepgym.agents.providers import (
    ANTHROPIC_MAX_TOKENS,
    SDK_RETRIES,
    account,
    anthropic_incomplete,
    anthropic_message,
    anthropic_unfinished_calls,
    api_error,
    openai_incomplete,
    validate_budget,
)
from deepgym.exceptions import ActionError, DeepGymError
from deepgym.models import Action, Observation
from deepgym.session import Session

_SYSTEM = (
    'Operate the desktop using screenshots and computer actions to complete the user task. '
    'When finished, stop calling tools and explain the result. If unable to finish, say so.'
)


def _image(screenshot: str | None) -> str:
    if not screenshot:
        raise ValueError('Computer agents require a desktop observation with a screenshot')
    return screenshot


def _step(session: Session, action: Action, tracker: BudgetTracker) -> Observation:
    tracker.check()
    if session.done:
        raise BudgetExceeded('session_done')
    if action.action_type == 'wait' and action.amount > tracker.remaining_seconds:
        raise BudgetExceeded('time_budget')
    observation = session.step(action, timeout=tracker.request_timeout).observation
    tracker.check()
    return observation


class AnthropicComputerAgent:
    """Drive Session with Anthropic's computer toolset; import the SDK only on run.

    Args:
        model: Provider model supporting computer_toolset_20260801.
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
        """Run the task and return its trajectory without grading.

        Raises:
            ImportError: If the optional anthropic extra is absent.
            ValueError: If required prices or the initial screenshot are missing.
        """
        validate_budget(budget, self.prices)
        screenshot = _image(initial.screenshot)
        import anthropic

        tracker = BudgetTracker(budget)
        messages: list[dict[str, Any]] = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': task},
                    {
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': 'image/png', 'data': screenshot},
                    },
                ],
            }
        ]
        client = anthropic.Anthropic(max_retries=SDK_RETRIES)
        try:
            while True:
                if session.done:
                    return tracker.result(session, 'session_done')
                tracker.before_turn()
                # Cache the growing prefix: without it every turn re-bills all earlier
                # screenshots.
                thinking = (
                    {
                        'thinking': {'type': 'adaptive', 'display': 'summarized'},
                        'output_config': {'effort': self.effort},
                    }
                    if self.effort
                    else {}
                )
                response = anthropic_message(
                    client,
                    tracker,
                    self.prices,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    cache_control={'type': 'ephemeral'},
                    **thinking,
                    system=_SYSTEM,
                    tools=[
                        {
                            'type': 'computer_toolset_20260801',
                            'configs': {'zoom': {'enabled': False}},
                        }
                    ],
                    messages=messages,
                    timeout=tracker.request_timeout,
                )
                if error := anthropic_incomplete(response):
                    return tracker.result(session, 'error', error=error)
                account(tracker, response.get('usage'), self.prices, 'anthropic')
                content = response['content']
                messages.append({'role': 'assistant', 'content': content})
                text = ' '.join(part['text'] for part in content if part['type'] == 'text')
                if response.get('stop_reason') == 'refusal':
                    return tracker.result(session, 'refusal', text)
                if error := anthropic_unfinished_calls(response):
                    return tracker.result(session, 'error', text, error=error)
                calls = [part for part in content if part['type'] == 'tool_use']
                if not calls:
                    if response.get('stop_reason') in {'max_tokens', 'pause_turn'}:
                        messages.append(
                            {'role': 'user', 'content': 'Continue from the output limit.'}
                        )
                        continue
                    if response.get('stop_reason') not in {'end_turn', 'stop_sequence'}:
                        return tracker.result(session, 'error', error='Response did not complete')
                    return tracker.result(session, 'completed', text)
                results = []
                observation: Observation | None = None
                failed = False
                for call in calls:
                    result: dict[str, Any] = {
                        'type': 'tool_result',
                        'tool_use_id': call['id'],
                        'toolset_name': call.get('toolset_name') or 'computer',
                        'content': 'OK',
                    }
                    try:
                        if failed:
                            raise ValueError('Not executed: an earlier action in this batch failed')
                        for action in anthropic_actions(call['name'], call.get('input') or {}):
                            observation = _step(session, action, tracker)
                    except (ValueError, ActionError) as exc:
                        failed = True
                        result.update(is_error=True, content=f'Action rejected: {exc}')
                    results.append(result)
                if session.done:
                    return tracker.result(session, 'session_done')
                if observation is None or not observation.screenshot:
                    observation = _step(session, Action(action_type='screenshot'), tracker)
                image = {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': _image(observation.screenshot),
                    },
                }
                # The API accepts only text in an error result, so the screenshot goes on the
                # last successful result, or after the results when every action failed.
                ok = [result for result in results if not result.get('is_error')]
                if ok:
                    ok[-1]['content'] = [{'type': 'text', 'text': ok[-1]['content']}, image]
                    messages.append({'role': 'user', 'content': results})
                else:
                    messages.append({'role': 'user', 'content': [*results, image]})
        except BudgetExceeded as exc:
            return tracker.result(session, exc.reason)
        except DeepGymError as exc:
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return tracker.result(
                session, reason, error=f'Session step failed: {type(exc).__name__}'
            )
        except anthropic.APIError as exc:
            tracker.add_usage(Usage(), Cost(provenance='Anthropic failed request; billing unknown'))
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return tracker.result(session, reason, error=f'Anthropic {api_error(exc)}')
        except (ValueError, KeyError, TypeError) as exc:
            return tracker.result(session, 'error', error=f'Invalid Anthropic response: {exc}')
        finally:
            client.close()


class OpenAIComputerAgent:
    """Drive Session with the OpenAI Responses computer tool, importing the SDK lazily.

    Args:
        model: Provider model supporting the Responses computer tool.
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
        """Run the task, stopping before any call with pending safety checks.

        Raises:
            ImportError: If the optional openai extra is absent.
            ValueError: If required prices or the initial screenshot are missing.
        """
        validate_budget(budget, self.prices)
        screenshot = _image(initial.screenshot)
        import openai

        tracker = BudgetTracker(budget)
        items: list[dict[str, Any]] = [
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': task},
                    {'type': 'input_image', 'image_url': f'data:image/png;base64,{screenshot}'},
                ],
            }
        ]
        previous: str | None = None
        client = openai.OpenAI(max_retries=SDK_RETRIES)
        try:
            while True:
                if session.done:
                    return tracker.result(session, 'session_done')
                tracker.before_turn()
                response = client.responses.create(
                    model=self.model,
                    instructions=_SYSTEM,
                    tools=[{'type': 'computer'}],
                    input=items,
                    previous_response_id=previous,
                    **({'reasoning': {'effort': self.effort}} if self.effort else {}),
                    timeout=tracker.request_timeout,
                ).model_dump(exclude_none=True)
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
                    return tracker.result(session, 'refusal', refusal)
                text = ' '.join(part['text'] for part in parts if part['type'] == 'output_text')
                calls = [item for item in output if item['type'] == 'computer_call']
                if any(call.get('pending_safety_checks') for call in calls):
                    return tracker.result(
                        session,
                        'safety_check',
                        error=('Provider safety checks require approval; no actions executed'),
                    )
                if error := openai_incomplete(response):
                    return tracker.result(session, 'error', error=error)
                if not calls:
                    return tracker.result(session, 'completed', text)
                items = []
                notes = []
                for call in calls:
                    actions = call.get('actions') or (
                        [call['action']] if call.get('action') else []
                    )
                    if not actions:
                        raise ValueError('computer_call contains no actions')
                    observation: Observation | None = None
                    try:
                        for data in actions:
                            for action in openai_actions(data):
                                observation = _step(session, action, tracker)
                    except (ValueError, ActionError) as exc:
                        notes.append(
                            f'Action rejected: {exc}. Remaining actions in this call skipped.'
                        )
                    if session.done:
                        return tracker.result(session, 'session_done')
                    if observation is None or not observation.screenshot:
                        observation = _step(session, Action(action_type='screenshot'), tracker)
                    screenshot = _image(observation.screenshot)
                    items.append(
                        {
                            'type': 'computer_call_output',
                            'call_id': call['call_id'],
                            'output': {
                                'type': 'computer_screenshot',
                                'image_url': f'data:image/png;base64,{screenshot}',
                            },
                        }
                    )
                if notes:
                    items.append({'role': 'user', 'content': ' '.join(notes)})
        except BudgetExceeded as exc:
            return tracker.result(session, exc.reason)
        except DeepGymError as exc:
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return tracker.result(
                session, reason, error=f'Session step failed: {type(exc).__name__}'
            )
        except openai.APIError as exc:
            tracker.add_usage(Usage(), Cost(provenance='OpenAI failed request; billing unknown'))
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            return tracker.result(session, reason, error=f'OpenAI {api_error(exc)}')
        except (ValueError, KeyError, TypeError) as exc:
            return tracker.result(session, 'error', error=f'Invalid OpenAI response: {exc}')
        finally:
            client.close()
