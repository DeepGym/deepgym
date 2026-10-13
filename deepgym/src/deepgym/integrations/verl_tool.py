"""verl multi-turn integration backed by a live DeepGym session.

verl's ToolAgentLoop calls create -> execute -> release around *every* tool call,
so a tool alone can't keep one sandbox for a whole rollout. DeepGymAgentLoop
fixes that: it gives each rollout an episode key, DeepGymSessionTool keeps one
session per episode across calls, and when the rollout ends the loop runs the
final verifier, sets ``reward_score``, and deletes the sandbox. The first tool
call starts the episode and returns the starting screen (the model can't see it
before then), so prompts should tell the model to begin with a screenshot.

verl turns a tool's exception into tool-error text and the rollout goes on
(``ToolAgentLoop._call_tool``), so a sandbox that fails to start, or a provider that
fails mid-action, would otherwise end the rollout with a final verifier judging a state
the model never got to shape. The tool remembers any such failure instead: a rollout
whose sandbox failed to start, whose action failed for any reason other than the action
itself (an invalid or disallowed action comes back as text), or whose step or final
verifier failed, gets no verdict. Like :mod:`deepgym.integrations.verl_mask`, the loop
keeps it out of training (see :func:`~deepgym.integrations.verl_mask.exclude_unscored`)
and logs ``error`` 1.0 in ``reward_extra_info``. A rollout where the model never called
the tool scores 0.0.

Agent loop config (``actor_rollout_ref.rollout.agent.agent_loop_config_path``)::

    - name: deepgym_agent
      _target_: deepgym.integrations.verl_tool.DeepGymAgentLoop

Tool config (``actor_rollout_ref.rollout.multi_turn.tool_config_path``)::

    tools:
      - class_name: deepgym.integrations.verl_tool.DeepGymSessionTool
        config:
          type: native
          environment: signup_form   # default; override per sample below
          mode: daytona             # default; 'local' runs model commands on this host

Dataset rows set ``agent_name: deepgym_agent`` and, optionally::

    extra_info:
      tools_kwargs:
        computer:
          create_kwargs: {environment: todo_list, variant: 3}

Written against verl 0.9.1. The tests drive it through a stand-in for verl's loop; it
was also run on Daytona sandboxes through verl 0.9.1's own ``ToolAgentLoop._call_tool``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

from deepgym.core import DeepGym
from deepgym.exceptions import ActionError, DeepGymError, SandboxError
from deepgym.integrations.verl_mask import exclude_unscored
from deepgym.models import Action, ActionType, MultiTurnEnvironment, Observation, RunResult
from deepgym.registry import load_environment
from deepgym.sandbox import error_result
from deepgym.session import Session

logger = logging.getLogger(__name__)

_STARTED_NOTE = (
    'Episode started. Your action was not applied; this is the starting screen. '
    'Repeat the action if it is still needed.'
)
_FAILED_NOTE = 'The environment failed, so this episode will not be scored.'

_COORD = {'type': 'integer', 'description': 'Screenshot pixel coordinate.'}

TOOL_SCHEMA = {
    'type': 'function',
    'function': {
        'name': 'computer',
        'description': (
            'Act in the environment. GUI actions use screenshot pixel coordinates. '
            'Call with action_type "done" when the task is complete.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'action_type': {
                    'type': 'string',
                    'enum': sorted(ActionType.__args__),  # type: ignore[attr-defined]
                    'description': 'What to do.',
                },
                'content': {
                    'type': 'string',
                    'description': 'Code, shell command, file contents, text to type, key '
                    'chord like "ctrl+s", or scroll direction up/down/left/right.',
                },
                'x': _COORD,
                'y': _COORD,
                'to_x': _COORD,
                'to_y': _COORD,
                'path': {'type': 'string', 'description': 'Relative path for write_file.'},
                'amount': {'type': 'integer', 'description': 'Scroll clicks or wait seconds.'},
            },
            'required': ['action_type'],
        },
    },
}


class DeepGymSessionTool(BaseTool):
    """Expose a DeepGym environment to verl as one stateful tool, one session per episode."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None) -> None:
        super().__init__(config, tool_schema or OpenAIFunctionToolSchema(**TOOL_SCHEMA))
        self._dg = DeepGym(mode=config.get('mode', 'daytona'))
        self._episode_of: dict[str, str] = {}
        self._specs: dict[str, tuple[str, int]] = {}
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._failed: dict[str, RunResult] = {}

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, ToolResponse]:
        """Bind this tool call to its rollout's episode; the sandbox opens on first execute.

        Raises:
            DeepGymError: If no episode key was passed (the loop isn't DeepGymAgentLoop).
        """
        instance_id = instance_id or uuid4().hex
        create_kwargs = kwargs.get('create_kwargs') or {}
        episode = create_kwargs.get('episode')
        if episode is None:
            raise DeepGymError(
                "DeepGymSessionTool needs agent_name 'deepgym_agent' "
                'so one sandbox spans the whole rollout'
            )
        self._episode_of[instance_id] = episode
        # Parquet rows carry missing keys as None, so fall back on falsy, not only absent.
        self._specs.setdefault(
            episode,
            (
                create_kwargs.get('environment') or self.config['environment'],
                int(create_kwargs.get('variant') or 0),
            ),
        )
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict]:
        """Apply one action. Bad calls come back as text so the model can correct itself.

        Raises:
            Exception: Whatever starting the sandbox, or applying a valid action, raised
                (a provider outage, a failed step verifier). verl shows it to the model as
                tool-error text; the episode is remembered as failed, gets no verdict, and
                takes no more actions.
        """
        episode = self._episode_of[instance_id]
        async with self._locks.setdefault(episode, asyncio.Lock()):
            if episode in self._failed:
                return ToolResponse(text=_FAILED_NOTE), 0.0, {}
            session = self._sessions.get(episode)
            if session is None:
                try:
                    obs = await self._open(episode)
                except Exception as exc:
                    self._failed[episode] = _no_verdict(exc)
                    raise
                note = '' if parameters.get('action_type') == 'screenshot' else _STARTED_NOTE
                return _to_response(obs, note), 0.0, {}
            if session.done:
                return ToolResponse(text='The episode is over.'), 0.0, {}
            try:
                action = Action.model_validate(parameters)
                obs, reward, done = await asyncio.to_thread(session.step, action)
            except ValidationError as exc:
                return ToolResponse(text=f'Invalid action: {exc.errors()[0]["msg"]}'), 0.0, {}
            except ActionError as exc:
                return ToolResponse(text=str(exc)), 0.0, {}
            except Exception as exc:
                self._failed[episode] = _no_verdict(exc)
                raise
            return _to_response(obs), reward, {'done': done}

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """Unbind the tool call; the episode's session lives on until finish()."""
        self._episode_of.pop(instance_id, None)

    async def finish(self, episode: str, *, score: bool) -> RunResult | None:
        """Score (if asked) and delete the episode's sandbox.

        Returns:
            The final verifier's result; a result without a verdict if the sandbox failed
            to start, an action failed (see :meth:`execute`) or the verifier raised; None
            if the episode never started (the model made no tool call) or *score* is False.
        """
        self._specs.pop(episode, None)
        self._locks.pop(episode, None)
        failed = self._failed.pop(episode, None)
        session = self._sessions.pop(episode, None)
        try:
            if not score:
                return None
            if failed is not None or session is None:
                # A failure decides the episode, whatever state it left behind.
                return failed
            return await asyncio.to_thread(session.score)
        except Exception as exc:
            return _no_verdict(exc)
        finally:
            if session is not None:
                try:
                    await asyncio.to_thread(session.close)
                except SandboxError as exc:
                    # The episode was judged; keep its result and leave the leak visible.
                    logger.warning('Could not delete sandbox %s: %s', session.sandbox_id, exc)

    async def _open(self, episode: str) -> Observation:
        """Create and start the episode's session; never lose it if we're cancelled meanwhile."""
        name, variant = self._specs[episode]
        task = asyncio.ensure_future(
            asyncio.to_thread(self._start, load_environment(name, variant=variant))
        )
        try:
            session, obs = await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(_close_started)
            raise
        self._sessions[episode] = session
        return obs

    def _start(self, env: MultiTurnEnvironment) -> tuple[Session, Observation]:
        session = self._dg.session(env)
        try:
            return session, session.start()
        except BaseException:
            session.close()
            raise


@register('deepgym_agent')
class DeepGymAgentLoop(ToolAgentLoop):
    """ToolAgentLoop where each rollout keeps one DeepGym sandbox and is scored by its verifier."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Run the rollout with a fresh episode, then score and delete its sandbox.

        Raises:
            DeepGymError: If the episode got no verdict under verl's v1 trainer (see
                :func:`~deepgym.integrations.verl_mask.exclude_unscored`).
        """
        tools = [tool for tool in self.tools.values() if isinstance(tool, DeepGymSessionTool)]
        if not tools:
            return await super().run(sampling_params, **kwargs)
        episode = uuid4().hex
        tools_kwargs = dict(kwargs.get('tools_kwargs') or {})
        for tool in tools:
            entry = dict(tools_kwargs.get(tool.name) or {})
            entry['create_kwargs'] = {**(entry.get('create_kwargs') or {}), 'episode': episode}
            tools_kwargs[tool.name] = entry
        try:
            output = await super().run(sampling_params, **{**kwargs, 'tools_kwargs': tools_kwargs})
        except BaseException:
            for tool in tools:
                await tool.finish(episode, score=False)
            raise
        finished = [await tool.finish(episode, score=True) for tool in tools]
        results = [result for result in finished if result is not None]
        failed = [result for result in results if not result.has_verdict]
        output.reward_score = sum((result.score for result in results), 0.0)
        output.extra_fields['reward_extra_info'] = {
            'score': output.reward_score,
            'passed': float(bool(results) and all(result.passed for result in results)),
            'error': float(bool(failed)),
        }
        if failed:
            exclude_unscored(output, self.config, f'{failed[0].error_type}: {failed[0].output}')
        return output


def _no_verdict(exc: Exception) -> RunResult:
    """Log *exc*, raised while starting, acting in or scoring an episode, as its lack of verdict."""
    logger.warning('DeepGym episode got no verdict: %s: %s', type(exc).__name__, exc)
    return error_result(exc)


def _close_started(task: asyncio.Future) -> None:
    """Close a session whose start finished after its caller was cancelled."""
    if task.cancelled() or task.exception() is not None:
        return
    session, _ = task.result()
    threading.Thread(target=session.close, daemon=True).start()


def _to_response(obs: Observation, note: str = '') -> ToolResponse:
    """Convert an observation to a verl ToolResponse, decoding any screenshot."""
    text = '\n'.join(part for part in (note, obs.content) if part) or None
    if obs.screenshot is None:
        return ToolResponse(text=text or '(no output)')
    from PIL import Image
    from verl.utils.dataset.vision_utils import process_image

    image = Image.open(io.BytesIO(base64.b64decode(obs.screenshot)))
    return ToolResponse(text=text, image=[process_image(image)])
