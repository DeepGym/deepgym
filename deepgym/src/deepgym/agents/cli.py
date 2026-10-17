"""Run installed, headless CLIs as unprivileged sandbox processes.

Phase 1 requires explicit consent to model-readable credentials. The fixed provider
public-address tunnel passes opaque TLS, not verified HTTP routes; a shared CDN can
serve other SNI/Host names. Keys can be used outside CLI accounting. These adapters
never grade, replay a parsed tool call, or run the CLI on the host.
"""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from deepgym.agents.base import AgentResult, Budget, BudgetTracker, Cost, TokenPrices
from deepgym.agents.cli_events import CliEvents
from deepgym.agents.versions import SUPPORTED_CLIS
from deepgym.cli_runtime import CliExecutionError, CliRequest
from deepgym.exceptions import ActionError, DeepGymError
from deepgym.models import Observation
from deepgym.session import Session


class _CliAgent(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    model: str = Field(min_length=1)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    allow_model_credentials: bool = False
    prices: TokenPrices | None = None
    provider: ClassVar[Literal['anthropic', 'openai']]

    def run(
        self,
        session: Session,
        initial: Observation,
        *,
        task: str,
        budget: Budget,
    ) -> AgentResult:
        """Run once and record a single Session action containing nested CLI events.

        Args:
            session: Started sandbox Session with supervised CLI support.
            initial: Observation returned by Session.start.
            task: Task supplied as one literal CLI argument.
            budget: Wall deadline, activity cap, and optional USD bound.

        Returns:
            Trajectory and reported usage/cost, including interruption or errors.

        Raises:
            ValueError: If credentials are missing, exposure was not allowed, or a
                finite USD budget cannot be enforced by this CLI.
            ActionError: If the environment disallows CLI actions or provider egress.
            DeepGymError: If containment cannot be finalized; the session is ungradable.
        """
        try:
            result = self._run(session, task, budget)
        finally:
            session.finalize_cli()
        return result.model_copy(update={'trajectory': session.trajectory.model_copy(deep=True)})

    def _run(self, session: Session, task: str, budget: Budget) -> AgentResult:
        tracker = BudgetTracker(budget)
        if self.provider == 'openai' and budget.max_cost_usd is not None:
            raise ValueError(
                'Codex CLI reports usage only after the entire task: a finite USD budget '
                'cannot be enforced in phase 1; use a capped relay or omit max_cost_usd'
            )
        if session.done:
            return tracker.result(session, 'session_done')
        if budget.max_cost_usd == 0:
            return tracker.result(session, 'cost_budget')
        if not self.allow_model_credentials:
            raise ValueError(
                'CLI agents expose the provider API key to model-generated commands; '
                'set allow_model_credentials=True to opt in'
            )
        if budget.max_cost_usd is not None and self.prices is None:
            raise ValueError(
                'A finite CLI USD budget requires explicit TokenPrices to monitor '
                'each assistant message; the native CLI budget flag is secondary'
            )
        source = 'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'
        key = self.api_key.get_secret_value() if self.api_key else os.environ.get(source)
        if not key:
            raise ValueError(f'{source} or api_key is required')
        if tracker.remaining_seconds <= 0:
            return tracker.result(session, 'time_budget')
        events = CliEvents(self.provider, tracker, self.prices)
        credential = source if self.provider == 'anthropic' else 'CODEX_API_KEY'
        request = CliRequest(
            argv=self._argv(task, budget),
            env={credential: key},
            provider=self.provider,
            timeout=tracker.remaining_seconds,
            allow_model_credentials=True,
            supported_version=SUPPORTED_CLIS[self.provider].version,
        )
        try:
            execution = session.run_cli(request, events.consume)
        except ActionError:
            raise
        except DeepGymError as exc:
            tracker.cost = tracker.cost.model_copy(
                update={
                    'provenance': tracker.cost.provenance
                    + '; CLI failure: usage may be incomplete',
                }
            )
            reason = 'time_budget' if tracker.remaining_seconds <= 0 else 'error'
            result = tracker.result(session, reason, events.final_text, type(exc).__name__)
            if isinstance(exc, CliExecutionError):
                result = result.model_copy(update={'cli_version': exc.cli_version})
            return result
        if execution.version_error:
            tracker.cost = Cost(usd=0, provenance='CLI version rejected before inference')
            reason = (
                'time_budget'
                if execution.stop_reason == 'time_budget' or tracker.remaining_seconds <= 0
                else 'error'
            )
            result = tracker.result(session, reason, error=execution.version_error)
        else:
            reason = events.finish(execution.exit_code, execution.stop_reason)
            result = tracker.result(session, reason, events.final_text, events.error)
        return result.model_copy(update={'cli_version': execution.cli_version})

    def _argv(self, task: str, budget: Budget) -> tuple[str, ...]:
        if self.provider == 'anthropic':
            args = (
                SUPPORTED_CLIS[self.provider].command,
                '--print',
                '--verbose',
                '--output-format',
                'stream-json',
                '--bare',
                '--setting-sources',
                '',
                '--strict-mcp-config',
                '--no-session-persistence',
                '--dangerously-skip-permissions',
                '--model',
                self.model,
            )
            if budget.max_cost_usd is not None:
                args += ('--max-budget-usd', str(budget.max_cost_usd))
            return (*args, '--', task)
        return (
            SUPPORTED_CLIS[self.provider].command,
            'exec',
            '--json',
            '--ephemeral',
            '--ignore-user-config',
            '--skip-git-repo-check',
            '--color',
            'never',
            '--dangerously-bypass-approvals-and-sandbox',
            '--model',
            self.model,
            '--',
            task,
        )


class ClaudeCodeAgent(_CliAgent):
    """Drive a sandbox with an installed Claude Code CLI.

    Opt in with ``allow_model_credentials=True``: the agent can read the real API
    key and make additional provider requests. The sandbox namespace and provider
    allowlist remain enforced, but this is not credential isolation or a hard bill
    cap. Claude's native dollar-budget flag complements host event policing.
    ``Budget.max_turns`` counts observed assistant messages; polling can overshoot.
    CLI-reported USD takes precedence over optional caller-rate estimates.
    A finite USD budget requires ``prices`` to stop on per-message reported usage;
    missing usage fails closed. In-flight calls can still exceed the dollar cap.
    """

    provider: ClassVar[Literal['anthropic']] = 'anthropic'


class CodexAgent(_CliAgent):
    """Drive a sandbox with an installed Codex CLI using explicit API credentials.

    Opt in with ``allow_model_credentials=True`` to expose the real provider key
    to model-generated commands. ``Budget.max_turns`` counts completed activity
    items, including messages, reasoning and tools; it is not a model-call count.
    The host stops at observed boundaries with polling latency. Codex reports token
    usage only after the whole task; finite USD limits therefore fail closed before
    launching. Optional ``prices`` estimate cost from reported usage with provenance.
    Interrupted runs can have unavailable usage/cost. Capped relay support is phase 2.
    """

    provider: ClassVar[Literal['openai']] = 'openai'
