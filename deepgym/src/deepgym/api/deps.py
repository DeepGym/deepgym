"""FastAPI dependency providers for the DeepGym API."""

from __future__ import annotations

import copy
import os
import threading
from functools import lru_cache

from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError
from deepgym.models import Environment, RunResult

# ponytail: one fixed cap for the whole process; make it configurable if buyers' quotas differ.
MAX_PARALLEL_RUNS = 100
# Every run the server starts, from any request or job, holds one of these while it executes.
_run_slots = threading.BoundedSemaphore(MAX_PARALLEL_RUNS)


class ServerDeepGym(DeepGym):
    """DeepGym for the API server: all requests share one limit on concurrent runs.

    A batch or eval request runs its outputs through :meth:`run`, so ``max_parallel`` only
    splits the server's slots and never adds to them.
    """

    _cancelled: threading.Event | None = None

    def for_job(self, cancelled: threading.Event) -> ServerDeepGym:
        """Return a client on the same backend whose runs stop starting once *cancelled* is set."""
        job_client = copy.copy(self)
        job_client._cancelled = cancelled
        return job_client

    def run(self, env: Environment, model_output: str, **kwargs: object) -> RunResult:
        """Run as :meth:`DeepGym.run` does, holding one of the server's run slots.

        Raises:
            DeepGymError: If the job this client was made for was cancelled before the run
                started.
        """
        self._refuse_if_cancelled()
        with _run_slots:
            self._refuse_if_cancelled()
            return super().run(env, model_output, **kwargs)

    def _refuse_if_cancelled(self) -> None:
        """Raise DeepGymError if this client's job has been cancelled."""
        if self._cancelled is not None and self._cancelled.is_set():
            raise DeepGymError('The job was cancelled before this run started')


@lru_cache
def get_deepgym() -> ServerDeepGym:
    """Return a singleton DeepGym client instance.

    In server mode the client is configured explicitly: if DAYTONA_API_KEY is
    set, mode is forced to ``'daytona'`` so that a Daytona init failure is
    fatal rather than silently falling back to local execution.  Otherwise
    local mode is used (already gated by ``--allow-local-exec``).

    Returns:
        A DeepGym client in the appropriate mode.
    """
    if os.getenv('DAYTONA_API_KEY'):
        return ServerDeepGym(mode='daytona')
    return ServerDeepGym(mode='local')
