"""Delete the Daytona sandboxes DeepGym created since a time, and fail if there were any.

The nightly live job runs this last: every sandbox a run creates should be gone when the
run ends, so one found here is a leak. Sandboxes without DeepGym's label are never touched.
It keeps looking for a while, since Daytona can finish a create after the client gave up.

    python scripts/sweep_daytona.py 2026-11-07T02:00:00+00:00
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

from daytona import Daytona

LABEL = 'deepgym_sandbox'
# Daytona has finished a create about 70 s after the client's 60 s timeout.
WATCH_S = 180.0
EVERY_S = 15.0


def ours(daytona: Daytona, since: datetime) -> list:
    """DeepGym's sandboxes created at or after *since*."""
    return [
        sandbox
        for sandbox in daytona.list()
        if LABEL in (sandbox.labels or {})
        and datetime.fromisoformat(str(sandbox.created_at)) >= since
    ]


def main(since: str) -> int:
    start = datetime.fromisoformat(since)
    daytona = Daytona()
    leaked: set[str] = set()
    end = time.monotonic() + WATCH_S
    while True:
        for sandbox in ours(daytona, start):
            leaked.add(sandbox.id)
            try:
                daytona.delete(sandbox, timeout=60)
            except Exception as exc:  # still being created: the next pass retries
                print(f'could not delete {sandbox.id} yet: {exc}')
        if time.monotonic() >= end:
            break
        time.sleep(EVERY_S)
    left = [sandbox.id for sandbox in ours(daytona, start)]
    print(f'leaked: {len(leaked)} {sorted(leaked)}; still there: {len(left)} {left}')
    return 1 if leaked else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
