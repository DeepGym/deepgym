#!/usr/bin/env bash
# Serve ./app and open it full-screen in Chromium. Exits once the page has loaded.
#
# --no-sandbox: Chromium can't sandbox its renderers here without loosening the agent. The
# agent runs with no-new-privs, so the setuid chrome-sandbox can't start, and Daytona's
# hosts refuse unprivileged user namespaces their id maps (for every non-root user), so the
# namespace sandbox can't either. Dropping no-new-privs would hand every agent process the
# image's setuid binaries. Unsandboxed is safe enough: the page is our own task app, served
# offline from loopback, and Chromium runs as the unprivileged agent, which already drives
# the whole desktop, so the verifier/root boundary doesn't rest on it. Kiosk mode shows no
# warning bar for the flag, so nothing needs hiding.
#
# /usr/local/bin/deepgym-chromium is the Chromium the desktop lockdown checked reads its
# managed policy, linked there by the lockdown. Running it, not `chromium` through this
# script's PATH (the image's), means the launch can't pick a different, unchecked binary.
set -euo pipefail
[ -x /usr/local/bin/deepgym-chromium ] || { echo 'the desktop lockdown found no Chromium' >&2; exit 1; }
(setsid nohup python3 _shared/state_server.py >/tmp/deepgym_server.log 2>&1 &)
(setsid nohup /usr/local/bin/deepgym-chromium --no-first-run --no-default-browser-check --disable-gpu \
  --disable-features=Translate --password-store=basic --kiosk --no-sandbox \
  --remote-debugging-port=9222 --user-data-dir=/tmp/deepgym_chrome \
  http://127.0.0.1:8000/ >/tmp/deepgym_chrome.log 2>&1 &)
for _ in $(seq 60); do
  if curl -sf http://127.0.0.1:9222/json | grep -q '"title": "DeepGym'; then
    sleep 1
    exit 0
  fi
  sleep 0.5
done
echo 'app did not load' >&2
exit 1
