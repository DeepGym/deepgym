"""Send one HTTP request described by a JSON file and print what happened as one JSON line.

The session runs this inside the sandbox (``python3 -I``) as the agent user, in the offline
network namespace, for an environment's declared API tool (see :mod:`deepgym.tools`). The
host built and checked the request; this only sends it. Nothing here comes from the model
but that request's data: no proxy is used and redirects come back as the response, never
followed, so the request reaches only its own URL. Stdlib only, as the image's python3 runs it.
"""

import base64
import http.client
import json
import sys
import time
import urllib.error
import urllib.request


class _Unfollowed(urllib.request.HTTPRedirectHandler):
    """Hand a 3xx back as the response instead of following its Location."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def main(path: str) -> int:
    """Send the request in *path*; print its record; return 0 if a response came back."""
    with open(path, encoding='utf-8') as file:
        spec = json.load(file)
    body = spec['body']
    request = urllib.request.Request(
        spec['url'],
        data=None if body is None else body.encode('utf-8'),
        headers=spec['headers'],
        method=spec['method'],
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _Unfollowed)
    record: dict[str, object] = {'request': spec['record']}
    started = time.monotonic()
    try:
        try:
            response = opener.open(request)
        except urllib.error.HTTPError as error:  # 4xx, 5xx and unfollowed 3xx are answers too
            response = error
        with response:
            raw = response.read()
    except (OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, 'reason', exc)
        record['error'] = f'No response: {type(reason).__name__}: {reason}'
        # Only a refused connection proves nothing was sent; any other loss may follow a write.
        record['uncertain'] = spec['mutation'] and not isinstance(reason, ConnectionRefusedError)
        status = 1
    else:
        answer: dict[str, object] = {
            'status': response.status,
            'reason': response.reason,
            'headers': [list(header) for header in response.headers.items()],
        }
        try:
            answer['body'] = raw.decode('utf-8')
        except UnicodeDecodeError:
            answer['body_base64'] = base64.b64encode(raw).decode('ascii')
        record['response'] = answer
        status = 0
    record['elapsed_ms'] = (time.monotonic() - started) * 1000
    # UTF-8 bytes whatever the sandbox locale, on one line: JSON escapes every newline.
    sys.stdout.buffer.write(json.dumps(record, ensure_ascii=False).encode('utf-8') + b'\n')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
