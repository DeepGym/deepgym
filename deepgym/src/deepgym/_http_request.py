"""Send one HTTP request, read as JSON from stdin, and print what happened as one JSON line.

The session runs this inside the sandbox (``python3 -I``) as the agent user, in the offline
network namespace, for an environment's declared API tool (see :mod:`deepgym.tools`). Root
pipes the request in from a file only root can read, so the app credentials it carries
never sit where the agent user could read them. The host built and checked every part of
the request; this only sends it with http.client, which parses no URL, uses no proxy and
follows no redirect: the request line carries exactly the target the host built, to the
host and port it named. Stdlib only, as the image's python3 runs it.
"""

import base64
import http.client
import json
import ssl
import sys
import time


def _answer(response: http.client.HTTPResponse, raw: bytes) -> dict:
    """The record of *response*: status, every header, and *raw* as its body."""
    answer: dict[str, object] = {
        'status': response.status,
        'reason': response.reason,
        'headers': [list(header) for header in response.getheaders()],
    }
    try:
        answer['body'] = raw.decode('utf-8')
    except UnicodeDecodeError:
        answer['body_base64'] = base64.b64encode(raw).decode('ascii')
    return answer


def main() -> int:
    """Send the request on stdin; print its record; return 0 if a whole response came back.

    A response cut short keeps what arrived (status, headers, the partial body), marked
    ``incomplete``.
    """
    spec = json.load(sys.stdin)
    if spec['scheme'] == 'https':
        context = ssl.create_default_context()
        connection = http.client.HTTPSConnection(spec['host'], spec['port'], context=context)
    else:
        connection = http.client.HTTPConnection(spec['host'], spec['port'])
    record: dict[str, object] = {'request': spec['record']}
    status = 1
    started = time.monotonic()
    try:
        connection.connect()
    except OSError as exc:  # nothing was sent
        record['error'] = f'No connection: {type(exc).__name__}: {exc}'
        record['uncertain'] = False
    else:
        body = spec['body']
        response = None
        try:
            connection.request(
                spec['method'],
                spec['target'],
                None if body is None else body.encode('utf-8'),
                spec['headers'],
            )
            response = connection.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException) as exc:
            # The request may have reached the app: a write is then neither known done nor not.
            kind = 'No' if response is None else 'Incomplete'
            record['error'] = f'{kind} response: {type(exc).__name__}: {exc}'
            record['uncertain'] = spec['mutation']
            if response is not None:
                partial = exc.partial if isinstance(exc, http.client.IncompleteRead) else b''
                record['response'] = {**_answer(response, partial), 'incomplete': True}
        else:
            record['response'] = _answer(response, raw)
            status = 0
    finally:
        connection.close()
    record['elapsed_ms'] = (time.monotonic() - started) * 1000
    # UTF-8 bytes whatever the sandbox locale, on one line: JSON escapes every newline.
    sys.stdout.buffer.write(json.dumps(record, ensure_ascii=False).encode('utf-8') + b'\n')
    return status


if __name__ == '__main__':
    sys.exit(main())
