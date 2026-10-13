"""Verifier for the API Request task.

Serve randomized slideshow documents from a local HTTP server, run the solution
once per document with its URL, and check that it prints that document's
``slideshow.title``. Titles, paths, extra fields and key order vary, and every run
draws new ones from a fresh seed (reported as ``seed`` for reproducing a run), so
only a solution that fetches and parses the response gets credit: hardcoded output,
keywords in the source and answers remembered from earlier runs score 0.
"""

import http.server
import json
import random
import secrets
import subprocess
import sys
import threading

from deepgym_solution import run_solution

DOCUMENTS = 5
WORDS = (
    'quarterly roadmap widget launch review pipeline summit pricing '
    'hiring design security metrics migration onboarding release budget'
).split()


def _phrase(rng):
    return ' '.join(rng.choice(WORDS).capitalize() for _ in range(rng.randint(2, 4)))


def _document(rng):
    """Return (title, document) with distractor titles and a shuffled key order."""
    title = _phrase(rng)
    slides = [{'title': _phrase(rng), 'type': rng.choice(['all', 'intro'])} for _ in range(3)]
    show = [('author', _phrase(rng)), ('date', f'2026-{rng.randint(1, 12):02d}-01')]
    show += [('slides', slides), ('title', title)]
    rng.shuffle(show)
    top = [('slideshow', dict(show)), ('title', f'API v{rng.randint(1, 9)}')]
    rng.shuffle(top)
    return title, dict(top)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve ``server.routes`` (path -> JSON bytes); anything else is a 404."""

    def do_GET(self):
        body = self.server.routes.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def verify(solution_path, test_cases_path=None):
    """Score the fraction of served documents whose title the solution prints exactly."""
    seed = secrets.randbits(64)  # fresh every run, so earlier runs reveal nothing
    rng = random.Random(seed)
    expected = {}
    routes = {}
    for _ in range(DOCUMENTS):
        path = f'/api/v{rng.randint(1, 3)}/slideshows/{rng.randrange(10**6)}'
        expected[path], document = _document(rng)
        routes[path] = json.dumps(document).encode('utf-8')

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    server.routes = routes
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cases = []
    try:
        for i, (path, title) in enumerate(expected.items()):
            url = f'http://127.0.0.1:{port}{path}'
            case = {'id': f'document_{i}', 'input_summary': path, 'expected_summary': title}
            try:
                proc = run_solution(solution_path, [url], loopback_port=port)
                printed = proc.stdout.strip()
                case['actual_summary'] = printed[:200]
                if proc.returncode != 0:
                    case['error'] = proc.stderr.strip()[-500:]
            except subprocess.TimeoutExpired:
                printed = None
                case['error'] = 'timed out'
            case['passed'] = printed == title
            case['score'] = 1.0 if case['passed'] else 0.0
            cases.append(case)
    finally:
        server.shutdown()
        server.server_close()

    passed = sum(case['passed'] for case in cases)
    return {
        'schema_version': '1.0',
        'score': passed / len(cases),
        'passed': passed == len(cases),
        'details': f'{passed}/{len(cases)} documents: printed slideshow.title exactly',
        'cases': cases,
        'seed': seed,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'cases': [],
                    'details': 'Usage: verifier.py <solution_path>',
                }
            )
        )
        sys.exit(1)

    try:
        result = verify(sys.argv[1])
    except Exception as e:
        result = {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Verifier error: {type(e).__name__}: {e}',
        }

    print(json.dumps(result))
