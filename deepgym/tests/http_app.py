"""A loopback HTTP app for API tool tests: real sockets, no sandbox, no provider call."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deepgym.models import MultiTurnEnvironment
from deepgym.session import LocalBackend, Session
from deepgym.tools import HttpRoute, HttpTool

API_KEY = 'local-app-key-not-a-secret'
BIG = 'x' * 300_000
"""A field long enough that any clipping of a response body would show."""

ROUTES = (
    HttpRoute(method='GET', path='/rest/bug/{id}'),
    HttpRoute(method='POST', path='/rest/bug/{id}/comment'),
    HttpRoute(method='GET', path='/rest/done'),
    HttpRoute(method='GET', path='/rest/redirect'),
    HttpRoute(method='POST', path='/rest/lost'),
    HttpRoute(method='GET', path='/rest/lost'),
    HttpRoute(method='POST', path='/rest/slow'),
    HttpRoute(method='GET', path='/rest/binary'),
)


class App:
    """A loopback server that records every request and answers a few fixed routes."""

    def __init__(self, redirect_to: str = 'http://127.0.0.1:9/') -> None:
        self.requests: list[dict] = []
        self.redirect_to = redirect_to
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self._handle()

            def do_POST(self) -> None:
                self._handle()

            def _handle(self) -> None:
                length = int(self.headers.get('Content-Length') or 0)
                body = self.rfile.read(length).decode('utf-8')
                app.requests.append(
                    {
                        'method': self.command,
                        'path': self.path,
                        'headers': {name.lower(): value for name, value in self.headers.items()},
                        'body': body,
                    }
                )
                route = self.path.split('?')[0]
                if route.endswith('/lost'):
                    self.close_connection = True  # the request arrived; the response never does
                    return
                if route.endswith('/slow'):
                    time.sleep(3)
                if route.endswith('/redirect'):
                    self._send(302, b'', {'Location': app.redirect_to})
                elif route.endswith('/done'):
                    self._send(200, b'DONE\n{"done": true}\n', {'Content-Type': 'text/plain'})
                elif route.endswith('/binary'):
                    self._send(200, b'\xff\x00\xfe', {})
                elif route.endswith('/comment'):
                    self._send(201, json.dumps({'id': 7}).encode(), {})
                elif route.startswith('/rest/bug/'):
                    bug = {'id': route.rsplit('/', 1)[1], 'summary': BIG}
                    link = {'Link': f'<{app.url}/rest/bug?page=2>; rel="next"'}
                    self._send(200, json.dumps({'bugs': [bug]}).encode(), link)
                else:
                    self._send(404, b'{"error": "not found"}', {})

            def _send(self, status: int, body: bytes, headers: dict[str, str]) -> None:
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        """The app's base URL."""
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def close(self) -> None:
        """Stop serving."""
        self.server.shutdown()
        self.server.server_close()


def tool(base_url: str) -> HttpTool:
    """The app's API tool, sending the app key with every request."""
    return HttpTool(
        name='tracker',
        description='Bug tracker REST API.',
        base_url=base_url,
        routes=ROUTES,
        headers={'X-API-KEY': API_KEY},
    )


def app_env(base_url: str, **settings: object) -> MultiTurnEnvironment:
    """An environment whose only tools call the app, with no step limit."""
    return MultiTurnEnvironment(
        task='Triage bug 1.',
        tools=(tool(base_url),),
        allowed_actions=frozenset({'tool', 'done'}),
        max_steps=None,
        **settings,
    )


def app_session(base_url: str, **settings: object) -> Session:
    """An unstarted local session of :func:`app_env`."""
    return Session(app_env(base_url, **settings), LocalBackend())
