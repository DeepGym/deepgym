"""Run the sandbox-side CLI supervisor and fixed-destination CONNECT transport.

This file is copied into a root-only directory and runs with the standard library.
It never imports the agent's work directory or executes a model-produced root command.
"""

import ipaddress
import json
import os
import selectors
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

_LOOPBACK_NO_PROXY = 'localhost,localhost.,127.0.0.1,::1,[::1],0:0:0:0:0:0:0:1,[0:0:0:0:0:0:0:1]'


def public_addresses(host: str) -> list[tuple[int, tuple]]:
    """Resolve an API name once; refuse any non-public answer before connecting.

    Args:
        host: The fixed provider API hostname.

    Returns:
        Numeric socket addresses, without a second DNS lookup.

    Raises:
        ValueError: If DNS returns a private, loopback, or otherwise reserved address.
    """
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    resolved = [ipaddress.ip_address(row[4][0]) for row in addresses]
    if not addresses or any(not address.is_global or address.is_multicast for address in resolved):
        raise ValueError('Provider API resolved to a non-public address')
    return [(row[0], row[4]) for row in addresses]


def _copy(left: socket.socket, right: socket.socket) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(left, selectors.EVENT_READ, right)
        selector.register(right, selectors.EVENT_READ, left)
        while True:
            ready = selector.select()
            for key, _ in ready:
                data = key.fileobj.recv(65536)
                if not data:
                    return
                key.data.sendall(data)


def _proxy_handler(host: str) -> type[socketserver.BaseRequestHandler]:
    class Connect(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.settimeout(15)
            header = b''
            try:
                while b'\r\n\r\n' not in header and len(header) <= 8192:
                    part = self.request.recv(1)
                    if not part:
                        return
                    header += part
                first = header.split(b'\r\n', 1)[0].decode('ascii')
                if b'\r\n\r\n' not in header or first not in (
                    f'CONNECT {host}:443 HTTP/1.1',
                    f'CONNECT {host}:443 HTTP/1.0',
                ):
                    self.request.sendall(b'HTTP/1.1 403 Forbidden\r\n\r\n')
                    return
                addresses = public_addresses(host)
                family, address = addresses[0]
                with socket.socket(family, socket.SOCK_STREAM) as upstream:
                    upstream.settimeout(30)
                    upstream.connect(address)
                    self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                    _copy(self.request, upstream)
            except (OSError, ValueError, UnicodeError):
                return

    return Connect


def _bridge(directory: Path) -> None:
    class Forward(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                    upstream.connect(str(directory / 'proxy.sock'))
                    _copy(self.request, upstream)
            except OSError:
                return

    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), Forward) as server:
        server.daemon_threads = True
        (directory / 'port').write_text(str(server.server_address[1]))
        server.serve_forever()


def _supervise(directory: Path) -> None:
    config = json.loads((directory / 'request.json').read_text())
    host = {'anthropic': 'api.anthropic.com', 'openai': 'api.openai.com'}[config['provider']]
    started = time.monotonic()
    deadline = started + config['timeout']
    bridge = None
    process = None
    server = socketserver.ThreadingUnixStreamServer(
        str(directory / 'proxy.sock'), _proxy_handler(host)
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    status = {'exit_code': -1, 'stop_reason': None}
    try:
        descriptor = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(descriptor, 0)
        finally:
            os.close(descriptor)
        status['stop_reason'] = _stop_reason(directory, deadline)
        if status['stop_reason'] is not None:
            return
        if config.get('supported_version') is not None:
            _check_version(config, status, deadline)
            if status.get('version_error'):
                return
        bridge = subprocess.Popen(
            [*config['namespace'], sys.executable, '-I', __file__, 'bridge', str(directory)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        while not (directory / 'port').exists():
            status['stop_reason'] = _stop_reason(directory, deadline)
            if status['stop_reason'] is not None:
                return
            if bridge.poll() is not None or time.monotonic() - started > 10:
                raise RuntimeError('Could not start the isolated provider proxy')
            time.sleep(0.01)
        proxy = f'http://127.0.0.1:{(directory / "port").read_text()}'
        environment = {
            **config['env'],
            'HTTP_PROXY': proxy,
            'HTTPS_PROXY': proxy,
            'http_proxy': proxy,
            'https_proxy': proxy,
            'NO_PROXY': _LOOPBACK_NO_PROXY,
            'no_proxy': _LOOPBACK_NO_PROXY,
        }
        with (directory / 'output').open('wb') as output:
            status['stop_reason'] = _stop_reason(directory, deadline)
            if status['stop_reason'] is not None:
                return
            process = subprocess.Popen(
                [*config['prefix'], *config['argv']],
                cwd=config['cwd'],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                status['stop_reason'] = _stop_reason(directory, deadline)
                if status['stop_reason'] is not None:
                    break
                if output.tell() > 32 * 1024 * 1024:
                    status['stop_reason'] = 'output_limit'
                    break
                time.sleep(0.02)
    except Exception as exc:
        status['error'] = str(exc)
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            status['exit_code'] = process.wait(timeout=10)
        if bridge is not None:
            if bridge.poll() is None:
                os.killpg(bridge.pid, signal.SIGKILL)
            bridge.wait(timeout=10)
        server.shutdown()
        server.server_close()
        temporary = directory / 'status.tmp'
        temporary.write_text(json.dumps(status))
        temporary.replace(directory / 'status.json')


def parse_cli_version(output: str) -> str | None:
    """Extract one unambiguous stable x.y.z release from a CLI version banner.

    Args:
        output: stdout from claude --version or codex --version.

    Returns:
        The release, or None for missing, malformed, or ambiguous banners.
    """
    versions = [
        word
        for word in output.split()
        if len(word.split('.')) == 3
        and all(part.isascii() and part.isdecimal() for part in word.split('.'))
    ]
    return versions[0] if len(versions) == 1 else None


def _check_version(config: dict, status: dict, deadline: float) -> None:
    command = config['argv'][0]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        status['stop_reason'] = 'time_budget'
        status['version_error'] = 'Wall budget exhausted before CLI version check'
        return
    failure = None
    try:
        observed = subprocess.run(
            [*config['prefix'], command, '--version'],
            cwd=config['cwd'],
            env=config['version_env'],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=min(10, remaining),
            check=False,
        )
        version = parse_cli_version(observed.stdout)
        if observed.returncode in (126, 127):
            failure = 'executable or interpreter is missing or not executable on the agent PATH'
        elif observed.returncode != 0:
            failure = f'version command exited with code {observed.returncode}'
    except OSError:
        version = None
        failure = 'executable or interpreter could not be started on the agent PATH'
    except subprocess.TimeoutExpired:
        version = None
        failure = 'version command timed out'
        if time.monotonic() >= deadline:
            status['stop_reason'] = 'time_budget'
    status['cli_version'] = version
    expected = config['supported_version']
    if failure:
        status['version_error'] = (
            f'Cannot start {command}: {failure}. Rebuild the sandbox image using '
            'deepgym.agents.cli_install_command() and ensure its root-owned Node '
            'interpreter is executable on the same unprivileged PATH (for example '
            '/usr/local/bin/node); do not load interactive nvm profiles.'
        )
    elif not cli_version_supported(version, expected):
        status['version_error'] = (
            f'Unsupported {command} version {version or "unknown"}; supported: {expected} or '
            'a later compatible release. Install one using '
            'deepgym.agents.cli_install_command().'
        )


def cli_version_supported(version: str | None, tested: str) -> bool:
    """Whether *version* is *tested* or a later compatible release, which keeps the flags
    and stream events the adapters use: the same major version, or below 1.0 the same
    minor version, since there a minor release may break them."""
    if version is None:
        return False
    have, want = (tuple(map(int, release.split('.'))) for release in (version, tested))
    line = 1 if want[0] else 2
    return have[:line] == want[:line] and have >= want


def _stop_reason(directory: Path, deadline: float) -> str | None:
    marker = directory / 'stop'
    if marker.exists():
        return marker.read_text()
    return 'time_budget' if time.monotonic() >= deadline else None


if __name__ == '__main__':
    operation, target = sys.argv[1:]
    if operation == 'bridge':
        _bridge(Path(target))
    else:
        _supervise(Path(target))
