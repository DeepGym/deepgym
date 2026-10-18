"""Offline kernel-model regressions for session-wide process containment."""

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepgym import _pid_namespace


class KernelModel:
    """Model PID 1 teardown separately from the worker's /proc inspection."""

    def __init__(self, root: Path) -> None:
        self.directory = root / 'control'
        self.directory.mkdir()
        self.proc = root / 'proc'
        self.proc.mkdir()
        self.init = 101
        self.uid = 997
        self.members: set[int] = set()
        self.events: list[str] = []
        self.signals: list[tuple[int, int]] = []
        self.closed: list[int] = []
        self.dead = False
        self.death_confirmed = True
        self.fork_at_kill = False
        self.add(self.init, 0, contained=True)
        self.add(14, self.uid, contained=True)
        self.add(900, 998, contained=False)
        namespace = self.proc / str(self.init) / 'ns'
        namespace.mkdir()
        (self.directory / 'pid').touch()
        (namespace / 'pid').hardlink_to(self.directory / 'pid')
        self.identity(self.uid)

    def identity(self, uid: int) -> None:
        (self.directory / 'identity.json').write_text(json.dumps({'pid': self.init, 'uid': uid}))

    def add(self, pid: int, uid: int, *, contained: bool) -> None:
        process = self.proc / str(pid)
        process.mkdir(exist_ok=True)
        (process / 'status').write_text(f'Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nState:\tS\n')
        if contained:
            self.members.add(pid)

    def fork_then_exit(self) -> None:
        self.add(15, self.uid, contained=True)
        (self.proc / '14' / 'status').unlink()
        self.members.discard(14)
        self.events.append('fork_then_exit')

    def path(self, path: str) -> object:
        if path == '/proc':

            def entries() -> list[Path]:
                self.events.append('enumerate')
                return list(self.proc.iterdir())

            return SimpleNamespace(iterdir=entries)
        if path.startswith('/proc/'):
            return self.proc / path.removeprefix('/proc/')
        return Path(path)

    def open_pidfd(self, pid: int) -> int:
        self.events.append('open_pidfd')
        if pid != self.init or not (self.proc / str(pid) / 'status').exists():
            raise ProcessLookupError(pid)
        return pid + 10000

    def send_signal(self, descriptor: int, sig: int) -> None:
        self.signals.append((descriptor, sig))
        assert descriptor == self.init + 10000
        assert sig == signal.SIGKILL
        if self.fork_at_kill:
            self.fork_then_exit()
        self.events.append('kernel_teardown')
        for pid in self.members:
            (self.proc / str(pid) / 'status').unlink(missing_ok=True)
        self.members.clear()
        self.dead = True

    def wait(self, readers: list[int], *args: object) -> tuple[list[int], list, list]:
        self.events.append('wait')
        return (readers if self.dead and self.death_confirmed else [], [], [])


@pytest.fixture
def kernel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> KernelModel:
    model = KernelModel(tmp_path)
    monkeypatch.setattr(_pid_namespace, 'Path', model.path)
    monkeypatch.setattr(
        _pid_namespace,
        'os',
        SimpleNamespace(pidfd_open=model.open_pidfd, close=model.closed.append),
    )
    monkeypatch.setattr(
        _pid_namespace,
        'signal',
        SimpleNamespace(SIGKILL=signal.SIGKILL, pidfd_send_signal=model.send_signal),
    )
    monkeypatch.setattr(_pid_namespace, 'select', SimpleNamespace(select=model.wait))
    return model


def test_kernel_teardown_covers_setup_child_fork_after_enumeration(kernel: KernelModel) -> None:
    # The old scan can miss Q: its snapshot contains P, which exits before status is read.
    snapshot = list(kernel.proc.iterdir())
    kernel.fork_then_exit()
    observed = []
    for entry in snapshot:
        try:
            if f'Uid:\t{kernel.uid}\t' in (entry / 'status').read_text():
                observed.append(entry.name)
        except FileNotFoundError:
            pass
    assert observed == []
    assert (kernel.proc / '15' / 'status').exists()

    # Fork again at SIGKILL delivery: namespace destruction must include the new child.
    (kernel.proc / '15' / 'status').unlink()
    kernel.members.discard(15)
    kernel.add(14, kernel.uid, contained=True)
    kernel.fork_at_kill = True
    kernel.events.clear()
    _pid_namespace.stop(kernel.directory)

    assert not (kernel.proc / '15' / 'status').exists()
    assert (kernel.proc / '900' / 'status').exists()
    assert kernel.signals == [(kernel.init + 10000, signal.SIGKILL)]
    assert kernel.events.index('fork_then_exit') < kernel.events.index('kernel_teardown')
    assert kernel.events.index('kernel_teardown') < kernel.events.index('enumerate')
    assert (kernel.directory / 'stopped').exists()


def test_live_agent_uid_outside_containment_fails_closed(kernel: KernelModel) -> None:
    kernel.add(902, kernel.uid, contained=False)
    with pytest.raises(RuntimeError, match='survived.*grading is unsafe'):
        _pid_namespace.stop(kernel.directory)
    assert not (kernel.directory / 'stopped').exists()
    assert (kernel.proc / '902' / 'status').exists()


def test_reused_init_pid_is_not_signalled(kernel: KernelModel) -> None:
    actual = kernel.proc / str(kernel.init) / 'ns' / 'pid'
    actual.unlink()
    actual.touch()
    with pytest.raises(RuntimeError, match='identity changed.*PID reuse'):
        _pid_namespace.stop(kernel.directory)
    assert kernel.signals == []
    assert kernel.closed == [kernel.init + 10000]
    assert not (kernel.directory / 'stopped').exists()


def test_missing_init_fails_closed(kernel: KernelModel) -> None:
    (kernel.proc / str(kernel.init) / 'status').unlink()
    with pytest.raises(ProcessLookupError):
        _pid_namespace.stop(kernel.directory)
    assert kernel.signals == []
    assert not (kernel.directory / 'stopped').exists()


def test_unconfirmed_namespace_death_fails_closed(kernel: KernelModel) -> None:
    kernel.death_confirmed = False
    with pytest.raises(RuntimeError, match='did not terminate'):
        _pid_namespace.stop(kernel.directory)
    assert 'enumerate' not in kernel.events
    assert kernel.closed == [kernel.init + 10000]
    assert not (kernel.directory / 'stopped').exists()


def test_repeated_stop_rechecks_survivors(kernel: KernelModel) -> None:
    _pid_namespace.stop(kernel.directory)
    kernel.add(902, kernel.uid, contained=False)
    with pytest.raises(RuntimeError, match='survived'):
        _pid_namespace.stop(kernel.directory)
    assert len(kernel.signals) == 1


@pytest.mark.parametrize('operation', ['create', 'verify_empty', 'stop'])
def test_root_uid_is_rejected_before_any_signal(kernel: KernelModel, operation: str) -> None:
    kernel.identity(0)
    with pytest.raises(ValueError, match='root'):
        if operation == 'create':
            _pid_namespace.create(kernel.directory, 0)
        elif operation == 'verify_empty':
            _pid_namespace.verify_empty(0)
        else:
            _pid_namespace.stop(kernel.directory)
    assert kernel.signals == []
    assert 'open_pidfd' not in kernel.events
