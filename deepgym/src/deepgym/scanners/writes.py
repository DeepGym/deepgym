"""The paths an action names as ones it writes, deletes or moves, read from its text.

Nothing is run, and only these forms are read:

* ``write_file``: its path.
* ``bash``: redirection targets (``>``, ``>>``, ``&>``, ...); the operands of ``rm``,
  ``rmdir``, ``unlink``, ``shred``, ``mv``, ``touch``, ``truncate``, ``tee``, ``mkdir``,
  ``chmod``, ``chown`` and ``chgrp``; the destination of ``cp``, ``install``, ``ln`` and
  ``rsync``; ``sed``, ``perl`` and ``ruby`` with ``-i``; the paths named after ``git rm``,
  ``git mv``, ``git checkout``, ``git restore`` and ``git clean`` (which deletes untracked
  files); and ``find ... -delete``. A command may follow ``if``, ``then``, ``do``, ``!``,
  ``{`` and the like, ``VAR=value``, and ``sudo``, ``env``, ``nice``, ``timeout``,
  ``nohup``, ``time``, ``command`` or ``exec`` with their options; ``bash -c`` /
  ``sh -c`` scripts are read the same way. A heredoc's body is data.
* ``code`` (Python): literal paths given to ``open`` with a write mode, to ``os`` and
  ``shutil`` calls that change files, to ``Path(...)`` methods that do, and literal
  shell commands given to ``os.system`` or ``subprocess``.
* ``cli``: the files Claude Code's Write, Edit, MultiEdit and NotebookEdit calls and
  Codex's file changes touched (those that failed are left out), and what the shell
  commands each ran name as above.

So an empty result means none of these forms named the path, not that nothing changed it.
Not read: commands that change the whole work tree (``git checkout .`` or a branch,
``git reset --hard``, ``git stash``, ``git clean`` without paths), patches (``git apply``,
``patch``), paths held in variables or built at run time, shell globs (read as written,
never expanded: ``rm t*/a.py``), scripts run from a file or heredoc, a ``cd`` before a
relative path, and text typed into a GUI. Diff the final workspace when you need to know
every file that changed.
"""

from __future__ import annotations

import ast
import posixpath
import re
import shlex
from collections.abc import Callable, Sequence
from itertools import takewhile
from typing import NamedTuple, cast

from pydantic import JsonValue

from deepgym.traces.schema import TraceStep

Write = tuple[str, tuple[int, int] | None]
"""A path an action changes, and where it is in the action's text (None if it isn't)."""

_SHELL = re.compile(
    r'(?P<skip>\\\n)'
    r'|(?P<comment>(?<![^\s;&|()<>])#[^\n]*)'
    r'|(?P<op>&&|\|\||;;|&>>?|[0-9]*>>|[0-9]*>[|&]?|<<<|<<-?|<>|<&?|[;&|()\n])'
    r'|(?P<word>(?:\\[\s\S]|\'[^\']*\'|"(?:\\[\s\S]|[^"\\])*"|[^\s;&|()<>\'"\\])+)'
)
_HEREDOC_BODY = re.compile(
    r'(?<!<)<<(?!<)-?[ \t]*([\'"]?)(\w+)\1[^\n]*\n(?P<body>.*?)(?:^\t*\2[ \t]*$|\Z)',
    re.MULTILINE | re.DOTALL,
)
_KEYWORDS = frozenset({'if', 'then', 'else', 'elif', 'do', 'while', 'until', '!', '{'})
_WRAPPERS = {
    'sudo': frozenset({
        '-u', '-g', '-U', '-C', '-h', '-p', '-r', '-t', '-D', '-R', '-T', '--user', '--group',
        '--other-user', '--close-from', '--host', '--prompt', '--role', '--type', '--chdir',
        '--chroot', '--command-timeout',
    }),
    'env': frozenset({'-u', '-C', '-S', '--unset', '--chdir', '--split-string'}),
    'nice': frozenset({'-n', '--adjustment'}),
    'timeout': frozenset({'-s', '-k', '--signal', '--kill-after'}),
    'time': frozenset({'-f', '-o', '--format', '--output'}),
    'exec': frozenset({'-a'}),
    'nohup': frozenset(),
    'command': frozenset(),
}  # fmt: skip
"""Commands that run the command after them, with their options that take the next word."""
_ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=.*', re.DOTALL)
_SHELLS = frozenset({'sh', 'bash', 'zsh', 'dash'})
_SCRIPT_FLAG = re.compile(r'-[A-Za-z]*c[A-Za-z]*')
_GIT_OPTIONS = frozenset({'-C', '-c', '--git-dir', '--work-tree', '--namespace'})
_GIT_CHANGES = {
    'rm': frozenset({'--pathspec-from-file'}),
    'mv': frozenset(),
    'checkout': frozenset({'-b', '-B', '--orphan', '--conflict', '--pathspec-from-file'}),
    'restore': frozenset({'-s', '--source', '--pathspec-from-file'}),
    'clean': frozenset({'-e', '--exclude'}),
}
"""git subcommands that change the paths they name, with their options that take the next word."""
_IN_PLACE = {'sed': 'efl', 'perl': 'IMmreExCdDFl0', 'ruby': 'IMmreExCdDFl0'}
"""In-place editors, with their short options that take the rest of a cluster as their
argument (so ``-Mstrict`` holds no ``-i``)."""
_TO_LAST = frozenset({'cp', 'install', 'ln', 'rsync'})
_CHANGES_ALL = frozenset({
    'rm', 'rmdir', 'unlink', 'shred', 'mv', 'touch', 'truncate', 'tee', 'mkdir', 'chmod',
    'chown', 'chgrp',
})  # fmt: skip

_PY_MODULES = frozenset({'os', 'shutil', 'subprocess', 'io', 'builtins'})
_PATH_TYPES = frozenset({'Path', 'PosixPath', 'PurePath', 'PurePosixPath'})
_PY_CHANGES = frozenset({
    'remove', 'unlink', 'rmdir', 'removedirs', 'rename', 'renames', 'replace', 'rmtree',
    'move', 'touch', 'truncate', 'chmod', 'chown', 'mkdir', 'makedirs', 'symlink_to',
    'hardlink_to',
})  # fmt: skip
_PY_WRITES = frozenset({'write_text', 'write_bytes'})  # Path methods: the arguments are content
_PY_COPIES = frozenset({'copy', 'copy2', 'copyfile', 'copytree'})
_PY_SHELL = frozenset({
    'system', 'popen', 'run', 'call', 'check_call', 'check_output', 'Popen', 'getoutput',
    'getstatusoutput',
})  # fmt: skip
_WRITE_MODES = frozenset('wax+')
_LITERAL_OPEN = re.compile(r'[rRbBuUfF]*(\'\'\'|"""|\'|")')
_NEWLINE = re.compile(r'\r\n|\r|\n')

_CLAUDE_EDITS = frozenset({'Write', 'Edit', 'MultiEdit', 'NotebookEdit'})


class _Word(NamedTuple):
    """One shell word (or Python string literal) and where it is in the text."""

    value: str
    span: tuple[int, int]
    """The word as written, quotes included."""
    start: int | None
    """Where ``value`` starts in the text when it is written there as is (bare, or inside
    one pair of quotes); None when quotes or escapes inside change it."""

    def cite(self) -> Write:
        """Return the word as a path, cited where its value is, or as written."""
        if self.start is None or not self.value:
            return self.value, self.span
        return self.value, (self.start, self.start + len(self.value))

    def inside(self, write: Write) -> Write:
        """Return *write*, found in this word's value, cited in the whole text."""
        path, span = write
        if self.start is None or span is None:
            return path, self.span
        return path, (self.start + span[0], self.start + span[1])


def action_writes(step: TraceStep) -> list[Write]:
    """Return the paths *step*'s action changes, as far as its text shows (see the module).

    Raises:
        ValueError: If a ``code`` action doesn't parse as Python here though its observation
            doesn't show Python refusing it (it may need a newer Python).
    """
    action = step.action
    match action.action_type:
        case 'write_file':
            return [(action.path or '', None)]
        case 'bash':
            return shell_writes(action.content)
        case 'cli':
            return [(path, None) for path in cli_writes(action.events)]
        case 'code':
            try:
                return python_writes(action.content)
            except SyntaxError as exc:
                shown = step.observation.content if step.observation else ''
                if step.exit_code not in (None, 0) and 'SyntaxError' in shown:
                    return []  # Python refused it, so it changed nothing
                raise ValueError(
                    f'step {step.index}: the code does not parse as Python here ({exc}), so '
                    'what it changes can not be read'
                ) from exc
        case _:
            return []


def shell_writes(text: str) -> list[Write]:
    """Return the paths shell *text* changes, with their spans in it (see the module)."""
    writes: list[Write] = []
    words: list[_Word] = []
    redirect: str | None = None
    for match in _SHELL.finditer(_HEREDOC_BODY.sub(_blank_body, text)):
        if match.lastgroup == 'op':
            op = match.group()
            if '>' in op or op.startswith('<'):
                redirect = op
                continue
            writes += _command_writes(words)
            words, redirect = [], None
        elif match.lastgroup == 'word':
            raw, (start, end) = match.group(), match.span()
            value = ''.join(shlex.split(raw))
            quoted = raw[0] == raw[-1] and raw[0] in '\'"' and raw[1:-1] == value
            word = _Word(
                value, (start, end), start if raw == value else start + 1 if quoted else None
            )
            if redirect is None:
                words.append(word)
            elif '>' in redirect:
                writes.append(word.cite())
            redirect = None
    return writes + _command_writes(words)


def _blank_body(heredoc: re.Match[str]) -> str:
    """Return *heredoc* with its body blanked, lines and offsets kept: it is data."""
    head, body = heredoc.start('body') - heredoc.start(), heredoc.end('body') - heredoc.start()
    whole = heredoc.group()
    return whole[:head] + re.sub(r'[^\n]', ' ', whole[head:body]) + whole[body:]


def _command_writes(words: Sequence[_Word]) -> list[Write]:
    """Return the paths one simple command (its words, redirections left out) changes."""
    words = _command(words)
    if not words:
        return []
    command, args = posixpath.basename(words[0].value), words[1:]
    if command in _SHELLS:
        return next(
            (
                [script.inside(write) for write in shell_writes(script.value)]
                for flag, script in zip(args, args[1:])
                if _SCRIPT_FLAG.fullmatch(flag.value)
            ),
            [],
        )
    if command == 'git':
        args = _past_options(args, _GIT_OPTIONS)
        takes = _GIT_CHANGES.get(args[0].value) if args else None
        changed = [] if takes is None else _operands(args[1:], takes)
    elif command in _IN_PLACE:
        changed = _operands(args) if _in_place(args, _IN_PLACE[command]) else []
    elif command == 'find':
        deletes = any(word.value == '-delete' for word in args)
        changed = list(takewhile(lambda w: not w.value.startswith('-'), args)) if deletes else []
    elif command in _TO_LAST:
        changed = _operands(args)[-1:]
    else:
        changed = _operands(args) if command in _CHANGES_ALL else []
    return [word.cite() for word in changed]


def _command(words: Sequence[_Word]) -> Sequence[_Word]:
    """Return a simple command from its name on: without the shell keywords, ``VAR=value``
    words and wrappers (``sudo -u root``, ``timeout 5``, ...) before it."""
    at = 0
    while at < len(words):
        value = words[at].value
        wrapper = posixpath.basename(value)
        if value in _KEYWORDS or _ASSIGNMENT.fullmatch(value):
            at += 1
        elif wrapper in _WRAPPERS:
            at = len(words) - len(_past_options(words[at + 1 :], _WRAPPERS[wrapper]))
            if wrapper == 'timeout':
                at += 1  # its duration
        else:
            break
    return words[at:]


def _past_options(words: Sequence[_Word], takes: frozenset[str]) -> Sequence[_Word]:
    """Return *words* after their leading options (an option in *takes* with the word after
    it), up to and including ``--``."""
    at = 0
    while at < len(words) and words[at].value.startswith('-'):
        option = words[at].value
        at += 2 if option in takes else 1
        if option == '--':
            break
    return words[at:]


def _in_place(words: Sequence[_Word], attached: str) -> bool:
    """Return whether *words* hold ``-i`` (alone or in a cluster such as ``-pi``, before any
    option in *attached*, whose argument is the rest of the cluster) or ``--in-place``."""
    for word in words:
        value = word.value
        if value == '--in-place' or value.startswith('--in-place='):
            return True
        if not value.startswith('-') or value.startswith('--'):
            continue
        for letter in value[1:]:
            if letter == 'i':
                return True
            if letter in attached:
                break
    return False


def _operands(words: Sequence[_Word], takes: frozenset[str] = frozenset()) -> list[_Word]:
    """Return the words that aren't options or an option's argument (the word after an
    option in *takes*): those not starting with '-', and all after '--'."""
    operands: list[_Word] = []
    ended = skip = False
    for word in words:
        if skip:
            skip = False
        elif ended or not word.value.startswith('-'):
            operands.append(word)
        elif word.value == '--':
            ended = True
        else:
            skip = word.value in takes
    return operands


def python_writes(code: str) -> list[Write]:
    """Return the literal paths Python *code* changes, with their spans in it (see the module).

    Raises:
        SyntaxError: If *code* doesn't parse.
    """
    tree = ast.parse(code)
    lines = _NEWLINE.split(code)
    starts = [0, *(match.end() for match in _NEWLINE.finditer(code))]

    def offset(line: int, column: int) -> int:  # ast columns count UTF-8 bytes
        return starts[line - 1] + len(lines[line - 1].encode()[:column].decode())

    def word(node: ast.expr | None) -> _Word | None:
        path = _path_call(node)
        if path is not None and path.args:  # Path('x'), pathlib.Path('x')
            node = path.args[0]
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            return None
        start = offset(node.lineno, node.col_offset)
        end = offset(cast(int, node.end_lineno), cast(int, node.end_col_offset))
        raw, value = code[start:end], node.value
        opening = _LITERAL_OPEN.match(raw)
        inner = None
        if opening is not None and raw[opening.end() :] == value + opening.group(1):
            inner = start + opening.end()
        return _Word(value, (start, end), inner)

    writes: list[Write] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, receiver = _callee(node.func)
        if name == 'open':
            at = 1 if receiver is None else 0  # open(path, mode), Path(path).open(mode)
            mode = node.args[at] if len(node.args) > at else _keyword(node, 'mode')
            writing = (
                isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and bool(_WRITE_MODES & set(mode.value))
            )
            paths = (node.args[:1] if receiver is None else [receiver]) if writing else []
        elif name in _PY_CHANGES:
            paths = [*node.args, receiver]
        elif name in _PY_WRITES:
            paths = [receiver]
        elif name in _PY_COPIES:
            paths = [*node.args[1:2], _keyword(node, 'dst')]
        elif name in _PY_SHELL and node.args:
            writes += _python_shell(node.args[0], word)
            continue
        else:
            continue
        writes += [found.cite() for path in paths if (found := word(path)) is not None]
    return sorted(writes, key=lambda write: write[1] or (0, 0))


def _callee(func: ast.expr) -> tuple[str | None, ast.Call | None]:
    """Return the name a call calls, and the ``Path(...)`` call it is a method of, if any.
    Methods of anything but ``os``, ``shutil``, ``subprocess``, ``io``, ``builtins`` and a
    ``Path(...)`` call have no name here (``text.replace`` is no file change)."""
    if isinstance(func, ast.Name):
        return func.id, None
    if not isinstance(func, ast.Attribute):
        return None, None
    receiver = _path_call(func.value)
    if receiver is not None:
        return func.attr, receiver
    if isinstance(func.value, ast.Name) and func.value.id in _PY_MODULES:
        return func.attr, None
    return None, None


def _path_call(node: ast.expr | None) -> ast.Call | None:
    """Return *node* if it calls ``Path``, ``PosixPath``, ``PurePath`` or ``PurePosixPath``,
    by name or from ``pathlib``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        name = func.attr if func.value.id == 'pathlib' else None
    else:
        name = None
    return node if name in _PATH_TYPES else None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    """Return the value of keyword argument *name* of *node*, if given."""
    return next((keyword.value for keyword in node.keywords if keyword.arg == name), None)


def _python_shell(
    command: ast.expr, word: Callable[[ast.expr | None], _Word | None]
) -> list[Write]:
    """Return what a literal shell command given to ``os.system`` or ``subprocess``
    changes: a string read as shell, or a list or tuple of strings read as one command."""
    if isinstance(command, (ast.List, ast.Tuple)):
        words = [found for item in command.elts if (found := word(item)) is not None]
        return _command_writes(words) if len(words) == len(command.elts) else []
    script = word(command)
    return [] if script is None else [script.inside(found) for found in shell_writes(script.value)]


def cli_writes(events: Sequence[dict[str, JsonValue]]) -> list[str]:
    """Return the paths a CLI run's events show it changed: Claude Code's Write, Edit,
    MultiEdit and NotebookEdit calls whose result isn't an error and Codex's file changes
    that didn't fail, plus what the shell commands each ran change (see
    :func:`shell_writes`; a failed command may have changed files before it failed)."""
    failed = {
        call
        for event in events
        if event.get('type') == 'user'
        for part in _parts(event.get('message'))
        if part.get('type') == 'tool_result'
        and part.get('is_error') is True
        and isinstance(call := part.get('tool_use_id'), str)
    }
    paths: list[JsonValue] = []
    for event in events:
        item = event.get('item')
        if event.get('type') == 'assistant':
            paths += _claude_writes(event.get('message'), failed)
        elif event.get('type') == 'item.completed' and isinstance(item, dict):
            changes = item.get('changes')
            done = item.get('status') != 'failed'
            if item.get('type') == 'file_change' and done and isinstance(changes, list):
                paths += [change.get('path') for change in changes if isinstance(change, dict)]
            elif item.get('type') == 'command_execution':
                paths += _shell_paths(item.get('command'))
    return [path for path in paths if isinstance(path, str)]


def _parts(message: JsonValue) -> list[dict[str, JsonValue]]:
    """Return the content blocks of a Claude Code message."""
    content = message.get('content') if isinstance(message, dict) else None
    return [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []


def _claude_writes(message: JsonValue, failed: set[str]) -> list[JsonValue]:
    """Return the paths a Claude Code assistant message's tool calls change, leaving out
    edits whose call id is in *failed*."""
    paths: list[JsonValue] = []
    for part in _parts(message):
        tool_input = part.get('input')
        if part.get('type') != 'tool_use' or not isinstance(tool_input, dict):
            continue
        if part.get('name') in _CLAUDE_EDITS and part.get('id') not in failed:
            paths.append(tool_input.get('file_path') or tool_input.get('notebook_path'))
        elif part.get('name') == 'Bash':
            paths += _shell_paths(tool_input.get('command'))
    return paths


def _shell_paths(command: JsonValue) -> list[JsonValue]:
    """Return the paths a shell command recorded in an event changes."""
    return [path for path, _ in shell_writes(command)] if isinstance(command, str) else []
