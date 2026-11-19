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
  shell commands given to ``os.system`` or ``subprocess``. Imports are resolved
  (``from pathlib import Path as P``, ``import shutil as sh``) regardless of scope or
  order: a name bound in several places takes every meaning, and module names and
  builtins keep their own, so a conditional ``import foo as os`` can't hide
  ``os.remove``. A literal counts through calls that keep the path (``str``,
  ``os.fspath``, ``os.path.abspath``, ``realpath``, ``normpath``, ``Path(...).resolve()``)
  or join literals (``Path('a') / 'b'``, ``os.path.join('a', 'b')``, ``joinpath``).
  Methods of anything else, such as ``str(...).replace``, change no file.
* ``cli``: the files Claude Code's Write, Edit, MultiEdit and NotebookEdit calls and
  Codex's file changes touched or tried to (an attempt the tool reported failing is
  marked ``failed``), and what the shell commands each ran name as above.

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
import builtins
import posixpath
import re
import shlex
from collections.abc import Sequence
from itertools import takewhile
from typing import NamedTuple, cast

from pydantic import JsonValue

from deepgym.traces.schema import TraceStep


class Write(NamedTuple):
    """A path an action changes, or tried to."""

    path: str
    span: tuple[int, int] | None
    """Where it is in the action's text; None if it isn't there."""
    failed: bool = False
    """True when the tool that tried to change it reported failing."""


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
_BUILTINS = frozenset(dir(builtins))
_PATH_TYPES = frozenset({'Path', 'PosixPath', 'PurePath', 'PurePosixPath'})
_PATH_CALLS = _PATH_TYPES | {f'pathlib.{name}' for name in _PATH_TYPES}
_SAME_PATH = frozenset({
    'str', 'os.fspath', 'os.path.abspath', 'os.path.realpath', 'os.path.normpath',
    'os.path.expanduser',
})  # fmt: skip
"""Calls that return the path they are given, as far as which file it names goes."""
_PATH_METHODS = frozenset({'resolve', 'absolute', 'expanduser'})
"""``Path`` methods that return the same path."""
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
            return Write(self.value, self.span)
        return Write(self.value, (self.start, self.start + len(self.value)))

    def inside(self, write: Write) -> Write:
        """Return *write*, found in this word's value, cited in the whole text."""
        if self.start is None or write.span is None:
            return write._replace(span=self.span)
        start, end = write.span
        return write._replace(span=(self.start + start, self.start + end))


def action_writes(step: TraceStep) -> list[Write]:
    """Return the paths *step*'s action changes, as far as its text shows (see the module).

    Raises:
        ValueError: If a ``code`` action doesn't parse as Python here though its observation
            doesn't show Python refusing it (it may need a newer Python).
    """
    action = step.action
    match action.action_type:
        case 'write_file':
            return [Write(action.path or '', None)]
        case 'bash':
            return shell_writes(action.content)
        case 'cli':
            return cli_writes(action.events)
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
    return _PythonSource(code).writes()


class _PythonSource:
    """Python code, read for the literal paths its calls change."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.tree = ast.parse(code)
        self.lines = _NEWLINE.split(code)
        self.starts = [0, *(match.end() for match in _NEWLINE.finditer(code))]
        self.aliases = _aliases(self.tree)

    def writes(self) -> list[Write]:
        """Return what each call to ``open``, ``os``, ``shutil``, ``subprocess`` or a
        ``Path`` method changes, by its position in the code."""
        writes: list[Write] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                writes += self._call_writes(node)
        return sorted(writes, key=lambda write: write.span or (0, 0))

    def _call_writes(self, node: ast.Call) -> list[Write]:
        """Return the paths one call changes under any meaning its name may have (see
        :meth:`_qualified`)."""
        func = node.func
        if isinstance(func, ast.Attribute) and self._is_path(func.value):
            return self._writes_as(node, func.attr, func.value)
        names = {
            name
            for module, _, name in (
                qualified.rpartition('.') for qualified in self._qualified(func)
            )
            if not module or module in _PY_MODULES  # not a method of anything else (str.replace)
        }
        writes = [write for name in sorted(names) for write in self._writes_as(node, name)]
        return list(dict.fromkeys(writes))

    def _writes_as(
        self, node: ast.Call, name: str, receiver: ast.expr | None = None
    ) -> list[Write]:
        """Return the paths *node* changes as a call of *name*: an ``os``, ``shutil``,
        ``subprocess``, ``io`` or builtin function, or a method of the ``Path`` *receiver*."""
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
        elif name in _PY_SHELL and receiver is None and node.args:
            return self._shell(node.args[0])
        else:
            return []
        return [found.cite() for path in paths if (found := self._word(path)) is not None]

    def _qualified(self, node: ast.expr) -> set[str]:
        """Return every dotted name *node* may refer to; empty if it isn't a name.

        Imports are resolved without regard to scope or order (``P`` from
        ``from pathlib import Path as P`` is ``pathlib.Path``), so a name bound in several
        places has each meaning, and a module name or builtin keeps its own: a conditional
        or function-local ``import foo as os`` can't hide ``os.remove``.
        """
        if isinstance(node, ast.Name):
            bound = self.aliases.get(node.id, set())
            own = not bound or node.id in _PY_MODULES or node.id in _BUILTINS
            return bound | {node.id} if own else bound
        if isinstance(node, ast.Attribute):
            return {f'{base}.{node.attr}' for base in self._qualified(node.value)}
        return set()

    def _is_path(self, node: ast.expr) -> bool:
        """Return whether *node* makes a ``Path``: a call of one, ``/`` on one, or a method
        of one that returns one (``resolve``, ``joinpath``, ...)."""
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, ast.Div) and self._is_path(node.left)
        if not isinstance(node, ast.Call):
            return False
        if self._qualified(node.func) & _PATH_CALLS:
            return True
        func = node.func
        methods = _PATH_METHODS | {'joinpath'}
        return (
            isinstance(func, ast.Attribute) and func.attr in methods and self._is_path(func.value)
        )

    def _word(self, node: ast.expr | None) -> _Word | None:
        """Return the literal path *node* names: a string literal, through calls that keep
        it (``str``, ``os.fspath``, ``os.path.abspath``, ``Path(...).resolve()``, ...) or
        join literals (``Path('a') / 'b'``, ``os.path.join('a', 'b')``); None otherwise."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self._literal(node)
        if isinstance(node, ast.BinOp) and self._is_path(node):
            return self._joined(node, [node.left, node.right])
        if not isinstance(node, ast.Call):
            return None
        qualified, func = self._qualified(node.func), node.func
        if qualified & (_SAME_PATH | _PATH_CALLS) and len(node.args) == 1:
            return self._word(node.args[0])
        if qualified & (_PATH_CALLS | {'os.path.join'}):
            return self._joined(node, node.args)
        if isinstance(func, ast.Attribute) and self._is_path(func.value):
            if func.attr in _PATH_METHODS and not node.args:
                return self._word(func.value)
            if func.attr == 'joinpath':
                return self._joined(node, [func.value, *node.args])
        return None

    def _joined(self, node: ast.expr, parts: Sequence[ast.expr]) -> _Word | None:
        """Return the path the literal *parts* join to, cited as the whole of *node*."""
        words = [found for part in parts if (found := self._word(part)) is not None]
        if not parts or len(words) < len(parts):
            return None
        return _Word(posixpath.join(*(word.value for word in words)), self._span(node), None)

    def _literal(self, node: ast.Constant) -> _Word:
        """Return a string literal as a word, its value placed when written as is."""
        start, end = self._span(node)
        raw, value = self.code[start:end], str(node.value)
        opening = _LITERAL_OPEN.match(raw)
        inner = None
        if opening is not None and raw[opening.end() :] == value + opening.group(1):
            inner = start + opening.end()
        return _Word(value, (start, end), inner)

    def _span(self, node: ast.expr) -> tuple[int, int]:
        """Return where *node* is in the code, in code points (ast counts UTF-8 bytes)."""
        end_line, end_column = cast(int, node.end_lineno), cast(int, node.end_col_offset)
        return self._offset(node.lineno, node.col_offset), self._offset(end_line, end_column)

    def _offset(self, line: int, column: int) -> int:
        """Return the code point offset of a 1-based line and a UTF-8 byte column."""
        return self.starts[line - 1] + len(self.lines[line - 1].encode()[:column].decode())

    def _shell(self, command: ast.expr) -> list[Write]:
        """Return what a literal shell command given to ``os.system`` or ``subprocess``
        changes: a string read as shell, or a list or tuple of strings read as one command."""
        if isinstance(command, (ast.List, ast.Tuple)):
            words = [found for item in command.elts if (found := self._word(item)) is not None]
            return _command_writes(words) if len(words) == len(command.elts) else []
        script = self._word(command)
        if script is None:
            return []
        return [script.inside(found) for found in shell_writes(script.value)]


def _aliases(tree: ast.AST) -> dict[str, set[str]]:
    """Return every dotted name each name is bound to by an import anywhere in *tree*."""
    aliases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition('.')[0]
                bound = alias.name if alias.asname else root
                aliases.setdefault(alias.asname or root, set()).add(bound)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                bound = f'{node.module}.{alias.name}'
                aliases.setdefault(alias.asname or alias.name, set()).add(bound)
    return aliases


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    """Return the value of keyword argument *name* of *node*, if given."""
    return next((keyword.value for keyword in node.keywords if keyword.arg == name), None)


def cli_writes(events: Sequence[dict[str, JsonValue]]) -> list[Write]:
    """Return the paths a CLI run's events show it changed or tried to: Claude Code's
    Write, Edit, MultiEdit and NotebookEdit calls and Codex's file changes, marked
    ``failed`` when the tool reported failing, plus what the shell commands each ran change
    (see :func:`shell_writes`)."""
    failed = {
        call
        for event in events
        if event.get('type') == 'user'
        for part in _parts(event.get('message'))
        if part.get('type') == 'tool_result'
        and part.get('is_error') is True
        and isinstance(call := part.get('tool_use_id'), str)
    }
    writes: list[Write] = []
    for event in events:
        item = event.get('item')
        if event.get('type') == 'assistant':
            writes += _claude_writes(event.get('message'), failed)
        elif event.get('type') == 'item.completed' and isinstance(item, dict):
            changes = item.get('changes')
            if item.get('type') == 'file_change' and isinstance(changes, list):
                refused = item.get('status') == 'failed'
                paths = [change.get('path') for change in changes if isinstance(change, dict)]
                writes += [Write(path, None, refused) for path in paths if isinstance(path, str)]
            elif item.get('type') == 'command_execution':
                writes += _shell_paths(item.get('command'))
    return writes


def _parts(message: JsonValue) -> list[dict[str, JsonValue]]:
    """Return the content blocks of a Claude Code message."""
    content = message.get('content') if isinstance(message, dict) else None
    return [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []


def _claude_writes(message: JsonValue, failed: set[str]) -> list[Write]:
    """Return the paths a Claude Code assistant message's tool calls change, marking edits
    whose call id is in *failed*."""
    writes: list[Write] = []
    for part in _parts(message):
        tool_input = part.get('input')
        if part.get('type') != 'tool_use' or not isinstance(tool_input, dict):
            continue
        if part.get('name') in _CLAUDE_EDITS:
            path = tool_input.get('file_path') or tool_input.get('notebook_path')
            if isinstance(path, str):
                writes.append(Write(path, None, part.get('id') in failed))
        elif part.get('name') == 'Bash':
            writes += _shell_paths(tool_input.get('command'))
    return writes


def _shell_paths(command: JsonValue) -> list[Write]:
    """Return the paths a shell command recorded in an event changes; not in the action's
    text, so without spans."""
    if not isinstance(command, str):
        return []
    return [write._replace(span=None) for write in shell_writes(command)]
