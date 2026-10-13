"""Advisory authoring lint over verifier source: warnings, never a safety check.

The judge must reach the solution only through ``deepgym_solution`` (load_solution,
run_solution, load_files, solution_source): that isolation is the boundary. This lint
reads the judge's code and warns where it looks otherwise:

* positive rules: the judge never calls one of those helpers, or hands the solution
  path to anything else (``open(solution_path)``, ``Path(solution_path)``, ...);
* negative rules: constructs that could run source in-process (exec, eval, compile,
  __import__, importlib, runpy, pickle, timeit, profilers, test runners), start a
  process (subprocess, os.system, os.exec*, ...), edit sys.path, or alias process and
  builtins modules so a launcher hides behind another name.

A pattern check over Python source can always be dodged (strings, encodings,
indirection), so a clean lint proves nothing and gates nothing: ``deepgym check``
prints findings as warnings and approval records keep them marked advisory.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import PurePosixPath

ADVISORY_NOTE = (
    'Advisory lint over the judge code. It can be dodged, so it is never evidence that '
    'the verifier isolates the solution; isolation comes from the judge reaching the '
    'solution only through load_solution, run_solution, load_files or solution_source.'
)
"""What deepgym check prints above lint warnings and approval records keep with them."""

_HELPERS = frozenset({'load_solution', 'run_solution', 'load_files', 'solution_source'})
"""The deepgym_solution helpers through which the judge should reach the solution."""
_PATH_TAKERS = frozenset({'load_solution', 'run_solution', 'solution_source'})
"""Helpers that take the solution path itself."""
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

_DYNAMIC = frozenset({'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'vars'})
"""Builtins that run code or reach namespaces by computed names."""
_RISKY_MODULES = frozenset(
    {
        'importlib',
        'runpy',
        'solution',
        'code',
        'codeop',
        'ctypes',
        'multiprocessing',
        'pickle',
        'marshal',
        'shelve',
        'timeit',
        'cProfile',
        'profile',
        'trace',
        'pdb',
        'bdb',
        'doctest',
        'pydoc',
        'pkgutil',
        'zipimport',
        'site',
        'unittest',
        'logging.config',
        'imp',
    }
)
"""Modules (and their submodules) that, imported at all, can execute source or import
files outside deepgym_solution: loaders, pickles, timers, profilers, debuggers, test
runners."""


def _risky_module(name: str) -> bool:
    """True if module *name* is, or is inside, one of :data:`_RISKY_MODULES`."""
    return any(name == risky or name.startswith(f'{risky}.') for risky in _RISKY_MODULES)


_SPAWNERS = {
    'subprocess': frozenset(
        {'run', 'Popen', 'call', 'check_call', 'check_output', 'getoutput', 'getstatusoutput'}
    ),
    'os': frozenset({'system', 'popen', 'startfile', 'fork', 'forkpty'}),
    'pty': frozenset({'spawn', 'fork'}),
    'asyncio': frozenset({'create_subprocess_exec', 'create_subprocess_shell'}),
    'asyncio.subprocess': frozenset({'create_subprocess_exec', 'create_subprocess_shell'}),
    'concurrent.futures': frozenset({'ProcessPoolExecutor'}),
}
"""Calls that start a process: the judge leaves running the solution to run_solution."""
_OS_SPAWN_PREFIXES = ('exec', 'spawn', 'posix_spawn')
_ANY_RECEIVER = frozenset({'subprocess_exec', 'subprocess_shell'})
"""Event-loop launchers, flagged whatever object they're called on."""
_GUARDED = frozenset({'builtins', '__builtins__', 'sys', *_SPAWNERS})
"""Modules usable only as ``module.attribute``: an alias, or the module object passed
around (assigned, returned, given to getattr), could hide a launcher."""
_GUARDED_NAMES = frozenset(name.rpartition('.')[2] for name in _GUARDED)


def _risky(qualified: str) -> bool:
    """True if dotted name *qualified* runs code dynamically, starts a process, or is
    sys.path or sys.modules."""
    name = qualified.removeprefix('builtins.').removeprefix('__builtins__.')
    module, _, attr = name.rpartition('.')
    return (
        name in _DYNAMIC
        or name in ('sys.path', 'sys.modules')
        or attr in _ANY_RECEIVER
        or attr in _SPAWNERS.get(module, ())
        or (module == 'os' and attr.startswith(_OS_SPAWN_PREFIXES))
    )


def _bindings(tree: ast.AST) -> dict[str, set[str]]:
    """Map every name an import in *tree*, in any scope, binds to what it may stand for."""
    bound: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                target = alias.name if alias.asname else root
                bound.setdefault(alias.asname or root, set()).add(target)
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                target = f'{node.module}.{alias.name}'
                bound.setdefault(alias.asname or alias.name, set()).add(target)
    return bound


def _qualified(node: ast.expr, bound: dict[str, set[str]]) -> set[str]:
    """Every dotted name a Name or Attribute chain may refer to.

    Conservative: a name stands for itself and for everything any import binds it to,
    whatever the scope, so an alias can't hide behind a same-named import elsewhere.
    """
    match node:
        case ast.Name(id=name):
            return {name} | bound.get(name, set())
        case ast.Attribute(value=value, attr=attr):
            return {f'{base}.{attr}' for base in _qualified(value, bound)}
    return set()


def _import_risks(node: ast.Import | ast.ImportFrom) -> list[str]:
    """What an import statement brings in that :func:`_risks` warns about."""
    if isinstance(node, ast.Import):
        return [
            f'imports {a.name}' if _risky_module(a.name) else f'imports {a.name} as {a.asname}'
            for a in node.names
            if _risky_module(a.name) or (a.asname and a.name in _GUARDED)
        ]
    module = node.module or ''
    found = [f'imports {module}'] if _risky_module(module) else []
    # A support module can re-export what it imports, so the name itself counts.
    return found + [
        f'imports {a.name} from {module}'
        for a in node.names
        if _risky(f'{module}.{a.name}')
        or _risky_module(f'{module}.{a.name}')
        or (a.name == '*' and module in _GUARDED)
        or (a.asname and (a.name in _GUARDED_NAMES or f'{module}.{a.name}' in _GUARDED))
    ]


def _risks(tree: ast.AST, filename: str) -> list[str]:
    """Lines of *tree* (from *filename*) that could import or run the solution in the judge.

    Importing a module that can execute source (importlib, runpy, ``solution``,
    pickle, timeit, profilers, debuggers, test runners: :data:`_RISKY_MODULES`), or a
    risky name itself (``from subprocess import run``, which a support
    module could re-export); any use, under any import alias, of exec, eval, compile,
    __import__, globals, vars, sys.path, sys.modules or a call that starts a process
    (subprocess.run, os.system, os.exec*, asyncio.create_subprocess_exec, ...); and
    any use of a process or builtins module other than ``module.attribute``.
    """
    bound = _bindings(tree)
    bases = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    found = []
    for node in ast.walk(tree):
        where = f'{filename}:{getattr(node, "lineno", 0)}'
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found += [f'{where} {risk}' for risk in _import_risks(node)]
        elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            names = _qualified(node, bound)
            if isinstance(node, ast.Attribute) and node.attr in _ANY_RECEIVER:
                names.add(node.attr)
            if risky := sorted(name for name in names if _risky(name)):
                found.append(f'{where} uses {risky[0]}')
            elif id(node) not in bases and (guarded := sorted(names & _GUARDED)):
                found.append(f'{where} passes {guarded[0]} around')
    return found


def _imports(tree: ast.AST) -> set[str]:
    """Names of every module *tree* imports: top-level packages, and relative ones."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {node.module.split('.')[0]} if node.module else set()
            names |= {alias.name for alias in node.names} if node.level else set()
    return names


def _callee(call: ast.Call) -> str:
    """The name a call's function goes by: ``f`` for ``f(...)`` and ``m.f(...)``."""
    match call.func:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return '<expression>'


def _is_solution_path(node: ast.AST) -> bool:
    """True for ``solution_path`` or ``sys.argv[1]``, the path the judge is given."""
    match node:
        case ast.Name(id='solution_path', ctx=ast.Load()):
            return True
        case ast.Subscript(
            value=ast.Attribute(value=ast.Name(id='sys'), attr='argv'), slice=ast.Constant(1)
        ):
            return True
    return False


def _path_warnings(tree: ast.AST, filename: str) -> list[str]:
    """Places *tree* hands the solution path to anything but a helper or its own functions.

    Allowed: an argument of load_solution, run_solution or solution_source; an argument
    of a function defined in the same file (which gets the same scrutiny under its own
    ``solution_path`` parameter); a plain assignment such as ``solution_path = sys.argv[1]``.
    """
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    local = {n.name for n in ast.walk(tree) if isinstance(n, (*_FUNCTIONS, ast.ClassDef))}
    found = []
    for node in ast.walk(tree):
        if not _is_solution_path(node):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.keyword):
            parent = parents.get(parent)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            continue
        where = f'{filename}:{node.lineno}'
        if not isinstance(parent, ast.Call) or parent.func is node:
            found.append(f'{where} uses the solution path outside deepgym_solution helpers')
        elif (callee := _callee(parent)) not in _PATH_TAKERS | local:
            found.append(f'{where} passes the solution path to {callee}()')
    return found


def lint_verifier(verifier: str, support: Mapping[str, str], *, prepended: bool) -> list[str]:
    """Advisory warnings about the judge's code: never evidence that it isolates the solution.

    The judge's code is *verifier* plus the *support* sources (path -> source) it runs:
    all of them when they are *prepended* to it (multi-turn), else every one with an
    imported name anywhere in its path (a module, or any file of an imported package),
    found through the verifier's imports and theirs. Support files it only reads, such
    as a repo-patch fixture, run on the solution's side and are left out.

    Warns when that code never calls a deepgym_solution helper, hands the solution path
    to anything else, or uses a construct that could run code in the judge (see the
    module docstring).

    Raises:
        SyntaxError: If one of them doesn't parse (``filename`` names it).
    """
    tree = ast.parse(verifier, filename='verifier.py')
    parsed = {'verifier.py': tree}
    wanted = _imports(tree)
    modules = {path: text for path, text in support.items() if prepended or path.endswith('.py')}
    while pending := [
        path
        for path in modules
        if path not in parsed
        and (prepended or wanted.intersection(PurePosixPath(path).with_suffix('').parts))
    ]:
        for path in pending:
            parsed[path] = ast.parse(modules[path], filename=path)
            wanted |= _imports(parsed[path])
    found = [
        warning
        for path, node in parsed.items()
        for warning in [*_risks(node, path), *_path_warnings(node, path)]
    ]
    helpers = {
        name
        for node in parsed.values()
        for sub in ast.walk(node)
        if (name := getattr(sub, 'id', None) or getattr(sub, 'attr', None)) in _HELPERS
    }
    if not helpers:
        listed = ', '.join(sorted(_HELPERS))
        found.insert(0, f'verifier.py never calls a deepgym_solution helper ({listed})')
    return found
