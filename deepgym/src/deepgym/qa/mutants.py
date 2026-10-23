"""Mutation testing: small edits of the reference that a sound verifier must fail.

Each mutant changes one thing in the reference's syntax tree (stdlib ``ast``):

* off-by-one: an integer constant +1 or -1, or a comparison boundary (``<`` and ``<=``)
* constant return: a computed return value replaced by a constant of the same kind
* swapped branch: an if/else (or ``a if c else b``) swapped, or a lone if negated
* dropped statement: one statement removed

A mutant survives when the verifier passes it or gives it full marks. Survivors
point at behavior the verifier never checks; the mutation score is the share of
judged mutants killed. Some mutants behave exactly like the reference (a redundant
guard dropped, say); mark such a line ``# pragma: no mutate`` to leave it alone.

Nothing here executes a mutant: each runs like a model solution, through
``DeepGym.run_batch``. On a local backend every judge run is confined (see
:func:`~deepgym.judge.local_judge`): the whole run sits in a bwrap sandbox, so a
mutant stays contained even if the verifier runs it in-process, and the sandbox's
PID namespace takes down everything the run started. The price: sandboxes don't
nest, so inside the run a mutant isn't kept from the judge's files, which is fine
for edits of your own reference. macOS's sandbox-exec can't reap a process a mutant
detaches, so there (and without bwrap) the check is inconclusive and nothing runs.
Sandbox providers run each judge in a disposable sandbox; mutants run only on those
that keep the whole sandbox offline (docker, modal, e2b), since apple and daytona
leave the judge online.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import json
from collections.abc import Callable, Iterator

from pydantic import BaseModel, ConfigDict

from deepgym.core import DeepGym
from deepgym.judge import local_isolation_kind
from deepgym.models import Environment, RunResult
from deepgym.qa.checks import solved
from deepgym.qa.report import MUTANTS_CHECK, QACheck
from deepgym.sandboxes import PROVIDERS, sandbox_class
from deepgym.verifier_template import is_main_guard

MAX_MUTANTS = 100
"""Above this, an evenly spaced sample runs, so a long reference stays quick to check."""

_WORKERS = 4
"""Mutants run four at a time: every run starts a judge and solution processes."""

_BOUNDARY: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}
_SYMBOL = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>='}
_EMPTY = {
    'list': '[]',
    'sorted': '[]',
    'dict': '{}',
    'set': 'set()',
    'frozenset': 'frozenset()',
    'tuple': '()',
    'str': "''",
    'bytes': "b''",
    'int': '0',
    'len': '0',
    'sum': '0',
    'float': '0.0',
}
"""A constant of each kind a return value can have, by type or builtin name."""
_KEPT = (
    ast.Pass,
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
)
"""Statements never dropped: removing them only breaks names, which proves nothing."""
_PRAGMA = '# pragma: no mutate'
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

Edit = Callable[[ast.AST], None]


class Mutant(BaseModel):
    """One mutated copy of the reference."""

    model_config = ConfigDict(frozen=True)

    line: int
    description: str
    """Where and what changed, e.g. 'line 7: off-by-one 1 -> 2'."""
    source: str


def _short(node: ast.AST) -> str:
    return ast.unparse(node)


def _kind(node: ast.expr | None) -> str | None:
    """The kind of value an annotation or expression gives: 'list', 'int', 'bool', ..."""
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            key = name.lower()
        case ast.Subscript(value=base) | ast.Call(func=base):
            return _kind(base)
        case ast.List() | ast.ListComp():
            key = 'list'
        case ast.Dict() | ast.DictComp():
            key = 'dict'
        case ast.Set() | ast.SetComp():
            key = 'set'
        case ast.Tuple():
            key = 'tuple'
        case ast.JoinedStr():
            key = 'str'
        case ast.Compare() | ast.BoolOp() | ast.UnaryOp(op=ast.Not()):
            key = 'bool'
        case _:
            return None
    return key if key in _EMPTY or key == 'bool' else None


def _constants(value: ast.expr, annotation: ast.expr | None) -> list[str]:
    """Constants a return of *value* could be replaced by, guided by the return annotation."""
    kind = _kind(annotation) or _kind(value)
    if kind == 'bool':
        return ['True', 'False']
    return [_EMPTY[kind]] if kind else ['None']


def _returns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.Return]:
    """Return statements of *function* itself, not of functions nested in it."""
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            yield node
        if not isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _set(field: str, value: object) -> Edit:
    return lambda node: setattr(node, field, value)


def _swap(node: ast.AST) -> None:
    node.body, node.orelse = node.orelse, node.body  # type: ignore[attr-defined]


def _negate(node: ast.AST) -> None:
    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)  # type: ignore[attr-defined]


def _replace_op(index: int, op: type[ast.cmpop]) -> Edit:
    def edit(node: ast.AST) -> None:
        node.ops[index] = op()  # type: ignore[attr-defined]

    return edit


def _drop(field: str, index: int) -> Edit:
    def edit(node: ast.AST) -> None:
        body = getattr(node, field)
        del body[index]
        if not body:
            body.append(ast.Pass())

    return edit


def _edits(node: ast.AST, annotation: ast.expr | None) -> Iterator[tuple[int, str, Edit]]:
    """Every single-step mutation of *node*, as (line, what changed, edit for a copy of it)."""
    line = getattr(node, 'lineno', 0)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        for value in (node.value + 1, node.value - 1):
            yield line, f'off-by-one {node.value} -> {value}', _set('value', value)
    elif isinstance(node, ast.Constant) and type(node.value) is bool:
        yield line, f'constant {node.value} -> {not node.value}', _set('value', not node.value)
    elif isinstance(node, ast.Compare):
        for index, op in enumerate(node.ops):
            if type(op) in _BOUNDARY:
                swapped = _BOUNDARY[type(op)]
                what = f'{_SYMBOL[type(op)]} -> {_SYMBOL[swapped]}'
                yield line, what, _replace_op(index, swapped)
    elif isinstance(node, ast.Return) and node.value is not None:
        if not isinstance(node.value, ast.Constant):
            for constant in _constants(node.value, annotation):
                value = ast.parse(constant, mode='eval').body
                yield line, f'return {_short(node.value)} -> {constant}', _set('value', value)
    elif isinstance(node, (ast.If, ast.IfExp)) and not (
        isinstance(node, ast.If) and is_main_guard(node)
    ):
        if node.orelse:
            yield line, f'swapped the branches of `{_short(node.test)}`', _swap
        else:
            yield line, f'negated `if {_short(node.test)}`', _negate
    for field in ('body', 'orelse', 'finalbody'):
        body = getattr(node, field, None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            docstring = isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            )
            if isinstance(statement, _KEPT) or docstring or is_main_guard(statement):
                continue
            yield statement.lineno, f'dropped `{_short(statement)}`', _drop(field, index)


def mutants(reference: str) -> list[Mutant]:
    """Every distinct single-step mutant of *reference*, in source order.

    Mutants identical to the reference (after ``ast.unparse``) or to an earlier
    mutant are dropped, and so are edits on lines marked ``# pragma: no mutate``.
    A reference that isn't Python, or is JSON data, gives no mutants. Above
    :data:`MAX_MUTANTS`, an evenly spaced sample is returned.

    Example:
        >>> [m.description for m in mutants('def f(x):\\n    return x + 1\\n')]
        ['line 2: dropped `return x + 1`', 'line 2: return x + 1 -> None',
         'line 2: off-by-one 1 -> 2', 'line 2: off-by-one 1 -> 0']
    """
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return []
    with contextlib.suppress(ValueError):  # JSON data parses as Python but isn't code
        json.loads(reference)
        return []
    pragma_lines = {n for n, text in enumerate(reference.splitlines(), 1) if _PRAGMA in text}
    nodes = list(ast.walk(tree))
    annotations = {
        id(ret): function.returns
        for function in nodes
        if isinstance(function, _FUNCTIONS)
        for ret in _returns(function)
    }
    seen = {ast.unparse(tree)}
    found: list[Mutant] = []
    for index, node in enumerate(nodes):
        for line, what, edit in _edits(node, annotations.get(id(node))):
            if line in pragma_lines:
                continue
            clone = copy.deepcopy(tree)
            edit(list(ast.walk(clone))[index])
            source = ast.unparse(ast.fix_missing_locations(clone))
            if source not in seen:
                seen.add(source)
                found.append(Mutant(line=line, description=f'line {line}: {what}', source=source))
    found.sort(key=lambda mutant: mutant.line)
    return found[:: max(1, -(-len(found) // MAX_MUTANTS))]


def _survived(result: RunResult) -> bool:
    return result.passed or solved(result)


def mutation_check(env: Environment, reference: str, dg: DeepGym) -> QACheck:
    """Run every mutant of *reference*; fail if any passes or scores full marks.

    A control run of the reference as ``ast.unparse`` prints it (comments and
    formatting dropped, like every mutant) goes first: if the verifier fails that,
    it depends on formatting and the mutants can't be judged (inconclusive).

    On a local backend the runs go through a judge confined by bwrap (the whole run
    in its sandbox). Without bwrap (macOS included: sandbox-exec can't reap what a
    mutant detaches), or on a provider that leaves the judge online, the check is
    inconclusive and nothing runs.

    Returns:
        'pass' when every mutant is killed, 'fail' when any survives, 'inconclusive'
        when some got no verdict (or the control failed, or nothing could run
        isolated), 'skip' with no mutants.
    """
    found = mutants(reference)
    if not found:
        details = 'reference is not Python code or has nothing to mutate'
        return QACheck(name=MUTANTS_CHECK, status='skip', details=details)
    runner = dg
    if dg.backend == 'local':
        kind = local_isolation_kind()
        if kind != 'bwrap':
            why = (
                "macOS's sandbox-exec can't take down processes a mutant leaves running"
                if kind == 'macos'
                else 'there is no bwrap here'
            )
            details = (
                'local mutation runs need bwrap (Linux), whose PID namespace dies with the '
                f'run; {why}, so no mutant ran. Use --mode docker, modal or e2b'
            )
            return QACheck(name=MUTANTS_CHECK, status='inconclusive', details=details)
        runner = DeepGym(mode='local', confine_local_judge=True)
    elif dg.backend not in PROVIDERS or not sandbox_class(dg.backend).network_blocked:
        details = (
            f'{dg.backend} sandboxes keep the judge itself online, so a mutant the verifier '
            'runs in-process could reach the network; no mutant ran. Use local mode '
            '(a confined judge), docker, modal or e2b'
        )
        return QACheck(name=MUTANTS_CHECK, status='inconclusive', details=details)
    sources = [ast.unparse(ast.parse(reference)), *(mutant.source for mutant in found)]
    control, *results = runner.run_batch(env, sources, max_parallel=_WORKERS).results
    if not (control.has_verdict and solved(control)):
        details = (
            f'the reference re-printed without comments or formatting scores {control.score:.2f} '
            f'in a confined run ({control.output.strip()}); the verifier depends on '
            'formatting, or on something the run cannot reach (writes outside its temp '
            'directory, the network, new processes on macOS), so mutants of it cannot be '
            'judged'
        )
        return QACheck(name=MUTANTS_CHECK, status='inconclusive', details=details)
    survivors = [m for m, r in zip(found, results) if r.has_verdict and _survived(r)]
    unjudged = [m for m, r in zip(found, results) if not r.has_verdict]
    judged = len(found) - len(unjudged)
    killed = judged - len(survivors)
    parts = [
        f'mutation score {killed}/{judged}'
        + (f' ({killed / judged:.0%})' if judged else '')
        + f' of {len(found)} mutants'
    ]
    if survivors:
        listed = '; '.join(mutant.description for mutant in survivors[:8])
        more = f' (+{len(survivors) - 8} more)' if len(survivors) > 8 else ''
        parts.append(f'survived: {listed}{more}')
    if unjudged:
        parts.append(f'{len(unjudged)} got no verdict (verifier error)')
    status = 'fail' if survivors else 'inconclusive' if unjudged else 'pass'
    return QACheck(name=MUTANTS_CHECK, status=status, details='; '.join(parts))
