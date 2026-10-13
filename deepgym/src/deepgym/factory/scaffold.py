"""``deepgym new``: render an environment folder from the templates shipped with deepgym."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Literal, get_args

Kind = Literal['function', 'script', 'repo-patch', 'multi-turn']
KINDS: tuple[Kind, ...] = get_args(Kind)

_TEMPLATES = Path(__file__).with_name('templates')
_NAME = re.compile(r'[a-z][a-z0-9_]*')


def scaffold(name: str, kind: Kind, parent: Path) -> Path:
    """Create the environment folder ``parent / name`` for *kind* and return it.

    Every ``*.tmpl`` file under templates/common/ and templates/<kind>/ is rendered
    with :class:`string.Template` (``$name``, ``$title``) to the same relative path
    without the suffix. Everything renders before anything is written, so a failure
    leaves no half-made folder.

    Args:
        name: Folder and environment name, lowercase snake_case.
        kind: function (load_solution), script (run_solution on stdin/stdout),
            repo-patch (a diff applied to fixture files) or multi-turn (an agent
            episode with a final verifier).
        parent: Directory to create the folder in.

    Returns:
        The new folder.

    Raises:
        ValueError: If *name* isn't lowercase snake_case, *kind* is unknown, or the
            folder already exists.
    """
    if not _NAME.fullmatch(name):
        raise ValueError(f'name {name!r} must be lowercase snake_case, e.g. word_count')
    if kind not in KINDS:
        raise ValueError(f'kind {kind!r} must be one of {", ".join(KINDS)}')
    values = {'name': name, 'title': name.replace('_', ' ').title()}
    rendered = {
        path.relative_to(root).with_suffix(''): Template(
            path.read_text(encoding='utf-8')
        ).substitute(values)
        for root in (_TEMPLATES / 'common', _TEMPLATES / kind)
        for path in sorted(root.rglob('*.tmpl'))
    }
    target = parent / name
    try:
        target.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError(f'{target} already exists') from exc
    for rel, text in rendered.items():
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        (target / rel).write_text(text, encoding='utf-8')
    return target
