"""Every name DeepGym imports from an optional SDK still exists in the installed release.

The SDKs are imported lazily, inside the functions that use them, so a release that renames
or drops a name breaks only when that code runs. The names are read from the source, so the
list can't fall behind it. Imports under ``if TYPE_CHECKING:`` never run and are left out;
imports inside ``try`` are kept, since a fallback there hides the break instead of fixing
it. An SDK that isn't installed is skipped, unless ``DEEPGYM_REQUIRE_SDKS`` is set, as the
nightly newest-dependencies workflow does after installing them all.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src' / 'deepgym'
# The packages behind DeepGym's optional extras (pyproject.toml).
OPTIONAL = frozenset(
    {
        'anthropic',
        'daytona',
        'datasets',
        'e2b',
        'fastapi',
        'huggingface_hub',
        'lm_eval',
        'modal',
        'openai',
        'pyarrow',
        'uvicorn',
    }
)


def _type_only(tree: ast.Module) -> set[int]:
    """Ids of the nodes under ``if TYPE_CHECKING:``."""
    skipped: set[int] = set()
    for node in ast.walk(tree):
        test = getattr(node, 'test', None)
        name = getattr(test, 'id', None) or getattr(test, 'attr', None)
        if isinstance(node, ast.If) and name == 'TYPE_CHECKING':
            skipped |= {id(inner) for stmt in node.body for inner in ast.walk(stmt)}
    return skipped


def _imported() -> dict[str, set[str]]:
    """Module -> names DeepGym's source imports from it at run time, per optional SDK."""
    found: dict[str, set[str]] = {}
    for path in SRC.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        type_only = _type_only(tree)
        for node in ast.walk(tree):
            if id(node) in type_only:
                continue
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split('.')[0] in OPTIONAL:
                    names = {a.name for a in node.names if a.name != '*'}
                    found.setdefault(node.module, set()).update(names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in OPTIONAL:
                        found.setdefault(alias.name, set())
    return found


IMPORTED = _imported()


def _has(module: object, parent: str, name: str) -> bool:
    if hasattr(module, name):
        return True
    try:  # a submodule isn't an attribute until it is imported
        importlib.import_module(f'{parent}.{name}')
    except ImportError:
        return False
    return True


def test_the_scan_finds_the_sdk_imports() -> None:
    assert {'daytona', 'e2b', 'modal'} <= {module.split('.')[0] for module in IMPORTED}


@pytest.mark.parametrize('module', sorted(IMPORTED))
def test_the_installed_sdk_has_what_deepgym_imports(module: str) -> None:
    root = module.split('.')[0]
    if importlib.util.find_spec(root) is None:
        assert not os.environ.get('DEEPGYM_REQUIRE_SDKS'), f'{root} is not installed'
        pytest.skip(f'{root} is not installed')
    imported = importlib.import_module(module)

    missing = sorted(name for name in IMPORTED[module] if not _has(imported, module, name))

    assert not missing, f'{module} no longer has {missing}'
