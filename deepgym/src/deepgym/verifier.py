"""Verifier utilities — reusable verifier definitions with protocol validation."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class Verifier(BaseModel):
    """A reusable verifier definition.

    A verifier is a piece of Python code that scores a candidate solution.
    It receives a solution file path (and optionally test cases) and returns
    a score, a bool, or a dict with ``{"score", "passed", "details"}`` keys.

    The DeepGym wrapper template (see :mod:`deepgym.verifier_template`)
    normalises any of those return types into the standard JSON output protocol.
    """

    name: str = Field(..., description='Human-readable verifier name.')
    code: str = Field(..., description='Python source code that implements verification logic.')
    description: str = Field(
        default='', description='Optional prose description of what the verifier checks.'
    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_file(self, path: str) -> None:
        """Write verifier code to *path*.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.code, encoding='utf-8')

    @classmethod
    def from_file(cls, path: str, name: str = '') -> Verifier:
        """Load a verifier from a Python file.

        Parameters
        ----------
        path:
            Path to a ``.py`` file containing verifier code.
        name:
            Optional name.  Defaults to the file stem (e.g. ``"check_sort"``
            for ``check_sort.py``).

        Returns
        -------
        Verifier
            A new :class:`Verifier` instance whose :attr:`code` is the file
            contents.
        """
        p = Path(path)
        code = p.read_text(encoding='utf-8')
        resolved_name = name or p.stem
        return cls(name=resolved_name, code=code)

    # ------------------------------------------------------------------
    # Protocol validation
    # ------------------------------------------------------------------

    def validate_protocol(self) -> bool:
        """Check that the verifier code follows the JSON output protocol.

        This performs lightweight **static** checks — it does not execute the
        code.  The checks are intentionally lenient: we only verify that the
        code is *likely* to produce structured output by looking for key
        indicators.

        Returns
        -------
        bool
            ``True`` if the code appears to follow the protocol, ``False``
            otherwise.
        """
        # 1. Must import (or at least reference) json somewhere.
        has_json = 'import json' in self.code or 'json.dumps' in self.code

        # 2. Must have a callable entry point — either a ``def verify`` / ``def _run_verifier``
        #    or an ``if __name__`` guard.
        has_entry = bool(
            re.search(r'def\s+(verify|_run_verifier)\s*\(', self.code)
            or re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', self.code)
        )

        return has_json and has_entry
