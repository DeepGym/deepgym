"""Runs with _checks.py (load_state, report) and PARAMS prepended."""

import os
from pathlib import Path

CONTENTS = {
    'report.pdf': 'Q3 revenue summary\n',
    'notes.txt': 'milk, eggs, bread\n',
    'photo.jpg': 'not really a photo\n',
}

docs = Path(os.environ['DEEPGYM_AGENT_HOME']) / 'Documents'
folder = docs / PARAMS['folder']
move, old, new, keep = PARAMS['move'], PARAMS['rename_from'], PARAMS['rename_to'], PARAMS['keep']


def has(path: Path, original: str) -> bool:
    return path.is_file() and path.read_text() == CONTENTS[original]


report(
    progress={
        'folder_created': folder.is_dir(),
        'file_moved': has(folder / move, move) and not (docs / move).exists(),
        'file_renamed': has(docs / new, old) and not (docs / old).exists(),
    },
    guards={
        'no_files_lost': (has(docs / move, move) or has(folder / move, move))
        and (has(docs / old, old) or has(docs / new, old)),
        'other_file_kept': has(docs / keep, keep),
        'nothing_else_in_folder': not folder.is_dir()
        or {p.name for p in folder.iterdir()} <= {move},
    },
)
