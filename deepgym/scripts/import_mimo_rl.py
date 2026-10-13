#!/usr/bin/env python3
"""Import the MiMo-V2.6-RL code tasks (XiaomiMiMo/MiMo-V2.6-RL-oss) into DeepGym format.

The code subset has 2,698 repo-level tasks. Each ships its own Docker image with the
repo checked out at ``cwd``, a hidden ``test_patch`` and a ``test_command``; the
reward is 1.0 if the test command exits 0 after the model's change. Each task
becomes a :class:`~deepgym.benchmark_envs.RepoPatchEnvironment` directory under
``~/.deepgym/environments/mimo_code/<instance_id>/``:

  task.md          the issue, where the repo is, and the answer format (a unified diff)
  test_patch.diff  the hidden tests, applied before the model's diff
  metadata.json    image (on Docker Hub), repo_dir, test_command, timeout

The other subsets are skipped because their rewards can't be reproduced honestly here:
cyber (ARVO) grades a proof-of-concept input with a submit script inside each image,
general scores weighted rubrics with LLM judges against mock-app sidecars, webdev uses
a visual LLM grader, and music scores ABC notation with MiMo's statistical style
scorer (abc2midi plus distance to a human baseline), not a check of the prompt.

The dataset is Apache-2.0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

DATASET_ID = 'XiaomiMiMo/MiMo-V2.6-RL-oss'
BENCHMARK = 'mimo_code'

_TASK_TEMPLATE = """{problem_statement}

---

The repository is checked out at `{repo_dir}`. Reply with a unified diff against it
(as `git diff` prints it, paths relative to the repository root) that resolves the
issue above. Hidden tests are applied before your diff and decide the score; your
edits to the files they touch are dropped.
"""


def load_image_map(path: Path) -> dict[str, str]:
    """Map each dataset image name to its Docker Hub image from image-mapping.jsonl."""
    rows = (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
    return {row['dataset_image']: row['dockerhub_image'] for row in rows}


def build_code_env(row: Mapping, images: Mapping[str, str], output_dir: Path) -> dict:
    """Write one code task's environment directory and return its registry entry.

    Args:
        row: A row of the dataset's code config (verl format).
        images: Dataset image name to Docker Hub image, from :func:`load_image_map`.
        output_dir: The benchmark folder the environment directory goes in.

    Returns:
        The task's metadata plus its registry ``path``.

    Raises:
        KeyError: If the row's image has no Docker Hub mapping.
    """
    instance = json.loads(row['extra_info']['instance_json'])
    instance_id = instance['instance_id']
    metadata = {
        'id': instance_id,
        'name': instance_id,
        'benchmark': BENCHMARK,
        'difficulty': 'hard',
        'domain': 'software-engineering',
        'tags': ['mimo', 'swe', 'patch'],
        'image': images[instance['docker_image']],
        'repo_dir': instance['cwd'],
        'test_command': instance['test_command'],
        'timeout': int(instance['verifier_timeout_sec']),
        'source': DATASET_ID,
        'license': 'Apache-2.0',
    }
    env_dir = output_dir / instance_id
    env_dir.mkdir(parents=True, exist_ok=True)
    task = _TASK_TEMPLATE.format(
        problem_statement=instance['problem_statement'].strip(), repo_dir=instance['cwd']
    )
    (env_dir / 'task.md').write_text(task, encoding='utf-8')
    # Bytes, not text mode: patches to CRLF files must keep their \r characters.
    (env_dir / 'test_patch.diff').write_bytes(instance['test_patch'].encode('utf-8'))
    (env_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return {**metadata, 'path': f'environments/{BENCHMARK}/{instance_id}'}


def main() -> None:
    """Download the code subset and write every task's environment plus registry.json."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('datasets is required to run the importer: pip install datasets')
        raise SystemExit(1) from None

    output_dir = Path.home() / '.deepgym' / 'environments' / BENCHMARK
    print(f'Loading {DATASET_ID} (code)...')
    rows = load_dataset(DATASET_ID, 'code', split='train')
    images = load_image_map(
        Path(hf_hub_download(DATASET_ID, 'image-mapping.jsonl', repo_type='dataset'))
    )
    registry = [build_code_env(row, images, output_dir) for row in rows]
    (output_dir / 'registry.json').write_text(
        json.dumps(registry, indent=2) + '\n', encoding='utf-8'
    )
    print(f'Generated {len(registry)} environments in {output_dir}')
    print('Each needs its own image, e.g. docker pull ' + registry[0]['image'])


if __name__ == '__main__':
    main()
