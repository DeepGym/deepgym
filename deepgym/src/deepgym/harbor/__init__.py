"""Harbor tasks (https://harborframework.com) in DeepGym, and DeepGym environments as Harbor tasks.

A Harbor task is a folder: instruction.md, task.toml, tests/test.sh, an image or
environment/Dockerfile, and often solution/solve.sh. DeepGym runs one as a
:class:`HarborEnvironment` whose model output is a bash script. The script runs as an
unprivileged, offline user; the grade comes from tests/test.sh run in a fresh sandbox
that receives only a copy of the work directory, so the agent can't touch the tests or
the reward.

Modules:
    task: HarborTask, one validated task folder.
    loader: load_tasks from a folder, a git repository or a Hugging Face dataset.
"""

from deepgym.harbor.loader import dataset_root, load_tasks, task_dirs
from deepgym.harbor.task import HarborTask

__all__ = [
    'HarborTask',
    'dataset_root',
    'load_tasks',
    'task_dirs',
]
