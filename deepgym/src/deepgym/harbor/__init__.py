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
    agent: the agent phase in its own sandbox.
    grading: grading in a fresh sandbox, and reading the reward.
    environment: HarborEnvironment, which ties them together.
    export: export_environment, a DeepGym environment as a Harbor task folder.
"""

from deepgym.harbor.environment import HarborEnvironment
from deepgym.harbor.export import export_environment
from deepgym.harbor.loader import dataset_root, load_tasks, task_dirs
from deepgym.harbor.task import HarborTask

__all__ = [
    'HarborEnvironment',
    'HarborTask',
    'dataset_root',
    'export_environment',
    'load_tasks',
    'task_dirs',
]
