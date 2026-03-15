"""Initialize a git repo, create a branch, add a file, and commit."""

import os
import shutil
import subprocess

REPO_DIR = '/tmp/my_repo'


def run(cmd, **kwargs):
    """Run a shell command in the repo directory."""
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
        **kwargs,
    )


def main():
    # Clean up and create fresh directory.
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    os.makedirs(REPO_DIR)

    # Initialize git repo.
    run('git init')
    run('git config user.email "test@deepgym.dev"')
    run('git config user.name "DeepGym Test"')

    # Create and switch to feature branch.
    run('git checkout -b feature')

    # Create the file.
    hello_path = os.path.join(REPO_DIR, 'hello.txt')
    with open(hello_path, 'w') as f:
        f.write('Hello, DeepGym!')

    # Stage and commit.
    run('git add hello.txt')
    run('git commit -m "Add hello.txt"')

    print('done')


if __name__ == '__main__':
    main()
