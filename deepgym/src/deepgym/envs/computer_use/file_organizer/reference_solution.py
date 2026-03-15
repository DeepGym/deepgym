"""Create a project directory structure with README files."""

import os

PROJECT_DIR = os.environ.get('PROJECT_DIR', '/tmp/project')

DIRS = {
    'src': '# Source Code\nMain application source files.',
    'tests': '# Tests\nUnit and integration tests.',
    'docs': '# Documentation\nProject documentation and guides.',
}


def main():
    for dirname, readme_content in DIRS.items():
        dirpath = os.path.join(PROJECT_DIR, dirname)
        os.makedirs(dirpath, exist_ok=True)
        readme_path = os.path.join(dirpath, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content + '\n')

    print('done')


if __name__ == '__main__':
    main()
