"""Tests for artifact-backed CyberGym patch environments."""

from __future__ import annotations

import tarfile
from pathlib import Path

from deepgym import DeepGym
from deepgym.cybergym_artifacts import CyberGymArtifacts, CyberGymPatchEnvironment, extract_patch


def _write_tar(source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_dir / 'src-vul', arcname='src-vul')


def test_extract_patch_preserves_trailing_context_space() -> None:
    patch = 'diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-a\n+b\n \n'

    assert extract_patch(patch) == patch


def test_cybergym_patch_environment_scores_reference_patch(tmp_path: Path) -> None:
    repo = tmp_path / 'repo' / 'src-vul' / 'toy'
    repo.mkdir(parents=True)
    (repo / 'hello.txt').write_text('bad\nkeep\n', encoding='utf-8')
    archive = tmp_path / 'repo-vul.tar.gz'
    _write_tar(tmp_path / 'repo', archive)
    patch = tmp_path / 'patch.diff'
    patch.write_text(
        'diff --git a/hello.txt b/hello.txt\n'
        '--- a/hello.txt\n'
        '+++ b/hello.txt\n'
        '@@ -1,2 +1,2 @@\n'
        '-bad\n'
        '+good\n'
        ' keep\n',
        encoding='utf-8',
    )
    artifacts = CyberGymArtifacts(task_id='toy:1', repo_vul=archive, patch=patch)
    row = {
        'task_id': 'toy:1',
        'project_name': 'toy',
        'project_language': 'c',
        'vulnerability_description': 'A parser bug requires a patch.',
    }
    env = CyberGymPatchEnvironment.from_row(row, artifacts=artifacts, timeout=30)

    result = DeepGym(mode='local').run(env, patch.read_text(encoding='utf-8'))

    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics['root'] == 'src-vul/toy'


def test_cybergym_patch_environment_rejects_non_patch(tmp_path: Path) -> None:
    repo = tmp_path / 'repo' / 'src-vul' / 'toy'
    repo.mkdir(parents=True)
    (repo / 'hello.txt').write_text('bad\n', encoding='utf-8')
    archive = tmp_path / 'repo-vul.tar.gz'
    _write_tar(tmp_path / 'repo', archive)
    artifacts = CyberGymArtifacts(task_id='toy:2', repo_vul=archive)
    env = CyberGymPatchEnvironment.from_row({'task_id': 'toy:2'}, artifacts=artifacts, timeout=30)

    result = DeepGym(mode='local').run(env, 'not a patch')

    assert result.passed is False
    assert result.error_type == 'invalid_patch'
