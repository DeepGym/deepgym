"""Tests for artifact-backed CyberGym patch environments."""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.cybergym_artifacts import CyberGymArtifacts, CyberGymPatchEnvironment, extract_patch

_ROOT = Path(__file__).resolve().parent.parent
_HEADER = 'diff --git a/hello.txt b/hello.txt\n--- a/hello.txt\n+++ b/hello.txt\n'
_REFERENCE = _HEADER + '@@ -1,2 +1,2 @@\n-bad\n+good\n keep\n'


def _write_tar(source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(source_dir / 'src-vul', arcname='src-vul')


def _toy_env(tmp_path: Path) -> CyberGymPatchEnvironment:
    """hello.txt holds 'bad\\nkeep\\n'; the reference patch turns 'bad' into 'good'."""
    repo = tmp_path / 'repo' / 'src-vul' / 'toy'
    repo.mkdir(parents=True)
    (repo / 'hello.txt').write_text('bad\nkeep\n', encoding='utf-8')
    archive = tmp_path / 'repo-vul.tar.gz'
    _write_tar(tmp_path / 'repo', archive)
    patch = tmp_path / 'patch.diff'
    patch.write_text(_REFERENCE, encoding='utf-8')
    artifacts = CyberGymArtifacts(task_id='toy:3', repo_vul=archive, patch=patch)
    return CyberGymPatchEnvironment.from_row({'task_id': 'toy:3'}, artifacts=artifacts, timeout=30)


def test_independent_fix_gets_full_credit(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    result = sandbox_dg.run(_toy_env(tmp_path), _HEADER + '@@ -1 +1 @@\n-bad\n+good\n')

    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.parametrize(
    'patch',
    [
        _HEADER + '-bad\n+good\n keep\n',  # the reference without its hunk header
        _HEADER + '-bad\n+good\n@@ -2 +2 @@\n-keep\n+kept\n',  # its lines outside any hunk
        _HEADER + '@@ -1,2 +1,2 @@\n bad\n keep\n',  # a hunk that changes nothing
    ],
)
def test_unapplied_lines_earn_nothing(tmp_path: Path, sandbox_dg: DeepGym, patch: str) -> None:
    """Patch text the repository never received used to score up to 1.0."""
    result = sandbox_dg.run(_toy_env(tmp_path), patch)

    assert result.score == 0.0
    assert result.passed is False
    assert result.error_type == 'invalid_patch'


@pytest.mark.parametrize('escape', ['absolute', 'dotdot'])
def test_patch_cannot_write_outside_the_repository(
    tmp_path: Path, sandbox_dg: DeepGym, escape: str
) -> None:
    victim = tmp_path / 'victim.txt'
    victim.write_text('bad\n', encoding='utf-8')
    target = str(victim) if escape == 'absolute' else '../' * 40 + str(victim).lstrip('/')
    patch = f'diff --git a/hello.txt b/hello.txt\n--- {target}\n+++ {target}\n'
    patch += '@@ -1 +1 @@\n-bad\n+pwned\n'

    result = sandbox_dg.run(_toy_env(tmp_path), patch)

    assert victim.read_text(encoding='utf-8') == 'bad\n'
    assert result.score == 0.0
    assert result.error_type == 'patch_apply_failed'


def test_extract_patch_preserves_trailing_context_space() -> None:
    patch = 'diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-a\n+b\n \n'

    assert extract_patch(patch) == patch


def test_cybergym_patch_environment_scores_reference_patch(
    tmp_path: Path, sandbox_dg: DeepGym
) -> None:
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

    result = sandbox_dg.run(env, patch.read_text(encoding='utf-8'))

    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics['root'] == 'src-vul/toy'


def test_cybergym_patch_environment_rejects_non_patch(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    repo = tmp_path / 'repo' / 'src-vul' / 'toy'
    repo.mkdir(parents=True)
    (repo / 'hello.txt').write_text('bad\n', encoding='utf-8')
    archive = tmp_path / 'repo-vul.tar.gz'
    _write_tar(tmp_path / 'repo', archive)
    artifacts = CyberGymArtifacts(task_id='toy:2', repo_vul=archive)
    env = CyberGymPatchEnvironment.from_row({'task_id': 'toy:2'}, artifacts=artifacts, timeout=30)

    result = sandbox_dg.run(env, 'not a patch')

    assert result.passed is False
    assert result.error_type == 'invalid_patch'


def test_documented_eval_modes_are_accepted() -> None:
    """docs/cyberbench.md once showed a --mode the eval script rejects."""
    docs = (_ROOT / 'docs' / 'cyberbench.md').read_text(encoding='utf-8')
    modes = re.findall(r'run_cybergym_artifact_eval\.py[^\n]*--mode (\S+)', docs)
    script = _ROOT / 'scripts' / 'run_cybergym_artifact_eval.py'

    assert modes
    for mode in modes:
        cmd = [sys.executable, str(script), '--mode', mode, '--help']
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f'--mode {mode}: {proc.stderr[-300:]}'
