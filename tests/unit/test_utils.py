"""Unit tests for src.utils."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.utils import find_project_root


def test_find_project_root_from_git_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / ".git").mkdir()
        child = root / "a" / "b"
        child.mkdir(parents=True)
        assert find_project_root(child) == root


def test_find_project_root_from_pyproject():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "pyproject.toml").touch()
        child = root / "src"
        child.mkdir()
        assert find_project_root(child) == root


def test_find_project_root_default_start():
    result = find_project_root()
    assert result.exists()
