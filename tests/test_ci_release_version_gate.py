# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "release_version_gate.py"
SPEC = importlib.util.spec_from_file_location("release_version_gate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_version_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_version_gate
SPEC.loader.exec_module(release_version_gate)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _write_pyproject(repo: Path, version: str) -> None:
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "gosh-memory"',
                f'version = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _repo_with_version(tmp_path: Path, version: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    _write_pyproject(repo, version)
    _commit(repo, f"release {version}")
    return repo


def _config(repo: Path, *, ref_name: str = "main", before_sha: str = ""):
    return release_version_gate.GateConfig(
        root=repo,
        ref_name=ref_name,
        before_sha=before_sha,
        should_run="true",
        repository="Futurizt/gosh.memory",
        token="",
        require_release_assets=False,
        release_timeout_seconds=0,
        release_interval_seconds=1,
    )


def test_release_version_gate_requires_version_bump_on_main(tmp_path: Path) -> None:
    repo = _repo_with_version(tmp_path, "0.3.5")
    before = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v0.3.5")
    _write_pyproject(repo, "0.3.5")
    (repo / "README.md").write_text("unreleased change\n", encoding="utf-8")
    _commit(repo, "main promotion without version bump")

    with pytest.raises(release_version_gate.GateError, match="must be bumped"):
        release_version_gate.run_gate(_config(repo, before_sha=before))


def test_release_version_gate_accepts_bumped_tagged_version(tmp_path: Path) -> None:
    repo = _repo_with_version(tmp_path, "0.3.5")
    before = _git(repo, "rev-parse", "HEAD")
    _write_pyproject(repo, "0.3.6")
    head = _commit(repo, "release 0.3.6")
    _git(repo, "tag", "v0.3.6", head)

    release_version_gate.run_gate(_config(repo, before_sha=before))


def test_release_version_gate_skips_dev_pushes(tmp_path: Path) -> None:
    repo = _repo_with_version(tmp_path, "0.3.6")

    release_version_gate.run_gate(_config(repo, ref_name="dev"))
