# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "scripts" / "mirror" / "mirror.sh"


SAMPLE_LICENSE = """Sample Project License

Copyright (c) 2026 GOSH.AI

Fixture-only license text used to verify release staging keeps the root
LICENSE file. It intentionally avoids project-license keywords because the
same tests are included in the noncommercial startrek release stage.
"""


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_rust_repo(path: Path, name: str) -> None:
    _write(path / "Cargo.toml", f'[package]\nname = "{name}"\nversion = "0.0.1"\nedition = "2021"\n')
    _write(path / "Cargo.lock", "")
    _write(path / "README.md", f"# {name}\n")
    _write(path / "LICENSE", SAMPLE_LICENSE)
    _write(path / "src" / "main.rs", "fn main() {}\n")


def _run_mirror(repo_name: str, source: Path, stage: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PUSH_SOURCE": "0",
        "PUSH_PUBLIC": "0",
        "REQUIRE_PUBLIC_SHAPE": "1",
        "MIRROR_GITHUB_TOKEN": "test-token",
        "MIRROR_GITHUB_ORG": "gosh-dot-ai",
        "STAGE_OUTPUT_DIR": str(stage),
    }
    return subprocess.run(
        [str(MIRROR), repo_name, str(source)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_cli_public_stage_requires_docs_license_and_tooling(tmp_path: Path) -> None:
    source = tmp_path / "gosh.cli"
    stage = tmp_path / "publish-stage-cli-missing"
    _base_rust_repo(source, "gosh-cli")
    _write(source / "clippy.toml")
    _write(source / "install.ps1")
    _write(source / "install.sh")
    _write(source / "taplo.toml")

    result = _run_mirror("gosh.cli", source, stage)

    assert result.returncode != 0
    assert "missing required path: docs" in result.stdout


def test_agent_public_stage_requires_docs_license_and_tooling(tmp_path: Path) -> None:
    source = tmp_path / "gosh.agent"
    stage = tmp_path / "publish-stage-agent-missing"
    _base_rust_repo(source, "gosh-agent")
    _write(source / "clippy.toml")
    _write(source / "rustfmt.toml")
    _write(source / "taplo.toml")

    result = _run_mirror("gosh.agent", source, stage)

    assert result.returncode != 0
    assert "missing required path: docs" in result.stdout


def test_cli_public_stage_keeps_required_files_and_excludes_specs(tmp_path: Path) -> None:
    source = tmp_path / "gosh.cli"
    stage = tmp_path / "publish-stage-cli"
    _base_rust_repo(source, "gosh-cli")
    _write(source / "docs" / "cli.md", "# CLI docs\n")
    _write(source / "clippy.toml")
    _write(source / "install.ps1")
    _write(source / "install.sh")
    _write(source / "taplo.toml")
    _write(source / "specs" / "private.md", "private\n")

    result = _run_mirror("gosh.cli", source, stage)

    assert result.returncode == 0, result.stdout
    for rel in ("docs", "LICENSE", "clippy.toml", "install.ps1", "install.sh", "taplo.toml"):
        assert (stage / rel).exists()
    assert not (stage / "specs").exists()
