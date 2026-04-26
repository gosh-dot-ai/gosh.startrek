# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ci" / "mirror" / "materialize_internal_ci.py"
PUBLIC_HARNESS_PATH = PROJECT_ROOT / "ci" / "mirror" / "run_public_harness.sh"
MIRROR_SCRIPT_PATH = PROJECT_ROOT / "ci" / "mirror" / "mirror.sh"
MIRROR_CONFIG_PATH = PROJECT_ROOT / "ci" / "mirror" / "mirror_config.json"
STARTREK_RELEASE_WORKFLOW_PATH = PROJECT_ROOT / "ci" / "mirror" / "startrek_release.yml"


def test_materialize_internal_ci_uses_current_ci_layout(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    stage_root.mkdir()

    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(PROJECT_ROOT), str(stage_root)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (stage_root / "ci" / "scripts" / "bootstrap_test_auth.py").exists()
    assert (stage_root / "ci" / "mirror" / "mirror.sh").exists()

    rendered = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((stage_root / ".github" / "workflows").glob("*.yml"))
    )
    assert "scripts/ci/" not in rendered
    assert "scripts/mirror/" not in rendered
    assert "ci/scripts/" in rendered
    assert "ci/mirror/" in rendered


def test_public_harness_uses_current_packaging_and_e2e_paths() -> None:
    script = PUBLIC_HARNESS_PATH.read_text(encoding="utf-8")

    assert "requirements.txt" not in script
    assert "test_live_runtime_e2e.py" not in script
    assert "test_mal_ama_live.py" not in script
    assert "pytest" not in script
    assert "cargo build" not in script
    assert 'python3 -m pip install -e "$MEMORY_REPO"' in script
    assert 'python3 -m compileall -q "$MEMORY_REPO/src"' in script
    assert 'importlib.import_module("src.cli")' in script
    assert 'importlib.import_module("src.mcp_server")' in script
    assert 'manifest = repo / "Cargo.toml"' in script
    assert "public harness packaging smoke passed" in script


def test_startrek_release_stage_excludes_memory_docs_and_preserves_public_readme() -> None:
    config = json.loads(MIRROR_CONFIG_PATH.read_text(encoding="utf-8"))
    startrek = config["repos"]["gosh.startrek"]
    cli = config["repos"]["gosh.cli"]
    agent = config["repos"]["gosh.agent"]
    script = MIRROR_SCRIPT_PATH.read_text(encoding="utf-8")
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "mirror.yml").read_text(encoding="utf-8")
    rc_gate = (PROJECT_ROOT / ".github" / "workflows" / "rc-gate.yml").read_text(encoding="utf-8")
    startrek_release = STARTREK_RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "rustfmt.toml" in cli["include"]
    assert "build.rs" in agent["include"]
    assert "STARTREK_RELEASE_WORKFLOW" in script
    assert 'install_startrek_public_workflows "$STAGE_DIR"' in script
    assert 'install_startrek_public_workflows "$PUBLIC_DIR"' in script
    assert 'tags: ["v*"]' in startrek_release
    assert "python -m build --sdist --wheel" in startrek_release
    assert "softprops/action-gh-release@v2" in startrek_release
    assert r"ci/mirror/startrek_release\.yml" in rc_gate
    assert "docs/" not in startrek["include"]
    assert "docs/" in startrek["exclude"]
    assert 'SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"' in script
    assert 'rm -rf "$STAGE_DIR/docs"' in script
    assert "startrek stage must not include gosh.memory docs" in script
    assert "STARTREK_README_BACKUP" in script
    assert 'find "$STARTREK_README_BACKUP" -maxdepth 1 -type f -iname "README*"' in script
    assert "No startrek changes to publish" in workflow
    assert "latest release already points at source" in workflow
    assert "No docs changes to publish" in workflow
