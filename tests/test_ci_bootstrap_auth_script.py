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
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ci" / "bootstrap_test_auth.py"


def test_bootstrap_script_does_not_print_tokens_when_writing_github_env(tmp_path: Path) -> None:
    env_file = tmp_path / "github.env"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--github-env", str(env_file)],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    )

    assert proc.stdout == ""
    contents = env_file.read_text(encoding="utf-8")
    assert "GOSH_TEST_BOOTSTRAP_ADMIN_TOKEN=" in contents
    assert "MEMORY_SERVER_TOKEN=" in contents


def test_bootstrap_script_prints_shell_exports_only_in_shell_mode(tmp_path: Path) -> None:
    env_file = tmp_path / "github.env"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--github-env", str(env_file), "--shell"],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
    )

    assert "export GOSH_TEST_BOOTSTRAP_ADMIN_TOKEN=" in proc.stdout
    assert "export MEMORY_SERVER_TOKEN=" in proc.stdout
