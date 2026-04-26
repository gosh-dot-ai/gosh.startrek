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
SCRIPT_PATH = PROJECT_ROOT / "ci" / "mirror" / "materialize_internal_ci.py"


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
