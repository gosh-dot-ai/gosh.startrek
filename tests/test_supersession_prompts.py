# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path

PROMPTS_DIR = Path("src/prompts/inference")
CONFLICT_BLOCK = "CONFLICT RESOLUTION"
EXEMPT = {"icl.md", "list_set.md"}  # short leaf prompts don't carry conflict blocks


def test_all_prompts_have_conflict_resolution():
    """Every inference prompt (except ICL) must contain CONFLICT RESOLUTION."""
    for md in sorted(PROMPTS_DIR.glob("*.md")):
        if md.name in EXEMPT:
            continue
        text = md.read_text()
        assert CONFLICT_BLOCK in text, f"{md.name} missing CONFLICT RESOLUTION block"


def test_icl_exempt():
    """ICL prompt should NOT have CONFLICT RESOLUTION."""
    icl = PROMPTS_DIR / "icl.md"
    if icl.exists():
        icl.read_text()
