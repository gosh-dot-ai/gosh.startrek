# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mirror_config() -> dict:
    config_path = PROJECT_ROOT / "ci" / "mirror" / "mirror_config.json"
    return json.loads(config_path.read_text())


def test_cli_public_mirror_preserves_docs_installers_and_tool_configs():
    include = set(_mirror_config()["repos"]["gosh.cli"]["include"])

    assert {
        "docs/",
        "clippy.toml",
        "install.ps1",
        "install.sh",
        "taplo.toml",
    }.issubset(include)


def test_agent_public_mirror_preserves_docs_and_tool_configs():
    include = set(_mirror_config()["repos"]["gosh.agent"]["include"])

    assert {
        "docs/",
        "clippy.toml",
        "rustfmt.toml",
        "taplo.toml",
    }.issubset(include)
