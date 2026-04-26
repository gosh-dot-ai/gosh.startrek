# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests._archive_repo import archive_repo_root


def _load_security_scan_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "security_scan.py"
    spec = importlib.util.spec_from_file_location("security_scan", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


security_scan = _load_security_scan_module()


def test_find_secret_assignment_detects_env_style_assignment():
    assert security_scan.find_secret_assignment('API_TOKEN="supersecretvalue"') == (
        "API_TOKEN",
        "supersecretvalue",
    )


def test_scan_text_skips_lowercase_dataset_tokens():
    path = archive_repo_root() / "validation" / "example.json"
    findings = security_scan.scan_text(
        path,
        '{"access_token": "human transcript token field"}',
        {},
    )
    assert findings == []


def test_scan_text_handles_large_single_line_dataset_payload():
    path = archive_repo_root() / "validation" / "huge.json"
    payload = '{"transcript":"' + ("word " * 20000) + '", "note":"no secrets here"}'
    findings = security_scan.scan_text(path, payload, {})
    assert findings == []
