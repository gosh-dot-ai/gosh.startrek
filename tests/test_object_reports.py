# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import pytest

from src.memory import _coerce_runtime_report
from src.object_flags import build_object_flag
from src.object_reports import build_report, build_report_entry, normalize_legacy_report_fields


def _bad_flag() -> dict:
    flag = build_object_flag(
        producer="test_producer",
        category="runtime",
        severity="low",
        message="bad flag",
        status="open",
    )
    flag["severity"] = "BAD"
    return flag


def test_build_report_rejects_invalid_report_flag():
    with pytest.raises(ValueError, match="severity is invalid"):
        build_report(
            report_kind="validation",
            producer="test_reporter",
            status="ok",
            entries=[],
            flags=[_bad_flag()],
        )


def test_build_report_entry_rejects_invalid_status():
    with pytest.raises(ValueError, match="unsupported report entry status"):
        build_report_entry(
            target_path="facts[0]",
            status="banana",
        )


def test_coerce_runtime_report_rejects_invalid_report_flag():
    with pytest.raises(ValueError, match="invalid extraction report"):
        _coerce_runtime_report(
            {
                "report_id": "report_bad",
                "report_kind": "extraction",
                "producer": "block_extractor",
                "status": "ok",
                "entries": [],
                "summary": None,
                "flags": [_bad_flag()],
            },
            producer="block_extractor",
            report_kind="extraction",
        )


def test_normalize_legacy_report_fields_rejects_invalid_nested_report_flag():
    with pytest.raises(ValueError, match="severity is invalid"):
        normalize_legacy_report_fields(
            {
                "extraction_report": {
                    "report_id": "report_bad",
                    "report_kind": "extraction",
                    "producer": "block_extractor",
                    "status": "ok",
                    "entries": [],
                    "summary": None,
                    "flags": [_bad_flag()],
                }
            },
            producer="runtime",
            report_kind="extraction",
        )


def test_normalize_legacy_report_fields_accepts_legacy_entry_level_flags():
    legacy_flag = {
        "flag_id": "flag_legacy_entry",
        "origin": "extraction",
        "producer": "block_extractor",
        "normalized_code": "shape.list_item_type_error",
        "raw_code": "invalid_fact_item_type",
        "severity": "high",
        "status": "dropped",
        "message": "legacy entry flag",
        "path": "facts[1]",
        "object_id": "f_legacy",
        "resolution": "dropped_after_failed_repair",
        "repair_attempted": True,
    }
    report = normalize_legacy_report_fields(
        {
            "producer": "block_extractor",
            "diagnostics": [{
                "target_path": "facts[1]",
                "status": "dropped",
                "repair_attempted": True,
                "issue": {
                    "normalized_code": "shape.list_item_type_error",
                    "raw_code": "invalid_fact_item_type",
                    "message": "legacy issue",
                },
                "flags": [legacy_flag],
            }],
        },
        producer="block_extractor",
        report_kind="extraction",
    )

    assert report["entries"][0]["flags"][0]["category"] == "extraction"
    assert report["entries"][0]["flags"][0]["code"] == "shape.list_item_type_error"
    assert report["entries"][0]["flags"][0]["details"]["raw_code"] == "invalid_fact_item_type"
