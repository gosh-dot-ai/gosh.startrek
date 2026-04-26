# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from src.object_flags import build_object_flag, validate_object_flags


def test_valid_minimal_flag_accepted():
    flag = build_object_flag(
        producer="test_producer",
        category="runtime",
        severity="low",
        message="test message",
        status="open",
    )
    assert validate_object_flags([flag]) is None


def test_valid_flag_with_optional_fields_accepted():
    flag = build_object_flag(
        producer="test_producer",
        category="extraction",
        severity="high",
        message="needs attention",
        status="resolved",
        code="shape.type_error",
        path="facts[0]",
        object_id="f1",
        resolution="repaired",
        repair_attempted=True,
        details={"raw_code": "invalid_fact_item_type"},
    )
    assert validate_object_flags([flag]) is None


def test_invalid_severity_rejected():
    flag = build_object_flag(
        producer="test_producer",
        category="runtime",
        severity="low",
        message="test",
        status="open",
    )
    flag["severity"] = "bad"
    assert validate_object_flags([flag]) == "flags[0].severity is invalid"


def test_missing_required_field_rejected():
    flag = build_object_flag(
        producer="test_producer",
        category="runtime",
        severity="low",
        message="test",
        status="open",
    )
    del flag["category"]
    assert "missing required keys" in validate_object_flags([flag])


def test_non_dict_details_rejected():
    flag = build_object_flag(
        producer="test_producer",
        category="runtime",
        severity="low",
        message="test",
        status="open",
    )
    flag["details"] = ["bad"]
    assert validate_object_flags([flag]) == "flags[0].details must be a dict or null"


def test_no_dependency_on_extraction_issue_taxonomy():
    flag = build_object_flag(
        producer="custom_reporter",
        category="membership",
        severity="info",
        message="membership refreshed",
        status="open",
        code="membership.sync",
        details={"source": "test"},
    )
    assert validate_object_flags([flag]) is None
