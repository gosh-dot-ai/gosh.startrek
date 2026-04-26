# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from src.extraction_repair import apply_patch_operations, parse_patch_operations


def test_parse_patch_operations_invalid_json_maps_to_repair_patch_invalid():
    operations, issues = parse_patch_operations(
        "not json at all",
        producer="test_producer",
        family="conversation",
        layer="top_level",
    )

    assert operations is None
    assert issues[0]["normalized_code"] == "repair.patch_invalid"
    assert issues[0]["raw_code"] == "json_decode_error"


def test_parse_patch_operations_rejects_missing_operations_list():
    operations, issues = parse_patch_operations(
        {"facts": []},
        producer="test_producer",
        family="conversation",
        layer="top_level",
    )

    assert operations is None
    assert issues == [{
        "normalized_code": "repair.patch_invalid",
        "raw_code": "missing_operations_list",
        "category": "repair",
        "layer": "top_level",
        "family": "conversation",
        "path": "operations",
        "object_id": None,
        "expected": "list",
        "actual": "NoneType",
        "message": "Patch repair output must include an operations list.",
        "repair_hint": "Return only {\"operations\": []}",
        "repairable": False,
        "severity": "medium",
    }]


def test_parse_patch_operations_rejects_operation_not_object():
    operations, issues = parse_patch_operations(
        {"operations": ["bad-op"]},
        producer="test_producer",
        family="conversation",
        layer="top_level",
    )

    assert operations is None
    assert issues[0]["normalized_code"] == "repair.patch_invalid"
    assert issues[0]["raw_code"] == "operation_not_object"
    assert issues[0]["path"] == "operations[0]"


def test_parse_patch_operations_rejects_unsupported_operation_action():
    operations, issues = parse_patch_operations(
        {"operations": [{"path": "facts[0]", "action": "mutate", "value": {}}]},
        producer="test_producer",
        family="conversation",
        layer="top_level",
    )

    assert operations is None
    assert issues[0]["normalized_code"] == "repair.patch_invalid"
    assert issues[0]["raw_code"] == "unsupported_operation_action"
    assert issues[0]["path"] == "facts[0]"


def test_apply_patch_operations_rejects_path_outside_allowed_roots():
    payload = {"facts": [{"local_id": "b1", "fact": "ok", "kind": "fact"}], "temporal_links": []}
    operations = [{"path": "temporal_links", "action": "append", "value": {"before": "b1", "after": "b1"}}]

    updated, issues = apply_patch_operations(
        payload,
        operations,
        allowed_roots={"facts[0]"},
        family="conversation",
        layer="top_level",
    )

    assert updated == payload
    assert issues[0]["normalized_code"] == "repair.path_not_allowed"
    assert issues[0]["raw_code"] == "operation_outside_allowed_roots"
    assert issues[0]["path"] == "temporal_links"


def test_apply_patch_operations_rejects_invalid_target_path_with_patch_invalid():
    payload = {"facts": [{"local_id": "b1", "fact": "ok", "kind": "fact"}], "temporal_links": []}
    operations = [{"path": "facts[9].fact", "action": "set", "value": "changed"}]

    updated, issues = apply_patch_operations(
        payload,
        operations,
        allowed_roots={"facts[9]"},
        family="conversation",
        layer="top_level",
    )

    assert updated == payload
    assert issues[0]["normalized_code"] == "repair.patch_invalid"
    assert issues[0]["raw_code"] == "operation_apply_failed"
    assert issues[0]["path"] == "facts[9].fact"
