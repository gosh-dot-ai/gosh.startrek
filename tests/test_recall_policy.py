# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path

from src.recall_policy import (
    FACT_LIKELIHOOD_HIGH,
    FACT_LIKELIHOOD_MEDIUM,
    FACT_LIKELIHOOD_UNCERTAIN,
    RAW_CONVERSATION_WINDOW_RADIUS,
    RAW_EPISODE_RETRIEVAL_LIMIT,
    RAW_SOURCE_WINDOW_BUDGET_CHARS,
    RECALL_EXTRACTION_POLICY_MIRROR,
)


def test_recall_policy_mirror_points_to_existing_prompts():
    repo_root = Path(__file__).resolve().parents[1]
    for family, axes in RECALL_EXTRACTION_POLICY_MIRROR.items():
        assert family in {"conversation", "document"}
        assert axes
        for axis_name, axis in axes.items():
            assert axis_name
            source_prompt = axis.get("source_prompt")
            assert isinstance(source_prompt, str)
            assert (repo_root / source_prompt).is_file()


def test_recall_policy_mirror_axes_have_extraction_contract_fields():
    for axes in RECALL_EXTRACTION_POLICY_MIRROR.values():
        for axis in axes.values():
            assert axis.get("rule_ids")
            assert all(isinstance(rule_id, str) and rule_id for rule_id in axis["rule_ids"])
            assert axis.get("source_requirements")
            assert all(
                isinstance(requirement, str) and requirement
                for requirement in axis["source_requirements"]
            )
            assert axis.get("evidence_contract")
            assert all(
                isinstance(contract, str) and contract
                for contract in axis["evidence_contract"]
            )


def test_recall_policy_public_constants_are_bounded_and_typed():
    assert {FACT_LIKELIHOOD_HIGH, FACT_LIKELIHOOD_MEDIUM, FACT_LIKELIHOOD_UNCERTAIN} == {
        "high",
        "medium",
        "uncertain",
    }
    assert 0 < RAW_CONVERSATION_WINDOW_RADIUS <= 2
    assert 0 < RAW_EPISODE_RETRIEVAL_LIMIT <= 10
    assert 0 < RAW_SOURCE_WINDOW_BUDGET_CHARS <= 8000
