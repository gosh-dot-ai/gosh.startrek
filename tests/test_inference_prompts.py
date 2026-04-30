# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path

PROMPTS_DIR = Path("src/prompts/inference")


def test_hybrid_prompt():
    p = PROMPTS_DIR / "hybrid.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "CONFLICT RESOLUTION" in text


def test_list_set_prompt():
    p = PROMPTS_DIR / "list_set.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "list or set of items" in text


def test_tool_prompt():
    p = PROMPTS_DIR / "tool.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "{sessions_in_context}" in text
    assert "{total_sessions}" in text
    assert "CONFLICT RESOLUTION" in text
    assert "get_more_context" in text
    assert "RECALL CONTINUATION AVAILABLE" in text
    assert 'page="next"' in text
    assert "recall_continuation handle" in text


def test_tool_prompt_uses_conditional_recency_policy_like_hybrid():
    tool_text = (PROMPTS_DIR / "tool.md").read_text()
    hybrid_text = (PROMPTS_DIR / "hybrid.md").read_text()

    assert "get_more_context" in tool_text
    assert "ALWAYS choose" not in tool_text
    assert "HIGHEST session number" not in tool_text
    assert "the most recent update. Ignore superseded values." not in tool_text

    shared_policy_fragments = [
        "Never use session number alone",
        "newer/current/replaces",
        "explicit date/version/status",
        "If the evidence does not prove which fact supersedes the other",
        "strongest directly relevant evidence",
    ]
    for fragment in shared_policy_fragments:
        assert fragment in tool_text
        assert fragment in hybrid_text


def test_summarize_with_metadata_prompt():
    p = PROMPTS_DIR / "summarize_with_metadata.md"
    assert p.exists()
    text = p.read_text()
    assert "{context}" in text
    assert "{question}" in text
    assert "{total_sessions}" in text
    assert "{sessions_in_context}" in text
    assert "{coverage_pct" in text
