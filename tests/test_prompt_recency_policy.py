# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path

import pytest

from src.memory import MemoryServer
from src.prompt_routing.hooks import build_payload_messages, resolve_prompt_key


PROMPTS_DIR = Path("src/prompts/inference")
PROSE_PROMPTS = (
    "lookup",
    "current",
    "temporal",
    "aggregate",
    "synthesize",
    "procedural",
    "prospective",
    "summarize",
    "summarize_with_metadata",
    "tool",
    "hybrid",
)
FORBIDDEN_POLICY_FRAGMENTS = (
    "ALWAYS choose",
    "HIGHEST session number",
    "highest session number — it is the most recent update",
    "Ignore superseded values",
)
REQUIRED_POLICY_FRAGMENTS = (
    "Never use session number alone",
    "newer/current/replaces",
    "explicit date/version/status",
    "If the evidence does not prove which fact supersedes the other",
)
PROFILE_CONFIGS = {
    "fast": {
        "model": "openai/gpt-4o-mini",
        "context_window": 128000,
        "max_output_tokens": 2000,
        "thinking_overhead": 0,
        "pricing": {
            "input_per_1k": 0.15,
            "output_per_1k": 0.60,
            "reasoning_per_1k": 0.0,
            "cache_read_per_1k": 0.0,
            "cache_write_per_1k": 0.0,
        },
    }
}
PAYLOAD_PROMPT_CASES = (
    ("default", False),
    ("supersession", False),
    ("temporal", False),
    ("counting", False),
    ("synthesize", False),
    ("rule", False),
    ("prospective", False),
    ("summarize", False),
    ("summarize_with_metadata", True),
    ("tool", True),
    ("hybrid", False),
)


def _assert_policy(text: str) -> None:
    for fragment in FORBIDDEN_POLICY_FRAGMENTS:
        assert fragment not in text
    for fragment in REQUIRED_POLICY_FRAGMENTS:
        assert fragment in text


def _make_server(tmp_path) -> MemoryServer:
    return MemoryServer(
        str(tmp_path),
        "prompt_recency_policy",
        profiles={1: "fast"},
        profile_configs=PROFILE_CONFIGS,
    )


def _recall_result(*, prompt_type: str, context: str, use_tool: bool) -> dict:
    return {
        "context": context,
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": context, "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": prompt_type,
        "recommended_profile": "fast",
        "recommended_prompt_type": prompt_type,
        "use_tool": use_tool,
        "sessions_in_context": 2,
        "total_sessions": 99,
        "coverage_pct": 2,
        "runtime_trace": {},
    }


def _payload_text(
    tmp_path,
    *,
    prompt_type: str,
    context: str,
    query: str,
    use_tool: bool = False,
) -> tuple[str, dict, dict]:
    ms = _make_server(tmp_path)
    payload, meta, _secret_ref = ms._build_payload(
        query=query,
        recall_result=_recall_result(prompt_type=prompt_type, context=context, use_tool=use_tool),
    )
    text = "\n".join(str(message.get("content") or "") for message in payload["messages"])
    return text, payload, meta


@pytest.mark.parametrize("prompt_name", PROSE_PROMPTS)
def test_prose_prompt_text_uses_universal_recency_policy(prompt_name):
    text = (PROMPTS_DIR / f"{prompt_name}.md").read_text()

    _assert_policy(text)
    if prompt_name in {"tool", "summarize_with_metadata"}:
        assert "get_more_context" in text
    else:
        assert "get_more_context" not in text


@pytest.mark.parametrize(("prompt_type", "use_tool"), PAYLOAD_PROMPT_CASES)
def test_payload_messages_use_universal_recency_policy(tmp_path, prompt_type, use_tool):
    text, payload, meta = _payload_text(
        tmp_path,
        prompt_type=prompt_type,
        use_tool=use_tool,
        context="[S12] The value is ALPHA.\n[S75] The value is BETA.",
        query="What is the value?",
    )

    _assert_policy(text)
    if prompt_type in {"tool", "summarize_with_metadata"}:
        assert "get_more_context" in text
        assert "tools" in payload
        assert any((tool.get("function") or {}).get("name") == "get_more_context" for tool in payload["tools"])
        assert meta["use_tool"] is True
    else:
        assert "get_more_context" not in text
        assert "tools" not in payload


BENCHMARK_DERIVED_CASES = (
    pytest.param(
        "LoCoMo",
        "session_id_unsafe",
        "lookup",
        "[S12] Person A likes activity X.\n[S75] Person A likes activity Y.",
        "What activity does Person A like?",
        "locomo/data/sources/qwen_sprint30/query_manifest.json: category 4 preference/commonality questions",
        id="locomo_unordered_preference_does_not_use_highest_session",
    ),
    pytest.param(
        "LoCoMo",
        "explicit_update",
        "current",
        "[S12] Person A likes activity X.\n[S75] Person A says: update, I no longer like X; I now like Y.",
        "What activity does Person A currently like?",
        "locomo/data/sources/qwen_sprint30/bundles: currently/now/update facts appear in conversation bundles",
        id="locomo_explicit_update_can_resolve_current_preference",
    ),
    pytest.param(
        "MRCR",
        "session_id_unsafe",
        "lookup",
        "[S3] In the requested reference passage, the access code is ALPHA.\n"
        "[S19] In an unrelated later passage, the access code is BETA.",
        "What access code appears in the requested reference passage?",
        "mrcr_v2/data/mrcr_test.jsonl: asks for a specific indexed generated passage/span",
        id="mrcr_direct_span_beats_later_distractor",
    ),
    pytest.param(
        "MRCR",
        "explicit_update",
        "current",
        "[S3] The access code is ALPHA.\n[S19] Correction to the earlier access code: use BETA instead.",
        "What access code should be used now?",
        "mrcr_v2/data/mrcr_test.jsonl: same generated-span question shape with correction/update fixture",
        id="mrcr_explicit_correction_can_resolve_current_value",
    ),
    pytest.param(
        "AMA",
        "session_id_unsafe",
        "lookup",
        "[S8] Active rule: key is win.\n[S42] An unrelated later observation moves the wall.",
        "Which active rule makes the key win?",
        "ama/data/ama_test.jsonl: babaisai active-rule trajectory observations",
        id="ama_active_rule_not_overridden_by_later_unrelated_observation",
    ),
    pytest.param(
        "AMA",
        "explicit_update",
        "current",
        "[S8] Active rule: key is win.\n[S42] Updated active rule: key no longer wins; door is win now.",
        "Which active rule is current?",
        "ama/data/ama_test.jsonl: babaisai active rules change across trajectory turns",
        id="ama_explicit_active_rule_update_can_resolve_state",
    ),
    pytest.param(
        "LongMemEval",
        "session_id_unsafe",
        "lookup",
        "[S5] The user stored the passport in the blue folder.\n"
        "[S88] The user bought a red folder for tax papers.",
        "Where is the passport stored?",
        "convomem/dataset_hf/legacy_benchmarks/longmemeval/preferences: long-range preference/lookup evidence",
        id="longmemeval_direct_entity_relevance_beats_later_distractor",
    ),
    pytest.param(
        "LongMemEval",
        "explicit_update",
        "current",
        "[S5] The user stored the passport in the blue folder.\n"
        "[S88] On 2024-04-20, the user moved the passport to the safe.",
        "Where is the passport currently stored?",
        "convomem/dataset_hf/legacy_benchmarks/longmemeval/knowledge_updates: explicit update evidence",
        id="longmemeval_explicit_dated_relocation_can_resolve_currentness",
    ),
    pytest.param(
        "Document",
        "session_id_unsafe",
        "lookup",
        "[S2] Policy limit is 10 units.\n[S9] Policy limit is 20 units.",
        "What is the policy limit?",
        "document-style version/status coverage; no direct benchmark corpus document/version example found",
        id="document_unordered_sections_do_not_use_highest_session",
    ),
    pytest.param(
        "Document",
        "explicit_update",
        "current",
        "[S2] Policy v3 current: limit is 10 units.\n[S9] Policy v2 archived: limit is 20 units.",
        "What is the current policy limit?",
        "document-style version/status coverage; no direct benchmark corpus document/version example found",
        id="document_version_status_can_resolve_currentness",
    ),
)


@pytest.mark.parametrize(
    ("benchmark_name", "risk_class", "prompt_type", "context", "query", "corpus_note"),
    BENCHMARK_DERIVED_CASES,
)
def test_benchmark_derived_recency_policy_contract(
    tmp_path,
    benchmark_name,
    risk_class,
    prompt_type,
    context,
    query,
    corpus_note,
):
    del benchmark_name, corpus_note
    text, _payload, _meta = _payload_text(
        tmp_path,
        prompt_type=prompt_type,
        context=context,
        query=query,
    )

    _assert_policy(text)
    if risk_class == "session_id_unsafe":
        assert "Never use session number alone" in text
        assert "strongest directly relevant evidence" in text
    else:
        assert "newer/current/replaces" in text
        assert "explicit date/version/status" in text


def test_codebase_leaf_prompts_do_not_receive_prose_recency_policy():
    codebase_cases = (
        (
            "codebase",
            "Explain the module behavior.",
            {
                "search_family": "codebase",
                "retrieval_families": ["codebase"],
                "runtime_trace": {"codebase_augmentation": {"mode": "whole_file"}},
            },
            "retrieved codebase memory evidence",
        ),
        (
            "code_chain",
            "Trace the dependency path and show what calls this symbol.",
            {
                "search_family": "codebase",
                "retrieval_families": ["codebase"],
                "runtime_trace": {
                    "query": {"code_query_mode": "precise_code"},
                    "codebase_augmentation": {"mode": "whole_file"},
                },
            },
            "code dependency, call-chain, or impact-trace question",
        ),
        (
            "risk_review",
            "What breaks if we change this function? Give the blast radius.",
            {
                "search_family": "codebase",
                "retrieval_families": ["codebase"],
                "runtime_trace": {
                    "query": {"code_query_mode": "precise_code"},
                    "codebase_augmentation": {"mode": "whole_file"},
                },
            },
            "code change risk or review question",
        ),
    )
    for expected_prompt, query, recall_result, expected_content in codebase_cases:
        assert resolve_prompt_key(prompt_type="lookup", query=query, recall_result=recall_result) == expected_prompt
        messages = build_payload_messages(
            prompt_type="lookup",
            context="--- SOURCE FILES ---\npath/to/file.py",
            query=query,
            recall_result={
                **recall_result,
                "sessions_in_context": 1,
                "total_sessions": 1,
                "coverage_pct": 100,
            },
            speakers="User and Assistant",
        )
        content = messages[0]["content"]
        assert expected_content in content
        assert "Never use session number alone" not in content
        assert "get_more_context" not in content


def test_container_exact_copy_prompt_path_is_not_given_prose_recency_policy():
    recall_result = {
        "terminal_render_candidate": {"candidate_id": "candidate-1"},
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    assert (
        resolve_prompt_key(
            prompt_type="lookup",
            query="Copy the second paragraph exactly.",
            recall_result=recall_result,
        )
        == "container_exact_copy"
    )
    messages = build_payload_messages(
        prompt_type="lookup",
        context="TERMINAL RENDER CANDIDATE: candidate-1",
        query="Copy the second paragraph exactly.",
        recall_result=recall_result,
        speakers="User and Assistant",
    )
    content = messages[0]["content"]
    assert "TERMINAL RENDER CANDIDATE" in content
    assert "Never use session number alone" not in content
    assert "get_more_context" not in content
