# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from src.inference import (
    INF_PROMPTS,
    get_inf_prompt,
    resolve_inference_prompt_key,
)


# ── get_inf_prompt basic contract ──

def test_get_inf_prompt_lookup():
    prompt = get_inf_prompt("lookup")
    assert "{context}" in prompt
    assert "{question}" in prompt


def test_get_inf_prompt_temporal():
    prompt = get_inf_prompt("temporal")
    assert "{context}" in prompt
    assert "{question}" in prompt
    assert "chronological" in prompt.lower()


def test_get_inf_prompt_aggregate():
    prompt = get_inf_prompt("aggregate")
    assert "COUNTING PROTOCOL" in prompt


def test_get_inf_prompt_current():
    prompt = get_inf_prompt("current")
    assert "MOST RECENT" in prompt or "UPDATED" in prompt


def test_get_inf_prompt_synthesize():
    prompt = get_inf_prompt("synthesize")
    assert "preference" in prompt.lower() or "pattern" in prompt.lower()


def test_get_inf_prompt_procedural():
    prompt = get_inf_prompt("procedural")
    assert "rules" in prompt.lower() or "policies" in prompt.lower()


def test_get_inf_prompt_prospective():
    prompt = get_inf_prompt("prospective")
    assert "planned" in prompt.lower() or "upcoming" in prompt.lower()


def test_get_inf_prompt_summarize():
    prompt = get_inf_prompt("summarize")
    assert "chronological" in prompt.lower()


def test_get_inf_prompt_unknown_returns_lookup():
    """Unknown query types fall back to the lookup prompt."""
    assert get_inf_prompt("nonexistent") == get_inf_prompt("lookup")


def test_get_inf_prompt_empty_returns_lookup():
    assert get_inf_prompt("") == get_inf_prompt("lookup")


# ── All prompts have required placeholders ──

def test_all_prompts_have_context_and_question():
    for qtype, prompt in INF_PROMPTS.items():
        assert "{context}" in prompt, f"{qtype} prompt missing {{context}}"
        assert "{question}" in prompt, f"{qtype} prompt missing {{question}}"


# ── Prompts are formattable ──

def test_lookup_prompt_formats():
    prompt = get_inf_prompt("lookup").format(context="some facts", question="what?")
    assert "some facts" in prompt
    assert "what?" in prompt


def test_summarize_prompt_formats():
    prompt = get_inf_prompt("summarize").format(context="facts here", question="summarize all")
    assert "facts here" in prompt
    assert "summarize all" in prompt


# ── Dict completeness ──

def test_inf_prompts_has_all_types():
    expected = {"lookup", "temporal", "aggregate", "current",
                "synthesize", "procedural", "prospective", "summarize", "icl",
                "hybrid", "tool", "summarize_with_metadata", "list_set",
                "slot_query", "compositional", "codebase", "codebase_mixed",
                "code_slot", "code_chain", "risk_review",
                "container_exact_copy"}
    assert set(INF_PROMPTS.keys()) == expected


def test_resolve_inference_prompt_key_uses_slot_query_leaf():
    operator_plan = {
        "slot_query": {"enabled": True},
        "list_set": {"enabled": False},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
    }
    assert resolve_inference_prompt_key("default", operator_plan) == "slot_query"


def test_resolve_inference_prompt_key_does_not_use_slot_query_leaf_for_bounded_chain_queries():
    operator_plan = {
        "slot_query": {"enabled": True},
        "bounded_chain": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "list_set": {"enabled": False},
        "local_anchor": {"enabled": False},
        "temporal_grounding": {"enabled": False},
    }
    assert resolve_inference_prompt_key("hybrid", operator_plan) == "hybrid"


def test_resolve_inference_prompt_key_uses_list_set_leaf():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert resolve_inference_prompt_key("hybrid", operator_plan) == "list_set"


def test_resolve_inference_prompt_key_maps_default_to_lookup_before_leafs():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert resolve_inference_prompt_key("default", operator_plan) == "list_set"


def test_resolve_inference_prompt_key_respects_disabled_leaf_plugin():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    assert (
        resolve_inference_prompt_key(
            "hybrid",
            operator_plan,
            plugin_state={"list_set": False},
        )
        == "hybrid"
    )


def test_get_inf_prompt_slot_query_is_short_leaf():
    prompt = get_inf_prompt("slot_query")
    assert "slot-filling or attribute question" in prompt
    assert "RAW SLOT CANDIDATES" in prompt


def test_get_inf_prompt_list_set_is_short_leaf():
    prompt = get_inf_prompt("list_set")
    assert "grounded list or set of items" in prompt
    assert "Return all distinct grounded items" in prompt


def test_get_inf_prompt_container_exact_copy_is_candidate_decision_leaf():
    prompt = get_inf_prompt("container_exact_copy")
    assert "TERMINAL RENDER CANDIDATE" in prompt
    assert "structured planner proof" in prompt
    assert "Decide from the proof fields, not from technical ids" in prompt
    assert "Do not write, reconstruct, summarize, or paraphrase" in prompt
    assert '"decision":"use_candidate"' in prompt


def test_resolve_inference_prompt_key_routes_terminal_render_candidate_to_exact_copy_leaf():
    prompt_key = resolve_inference_prompt_key(
        "lookup",
        {"ordinal": {"enabled": True}},
        recall_result={
            "search_family": "document",
            "retrieval_families": ["document"],
            "terminal_render_candidate": {"status": "available"},
            "runtime_trace": {
                "container_graph": {
                    "render_mode": "exact_copy",
                    "exact_copy_validated": True,
                },
            },
        },
    )
    assert prompt_key == "container_exact_copy"


def test_resolve_inference_prompt_key_requires_terminal_render_candidate_payload():
    prompt_key = resolve_inference_prompt_key(
        "lookup",
        {"ordinal": {"enabled": True}},
        recall_result={
            "search_family": "document",
            "retrieval_families": ["document"],
            "runtime_trace": {
                "container_graph": {
                    "render_mode": "exact_copy",
                    "exact_copy_validated": True,
                },
            },
        },
    )
    assert prompt_key == "lookup"


def test_resolve_inference_prompt_key_routes_mixed_code_queries_to_codebase_mixed():
    prompt_key = resolve_inference_prompt_key(
        "hybrid",
        {},
        query=(
            "Using all available memory sources, find the incident codename mentioned in chat, "
            "the owner team named in the document, and the exact Python qualified name."
        ),
        recall_result={
            "retrieval_families": ["conversation", "document", "codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "mixed_code_plus_prose"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
        },
    )
    assert prompt_key == "codebase_mixed"


def test_resolve_inference_prompt_key_routes_pure_code_slot_queries_to_code_slot():
    prompt_key = resolve_inference_prompt_key(
        "lookup",
        {},
        query="What is the exact Python qualified name of the function?",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
        },
    )
    assert prompt_key == "code_slot"


def test_resolve_inference_prompt_key_does_not_treat_requested_codebase_lane_as_retrieved_evidence():
    prompt_key = resolve_inference_prompt_key(
        "lookup",
        {},
        query="What is the exact Python qualified name of the function?",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": [],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
            },
        },
    )
    assert prompt_key == "lookup"


def test_resolve_inference_prompt_key_preserves_operator_leaf_priority_over_code_leafs():
    operator_plan = {
        "list_set": {"enabled": True},
        "ordinal": {"enabled": False},
        "commonality": {"enabled": False},
        "compare_diff": {"enabled": False},
        "bounded_chain": {"enabled": False},
    }
    prompt_key = resolve_inference_prompt_key(
        "lookup",
        operator_plan,
        query="What parameters does this API take?",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
        },
    )
    assert prompt_key == "list_set"
