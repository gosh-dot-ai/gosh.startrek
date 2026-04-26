# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import numpy as np

from src.memory import MemoryServer, _route_prompt_type
from src.prompt_routing.hooks import build_payload_messages, resolve_prompt_key
from tests._memory_llm_mocks import patch_memory_llm_runtime


async def _store(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.store(*args, **kwargs)

# ── Summarize branch ──

def test_summarize():
    assert _route_prompt_type("summarize", [{"id": "f1"}], 100, 50, "") == (
        "summarize_with_metadata", True
    )


# ── ICL branch ──

def test_icl():
    assert _route_prompt_type("icl", [{"id": "f1"}], 100, 50, "") == (
        "icl", False
    )


# ── Tool mode: low coverage with facts, >20 sessions, <30% coverage ──

def test_low_coverage_tool():
    """50 sessions, 10 in context (20% < 30%) → tool mode."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 50, 10, "RETRIEVED FACTS only"
    ) == ("tool", True)


# ── Negative: low session count should NOT trigger tool mode ──

def test_low_session_count_no_tool():
    """Only 5 sessions — even with <30% coverage, no tool mode."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 1, "RETRIEVED FACTS only"
    ) != ("tool", True)


# ── Negative: empty resolved_facts should NOT trigger tool mode ──

def test_empty_facts_no_tool():
    """No resolved_facts — tool mode requires facts to exist."""
    pt, ut = _route_prompt_type("lookup", [], 50, 10, "RETRIEVED FACTS only")
    assert pt != "tool"
    assert ut is False


# ── Hybrid: high coverage + RAW CONTEXT present ──

def test_high_coverage_hybrid():
    """5 sessions, 3 in context (60%) with RAW CONTEXT → hybrid."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3, "RAW CONTEXT blah"
    ) == ("hybrid", False)


# ── Fallback: no raw context, no special type → pass through ──

def test_no_raw_context_fallback():
    """No RAW CONTEXT marker → fallback to resolved_type."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3, "RETRIEVED FACTS only"
    ) == ("lookup", False)


# ── Summarize takes priority over low-coverage tool mode ──

def test_summarize_overrides_low_coverage():
    """summarize type wins even with low coverage params."""
    assert _route_prompt_type(
        "summarize", [{"id": "f1"}], 50, 5, "RAW CONTEXT blah"
    ) == ("summarize_with_metadata", True)


# ── ICL takes priority over everything below it ──

def test_icl_overrides_low_coverage():
    """icl type wins even with low coverage params."""
    assert _route_prompt_type(
        "icl", [{"id": "f1"}], 50, 5, "RAW CONTEXT blah"
    ) == ("icl", False)


# ── Episode hybrid marker recognized by _route_prompt_type ──

def test_episode_hybrid_marker():
    """Episode raw-text marker triggers hybrid mode like RAW CONTEXT does."""
    assert _route_prompt_type(
        "lookup", [{"id": "f1"}], 5, 3,
        "Some context\n--- SOURCE EPISODE RAW TEXT ---\nraw text here"
    ) == ("hybrid", False)


def test_document_hybrid_marker():
    """Document section marker must still route the fact path to hybrid."""
    assert _route_prompt_type(
        "lookup",
        [{"id": "f1"}],
        5,
        3,
        "Some context\n--- SOURCE DOCUMENT SECTIONS ---\nsection text here",
    ) == ("hybrid", False)


def test_episode_path_does_not_enter_tool_mode():
    """Episode recall should preserve its legacy no-tool behavior."""
    assert _route_prompt_type(
        "lookup",
        [{"id": "f1"}],
        50,
        10,
        "Some context\n--- SOURCE EPISODE RAW TEXT ---\nraw text here",
        allow_tool_mode=False,
    ) == ("hybrid", False)


# ── Real-pipeline integration: store → recall → routing ──

DIM = 3072


def _patch_llm_and_embeddings(monkeypatch):
    """Mock only LLM extraction and embedding API calls — NOT retrieval/routing."""

    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"The project uses Rust for the agent executor.",
             "kind": "fact", "entities": ["Rust"], "tags": ["tech"], "session": sn},
            {"id": f"f{sn}_1", "fact": f"Memory subsystem is written in Python.",
             "kind": "fact", "entities": ["Python"], "tags": ["tech"], "session": sn},
        ], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Project stack: Rust agent, Python memory.",
             "kind": "summary", "entities": ["Rust", "Python"], "tags": ["tech"]}
        ])

    async def mock_cross(**kwargs):
        return ("conv", "e", [
            {"id": "x0", "fact": "Multi-language architecture: Rust + Python.",
             "kind": "profile", "entities": ["Rust", "Python"], "tags": ["arch"]}
        ])

    async def mock_embed(texts, **kw):
        # Deterministic embeddings seeded by text content for consistent retrieval
        vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % (2**31)
            rng = np.random.RandomState(seed)
            vecs.append(rng.randn(DIM).astype(np.float32))
        return np.array(vecs)

    async def mock_embed_q(text, **kw):
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    patch_memory_llm_runtime(monkeypatch)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)


def test_real_pipeline_store_recall_routing(tmp_path, monkeypatch):
    """End-to-end: store() real data, recall() through real retrieval + routing.

    Mocks only LLM extraction and embedding API calls.
    Retrieval (retrieve_adaptive), query-type detection, and prompt routing
    all run through real code paths.
    """
    _patch_llm_and_embeddings(monkeypatch)

    ms = MemoryServer(str(tmp_path), "pipeline_test")

    # Store a session through the real store() path
    store_result = asyncio.run(_store(ms,
        content="User: What language is the agent written in?\n"
                "Assistant: The agent executor is written in Rust.\n"
                "User: And the memory subsystem?\n"
                "Assistant: The memory subsystem is Python-based.",
        session_num=1,
        session_date="2024-06-01",
    ))
    assert store_result.get("facts_extracted", 0) > 0

    # build_index runs real embedding + data_dict construction
    asyncio.run(ms.build_index())

    # recall() runs real retrieval + real routing — nothing mocked here
    result = asyncio.run(ms.recall("What language is the agent written in?"))

    # Core recall shape assertions
    assert "context" in result
    assert "query_type" in result
    assert "recommended_prompt_type" in result
    assert "use_tool" in result
    assert isinstance(result["use_tool"], bool)

    # The real retrieval+rendering path for this corpus should stay hybrid
    assert result["recommended_prompt_type"] == "hybrid"

    # Retrieved facts should exist (real retrieval found something)
    assert len(result.get("retrieved", [])) > 0

    # Context should contain actual content from stored facts
    assert len(result["context"]) > 0


def test_resolve_prompt_key_routes_pure_code_lookup_to_code_slot():
    prompt_key = resolve_prompt_key(
        prompt_type="lookup",
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


def test_codebase_payload_routing_does_not_use_tool_prompt_affordance():
    messages = build_payload_messages(
        prompt_type="lookup",
        context="pkg.service.cinder_signature\n--- SOURCE FILES ---",
        query="What is the exact Python qualified name of the function?",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
            "sessions_in_context": 1,
            "total_sessions": 1,
            "coverage_pct": 100,
        },
        speakers="User and Assistant",
    )

    content = messages[0]["content"]
    assert "exact code object lookup question" in content
    assert "get_more_context" not in content


def test_resolve_prompt_key_routes_mixed_code_and_prose_to_codebase_mixed():
    prompt_key = resolve_prompt_key(
        prompt_type="hybrid",
        query=(
            "Using all available memory sources, find the incident codename mentioned in chat, "
            "the owner team named in the document, and the exact Python qualified name."
        ),
        recall_result={
            "search_family": "auto",
            "retrieval_families": ["conversation", "document", "codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "mixed_code_plus_prose"},
                "codebase_augmentation": {"mode": "whole_file"},
                "mixed_family_merge": {"merged_families": ["conversation", "document", "codebase"]},
            },
        },
    )

    assert prompt_key == "codebase_mixed"


def test_resolve_prompt_key_routes_chain_queries_to_code_chain():
    prompt_key = resolve_prompt_key(
        prompt_type="lookup",
        query="Trace the dependency path and show what calls this symbol.",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
        },
    )

    assert prompt_key == "code_chain"


def test_resolve_prompt_key_routes_risk_queries_to_risk_review():
    prompt_key = resolve_prompt_key(
        prompt_type="lookup",
        query="What breaks if we change this function? Give the blast radius.",
        recall_result={
            "search_family": "codebase",
            "retrieval_families": ["codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "precise_code"},
                "codebase_augmentation": {"mode": "whole_file"},
            },
        },
    )

    assert prompt_key == "risk_review"


def test_build_payload_messages_uses_codebase_mixed_prompt_body():
    messages = build_payload_messages(
        prompt_type="hybrid",
        context="CINDER-42\nPlatform Reliability\npkg.service.cinder_signature\n--- SOURCE FILES ---",
        query=(
            "Using all available memory sources, find the incident codename mentioned in chat, "
            "the owner team named in the document, and the exact Python qualified name."
        ),
        recall_result={
            "search_family": "auto",
            "retrieval_families": ["conversation", "document", "codebase"],
            "runtime_trace": {
                "query": {"code_query_mode": "mixed_code_plus_prose"},
                "codebase_augmentation": {"mode": "whole_file"},
                "mixed_family_merge": {"merged_families": ["conversation", "document", "codebase"]},
            },
            "sessions_in_context": 1,
            "total_sessions": 3,
            "coverage_pct": 33,
        },
        speakers="User and Assistant",
    )

    content = messages[0]["content"]
    assert "requires BOTH prose memory evidence and codebase evidence" in content
    assert "CINDER-42" in content
    assert "pkg.service.cinder_signature" in content
