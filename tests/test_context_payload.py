# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import numpy as np

from src.memory import MemoryServer
import src.mcp_server as mcp_mod
from tests._memory_llm_mocks import patch_memory_llm_runtime
from tests._mcp_auth import auth_token_for_owner, install_test_verified_auth


DIM = 3072

def _pricing(input_per_1k, output_per_1k, **extra):
    pricing = {
        "input_per_1k": input_per_1k,
        "output_per_1k": output_per_1k,
        "reasoning_per_1k": 0.0,
        "cache_read_per_1k": 0.0,
        "cache_write_per_1k": 0.0,
    }
    pricing.update(extra)
    return pricing


def _profile(model, context_window, max_output_tokens, *, thinking_overhead=0, **extra):
    profile = {
        "model": model,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "thinking_overhead": thinking_overhead,
        "pricing": _pricing(0.1, 0.4),
    }
    profile.update(extra)
    return profile


PROFILES = {1: "fast", 2: "fast", 3: "balanced", 4: "max", 5: "max"}
PROFILE_CONFIGS = {
    "fast": _profile(
        "openai/gpt-4o-mini",
        128000,
        2000,
        pricing=_pricing(0.15, 0.60),
    ),
    "balanced": _profile(
        "google/gemini-2.0-flash",
        128000,
        2000,
        pricing=_pricing(0.10, 0.40),
    ),
    "max": _profile(
        "anthropic/claude-sonnet-4-6",
        200000,
        4096,
        pricing=_pricing(3.0, 15.0, cache_read_per_1k=0.3, cache_write_per_1k=3.75),
    ),
}


async def _store(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.store(*args, **kwargs)


def _patch_all(monkeypatch):
    async def mock_extract(**kwargs):
        sn = kwargs.get("session_num", 1)
        return ("conv", sn, "2024-06-01", [
            {"id": f"f{sn}_0", "fact": f"Fact {sn}", "kind": "fact",
             "entities": [], "tags": [], "session": sn}], [])

    async def mock_consolidate(**kwargs):
        return ("conv", 1, "2024-06-01", [])

    async def mock_cross(**kwargs):
        return ("conv", "e", [])

    async def mock_embed(texts, **kw):
        return np.random.randn(len(texts), DIM).astype(np.float32)

    async def mock_embed_q(text, **kw):
        return np.random.randn(DIM).astype(np.float32)

    monkeypatch.setattr("src.memory.extract_session", mock_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda f, l: None)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_q)
    patch_memory_llm_runtime(monkeypatch)


def _make_server(tmp_path, monkeypatch, *, profiles=True, inference_leaf_plugins=None):
    _patch_all(monkeypatch)
    ms = MemoryServer(
        str(tmp_path),
        "ctx_payload",
        profiles=PROFILES if profiles else None,
        profile_configs=PROFILE_CONFIGS if profiles else None,
        inference_leaf_plugins=inference_leaf_plugins,
    )
    asyncio.run(_store(ms,
        "User: We chose anodized aluminum for the casing.\nAssistant: Noted.",
        session_num=1,
        session_date="2024-06-01",
    ))
    ms._all_granular[0]["fact"] = "We chose anodized aluminum for the casing."
    ms._all_granular[0]["kind"] = "decision"
    ms._all_granular[0]["status"] = "active"
    ms._all_granular[0]["_session_content_complexity"] = 0.55
    asyncio.run(ms.build_index())
    return ms


def test_plan_inference_payload_present_with_profiles(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.plan_inference("What material did we choose for the casing?"))
    assert "payload" in result
    assert "payload_meta" in result
    assert result["payload"]["model"] == PROFILE_CONFIGS["balanced"]["model"]
    assert result["payload_meta"]["provider_family"] == "google"
    assert result["payload_meta"]["pricing"] == PROFILE_CONFIGS["balanced"]["pricing"]


def test_recall_payload_absent_without_profiles(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=False)
    result = asyncio.run(ms.recall("What material did we choose for the casing?"))
    assert "payload" not in result
    assert "payload_meta" not in result
    assert "complexity_hint" in result


def test_payload_messages_contain_context_and_question(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.plan_inference("What material did we choose for the casing?"))
    message = result["payload"]["messages"][0]["content"]
    assert "anodized aluminum" in message
    assert "What material did we choose for the casing?" in message


def test_payload_messages_use_list_set_prompt_for_list_queries(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=False)
    messages = ms._build_payload_messages(
        prompt_type="default",
        context="RETRIEVED FACTS:\n[1] Gina is working on a wildlife documentary project.\n\nRAW SLOT CANDIDATES:\n[Q1] wildlife documentary",
        query="What kind of project is Gina doing?",
        recall_result={
            "sessions_in_context": 1,
            "total_sessions": 8,
            "coverage_pct": 12,
            "query_operator_plan": {
                "slot_query": {"enabled": True},
                "list_set": {"enabled": False},
                "ordinal": {"enabled": False},
                "commonality": {"enabled": False},
                "compare_diff": {"enabled": False},
            },
        },
        speakers="User and Assistant",
    )
    content = messages[0]["content"]
    assert "slot-filling or attribute question" in content
    assert "RAW SLOT CANDIDATES" in content




def test_payload_temperature_from_profile_or_zero(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    ms._profile_configs["balanced"]["temperature"] = 0.3
    result = asyncio.run(ms.plan_inference("What material did we choose for the casing?"))
    assert result["payload"]["temperature"] == 0.3


def test_payload_cost_estimate_uses_nested_profile_pricing(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    cost = ms._estimate_payload_cost(
        payload={"model": PROFILE_CONFIGS["fast"]["model"], "max_tokens": 500},
        payload_meta={
            "pricing": _pricing(0.15, 0.60),
            "message_tokens_est": 1000,
            "tool_tokens_est": 0,
            "backend": "api",
        },
    )
    expected = (1000 / 1000 * 0.15) + (500 / 1000 * 0.60)
    assert abs(cost - expected) < 1e-9


def test_payload_cost_estimate_falls_back_to_profile_pricing_when_payload_meta_pricing_missing(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    cost = ms._estimate_payload_cost(
        payload={"model": PROFILE_CONFIGS["fast"]["model"], "max_tokens": 500},
        payload_meta={
            "profile_used": "fast",
            "message_tokens_est": 1000,
            "tool_tokens_est": 0,
            "backend": "api",
        },
    )
    expected = (1000 / 1000 * 0.15) + (500 / 1000 * 0.60)
    assert abs(cost - expected) < 1e-9


def test_payload_tools_present_only_when_use_tool(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_payload_tools",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
        "runtime_trace": {"caller": {"kept": True}},
    }
    payload, meta, _secret_ref = ms._build_payload(query="Need more detail", recall_result=recall_result)
    assert "tools" in payload
    assert meta["use_tool"] is True

    payload2, meta2, _secret_ref2 = ms._build_payload(
        query="Need more detail",
        recall_result=recall_result,
        use_tool=False,
    )
    assert "tools" not in payload2
    assert meta2["use_tool"] is False


def test_tool_payload_includes_get_more_context_but_hybrid_payload_does_not(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_payload_tool_hybrid_prompts",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    base_recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
        "runtime_trace": {},
    }

    tool_result = {
        **base_recall_result,
        "recommended_prompt_type": "tool",
        "use_tool": True,
    }
    tool_payload, tool_meta, _tool_secret_ref = ms._build_payload(
        query="Need more detail",
        recall_result=tool_result,
    )
    tool_context = "\n".join(str(message.get("content") or "") for message in tool_payload["messages"])
    assert "get_more_context" in tool_context
    assert any((tool.get("function") or {}).get("name") == "get_more_context" for tool in tool_payload["tools"])
    assert tool_meta["use_tool"] is True

    hybrid_result = {
        **base_recall_result,
        "recommended_prompt_type": "hybrid",
        "use_tool": False,
    }
    hybrid_payload, hybrid_meta, _hybrid_secret_ref = ms._build_payload(
        query="Need more detail",
        recall_result=hybrid_result,
    )
    hybrid_context = "\n".join(str(message.get("content") or "") for message in hybrid_payload["messages"])
    assert "get_more_context" not in hybrid_context
    assert "tools" not in hybrid_payload
    assert hybrid_meta["use_tool"] is False


def test_payload_provider_specific_shapes(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_provider_shapes",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    messages = [{"role": "user", "content": "hello"}]
    openai_payload, _ = ms._build_provider_payload(
        model="openai/gpt-4o-mini",
        messages=messages,
        max_tokens=200,
        temperature=0,
        use_tool=True,
    )
    anthropic_payload, _ = ms._build_provider_payload(
        model="anthropic/claude-sonnet-4-6",
        messages=messages,
        max_tokens=200,
        temperature=0,
        use_tool=True,
    )
    assert openai_payload["tools"][0]["type"] == "function"
    assert "function" in openai_payload["tools"][0]
    assert anthropic_payload["tools"][0]["input_schema"]["type"] == "object"


def test_context_for_legacy_drops_payload(tmp_path, monkeypatch):
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    result = asyncio.run(ms.context_for("What material did we choose?", token_budget=4000))
    assert "payload" not in result
    assert "payload_meta" not in result


def test_memory_plan_inference_mcp_returns_payload(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    install_test_verified_auth(monkeypatch)
    mcp_mod.registry.clear()
    owner_token = auth_token_for_owner("mcp-payload-owner")
    store_result = asyncio.run(mcp_mod.memory_store(
        key="mcp_payload",
        content="User: Short fact.\nAssistant: saved.",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        token=owner_token,
    ))
    assert store_result.get("status") == "ok"
    ms = mcp_mod.registry["mcp_payload"]
    asyncio.run(ms.set_profiles(PROFILES, PROFILE_CONFIGS))
    asyncio.run(ms.build_index())
    result = asyncio.run(mcp_mod.memory_plan_inference(
        key="mcp_payload",
        query="What is stored?",
        token_budget=200000,
        token=owner_token,
    ))
    assert "payload" in result
    assert "payload_meta" in result
    selected_profile = result["payload_meta"]["profile_used"]
    assert result["payload_meta"]["pricing"] == PROFILE_CONFIGS[selected_profile]["pricing"]


def test_memory_recall_mcp_drops_payload_when_truncated(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    install_test_verified_auth(monkeypatch)
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    ms = _make_server(tmp_path, monkeypatch, profiles=True)
    mcp_mod.registry["mcp_trunc"] = ms
    result = asyncio.run(mcp_mod.memory_recall(
        key="mcp_trunc",
        query="What material did we choose for the casing?",
        token_budget=1,
        token=auth_token_for_owner("system"),
    ))
    assert "payload" not in result
    assert "payload_meta" not in result


def test_truncation_preserves_tier1_and_drops_raw_first(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_trunc",
        profiles={1: "fast"},
        profile_configs={"fast": _profile("openai/gpt-4o-mini", 600, 20)},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "[1] critical decision", "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [{"text": "[2] ordinary fact", "rank": 1, "source": "fact"}],
            "tier4": [{"text": "[Raw S1]\n" + ("x" * 800), "rank": 2, "source": "raw"}],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
        "runtime_trace": {"caller": {"kept": True}},
    }
    finalized, trace = ms._finalize_recall_evidence_context(
        query="test",
        recall_result=recall_result,
    )
    assert "[1] critical decision" in finalized["context"]
    assert "[Raw S1]" not in finalized["context"]
    assert trace["truncation"]["removed"]["tier4"] >= 1
    assert trace["budget_exceeded"] is False
    assert recall_result["context"] == ""
    assert "evidence_context" not in recall_result["runtime_trace"]
    assert recall_result["runtime_trace"]["caller"]["kept"] is True

    payload, meta, _secret_ref = ms._build_payload(query="test", recall_result=finalized)
    assert payload["messages"]
    assert "[1] critical decision" in finalized["context"]
    assert meta["truncation"]["removed"]["tier4"] >= 1


def test_valid_profile_finalization_renders_raw_window_and_continuation_segments(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_raw_window_finalized",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "[1] Project Alpha workflow exists.", "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [],
            "tier4": [
                {
                    "text": "RAW CONVERSATION EVIDENCE:\n[conversation] assistant: Project Alpha releases at night.",
                    "rank": 0,
                    "source": "raw_window",
                },
            ],
        },
        "recall_continuation": {
            "available": True,
            "handle": "test-handle",
            "next_page": 2,
            "anchor_terms": ["project", "alpha"],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 4,
        "coverage_pct": 25,
        "runtime_trace": {},
    }

    finalized, trace = ms._finalize_recall_evidence_context(
        query="When should Project Alpha release?",
        recall_result=recall_result,
    )

    assert trace["finalized"] is True
    assert "RAW CONVERSATION EVIDENCE:" in finalized["context"]
    assert "Project Alpha releases at night." in finalized["context"]
    assert "RECALL CONTINUATION AVAILABLE:" in finalized["context"]
    assert 'page="next"' in finalized["context"]
    assert finalized["context"].count("RECALL CONTINUATION AVAILABLE:") == 1


def test_inference_plan_finalizes_unfinalized_context_packet_before_payload(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_plan_finalizes_packet",
        profiles={1: "fast"},
        profile_configs={"fast": _profile("openai/gpt-4o-mini", 400, 20)},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "[1] critical decision", "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [{"text": "[2] ordinary fact", "rank": 1, "source": "fact"}],
            "tier4": [{"text": "[Raw S1]\n" + ("x" * 800), "rank": 2, "source": "raw"}],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
        "runtime_trace": {"caller": {"kept": True}},
    }

    plan = ms._build_inference_plan_from_recall_result(
        query="test",
        recall_result=recall_result,
    )

    payload_text = "\n".join(str(message.get("content") or "") for message in plan["payload"]["messages"])
    assert "[1] critical decision" in payload_text
    assert "[Raw S1]" not in payload_text
    assert plan["payload_meta"]["truncation"]["removed"]["tier4"] >= 1
    assert recall_result["context"] == ""
    assert "evidence_context" not in recall_result["runtime_trace"]
    assert recall_result["runtime_trace"]["caller"]["kept"] is True
    assert "_payload_secret_ref" not in recall_result


def test_tool_payload_has_visible_continuation_instruction_with_answer_contract(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_tool_continuation_visible",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "[1] Project Alpha workflow checkpoint.", "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [],
            "tier4": [],
        },
        "recall_continuation": {
            "available": True,
            "handle": "test-handle",
            "next_page": 2,
            "anchor_terms": ["project", "alpha"],
        },
        "_recall_continuation_pages": [
            {"page": 2, "context": "RECALL CONTINUATION PAGE 2:\n[1] Project Alpha schedule fact."},
        ],
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "tool",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 4,
        "coverage_pct": 25,
        "runtime_trace": {},
    }

    finalized, _trace = ms._finalize_recall_evidence_context(
        query="When should Project Alpha release?",
        recall_result=recall_result,
    )
    assert finalized["answer_contract"]["recall_continuation"]["available"] is True

    plan = ms._build_inference_plan_from_recall_result(
        query="When should Project Alpha release?",
        recall_result=finalized,
        use_tool=True,
    )
    payload_text = "\n".join(str(message.get("content") or "") for message in plan["payload"]["messages"])

    assert "RECALL CONTINUATION AVAILABLE:" in payload_text
    assert 'page="next"' in payload_text
    assert "recall_continuation handle" in payload_text
    assert any((tool.get("function") or {}).get("name") == "get_more_context" for tool in plan["payload"]["tools"])
    assert plan["payload_meta"]["use_tool"] is True


def test_public_answer_contract_documents_prompt_template_and_reference_date(tmp_path):
    ms = MemoryServer(str(tmp_path), "ctx_contract_reference")
    recall_result = {
        "context": "Context block",
        "query_type": "lookup",
        "retrieved": [
            {"fact": "Older fact", "session_date": "2024-04-01"},
            {"fact": "Latest fact", "session_date": "2024-06-15"},
        ],
        "sessions_in_context": 2,
        "total_sessions": 2,
        "coverage_pct": 100,
    }

    contract = ms._build_public_answer_contract(
        query="What happened?",
        recall_result=recall_result,
    )

    assert contract["prompt_template_public"] is True
    assert "{context}" in contract["prompt_template"]
    assert contract["variables"]["reference_date"] == "2024-06-15"


def test_budget_exceeded_when_tier1_alone_too_large(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_budget_exceeded",
        profiles={1: "fast"},
        profile_configs={"fast": _profile("openai/gpt-4o-mini", 80, 20)},
    )
    recall_result = {
        "context": "",
        "_context_packet": {
            "tier1": [{"text": "critical " * 200, "rank": 0, "source": "fact"}],
            "tier2": [],
            "tier3": [],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    _payload, meta, _secret_ref = ms._build_payload(query="test", recall_result=recall_result)
    assert meta["budget_exceeded"] is True


def test_routing_summarize_sets_use_tool_in_payload(tmp_path):
    """When recommended_prompt_type is summarize_with_metadata, payload gets tools."""
    ms = MemoryServer(
        str(tmp_path),
        "ctx_routing_summarize",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "summarize",
        "recommended_profile": "fast",
        "recommended_prompt_type": "summarize_with_metadata",
        "use_tool": True,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta, _secret_ref = ms._build_payload(query="Summarize the project", recall_result=recall_result)
    assert "tools" in payload
    assert meta["use_tool"] is True
    assert meta["prompt_type"] == "summarize_with_metadata"


def test_routing_default_no_tools_in_payload(tmp_path):
    """When use_tool is False, payload has no tools and prompt_type passes through."""
    ms = MemoryServer(
        str(tmp_path),
        "ctx_routing_default",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta, _secret_ref = ms._build_payload(query="What happened?", recall_result=recall_result)
    assert "tools" not in payload
    assert meta["use_tool"] is False


def test_build_payload_returns_runtime_secret_ref_without_stashing_on_recall_result(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_secret_ref",
        profiles={1: "fast"},
        profile_configs={
            "fast": {
                **PROFILE_CONFIGS["fast"],
                "secret_ref": {"name": "fast-runtime-secret", "scope": "system-wide"},
            }
        },
    )
    recall_result = {
        "context": "Context block",
        "_context_packet": {
            "tier1": [],
            "tier2": [],
            "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
            "tier4": [],
        },
        "query_type": "default",
        "recommended_profile": "fast",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "sessions_in_context": 1,
        "total_sessions": 1,
        "coverage_pct": 100,
    }
    payload, meta, secret_ref = ms._build_payload(query="What happened?", recall_result=recall_result)
    assert payload["model"] == PROFILE_CONFIGS["fast"]["model"]
    assert meta["profile_used"] == "fast"
    assert secret_ref["name"] == "fast-runtime-secret"
    assert "_payload_secret_ref" not in recall_result


def test_ask_uses_existing_payload_when_no_overrides(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_ask_reuse",
        profiles={1: "fast"},
        profile_configs={"fast": PROFILE_CONFIGS["fast"]},
    )

    async def fake_recall(*args, **kwargs):
        return {
            "context": "Context block",
            "query_type": "default",
            "recommended_profile": "fast",
            "use_tool": False,
            "recommended_prompt_type": "lookup",
            "sessions_in_context": 1,
            "total_sessions": 1,
            "coverage_pct": 100,
            "payload": {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 100,
                "temperature": 0,
                "seed": 42,
            },
            "payload_meta": {
                "profile_used": "fast",
                "profile_fallback": False,
                "context_tokens": 10,
                "message_tokens_est": 20,
                "tool_tokens_est": 0,
                "memory_budget": 1000,
                "budget_exceeded": False,
                "prompt_type": "lookup",
                "use_tool": False,
                "truncation": None,
                "provider": "openai",
                "provider_family": "openai_compatible",
            },
        }

    async def fake_send(payload, *, caller_id=None, secret_ref=None):
        return "payload answer", False, []

    def fail_build_payload(**kwargs):
        raise AssertionError("_build_payload should not be called when ready payload exists")

    ms.recall = fake_recall
    ms._send_payload = fake_send
    ms._build_payload = fail_build_payload

    result = asyncio.run(ms.ask("test"))
    assert result["answer"] == "payload answer"


def test_ask_forwards_runtime_secret_ref_to_send_payload(tmp_path):
    ms = MemoryServer(
        str(tmp_path),
        "ctx_secret_forward",
        profiles={1: "fast"},
        profile_configs={
            "fast": {
                **PROFILE_CONFIGS["fast"],
                "secret_ref": {"name": "fast-runtime-secret", "scope": "system-wide"},
            }
        },
    )
    captured = {}

    async def fake_recall(*args, **kwargs):
        return {
            "context": "Context block",
            "_context_packet": {
                "tier1": [],
                "tier2": [],
                "tier3": [{"text": "Context block", "rank": 0, "source": "fact"}],
                "tier4": [],
            },
            "query_type": "default",
            "recommended_profile": "fast",
            "recommended_prompt_type": "lookup",
            "use_tool": False,
            "sessions_in_context": 1,
            "total_sessions": 1,
            "coverage_pct": 100,
            "retrieved": [],
            "retrieval_families": ["conversation"],
            "search_family": "auto",
            "runtime_trace": {},
        }

    async def fake_send(payload, *, caller_id=None, secret_ref=None):
        captured["secret_ref"] = secret_ref
        return "payload answer", False, []

    ms.recall = fake_recall
    ms._send_payload = fake_send

    result = asyncio.run(ms.ask("test"))
    assert result["answer"] == "payload answer"
    assert captured["secret_ref"]["name"] == "fast-runtime-secret"
