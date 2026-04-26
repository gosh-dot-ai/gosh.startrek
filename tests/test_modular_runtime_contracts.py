# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import asyncio

import pytest

from src.episode_retrieval import route_retrieval_families
from src.episodes import validate_episode
from src.ingest import ingest_input
from src.inference import get_inf_prompt
from src.prompt_routing.hooks import build_payload_messages, resolve_prompt_key
from src.query_executors import registry as executor_registry
from src.query_executors.registry import (
    get_default_query_executors,
    register_query_executor,
    run_default_query_executor_chain,
)
from src.runtime_contracts import (
    assert_overlay_post_extraction_only,
    validate_overlay_delta_only,
)
from src.source_adapters.conversation import ConversationAdapter
from src.source_adapters.document import DocumentAdapter
from src.source_loader import LoadedSource


@pytest.mark.asyncio
async def test_ingest_routes_via_source_adapter_registry(monkeypatch):
    loaded = LoadedSource(
        raw_text="repo file listing",
        transport="path",
        locator="/tmp/repo",
        filename="repo",
        mime="inode/directory",
        is_directory=True,
        is_repo=True,
        fetch_metadata={"entry_count": 2},
    )
    seen: dict[str, object] = {}

    class FakeAdapter:
        ingestable = True

        async def ingest(self, server, **kwargs):
            seen["server"] = server
            seen.update(kwargs)
            return {"status": "ok", "adapter": "fake"}

    async def fake_load_source(**kwargs):
        assert kwargs == {"text": None, "path": "/tmp/repo", "url": None}
        return loaded

    monkeypatch.setattr("src.ingest.load_source", fake_load_source)
    monkeypatch.setattr(
        "src.ingest.detect_source_family",
        lambda *args, **kwargs: ("codebase", {"family": "codebase", "signals": ["repo_markers"]}),
    )
    monkeypatch.setattr("src.ingest.normalize_text", lambda text, family: f"{family}:{text}")
    monkeypatch.setattr("src.ingest.get_source_adapter", lambda family: FakeAdapter())

    server = object()
    result = await ingest_input(
        server,
        path="/tmp/repo",
        scope="agent-private",
        metadata={"bench": "guard"},
        source_id="repo-main",
    )

    assert seen["server"] is server
    assert seen["loaded"] == loaded
    assert seen["normalized_text"] == "codebase:repo file listing"
    assert seen["source_id"] == "repo-main"
    assert seen["metadata"] == {"bench": "guard"}
    assert result["source_family"] == "codebase"
    assert result["transport"] == "path"
    assert result["locator"] == "/tmp/repo"


@pytest.mark.asyncio
async def test_conversation_adapter_preserves_current_ingest_behavior():
    calls: dict[str, object] = {}

    class FakeServer:
        _raw_sessions = [{"session_num": 2}, {"session_num": 7}, {"session_num": "bad"}]

        async def store(self, **kwargs):
            calls.update(kwargs)
            return {"status": "ok", "facts_extracted": 3}

    loaded = LoadedSource(
        raw_text="User: hello\nAssistant: hi",
        transport="text",
        locator="(inline)",
        filename="chat.log",
        mime="text/plain",
        is_directory=False,
        is_repo=False,
        fetch_metadata={},
    )

    result = await ConversationAdapter().ingest(
        FakeServer(),
        loaded=loaded,
        normalized_text="User: hello\nAssistant: hi",
        metadata={"thread": "demo"},
        retention_ttl=60,
        target=["agent:demo"],
        agent_id="demo-agent",
        swarm_id="demo-swarm",
        scope="agent-private",
        source_id=None,
        owner_id="agent:demo-agent",
        read=["agent:demo-agent"],
        write=["agent:demo-agent"],
        caller_id="agent:demo-agent",
        caller_principal_kind="agent",
    )

    assert result["facts_extracted"] == 3
    assert calls["content"] == "User: hello\nAssistant: hi"
    assert calls["session_num"] == 8
    assert calls["session_date"] == ""
    assert calls["speakers"] == "User and Assistant"
    assert calls["source_id"] == "chat.log"
    assert calls["metadata"] == {"thread": "demo"}
    assert calls["scope"] == "agent-private"


@pytest.mark.asyncio
async def test_document_adapter_preserves_current_ingest_behavior():
    calls: dict[str, object] = {}

    class FakeServer:
        async def ingest_document(self, **kwargs):
            calls.update(kwargs)
            return {"status": "ok", "facts_extracted": 2}

    loaded = LoadedSource(
        raw_text="# Spec\n\nSystem layout.",
        transport="path",
        locator="/tmp/spec.md",
        filename="spec.md",
        mime="text/markdown",
        is_directory=False,
        is_repo=False,
        fetch_metadata={},
    )

    result = await DocumentAdapter().ingest(
        FakeServer(),
        loaded=loaded,
        normalized_text="# Spec\n\nSystem layout.",
        metadata={"doc": "spec"},
        retention_ttl=120,
        target=["team:runtime"],
        agent_id="demo-agent",
        swarm_id="demo-swarm",
        scope="agent-private",
        source_id=None,
        owner_id="agent:demo-agent",
        read=["agent:demo-agent"],
        write=["agent:demo-agent"],
        caller_id="agent:demo-agent",
        caller_principal_kind="agent",
    )

    assert result["facts_extracted"] == 2
    assert result["source_id"] == "spec.md"
    assert calls["content"] == "# Spec\n\nSystem layout."
    assert calls["source_id"] == "spec.md"
    assert calls["family"] == "document"
    assert calls["source_meta"] == {
        "ingest_transport": "path",
        "ingest_locator": "/tmp/spec.md",
        "ingest_mime": "text/markdown",
    }

def test_validate_episode_accepts_registered_family():
    episode = {
        "episode_id": "repo_e01",
        "source_type": "codebase",
        "source_id": "mini_repo",
        "source_date": "2026-04-01",
        "topic_key": "billing",
        "state_label": "module",
        "currentness": "unknown",
        "raw_text": "[File src/billing.py]\ndef parse_invoice_total(amount): ...",
        "provenance": {"raw_span": [0, 56]},
    }

    assert validate_episode(episode) == []


def test_route_retrieval_families_uses_family_capabilities():
    routed = route_retrieval_families(
        "Which file defines the parse invoice total function?",
        ["conversation", "document", "codebase"],
    )

    assert routed == ["codebase"]


def test_default_query_executor_chain_preserves_current_order(monkeypatch):
    expected_order = [
        "temporal",
        "container_graph",
        "conversation_structural",
        "codebase_structural",
        "document_structural",
        "coverage_recovery",
        "semantic_rescue",
    ]
    assert [executor.name for executor in get_default_query_executors()] == expected_order

    seen: list[str] = []

    class FakeExecutor:
        def __init__(self, name: str, *, halted: bool = False, skip_when_augmented: bool = False, facts=None):
            self.name = name
            self.priority = 0
            self.halted = halted
            self.skip_when_augmented = skip_when_augmented
            self.facts = facts

        def supports(self, query_type, search_family, packet):
            return True

        def should_skip(self, *, packet, query, query_type, episode_lookup, augmented_facts):
            return self.skip_when_augmented and augmented_facts is not None

        async def augment(self, server, *, packet, query, query_type, episode_lookup, fact_filter):
            seen.append(self.name)
            return packet, self.facts

        def should_halt(self, *, packet, query, query_type, episode_lookup, augmented_facts):
            return self.halted

    original_registry = dict(executor_registry._REGISTERED_QUERY_EXECUTORS)
    executor_registry._REGISTERED_QUERY_EXECUTORS.clear()
    register_query_executor(FakeExecutor("custom_first"))
    register_query_executor(FakeExecutor("custom_second", facts=[{"id": "f1"}]))
    register_query_executor(FakeExecutor("custom_third", halted=True))
    register_query_executor(FakeExecutor("custom_skipped", skip_when_augmented=True))

    try:
        packet, augmented = asyncio.run(
            run_default_query_executor_chain(
                object(),
                query="Which file defines the parse invoice total function?",
                query_type="lookup",
                packet={"search_family": "codebase"},
                episode_lookup={},
                fact_filter=lambda _fact: True,
            )
        )
    finally:
        executor_registry._REGISTERED_QUERY_EXECUTORS.clear()
        executor_registry._REGISTERED_QUERY_EXECUTORS.update(original_registry)

    assert packet == {"search_family": "codebase"}
    assert augmented == [{"id": "f1"}]
    assert seen == ["custom_first", "custom_second", "custom_third"]


def test_prompt_routing_hook_preserves_existing_prompt_mapping():
    prompt = get_inf_prompt("aggregate").format(
        context="Billing facts",
        question="How many invoice parsers exist?",
        speakers="User and Assistant",
        sessions_in_context=0,
        total_sessions=0,
        coverage_pct=100,
        reference_date="2023-01-01",
    )
    messages = build_payload_messages(
        prompt_type="counting",
        context="Billing facts",
        query="How many invoice parsers exist?",
        recall_result={},
        speakers="User and Assistant",
    )

    assert messages == [{"role": "user", "content": prompt}]
    assert resolve_prompt_key(
        prompt_type="lookup",
        query="What project is Gina doing?",
        recall_result={"query_operator_plan": {"slot_query": {"enabled": True}}},
    ) == "slot_query"
    assert resolve_prompt_key(
        prompt_type="lookup",
        query="Which commit modifies src/billing.py?",
        recall_result={"search_family": "codebase", "retrieval_families": ["codebase"]},
    ) == "codebase"


def test_overlay_plugin_contract_is_post_extraction_only():
    assert_overlay_post_extraction_only(
        base_facts=[{"id": "f1"}, {"id": "f2"}],
        extracted_facts=[{"id": "f1"}, {"id": "f2"}, {"id": "f3"}],
    )

    with pytest.raises(ValueError, match="rewrite or erase"):
        assert_overlay_post_extraction_only(
            base_facts=[{"id": "f1"}, {"id": "f2"}],
            extracted_facts=[{"id": "f2"}, {"id": "f3"}],
        )


def test_overlay_plugin_contract_is_delta_only():
    normalized = validate_overlay_delta_only({
        "facts": [{"id": "f1"}],
        "temporal_links": [{"before": "f1", "after": "f2"}],
    })

    assert normalized == {
        "facts": [{"id": "f1"}],
        "temporal_links": [{"before": "f1", "after": "f2"}],
    }
    with pytest.raises(ValueError, match="facts/temporal_links deltas"):
        validate_overlay_delta_only({"facts": [], "records": []})


def test_overlay_plugin_cannot_write_into_all_granular():
    with pytest.raises(ValueError, match="_all_granular"):
        validate_overlay_delta_only({"facts": [], "_all_granular": []})
