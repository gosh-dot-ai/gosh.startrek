# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
from copy import deepcopy
import math

import numpy as np
import pytest

import src.mcp_server as mcp_mod
from src.mcp_server import (
    memory_get_config,
    memory_plan_inference,
    memory_recall,
    memory_set_config,
    memory_set_profiles,
)
from src.memory import MemoryServer
from tests._mcp_auth import auth_token_for_agent, install_test_verified_auth

CONFIG = {
    "schema_version": 1,
    "embedding_model": "openai/text-embedding-3-large",
    "embedding_secret_ref": {"name": "team-embed-main", "scope": "system-wide"},
    "librarian_profile": "anthropic/claude-sonnet-4-6",
    "librarian_secret_ref": {"name": "team-extract-main", "scope": "system-wide"},
    "inference_secret_ref": {"name": "team-infer-default", "scope": "system-wide"},
    "judge_secret_ref": {"name": "team-judge-main", "scope": "system-wide"},
    "profiles": {
        1: "fast",
        2: "fast",
        3: "balanced",
        4: "strong",
        5: "strong",
    },
    "profile_configs": {
        "fast": {
            "model": "openai/gpt-4o-mini",
            "secret_ref": {"name": "team-openai-fast", "scope": "system-wide"},
            "pricing": {"input_per_1k": 0.15, "output_per_1k": 0.60},
        },
        "balanced": {
            "model": "google/gemini-2.0-flash",
            "secret_ref": {"name": "team-google-balanced", "scope": "system-wide"},
            "pricing": {"input_per_1k": 0.10, "output_per_1k": 0.40},
        },
        "strong": {
            "model": "anthropic/claude-sonnet-4-6",
            "secret_ref": {"name": "team-anthropic-strong", "scope": "swarm-shared", "swarm_id": "alpha"},
            "pricing": {
                "input_per_1k": 3.0,
                "output_per_1k": 15.0,
                "cache_read_per_1k": 0.3,
                "cache_write_per_1k": 3.75,
            },
        },
    },
    "retrieval": {
        "search_family": "auto",
        "default_token_budget": 4000,
    },
}


@pytest.fixture(autouse=True)
def reset_state(tmp_path, monkeypatch):
    mcp_mod.data_dir = str(tmp_path)
    mcp_mod.registry.clear()
    mcp_mod.courier_registry.clear()
    install_test_verified_auth(monkeypatch)
    yield


def test_memory_server_config_round_trip(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg")
    asyncio.run(server.set_config(CONFIG))
    result = server.get_config()
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["embedding_secret_ref"] == CONFIG["embedding_secret_ref"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["librarian_secret_ref"] == CONFIG["librarian_secret_ref"]
    assert result["inference_secret_ref"] == CONFIG["inference_secret_ref"]
    assert result["judge_secret_ref"] == CONFIG["judge_secret_ref"]
    assert result["profiles"][3] == "balanced"
    assert result["profile_configs"]["strong"]["model"] == "anthropic/claude-sonnet-4-6"
    assert result["profile_configs"]["strong"]["secret_ref"]["name"] == "team-anthropic-strong"
    assert result["profile_configs"]["strong"]["pricing"]["input_per_1k"] == 3.0
    assert "input_cost_per_1k" not in result["profile_configs"]["strong"]


def test_memory_server_config_persists_across_restart(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_restart")
    asyncio.run(server.set_config(CONFIG))

    restarted = MemoryServer(str(tmp_path), "cfg_restart")
    result = restarted.get_config()

    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["embedding_secret_ref"] == CONFIG["embedding_secret_ref"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"][5] == "strong"
    assert result["profile_configs"]["strong"]["secret_ref"]["swarm_id"] == "alpha"


def test_set_profiles_updates_only_inference_subset(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_subset")
    asyncio.run(server.set_config(CONFIG))

    asyncio.run(
        server.set_profiles(
            {1: "cheap", 2: "cheap", 3: "cheap", 4: "best", 5: "best"},
            {
                "cheap": {
                    "model": "openai/gpt-4o-mini",
                    "secret_ref": {"name": "cheap-primary", "scope": "system-wide"},
                    "pricing": {"input_per_1k": 0.15, "output_per_1k": 0.60},
                },
                "best": {
                    "model": "openai/gpt-4.1",
                    "secret_ref": {"name": "best-secondary", "scope": "system-wide"},
                    "pricing": {"input_per_1k": 2.0, "output_per_1k": 8.0},
                },
            },
        )
    )

    result = server.get_config()
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["embedding_secret_ref"] == CONFIG["embedding_secret_ref"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"][1] == "cheap"
    assert result["profile_configs"]["best"]["model"] == "openai/gpt-4.1"
    assert result["profile_configs"]["cheap"]["secret_ref"]["name"] == "cheap-primary"
    assert result["profile_configs"]["best"]["secret_ref"]["name"] == "best-secondary"
    assert result["profile_configs"]["best"]["pricing"]["output_per_1k"] == 8.0


def test_set_config_rejects_invalid_schema_version(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_schema")
    with pytest.raises(ValueError, match="schema_version must be 1"):
        asyncio.run(server.set_config({**CONFIG, "schema_version": 2}))


def test_set_config_rejects_unknown_top_level_key(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_key")
    with pytest.raises(ValueError, match="unknown memory config keys"):
        asyncio.run(server.set_config({**CONFIG, "unexpected": True}))


def test_set_config_rejects_bad_retrieval_budget(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_invalid_retrieval")
    bad = {
        **CONFIG,
        "retrieval": {
            "search_family": "auto",
            "default_token_budget": 0,
        },
    }
    with pytest.raises(ValueError, match="default_token_budget must be positive int"):
        asyncio.run(server.set_config(bad))


def test_set_config_rejects_profiles_without_profile_configs(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_missing_profile_configs")
    bad = deepcopy(CONFIG)
    bad["profile_configs"] = {}
    with pytest.raises(ValueError, match="profile_configs are required when profiles are configured"):
        asyncio.run(server.set_config(bad))


def test_set_config_rejects_missing_pricing_input_per_1k(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_missing_pricing_input")
    bad = deepcopy(CONFIG)
    bad["profile_configs"] = deepcopy(CONFIG["profile_configs"])
    bad["profile_configs"]["fast"]["pricing"] = {"output_per_1k": 0.6}
    with pytest.raises(ValueError, match="pricing.input_per_1k"):
        asyncio.run(server.set_config(bad))


def test_set_config_rejects_negative_pricing_field(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_negative_pricing")
    bad = deepcopy(CONFIG)
    bad["profile_configs"] = deepcopy(CONFIG["profile_configs"])
    bad["profile_configs"]["balanced"]["pricing"] = {
        "input_per_1k": -0.1,
        "output_per_1k": 0.4,
    }
    with pytest.raises(ValueError, match="pricing.input_per_1k must be >= 0"):
        asyncio.run(server.set_config(bad))


def test_set_config_rejects_infinite_pricing_field(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_infinite_pricing")
    bad = deepcopy(CONFIG)
    bad["profile_configs"] = deepcopy(CONFIG["profile_configs"])
    bad["profile_configs"]["balanced"]["pricing"] = {
        "input_per_1k": math.inf,
        "output_per_1k": 0.4,
    }
    with pytest.raises(ValueError, match="pricing.input_per_1k must be finite"):
        asyncio.run(server.set_config(bad))


def test_set_config_normalizes_legacy_flat_pricing_to_nested_pricing(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_legacy_pricing")
    legacy = deepcopy(CONFIG)
    legacy["profile_configs"] = deepcopy(CONFIG["profile_configs"])
    legacy["profile_configs"]["fast"] = {
        "model": "openai/gpt-4o-mini",
        "secret_ref": {"name": "team-openai-fast", "scope": "system-wide"},
        "input_cost_per_1k": 0.15,
        "output_cost_per_1k": 0.60,
    }

    asyncio.run(server.set_config(legacy))

    result = server.get_config()
    fast = result["profile_configs"]["fast"]
    assert fast["pricing"] == {
        "input_per_1k": 0.15,
        "output_per_1k": 0.60,
        "reasoning_per_1k": 0.0,
        "cache_read_per_1k": 0.0,
        "cache_write_per_1k": 0.0,
    }
    assert "input_cost_per_1k" not in fast
    assert "output_cost_per_1k" not in fast


async def _seed_instance(key: str, agent_id: str = "owner") -> None:
    from httpx import ASGITransport, AsyncClient

    app = mcp_mod.create_app(app_data_dir=mcp_mod.data_dir)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/admin/memory/init",
            json={"key": key},
            headers={
                "X-GOSH-MEMORY-TOKEN": mcp_mod.SERVER_TOKEN,
                "Authorization": f"Bearer {auth_token_for_agent(agent_id)}",
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_memory_set_get_config_mcp_round_trip():
    await _seed_instance("cfg_mcp")
    response = await memory_set_config(key="cfg_mcp", config=CONFIG, agent_id="owner", token=auth_token_for_agent("owner"))
    assert response["status"] == "ok"
    result = await memory_get_config(key="cfg_mcp", agent_id="owner", token=auth_token_for_agent("owner"))
    assert result["schema_version"] == 1
    assert result["embedding_model"] == CONFIG["embedding_model"]


@pytest.mark.asyncio
async def test_memory_get_config_missing_instance_returns_not_found():
    result = await memory_get_config(key="missing_cfg", agent_id="owner", token=auth_token_for_agent("owner"))
    assert result["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_set_config_missing_instance_returns_not_found():
    result = await memory_set_config(key="missing_cfg", config=CONFIG, agent_id="owner", token=auth_token_for_agent("owner"))
    assert result["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_set_profiles_missing_instance_returns_not_found():
    result = await memory_set_profiles(
        key="missing_profiles",
        profiles={1: "fast"},
        profile_configs={
            "fast": {
                "model": "openai/gpt-4o-mini",
                "pricing": {"input_per_1k": 0.15, "output_per_1k": 0.60},
            }
        },
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )
    assert result["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_memory_recall_threads_token_budget():
    await _seed_instance("cfg_budget")
    await memory_set_config(key="cfg_budget", config=CONFIG, agent_id="owner", token=auth_token_for_agent("owner"))
    server = mcp_mod.registry["cfg_budget"]
    captured = {}

    async def mock_recall(**kwargs):
        captured.update(kwargs)
        return {
            "context": "ctx",
            "retrieved": [],
            "query_type": "lookup",
            "complexity_hint": {"score": 0.1, "level": 1},
        }

    server.recall = mock_recall

    result = await memory_recall(
        key="cfg_budget",
        query="hello",
        agent_id="owner",
        token_budget=123,
        token=auth_token_for_agent("owner"),
    )

    assert captured["token_budget"] == 123
    assert result["token_estimate"] == 0


@pytest.mark.asyncio
async def test_memory_plan_inference_exposes_opaque_secret_ref():
    await _seed_instance("cfg_secret_ref")
    await memory_set_config(
        key="cfg_secret_ref",
        config=CONFIG,
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )
    server = mcp_mod.registry["cfg_secret_ref"]
    expected_secret_ref = CONFIG["profile_configs"]["fast"]["secret_ref"]

    async def mock_plan_inference(**kwargs):
        return {
            "recommended_profile": "fast",
            "payload": {"model": "openai/gpt-4o-mini", "messages": []},
            "payload_meta": {
                "provider_family": "openai",
                "profile_used": "fast",
                "pricing": CONFIG["profile_configs"]["fast"]["pricing"],
            },
            "secret_ref": expected_secret_ref,
        }

    server.plan_inference = mock_plan_inference

    result = await memory_plan_inference(
        key="cfg_secret_ref",
        query="hello",
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )

    assert result["secret_ref"] == expected_secret_ref
    assert "secret_ref" not in result["payload_meta"]
    assert result["payload_meta"]["pricing"] == CONFIG["profile_configs"]["fast"]["pricing"]


@pytest.mark.asyncio
async def test_memory_recall_drops_payload_meta_and_secret_ref():
    await _seed_instance("cfg_secret_ref_missing")
    await memory_set_config(
        key="cfg_secret_ref_missing",
        config=CONFIG,
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )
    server = mcp_mod.registry["cfg_secret_ref_missing"]

    async def mock_recall(**kwargs):
        return {
            "context": "ctx",
            "retrieved": [],
            "query_type": "lookup",
            "complexity_hint": {"score": 0.1, "level": 1},
            "payload": {"model": "openai/gpt-4o-mini", "messages": []},
            "payload_meta": {
                "provider_family": "openai",
                "profile_used": "fast",
            },
            "_payload_secret_ref": CONFIG["profile_configs"]["fast"]["secret_ref"],
        }

    server.recall = mock_recall

    result = await memory_recall(
        key="cfg_secret_ref_missing",
        query="hello",
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )

    assert "payload" not in result
    assert "payload_meta" not in result


@pytest.mark.asyncio
async def test_memory_set_profiles_keeps_non_profile_config_fields():
    await _seed_instance("cfg_keep")
    await memory_set_config(key="cfg_keep", config=CONFIG, agent_id="owner", token=auth_token_for_agent("owner"))
    await memory_set_profiles(
        key="cfg_keep",
        profiles={1: "cheap"},
        profile_configs={
            "cheap": {
                "model": "openai/gpt-4o-mini",
                "pricing": {"input_per_1k": 0.15, "output_per_1k": 0.60},
            }
        },
        agent_id="owner",
        token=auth_token_for_agent("owner"),
    )
    result = await memory_get_config(key="cfg_keep", agent_id="owner", token=auth_token_for_agent("owner"))
    assert result["embedding_model"] == CONFIG["embedding_model"]
    assert result["embedding_secret_ref"] == CONFIG["embedding_secret_ref"]
    assert result["librarian_profile"] == CONFIG["librarian_profile"]
    assert result["profiles"] == {1: "cheap"}


def test_set_config_rejects_invalid_secret_ref_shape(tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_bad_secret_ref")
    bad = {**CONFIG, "embedding_secret_ref": {"scope": "system-wide"}}
    with pytest.raises(ValueError, match="embedding_secret_ref.name must be a non-empty string"):
        asyncio.run(server.set_config(bad))


@pytest.mark.asyncio
async def test_build_index_uses_runtime_embedding_dim_for_empty_tiers(monkeypatch, tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_dim")
    await server.set_config({**CONFIG, "embedding_model": "custom/1536"})
    server._all_granular = [
        {"id": "g1", "fact": "alpha", "conv_id": "cfg_dim", "session": 1}
    ]
    server._all_cons = []
    server._all_cross = []
    captured = []

    async def fake_embed_texts(texts, **kwargs):
        captured.append(kwargs)
        return np.ones((len(texts), 1536), dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_texts", fake_embed_texts)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)

    await server.build_index()

    assert captured
    assert all(call["model"] == "custom/1536" for call in captured)
    assert server._data_dict["atomic_embs"].shape[1] == 1536
    assert server._data_dict["cons_embs"].shape[1] == 1536
    assert server._data_dict["cross_embs"].shape[1] == 1536


@pytest.mark.asyncio
async def test_recall_uses_runtime_embedding_model_for_query(monkeypatch, tmp_path):
    server = MemoryServer(str(tmp_path), "cfg_query_model")
    await server.set_config({**CONFIG, "embedding_model": "runtime/query-model"})
    source_id = "conv_john"
    server._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [
                {
                    "episode_id": "conv_e1",
                    "source_type": "conversation",
                    "source_id": source_id,
                    "source_date": "2024-06-01",
                    "topic_key": "session",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "John is doing kickboxing.",
                    "provenance": {"raw_span": [0, 24]},
                },
                {
                    "episode_id": "conv_e2",
                    "source_type": "conversation",
                    "source_id": source_id,
                    "source_date": "2024-06-01",
                    "topic_key": "session",
                    "state_label": "session",
                    "currentness": "unknown",
                    "raw_text": "John is going to do taekwondo.",
                    "provenance": {"raw_span": [0, 31]},
                },
            ],
        }]
    }
    facts = [
        {
            "id": "kick",
            "session": 1,
            "fact": "John is doing kickboxing",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e1", "episode_source_id": source_id},
        },
        {
            "id": "tae",
            "session": 2,
            "fact": "John is going to do taekwondo",
            "entities": ["John"],
            "source_id": source_id,
            "metadata": {"episode_id": "conv_e2", "episode_source_id": source_id},
        },
    ]
    server._all_granular = facts
    server._all_cons = []
    server._all_cross = []
    server._temporal_index_dirty = False
    server._raw_sessions = [
        {"session_num": 1, "format": "conversation", "source_id": source_id},
        {"session_num": 2, "format": "conversation", "source_id": source_id},
    ]
    server._data_dict = {
        "atomic_embs": np.array([[1.0, 0.0], [0.9, 0.1]], dtype=float),
        "cons_embs": np.zeros((0, 2)),
        "cross_embs": np.zeros((0, 2)),
        "fact_lookup": {fact["id"]: fact for fact in facts},
    }
    server._fact_lookup = {fact["id"]: fact for fact in facts}
    captured = []

    async def fake_embed_query(_text, model=None, provider=None):
        captured.append({"model": model, "provider": provider})
        return np.array([1.0, 0.0], dtype=float)

    monkeypatch.setattr("src.memory.embed_query", fake_embed_query)

    result = await server.recall("What martial arts has John done?")

    assert {fact["id"] for fact in result["retrieved"]} == {"kick", "tae"}
    assert captured
    assert all(call["model"] == "runtime/query-model" for call in captured)
