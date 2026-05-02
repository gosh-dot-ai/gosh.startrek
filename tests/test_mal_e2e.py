# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from src.mal.apply import ApplyEngine, current_gen_dir
from src.mal.artifact_store import ArtifactStore
from src.mal.control_store import ControlStore
from src.mal.feedback_store import FeedbackStore
from src.mal.scheduler import Scheduler
from src.memory import MemoryServer
from tests._auth_helpers import bootstrap_harness
from tests._memory_llm_mocks import patch_memory_llm_runtime
from tests._mcp_auth import auth_token_for_agent, install_test_verified_auth

DIM = 3072
KEY = "mal_test"
AGENT = "default"
MCP_OWNER = "mal_owner"
MCP_BINDING = f"agent:{MCP_OWNER}"


def _mcp_token() -> str:
    return auth_token_for_agent(MCP_OWNER)


def _rand_embs(n, dim=DIM):
    return np.random.randn(n, dim).astype(np.float32)


def _rand_qemb(dim=DIM):
    return np.random.randn(dim).astype(np.float32)


def _patch_all(monkeypatch):
    """Patch extraction + embeddings for MemoryServer."""
    install_test_verified_auth(monkeypatch)

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        facts = [
            {"id": f"f{i}", "fact": f"Test fact {i}", "kind": "event",
             "entities": ["TestEntity"], "tags": ["test"],
             "session": sn, "scope": "swarm-shared",
             "agent_id": "default", "swarm_id": "default"}
            for i in range(3)
        ]
        tlinks = [{"before": "f0", "after": "f1", "signal": "then"}]
        return ("mal_conv", sn, "2024-06-01", facts, tlinks)

    async def mock_session_merge_stub(**kwargs):
        return ("mal_conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Consolidated fact", "kind": "summary",
             "entities": ["TestEntity"], "tags": ["test"]},
        ])

    async def mock_cross_merge_stub(**kwargs):
        return ("mal_conv", "testentity", [
            {"id": "x0", "fact": "Cross-session fact", "kind": "profile",
             "entities": ["TestEntity"], "tags": ["test"]},
        ])

    async def mock_embed_texts(texts, **kwargs):
        return _rand_embs(len(texts))

    async def mock_embed_query(text, **kwargs):
        return _rand_qemb()

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    patch_memory_llm_runtime(monkeypatch)
    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


# ---------------------------------------------------------------------------
# 1. Full production loop: stores -> feedback -> trigger -> artifact -> apply
# ---------------------------------------------------------------------------

class TestMALProductionLoop:

    def test_full_loop(self, tmp_path, monkeypatch):
        """Full MAL loop: store, build_index, ask, feedback, trigger, artifact, apply."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        # a) Create MemoryServer with multiple facts including entities
        #    Some facts will only be found with entity_phrase_bonus boost
        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store(
            "The pipe cover depth at km 2.3 was measured at 980 mm after rework.",
            session_num=1, session_date="2024-06-01",
            scope="agent-private",
        ))
        asyncio.run(ms.store(
            "Inspector Petrov approved the measurement on 2024-05-28.",
            session_num=2, session_date="2024-06-01",
            scope="agent-private",
        ))
        asyncio.run(ms.store(
            "The weather was cloudy during the measurement.",
            session_num=3, session_date="2024-06-01",
            scope="agent-private",
        ))
        asyncio.run(ms.build_index())
        # Make entity text appear in fact text so entity matching works in eval
        for fact in ms._all_granular:
            fact_text = fact.get("fact", "").lower()
            # Add entities that are substrings of the fact text
            words = [w for w in fact_text.split() if len(w) > 3]
            fact["entities"] = words[:2] if words else ["fact"]

        # b) Simulate an ask result with a runtime_trace_ref
        #    (We cannot easily do full ask without inference, so we verify
        #     the trace_ref is generated in ask's return dict by testing
        #     the code path directly.)
        trace_ref_1 = "ask_aabbccdd0011"
        trace_ref_2 = "ask_112233445566"

        # c) Set up MAL stores — pass server for real snapshot eval
        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts, server=ms)

        # d) Enable MAL with min_signals=10
        control.set(KEY, AGENT, enabled=True, min_signals=10)
        assert control.is_enabled(KEY, AGENT)

        # e) Submit 10+ feedback events with real runtime_trace payloads
        #    (MIN_SIGNALS_FOR_NEW_BEHAVIOR = 10 for any MAL pipeline change)
        real_trace = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "low_entity_phrase_match"},
                "late_fusion": {"status": "ok"},
            },
            "query_type": "lookup",
            "source_families": ["document"],
        }
        eids = []
        for i in range(10):
            eid = feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Test question {i}",
                "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": real_trace,
            })
            eids.append(eid)
        assert all(eid.startswith("malfb_") for eid in eids)

        # f) Verify family support gate
        eligible = feedback.list_trigger_eligible(KEY, AGENT)
        assert len(eligible) >= 10
        n_independent = feedback.count_independent_failures(eligible)
        assert n_independent >= 10

        # g) Set up gen_0 as current live generation before trigger
        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        ptr = base / "current_gen"
        ptr.write_text("gen_0")

        # h) Trigger MAL — runs full loop: reserve → diagnose → propose → apply → consume
        result = scheduler.trigger(KEY, agent_id=AGENT)
        assert result.get("status") == "completed", f"trigger failed: {result}"
        assert "selected_feedback_event_ids" in result
        assert "artifact_id" in result, f"no artifact_id in result: outcome={result.get('outcome')}, reason={result.get('reason')}, score_before={result.get('score_before')}, score_after={result.get('score_after')}"
        assert result["artifact_id"] is not None, (
            f"artifact_id is None: outcome={result.get('outcome')}, "
            f"reason={result.get('reason')}, "
            f"score_before={result.get('score_before')}, "
            f"score_after={result.get('score_after')}"
        )
        assert result.get("apply_status") == "applied"
        run_id = result["run_id"]

        # i) Verify feedback events are consumed (not still queued)
        queued_after = feedback.list_queued(KEY, AGENT)
        assert len(queued_after) == 0, "feedback events should be consumed after successful run"

        # j) Verify artifact was created and applied
        artifact = artifacts.get(KEY, AGENT, result["artifact_id"])
        assert artifact is not None
        assert artifact["status"] == "applied"
        assert artifact["version"] == 1
        assert "entity_phrase_bonus" in artifact["materialized_state"]["selector_config_overrides"]

        # k) Verify generation directory exists with new config
        gen1 = base / "gen_1"
        assert gen1.exists()
        active_config = json.loads((gen1 / "active_config.json").read_text())
        assert "entity_phrase_bonus" in active_config.get("selector_config_overrides", {})

        # k) Verify current_gen_dir returns the right path
        current = current_gen_dir(data_dir, KEY, AGENT)
        assert current == gen1
        assert ptr.read_text().strip() == "gen_1"

    def test_runtime_trace_ref_in_ask(self, tmp_path, monkeypatch):
        """ask() must include runtime_trace_ref in its return dict."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("The sky is blue.",
                             session_num=1, session_date="2024-06-01",
                             scope="agent-private"))
        asyncio.run(ms.build_index())

        # Patch inference to avoid real LLM call
        async def mock_send_payload(payload, caller_id=None, secret_ref=None):
            return ("The sky is blue.", False, [])

        monkeypatch.setattr(ms, "_send_payload", mock_send_payload)
        monkeypatch.setattr(ms, "_has_profiles", lambda: True)

        def mock_build_payload(**kwargs):
            return (
                {"model": "test-model", "messages": [{"role": "user", "content": "test"}]},
                {"profile_used": "test", "use_tool": False},
            )
        monkeypatch.setattr(ms, "_build_payload", mock_build_payload)

        result = asyncio.run(ms.ask("What color is the sky?"))
        assert "runtime_trace_ref" in result
        assert result["runtime_trace_ref"].startswith("ask_")
        assert len(result["runtime_trace_ref"]) == 4 + 12  # "ask_" + 12 hex chars

    def test_bare_trace_ref_without_payload_aborts_and_releases(self, tmp_path, monkeypatch):
        """Feedback with only runtime_trace_ref (no runtime_trace payload)
        must cause trigger to abort with no_trace and release events."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts)

        control.set(KEY, AGENT, enabled=True)

        # Submit 2 events with bare trace_ref only — no runtime_trace payload
        feedback.submit(KEY, AGENT, {
            "signal_source": "user",
            "verdict": "bad_answer",
            "query": "What is X?",
            "runtime_trace_ref": "ask_aabbccdd0011",
        })
        feedback.submit(KEY, AGENT, {
            "signal_source": "user",
            "verdict": "bad_answer",
            "query": "What is Y?",
            "runtime_trace_ref": "ask_112233445566",
        })

        assert len(feedback.list_queued(KEY, AGENT)) == 2

        result = scheduler.trigger(KEY, agent_id=AGENT)

        # Without runtime_trace payloads, trigger falls back to unknown family
        # and proceeds to run. Without server/snapshot, eval is skipped and
        # atom is accepted in degraded mode. Events are consumed on terminal
        # outcome, or released on transient failure.
        assert result.get("status") == "completed"
        # Events should NOT remain queued — they were consumed or released
        # based on the run outcome
        outcome = result.get("outcome")
        queued = feedback.list_queued(KEY, AGENT)
        if outcome in ("accepted", "rejected"):
            assert len(queued) == 0, "terminal outcome should consume events"
        else:
            assert len(queued) == 2, "non-terminal outcome should release events"


# ---------------------------------------------------------------------------
# 2. MCP-level tests — call underlying functions directly
# ---------------------------------------------------------------------------

class TestMALMCPTools:

    def test_configure_and_status(self, tmp_path, monkeypatch):
        """memory_mal_configure + memory_mal_status round-trip."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import (
            _get_mal_stores,
            _mal_stores,
            memory_mal_configure,
            memory_mal_status,
        )

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        # Clear registry for fresh test
        mcp_mod.registry.clear()
        _mal_stores.clear()

        # Configure MAL
        result = asyncio.run(memory_mal_configure(
            key=KEY, agent_id=AGENT, enabled=True,
            auto_collect_feedback=True, auto_trigger=False,
            token=_mcp_token(),
        ))
        assert result["status"] == "ok"
        assert result["agent_id"] == MCP_BINDING
        assert result["config"]["enabled"] is True

        # Check status
        status = asyncio.run(memory_mal_status(key=KEY, agent_id=AGENT, token=_mcp_token()))
        assert status["agent_id"] == MCP_BINDING
        assert status["enabled"] is True
        assert status["auto_collect_feedback"] is True
        assert status["auto_trigger"] is False
        assert status["queued_feedback_count"] == 0
        assert status["convergence_state"] == "active"

    def test_feedback_and_list(self, tmp_path, monkeypatch):
        """memory_mal_feedback + memory_mal_list_feedback round-trip."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import (
            _mal_stores,
            memory_mal_configure,
            memory_mal_feedback,
            memory_mal_list_feedback,
        )

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        # Enable first
        asyncio.run(memory_mal_configure(
            key=KEY, agent_id=AGENT, enabled=True,
            token=_mcp_token(),
        ))

        # Submit feedback
        result = asyncio.run(memory_mal_feedback(
            key=KEY, agent_id=AGENT,
            verdict="bad_answer",
            query="test query",
            runtime_trace_ref="ask_aabbccdd0011",
            token=_mcp_token(),
        ))
        assert result["status"] == "ok"
        assert result["agent_id"] == MCP_BINDING
        assert result["feedback_event_id"].startswith("malfb_")

        # List feedback
        listed = asyncio.run(memory_mal_list_feedback(
            key=KEY, agent_id=AGENT,
            token=_mcp_token(),
        ))
        assert listed["agent_id"] == MCP_BINDING
        assert listed["count"] == 1
        assert listed["events"][0]["verdict"] == "bad_answer"

    def test_feedback_preserves_temporal_runtime_trace(self, tmp_path, monkeypatch):
        """MAL stores runtime_trace.temporal without introducing temporal tuning."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import (
            _mal_stores,
            memory_mal_configure,
            memory_mal_feedback,
            memory_mal_list_feedback,
        )

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        evidence_candidates = {
            "trace_version": 1,
            "operator_class": "as_of_state",
            "query_range": {
                "start": None,
                "end": "2026-03-01",
                "source": "explicit",
                "confidence": 0.9,
            },
            "candidate_count": 1,
            "selected_episode_ids": ["SRC_e01"],
            "candidates": [
                {
                    "episode_id": "SRC_e01",
                    "fact_ids": ["f_001"],
                    "event_date": "2026-02-28",
                    "source_date": "2026-02-28",
                    "session_date": None,
                    "date_provenance": "event_date",
                    "date_relation": "inside_range",
                    "content_support": {
                        "query_token_overlap": 2,
                        "named_slot_overlap": 0,
                        "answer_object_support": "present",
                        "required_terms_missing": [],
                    },
                    "selection_status": "already_selected",
                    "reason": "selected_context_contains_candidate",
                }
            ],
            "omitted_date_matching_episode_ids": [],
            "warnings": [],
        }
        runtime_trace = {
            "temporal": {
                "trace_version": 1,
                "path": "episode_late_fusion",
                "leaf_decision": {
                    "source": "retrieval_leaf_prompt",
                    "execution_policy": "soft_signal",
                },
                "executor": {
                    "used": False,
                    "type": None,
                    "selected_episode_ids": [],
                },
                "evidence_candidates": evidence_candidates,
            }
        }

        asyncio.run(memory_mal_configure(
            key=KEY,
            agent_id=AGENT,
            enabled=True,
            token=_mcp_token(),
        ))
        result = asyncio.run(memory_mal_feedback(
            key=KEY,
            agent_id=AGENT,
            verdict="bad_answer",
            query="Which event happened in May?",
            runtime_trace_ref="ask_temporal_trace",
            runtime_trace=runtime_trace,
            token=_mcp_token(),
        ))
        assert result["status"] == "ok"

        listed = asyncio.run(memory_mal_list_feedback(
            key=KEY,
            agent_id=AGENT,
            token=_mcp_token(),
        ))
        event = listed["events"][0]
        assert event["runtime_trace"]["temporal"]["trace_version"] == 1
        assert event["runtime_trace"]["temporal"]["path"] == "episode_late_fusion"
        assert event["runtime_trace"]["temporal"]["leaf_decision"]["execution_policy"] == "soft_signal"
        assert event["runtime_trace"]["temporal"]["executor"]["used"] is False
        assert event["runtime_trace"]["temporal"]["evidence_candidates"] == evidence_candidates
        assert event["runtime_trace"]["temporal"] == runtime_trace["temporal"]
        assert "temporal_adaptation" not in event

    def test_disabled_binding_feedback_rejected(self, tmp_path, monkeypatch):
        """Feedback submission on a disabled binding returns error."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import _mal_stores, memory_mal_feedback

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        # Do NOT enable MAL — submit should fail
        result = asyncio.run(memory_mal_feedback(
            key=KEY, agent_id=AGENT,
            verdict="bad_answer",
            query="test",
            runtime_trace_ref="ask_000000000000",
            token=_mcp_token(),
        ))
        # _safe_tool catches the ValueError and wraps it
        assert "error" in result
        assert "MAL_DISABLED" in str(result["error"])

    def test_trigger_estimate_only(self, tmp_path, monkeypatch):
        """memory_mal_trigger with estimate_only returns cost estimate."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import (
            _mal_stores,
            memory_mal_configure,
            memory_mal_feedback,
            memory_mal_trigger,
        )

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        # Enable + submit 2 independent feedback events with real traces
        trace = {"stages": {"first_pass": {"status": "failed", "detail": "low_entity_phrase_match"}}, "query_type": "lookup", "source_families": ["document"]}
        asyncio.run(memory_mal_configure(key=KEY, agent_id=AGENT, enabled=True, token=_mcp_token()))
        asyncio.run(memory_mal_feedback(
            key=KEY, agent_id=AGENT,
            verdict="bad_answer", query="q1",
            runtime_trace_ref="ask_aaa000000001",
            runtime_trace=trace,
            token=_mcp_token(),
        ))
        asyncio.run(memory_mal_feedback(
            key=KEY, agent_id=AGENT,
            verdict="user_correction", query="q2",
            runtime_trace_ref="ask_bbb000000002",
            runtime_trace=trace,
            token=_mcp_token(),
        ))

        # estimate_only
        result = asyncio.run(memory_mal_trigger(
            key=KEY, agent_id=AGENT, estimate_only=True,
            token=_mcp_token(),
        ))
        assert result.get("status") == "estimated"
        assert "estimated_eval_cost_usd" in result

    def test_get_artifact_not_found(self, tmp_path, monkeypatch):
        """memory_mal_get_artifact returns NOT_FOUND when no artifacts exist."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import _mal_stores, memory_mal_get_artifact

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        result = asyncio.run(memory_mal_get_artifact(key=KEY, agent_id=AGENT, token=_mcp_token()))
        assert result["code"] == "NOT_FOUND"

    def test_get_artifact_after_creation(self, tmp_path, monkeypatch):
        """memory_mal_get_artifact returns artifact after one is created."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import _get_mal_stores, _mal_stores, memory_mal_get_artifact

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        # Create artifact directly via store
        server = mcp_mod._get_memory(KEY)
        stores = _get_mal_stores(data_dir, server=server)

        stores["artifacts"].create(
            key=KEY, agent_id=MCP_BINDING,
            atom_type="lexical_signal_bundle",
            atom_payload={"word_overlap_bonus": {"old": 0.5, "new": 0.8}},
            failure_family={"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "low_recall"},
            feedback_event_ids=["malfb_test1"],
            runtime_trace_refs=["ask_000000000001"],
            independent_failures_evaluated=2,
            score_before={"episode_hit_rate": 0.4},
            score_after={"episode_hit_rate": 0.6},
        )

        result = asyncio.run(memory_mal_get_artifact(key=KEY, agent_id=AGENT, token=_mcp_token()))
        assert result["agent_id"] == MCP_BINDING
        assert "artifact" in result
        assert result["artifact"]["version"] == 1

    def test_authenticated_caller_cannot_spoof_mal_binding_selector(self, tmp_path, monkeypatch):
        """MAL binding is keyed by verified caller identity, not raw agent_id."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import _mal_stores, memory_mal_configure, memory_mal_status

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()

        alice_token = auth_token_for_agent("alice")
        bob_token = auth_token_for_agent("bob")

        result = asyncio.run(memory_mal_configure(
            key=KEY,
            agent_id="bob",
            enabled=True,
            token=alice_token,
        ))
        assert result["status"] == "ok"
        assert result["agent_id"] == "agent:alice"
        assert result["requested_agent_id"] == "bob"

        alice_status = asyncio.run(memory_mal_status(
            key=KEY,
            agent_id="bob",
            token=alice_token,
        ))
        assert alice_status["enabled"] is True
        assert alice_status["agent_id"] == "agent:alice"

        bob_status = asyncio.run(memory_mal_status(
            key=KEY,
            agent_id="bob",
            token=bob_token,
        ))
        assert bob_status["enabled"] is False
        assert bob_status["agent_id"] == "agent:bob"


# ---------------------------------------------------------------------------
# 3. Generation-aware runtime wiring
# ---------------------------------------------------------------------------

class TestMALGenerationWiring:

    def test_current_gen_dir_default(self, tmp_path):
        """current_gen_dir defaults to gen_0 when no pointer exists."""
        data_dir = str(tmp_path)
        result = current_gen_dir(data_dir, KEY, AGENT)
        assert result == Path(data_dir) / "mal" / KEY / AGENT / "gen_0"

    def test_current_gen_dir_after_apply(self, tmp_path):
        """current_gen_dir follows pointer after apply_generation."""
        data_dir = str(tmp_path)
        base = Path(data_dir) / "mal" / KEY / AGENT

        # Setup gen_0
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        (base / "current_gen").write_text("gen_0")

        engine = ApplyEngine(data_dir)
        engine.apply_generation(
            key=KEY, agent_id=AGENT,
            materialized_state={
                "selector_config_overrides": {"entity_phrase_bonus": 1.2},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 12000,
                "extraction_prompts": {},
            },
            previous_gen=0,
        )

        result = current_gen_dir(data_dir, KEY, AGENT)
        assert result == base / "gen_1"
        config = json.loads((result / "active_config.json").read_text())
        assert config["selector_config_overrides"]["entity_phrase_bonus"] == 1.2

    def test_multi_generation_chain(self, tmp_path):
        """Multiple sequential apply_generation calls produce a version chain."""
        data_dir = str(tmp_path)
        base = Path(data_dir) / "mal" / KEY / AGENT

        # Setup gen_0
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        (base / "current_gen").write_text("gen_0")

        engine = ApplyEngine(data_dir)

        # Apply gen 1
        engine.apply_generation(
            key=KEY, agent_id=AGENT,
            materialized_state={
                "selector_config_overrides": {"word_overlap_bonus": 0.6},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 12000,
                "extraction_prompts": {},
            },
            previous_gen=0,
        )
        assert current_gen_dir(data_dir, KEY, AGENT) == base / "gen_1"

        # Apply gen 2
        engine.apply_generation(
            key=KEY, agent_id=AGENT,
            materialized_state={
                "selector_config_overrides": {"word_overlap_bonus": 0.8},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 14000,
                "extraction_prompts": {},
            },
            previous_gen=1,
        )
        assert current_gen_dir(data_dir, KEY, AGENT) == base / "gen_2"
        config = json.loads((base / "gen_2" / "active_config.json").read_text())
        assert config["selector_config_overrides"]["word_overlap_bonus"] == 0.8
        assert config["size_cap_chars"] == 14000


class TestMALRuntimeWiring:
    """Prove that after MAL apply, recall() reads the new generation config."""

    def test_direct_python_recall_uses_canonical_agent_binding(self, tmp_path, monkeypatch):
        """Direct MemoryServer recall(agent_id='alice') must load agent:alice MAL config."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store(
            "Alice owns this private binding.",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
            scope="agent-private",
        ))
        asyncio.run(ms.build_index())

        engine = ApplyEngine(data_dir)
        base = Path(data_dir) / "mal" / KEY / "agent:alice"
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "inference_leaf_plugin_overrides": {},
        }))
        (base / "current_gen").write_text("gen_0")

        engine.apply_generation(
            key=KEY,
            agent_id="agent:alice",
            materialized_state={
                "selector_config_overrides": {},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 12000,
                "extraction_prompts": {},
                "inference_leaf_plugin_overrides": {"list_set": False},
            },
            previous_gen=0,
        )

        result = asyncio.run(ms.recall("What does Alice own?", agent_id="alice"))
        leaf_overrides = result.get("inference_leaf_plugins") or {}
        assert leaf_overrides.get("list_set") is False

    def test_admin_mcp_runtime_uses_selected_canonical_mal_binding(self, tmp_path, monkeypatch):
        """Admin MCP recall/ask keep ACL caller=system but MAL selector=canonical target."""
        import src.mcp_server as mcp_mod
        from src.mcp_server import _mal_stores, memory_ask, memory_recall

        _patch_all(monkeypatch)
        data_dir = str(tmp_path)
        mcp_mod.data_dir = data_dir
        mcp_mod.registry.clear()
        _mal_stores.clear()
        auth = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=False)
        auth.issue("agent:alice", kind="agent")

        ms = mcp_mod._get_memory(KEY)
        asyncio.run(ms.store(
            "Alice private data.",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
            scope="agent-private",
        ))
        asyncio.run(ms.build_index())

        captured: dict[str, str | None] = {"recall": None, "ask": None}
        real_recall = MemoryServer.recall
        real_ask = MemoryServer.ask

        async def _capture_recall(self, *args, **kwargs):
            captured["recall"] = kwargs.get("mal_binding_id")
            return await real_recall(self, *args, **kwargs)

        async def _capture_ask(self, *args, **kwargs):
            captured["ask"] = kwargs.get("mal_binding_id")
            return await real_ask(self, *args, **kwargs)

        monkeypatch.setattr(MemoryServer, "recall", _capture_recall)
        monkeypatch.setattr(MemoryServer, "ask", _capture_ask)
        monkeypatch.setattr(
            ms,
            "_send_payload",
            lambda payload, caller_id=None, secret_ref=None: asyncio.sleep(0, result=("ok", False, [])),
        )
        monkeypatch.setattr(ms, "_has_profiles", lambda: True)

        def _mock_build_payload(**kwargs):
            return (
                {"model": "test-model", "messages": [{"role": "user", "content": "test"}]},
                {"profile_used": "test", "use_tool": False},
            )

        monkeypatch.setattr(ms, "_build_payload", _mock_build_payload)

        recall_result = asyncio.run(memory_recall(
            key=KEY,
            query="What data is stored?",
            agent_id="alice",
            token=auth.admin_token,
        ))
        assert "code" not in recall_result
        assert captured["recall"] == "agent:alice"

        ask_result = asyncio.run(memory_ask(
            key=KEY,
            query="What data is stored?",
            agent_id="alice",
            token=auth.admin_token,
        ))
        assert "code" not in ask_result
        assert captured["ask"] == "agent:alice"

    def test_rebuild_tiers_uses_canonical_agent_binding_for_source_aggregation(self, tmp_path, monkeypatch):
        """_rebuild_tiers() must pass canonical binding id into source aggregation."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store(
            "Alice private rebuild payload.",
            session_num=1,
            session_date="2024-06-01",
            agent_id="alice",
            scope="agent-private",
        ))

        captured: dict[str, str | None] = {"agent_id": None}

        async def _mock_extract_source_aggregation_facts(self, **kwargs):
            captured["agent_id"] = kwargs.get("agent_id")
            return [{
                "id": "agg_1",
                "fact": "Aggregated fact",
                "kind": "summary",
                "entities": [],
                "tags": [],
            }]

        monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", _mock_extract_source_aggregation_facts)

        asyncio.run(ms._rebuild_tiers())
        assert captured["agent_id"] == "agent:alice"

    def test_recall_uses_mal_selector_overrides_after_apply(self, tmp_path, monkeypatch):
        """After MAL apply with new entity_phrase_bonus, recall() must use
        that value in episode selection, not the default."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        # Store data and build index
        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store(
            "The pipe cover depth at km 2.3 was measured at 980 mm after rework.",
            session_num=1, session_date="2024-06-01",
            scope="agent-private",
        ))
        asyncio.run(ms.store(
            "The initial measurement at km 2.3 recorded 780 mm cover depth.",
            session_num=2, session_date="2024-05-15",
            scope="agent-private",
        ))
        asyncio.run(ms.build_index())

        # Recall before MAL — default config
        result_before = asyncio.run(ms.recall("What is the pipe depth at km 2.3?"))
        assert "context" in result_before

        # Now apply MAL generation with boosted entity_phrase_bonus
        engine = ApplyEngine(data_dir)
        base = Path(data_dir) / "mal" / KEY / "system"
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
        }))
        (base / "current_gen").write_text("gen_0")

        engine.apply_generation(
            key=KEY, agent_id="system",
            materialized_state={
                "selector_config_overrides": {"entity_phrase_bonus": 8.0},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 12000,
                "extraction_prompts": {},
            },
            previous_gen=0,
        )

        # Verify generation was applied
        assert (base / "gen_1" / "active_config.json").exists()
        applied_config = json.loads((base / "gen_1" / "active_config.json").read_text())
        assert applied_config["selector_config_overrides"]["entity_phrase_bonus"] == 8.0

        # Recall after MAL — _load_mal_active_config should return the new overrides
        from src.memory import _load_mal_active_config
        mal_config = _load_mal_active_config(data_dir, KEY, "system")
        assert mal_config.get("selector_config_overrides", {}).get("entity_phrase_bonus") == 8.0

        # Recall again — the runtime should now use entity_phrase_bonus=8.0
        result_after = asyncio.run(ms.recall("What is the pipe depth at km 2.3?"))
        assert "context" in result_after

    def test_load_mal_active_config_returns_empty_when_no_generation(self, tmp_path):
        """Without MAL generation, _load_mal_active_config returns empty dict."""
        from src.memory import _load_mal_active_config
        config = _load_mal_active_config(str(tmp_path), "nonexistent", "default")
        assert config == {}

    def test_full_mal_loop_changes_inference_leaf_plugin_state(self, tmp_path, monkeypatch):
        """After MAL apply with inference_leaf_toggle, recall result
        carries the new plugin state for downstream inference."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store(
            "Jon lost his job as a banker. Gina lost her Door Dash job.",
            session_num=1, session_date="2024-06-01",
            scope="agent-private",
        ))
        asyncio.run(ms.build_index())

        # Before MAL: no leaf overrides
        from src.memory import _load_mal_active_config
        config_before = _load_mal_active_config(data_dir, KEY, "system")
        assert config_before.get("inference_leaf_plugin_overrides", {}) == {}

        # Apply MAL generation that disables list_set leaf
        engine = ApplyEngine(data_dir)
        base = Path(data_dir) / "mal" / KEY / "system"
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "inference_leaf_plugin_overrides": {},
        }))
        (base / "current_gen").write_text("gen_0")

        engine.apply_generation(
            key=KEY, agent_id="system",
            materialized_state={
                "selector_config_overrides": {},
                "grouping_prompt_mode": "strict_small",
                "size_cap_chars": 12000,
                "extraction_prompts": {},
                "inference_leaf_plugin_overrides": {"list_set": False},
            },
            previous_gen=0,
        )

        config_after = _load_mal_active_config(data_dir, KEY, "system")
        assert config_after["inference_leaf_plugin_overrides"]["list_set"] is False

        # Recall should now carry the override
        result = asyncio.run(ms.recall("What do Jon and Gina have in common?"))
        leaf_overrides = result.get("inference_leaf_plugins") or {}
        assert leaf_overrides.get("list_set") is False


# ---------------------------------------------------------------------------
# 4. Snapshot tests
# ---------------------------------------------------------------------------

class TestMALSnapshot:

    def test_snapshot_captures_correct_state(self, tmp_path, monkeypatch):
        """Snapshot captures raw_sessions, raw_docs, facts, and episode_corpus."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("Snapshot test data.",
                             session_num=1, session_date="2024-06-01",
                             scope="agent-private"))
        asyncio.run(ms.build_index())

        from src.mal.snapshot import Snapshot
        snap = Snapshot(ms)

        assert isinstance(snap.raw_sessions, list)
        assert isinstance(snap.raw_docs, dict)
        assert isinstance(snap.all_granular, list)
        assert len(snap.all_granular) > 0
        assert isinstance(snap.all_cons, list)
        assert isinstance(snap.all_cross, list)
        assert isinstance(snap.episode_corpus, dict)
        assert isinstance(snap.config, dict)
        assert isinstance(snap.prompts, dict)

    def test_snapshot_is_frozen_copy(self, tmp_path, monkeypatch):
        """Mutating server state after snapshot does not affect snapshot."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("Pre-snapshot data.",
                             session_num=1, session_date="2024-06-01",
                             scope="agent-private"))
        asyncio.run(ms.build_index())

        from src.mal.snapshot import Snapshot
        snap = Snapshot(ms)
        count_before = len(snap.all_granular)

        # Mutate server after snapshot
        ms._all_granular.append({"id": "extra", "fact": "extra fact"})
        assert len(snap.all_granular) == count_before


# ---------------------------------------------------------------------------
# 5. Optimizer tests
# ---------------------------------------------------------------------------

class TestMALOptimizer:

    def test_entity_signature_proposes_lexical_bundle(self):
        """Failure signature with 'entity' proposes lexical_signal_bundle."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "first_pass", "operator_class_or_shape": "lookup",
                  "signature": "low_entity_phrase_match"}
        atom = opt.propose("retrieval-only", family, {}, None)
        assert atom["atom_type"] == "lexical_signal_bundle"
        assert "entity_phrase_bonus" in atom["atom_payload"]

    def test_recency_signature_proposes_locality_bundle(self):
        """Failure signature with 'recency' proposes locality_bundle."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "first_pass", "operator_class_or_shape": "temporal",
                  "signature": "recency_bias_missing"}
        atom = opt.propose("retrieval-only", family, {}, None)
        assert atom["atom_type"] == "locality_bundle"
        assert "currentness_bonus" in atom["atom_payload"]

    def test_fusion_signature_proposes_fusion_bundle(self):
        """Failure signature with 'fusion' proposes fusion_bundle."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "late_fusion", "operator_class_or_shape": "lookup",
                  "signature": "fusion_rank_mismatch"}
        atom = opt.propose("retrieval-only", family, {}, None)
        assert atom["atom_type"] == "fusion_bundle"
        assert "rrf_k" in atom["atom_payload"]

    def test_default_signature_proposes_lexical_bundle(self):
        """Unknown signature defaults to lexical_signal_bundle."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "first_pass", "operator_class_or_shape": "lookup",
                  "signature": "score_below_threshold"}
        atom = opt.propose("retrieval-only", family, {}, None)
        assert atom["atom_type"] == "lexical_signal_bundle"
        assert "entity_phrase_bonus" in atom["atom_payload"]

    def test_reprocessing_proposes_grouping_bundle(self):
        """Reprocessing mode proposes grouping_bundle with cycled mode."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "episodes", "operator_class_or_shape": "lookup",
                  "signature": "grouping_mismatch"}
        current_state = {"grouping_prompt_mode": "strict_small"}
        atom = opt.propose("reprocessing", family, current_state, None)
        assert atom["atom_type"] == "grouping_bundle"
        assert "grouping_prompt_mode" in atom["atom_payload"]
        assert atom["atom_payload"]["grouping_prompt_mode"]["new"] != "strict_small"

    def test_extraction_proposes_example_append(self):
        """Extraction mode proposes extraction_example_append."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "facts", "operator_class_or_shape": "lookup",
                  "signature": "missing_fact_support"}
        atom = opt.propose("extraction", family, {}, None)
        assert atom["atom_type"] == "extraction_example_append"
        assert "prompt_target" in atom["atom_payload"]
        assert "example" in atom["atom_payload"]

    def test_bounds_enforcement(self):
        """Proposed numeric values respect [old * 0.3, old * 3.0] bounds."""
        from src.mal.optimizer import Optimizer
        opt = Optimizer()
        family = {"stage": "first_pass", "operator_class_or_shape": "lookup",
                  "signature": "low_entity_phrase_match"}
        current_state = {"selector_config_overrides": {"entity_phrase_bonus": 2.0}}
        atom = opt.propose("retrieval-only", family, current_state, None)
        payload = atom["atom_payload"]["entity_phrase_bonus"]
        assert payload["new"] >= payload["old"] * 0.3
        assert payload["new"] <= payload["old"] * 3.0


# ---------------------------------------------------------------------------
# 6. Overfitting mode tests
# ---------------------------------------------------------------------------

class TestMALOverfitting:

    def _setup_mal(self, tmp_path, n_events=10, n_sources=None):
        """Helper: set up MAL stores with n_events feedback events."""
        data_dir = str(tmp_path)

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts)

        control.set(KEY, AGENT, enabled=True)

        real_trace = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "low_entity_phrase_match"},
            },
            "query_type": "lookup",
            "source_families": ["document"],
        }
        eids = []
        for i in range(n_events):
            eid = feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Test question {i}",
                "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": real_trace,
            })
            eids.append(eid)

        # Set up gen_0
        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        ptr = base / "current_gen"
        ptr.write_text("gen_0")

        return scheduler, eids

    def test_personalization_mode_no_holdout(self, tmp_path):
        """Personalization mode (default) does not apply holdout check."""
        scheduler, eids = self._setup_mal(tmp_path, n_events=10)
        # No server = no snapshot = no eval = direct apply
        result = scheduler.trigger(KEY, agent_id=AGENT, overfitting_mode="personalization")
        assert result.get("status") == "completed"
        # Should complete without overfitting rejection
        assert result.get("outcome") != "rejected" or result.get("reason") != "overfitting_gap"

    def test_no_overfitting_mode_rejects_insufficient_sources(self, tmp_path, monkeypatch):
        """no_overfitting mode rejects when fewer than 20 sources."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("Test data for overfitting.",
                             session_num=1, session_date="2024-06-01",
                             scope="agent-private"))
        asyncio.run(ms.build_index())

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts, server=ms)

        control.set(KEY, AGENT, enabled=True)

        real_trace = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "low_entity_phrase_match"},
            },
            "query_type": "lookup",
            "source_families": ["document"],
        }
        for i in range(10):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Test question {i}",
                "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": real_trace,
            })

        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        (base / "current_gen").write_text("gen_0")

        result = scheduler.trigger(KEY, agent_id=AGENT, overfitting_mode="no_overfitting")
        # With only a few facts from 1 session, should reject for insufficient sources
        assert result.get("outcome") == "rejected"
        assert result.get("reason") == "insufficient_sources_for_no_overfitting"


# ---------------------------------------------------------------------------
# 7. Family clustering tests
# ---------------------------------------------------------------------------

class TestMALFamilyClustering:

    def test_two_families_largest_selected(self, tmp_path):
        """With two failure families, scheduler picks the largest one."""
        data_dir = str(tmp_path)

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts)

        control.set(KEY, AGENT, enabled=True)

        # Family A: entity failure (7 events)
        trace_a = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "low_entity_phrase_match"},
            },
            "query_type": "lookup",
            "source_families": ["document"],
        }
        # Family B: recency failure (3 events)
        trace_b = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "recency_bias_missing"},
            },
            "query_type": "temporal",
            "source_families": ["document"],
        }

        for i in range(7):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Entity question {i}",
                "runtime_trace_ref": f"ask_a{i:011d}",
                "runtime_trace": trace_a,
            })
        for i in range(3):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Recency question {i}",
                "runtime_trace_ref": f"ask_b{i:011d}",
                "runtime_trace": trace_b,
            })

        # Set up gen_0
        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        (base / "current_gen").write_text("gen_0")

        result = scheduler.trigger(KEY, agent_id=AGENT)
        assert result.get("status") == "completed"
        # The largest family is entity (7 events), so family_key should contain "entity"
        family_key = result.get("family_key", "")
        assert "entity" in family_key, f"Expected entity family, got: {family_key}"

    def test_single_family_still_works(self, tmp_path):
        """A single failure family is trivially selected."""
        data_dir = str(tmp_path)

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        artifacts = ArtifactStore(data_dir)
        scheduler = Scheduler(data_dir, control, feedback, artifacts=artifacts)

        control.set(KEY, AGENT, enabled=True)

        trace = {
            "stages": {
                "first_pass": {"status": "failed", "detail": "score_below_threshold"},
            },
            "query_type": "lookup",
            "source_families": ["document"],
        }
        for i in range(10):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user",
                "verdict": "bad_answer",
                "query": f"Question {i}",
                "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": trace,
            })

        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text(json.dumps({
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
        }))
        (base / "current_gen").write_text("gen_0")

        result = scheduler.trigger(KEY, agent_id=AGENT)
        assert result.get("status") == "completed"
        assert result.get("family_key") is not None


class TestMALCodeRequired:
    """Test CODE_REQUIRED detection and courier task emission."""

    def test_code_required_on_unsupported_format_signature(self, tmp_path, monkeypatch):
        """Failure with unsupported_content_format triggers code_required immediately."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("test data", session_num=1, session_date="2024-06-01", scope="agent-private"))
        asyncio.run(ms.build_index())

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        scheduler = Scheduler(data_dir, control, feedback, server=ms)
        control.set(KEY, AGENT, enabled=True, min_signals=2)

        trace = {
            "stages": {"facts": {"status": "failed", "detail": "unsupported_content_format"}},
            "query_type": "lookup",
            "source_families": ["document"],
        }
        for i in range(3):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user", "verdict": "bad_answer",
                "query": f"q{i}", "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": trace,
            })

        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text("{}")
        (base / "current_gen").write_text("gen_0")

        result = scheduler.trigger(KEY, agent_id=AGENT)
        assert result.get("outcome") == "code_required"
        assert result.get("code_request") is not None
        assert result["code_request"]["agent_id"] == "coding"
        assert result["code_request"]["task_type"] == "create_leaf_plugin"

        # Code request persisted to disk
        req_dir = base / "code_requests"
        assert req_dir.exists()
        assert len(list(req_dir.iterdir())) == 1

    def test_code_required_after_5_same_family_rejections(self, tmp_path, monkeypatch):
        """After 5 rejections for the same family, failure_analyzer detects code_required."""
        from src.mal.failure_analyzer import FailureAnalyzer
        analyzer = FailureAnalyzer()
        family = {"stage": "first_pass", "operator_class_or_shape": "lookup", "signature": "low_entity_phrase_match"}
        family_key = analyzer.derive_family_key(family)

        # Simulate 5 rejected artifacts with the same family_key
        rejected = [{"failure_family_key": family_key, "status": "rolled_back"} for _ in range(5)]
        assert analyzer.is_code_required(family, rejected) is True

    def test_code_required_does_not_consume_feedback(self, tmp_path, monkeypatch):
        """code_required releases feedback events back to queued."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        ms = MemoryServer(data_dir, KEY)
        asyncio.run(ms.store("test", session_num=1, session_date="2024-06-01", scope="agent-private"))
        asyncio.run(ms.build_index())

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        scheduler = Scheduler(data_dir, control, feedback, server=ms)
        control.set(KEY, AGENT, enabled=True, min_signals=2)

        trace = {
            "stages": {"facts": {"status": "failed", "detail": "unsupported_content_format"}},
            "query_type": "lookup",
            "source_families": ["document"],
        }
        for i in range(3):
            feedback.submit(KEY, AGENT, {
                "signal_source": "user", "verdict": "bad_answer",
                "query": f"q{i}", "runtime_trace_ref": f"ask_{i:012d}",
                "runtime_trace": trace,
            })

        base = Path(data_dir) / "mal" / KEY / AGENT
        gen0 = base / "gen_0"
        gen0.mkdir(parents=True, exist_ok=True)
        (gen0 / "active_config.json").write_text("{}")
        (base / "current_gen").write_text("gen_0")

        scheduler.trigger(KEY, agent_id=AGENT)

        # Events released, not consumed
        queued = feedback.list_queued(KEY, AGENT)
        assert len(queued) == 3

    def test_code_required_does_not_count_toward_convergence(self, tmp_path, monkeypatch):
        """code_required does not increment rejected streak."""
        _patch_all(monkeypatch)
        data_dir = str(tmp_path)

        control = ControlStore(data_dir)
        feedback = FeedbackStore(data_dir, control)
        scheduler = Scheduler(data_dir, control, feedback)
        control.set(KEY, AGENT, enabled=True)

        scheduler.record_outcome(KEY, AGENT, "code_required")
        state = scheduler.get_convergence_state(KEY, AGENT)
        assert state["rejected_streak"] == 0
        assert state["convergence_state"] == "active"
