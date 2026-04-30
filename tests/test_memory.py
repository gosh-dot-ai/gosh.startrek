# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import json
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import src.memory as memory_mod
from src.episode_features import extract_query_features
from src.inference import get_inf_prompt, get_more_context
from src.local_cli_backend import LocalCliTimeoutError
from src.memory import MemoryServer, _augment_commonality_facts, build_hybrid_context

DIM = 3072


def _rand_embs(n, dim=DIM):
    return np.random.randn(n, dim).astype(np.float32)


def _rand_qemb(dim=DIM):
    return np.random.randn(dim).astype(np.float32)


def _fake_extract_result(n_facts=3, **tag_overrides):
    """Return (conv_id, session_num, session_date, facts, tlinks)."""
    facts = []
    for i in range(n_facts):
        f = {
            "id": f"f{i}",
            "fact": f"Test fact number {i}",
            "kind": "event",
            "entities": ["Alice"],
            "tags": ["test"],
            "session": 1,
            "scope": "swarm-shared",
            "agent_id": "default",
            "swarm_id": "default",
        }
        f.update(tag_overrides)
        facts.append(f)
    tlinks = [{"before": "f0", "after": "f1", "signal": "then"}]
    return ("test_conv", 1, "2024-06-01", facts, tlinks)


async def _store(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.store(*args, **kwargs)


async def _write(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.write(*args, **kwargs)


async def _ingest_document(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.ingest_document(*args, **kwargs)


async def _ingest_asserted_facts(ms: MemoryServer, *args, **kwargs):
    kwargs.setdefault("scope", "agent-private")
    return await ms.ingest_asserted_facts(*args, **kwargs)


@pytest.mark.asyncio
async def test_direct_live_write_caller_principal_derives_and_validates_agent_identity(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "direct_live_identity")

    stored = await ms.store(
        "Hello",
        session_num=1,
        session_date="2024-06-01",
        scope="agent-private",
        caller_id="agent:petya",
        caller_principal_kind="agent",
    )
    assert stored["status"] == "ok"
    assert ms._all_granular[0]["agent_id"] == "petya"
    assert ms._raw_sessions[0]["agent_id"] == "petya"

    with pytest.raises(ValueError, match="agent_id"):
        await ms.write(
            message_id="mismatch",
            session_id="s1",
            content="hello",
            content_family="chat",
            timestamp_ms=1712000000000,
            agent_id="default",
            scope="agent-private",
            caller_id="agent:petya",
            caller_principal_kind="agent",
        )

    with pytest.raises(PermissionError, match="agent principal"):
        await ms.ingest_document(
            "Document body",
            source_id="DOC-1",
            scope="agent-private",
            caller_id="user:mitja",
            caller_principal_kind="user",
        )

    with pytest.raises(PermissionError, match="agent principal"):
        await ms.ingest_asserted_facts(
            facts=[{
                "id": "f1",
                "fact": "Imported fact",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": 1,
            }],
            raw_sessions=[{
                "raw_session_id": "rs1",
                "session_num": 1,
                "session_date": "2024-06-01",
                "content": "Imported session",
            }],
            scope="agent-private",
            caller_id="service:ci",
            caller_principal_kind="service",
            enrich_l0=False,
        )


def test_direct_share_uses_canonical_principal_acl_not_agent_id(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "principal_share_acl")

    result = asyncio.run(ms.ingest_asserted_facts(
        facts=[{
            "id": "f1",
            "fact": "Shared directly to Petya",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "Shared directly to Petya",
        }],
        scope="agent-private",
        owner_id="agent:alice",
        read=["agent:petya"],
        write=[],
        enrich_l0=False,
    ))

    assert result["granular_added"] == 1
    fact = ms._all_granular[0]
    assert fact["read"] == ["agent:petya"]
    assert ms._acl_allows(fact, "agent:petya", [], "user")
    assert not ms._acl_allows(fact, "agent:vasya", [], "user")


def test_legacy_conversation_write_log_entry_uses_store_not_document_ingest(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "legacy_conv_write_log")

    result = asyncio.run(
        ms._extract_write_log_entry(
            {
                "message_id": "raw:part-001",
                "session_id": "part-001",
                "content": "User: hello\nAssistant: hi",
                "content_family": "conversation",
                "timestamp_ms": 1712000000000,
                "agent_id": "agent-a",
                "swarm_id": "swarm-a",
                "scope": "swarm-shared",
                "owner_id": "agent:agent-a",
                "read": ["swarm:swarm-a"],
                "write": ["swarm:swarm-a"],
                "metadata": {
                    "logical_source_id": "q45_case",
                    "part_source_id": "q45_case_p001",
                    "part_idx": 1,
                },
            }
        )
    )

    assert result["facts_extracted"] >= 0
    assert not ms._raw_docs
    assert "q45_case" in ms._source_records
    assert ms._source_records["q45_case"]["family"] == "conversation"
    assert all(not source_id.startswith("part-001") for source_id in ms._source_records)
    doc_ids = [str(doc.get("doc_id") or "") for doc in ms._episode_corpus.get("documents", [])]
    assert doc_ids == ["conversation:q45_case"]


@pytest.mark.asyncio
async def test_write_does_not_wait_for_file_lock(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_no_file_lock")

    def _fake_append_write_log(**kwargs):
        return {"message_id": kwargs["message_id"], "extraction_state": "pending", "inserted": True}

    monkeypatch.setattr(ms._storage, "append_write_log", _fake_append_write_log)

    await ms._file_lock.acquire()
    try:
        result = await asyncio.wait_for(
            _write(ms,
                content="hello",
                content_family="chat",
                session_id="s1",
                message_id="m1",
                timestamp_ms=1712000000000,
            ),
            timeout=0.1,
        )
    finally:
        ms._file_lock.release()

    assert result["message_id"] == "m1"
    assert result["inserted"] is True

@pytest.mark.asyncio
async def test_write_remains_available_while_worker_extracts(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_during_extract")
    await _write(ms,
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_extract(entry):
        started.set()
        await release.wait()
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _slow_extract)

    worker_task = asyncio.create_task(ms.process_write_log_once(batch_size=1))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    second = await asyncio.wait_for(
        _write(ms,
            content="second pending write",
            content_family="chat",
            session_id="s2",
            message_id="m2",
            timestamp_ms=1712000001000,
        ),
        timeout=0.5,
    )
    status = ms.write_status("m2")

    release.set()
    processed = await asyncio.wait_for(worker_task, timeout=1.0)

    assert processed == 1
    assert second["inserted"] is True
    assert status is not None
    assert status["extraction_state"] == "pending"



@pytest.mark.asyncio
async def test_write_log_worker_does_not_complete_on_stale_pending_extraction_raw_session(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_retry_pending_extraction")
    await _write(ms,
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )
    async with ms._file_lock:
        ms._raw_sessions.append({
            "raw_session_id": "rs-stale",
            "message_id": "m1",
            "status": "pending_extraction",
        })

    async def _boom(_entry):
        raise RuntimeError("boom")

    monkeypatch.setattr(ms, "_extract_write_log_entry", _boom)

    processed = await ms.process_write_log_once(batch_size=1)
    status = ms.write_status("m1")

    assert processed == 0
    assert status is not None
    assert status["extraction_state"] == "pending"
    assert status["extraction_attempts"] == 1


@pytest.mark.asyncio
async def test_write_log_worker_skips_sync_store_entry_while_store_is_active(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_store_worker_race")

    release = asyncio.Event()
    first_extract_started = asyncio.Event()
    extract_calls: list[int] = []

    async def _blocking_extract_session(**kwargs):
        extract_calls.append(int(kwargs["session_num"]))
        if len(extract_calls) == 1:
            first_extract_started.set()
            await release.wait()
        return _fake_extract_result(1, session=int(kwargs["session_num"]))

    monkeypatch.setattr("src.memory.extract_session", _blocking_extract_session)

    store_task = asyncio.create_task(
        _store(ms,
            content="User: My favorite database is PostgreSQL because I trust MVCC. Assistant: Noted.",
            session_num=1,
            session_date="2026-03-31",
            speakers="User and Assistant",
        )
    )

    await asyncio.wait_for(first_extract_started.wait(), timeout=1.0)
    pending_entries = ms._storage.list_write_log_entries(states=["pending"], order="asc")
    assert len(pending_entries) == 1
    message_id = str(pending_entries[0]["message_id"])
    assert ms._raw_sessions == []

    processed = await asyncio.wait_for(ms.process_write_log_once(batch_size=1), timeout=1.0)
    status = ms.write_status(message_id)

    assert processed == 0
    assert ms._raw_sessions == []
    assert status is not None
    assert status["extraction_state"] == "pending"
    assert extract_calls == [1]

    release.set()
    result = await asyncio.wait_for(store_task, timeout=1.0)

    assert result["facts_extracted"] == 1
    assert len([rs for rs in ms._raw_sessions if rs.get("message_id") == message_id]) == 1
    assert ms.write_status(message_id)["extraction_state"] == "complete"


@pytest.mark.asyncio
async def test_write_log_worker_skips_sync_document_entry_while_ingest_is_active(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_document_worker_race")

    release = asyncio.Event()
    first_extract_started = asyncio.Event()
    extract_calls: list[int] = []

    async def _blocking_extract_session(**kwargs):
        extract_calls.append(int(kwargs["session_num"]))
        if len(extract_calls) == 1:
            first_extract_started.set()
            await release.wait()
        return _fake_extract_result(1, session=int(kwargs["session_num"]))

    async def _mock_group_document(model, source_id, title, source_date, block_dicts, grouping_config, sem):
        from src.memory import build_singleton_episodes

        return (
            build_singleton_episodes(source_id, source_date, block_dicts),
            {"mode": "singleton"},
            "singleton",
        )

    async def _mock_extract_source_aggregation_facts(self, **kwargs):
        return []

    monkeypatch.setattr("src.memory.extract_session", _blocking_extract_session)
    monkeypatch.setattr("src.memory.group_document", _mock_group_document)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", _mock_extract_source_aggregation_facts)

    ingest_task = asyncio.create_task(
        _ingest_document(
            ms,
            content="Document body. " * 400,
            source_id="doc-race-1",
            message_id="doc-race-msg-1",
        )
    )

    await asyncio.wait_for(first_extract_started.wait(), timeout=1.0)
    pending_entries = [
        entry for entry in ms._storage.list_write_log_entries(states=["pending"], order="asc")
        if entry["message_id"] == "doc-race-msg-1"
    ]
    assert len(pending_entries) == 1

    processed = await asyncio.wait_for(ms.process_write_log_once(batch_size=1), timeout=1.0)
    status = ms.write_status("doc-race-msg-1")

    assert processed == 0
    assert status is not None
    assert status["extraction_state"] == "pending"
    assert extract_calls
    assert extract_calls[0] == 1

    release.set()
    result = await asyncio.wait_for(ingest_task, timeout=5.0)

    assert result["facts_extracted"] > 0
    assert ms.write_status("doc-race-msg-1")["extraction_state"] == "complete"


@pytest.mark.asyncio
async def test_write_and_raw_recall_meet_latency_targets_while_worker_busy(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_latency")
    await _write(ms,
        content="first pending write",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
    )

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_extract(entry):
        started.set()
        await release.wait()
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _slow_extract)

    worker_task = asyncio.create_task(ms.process_write_log_once(batch_size=1))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    t0 = time.perf_counter()
    receipt = await _write(ms,
        content="latency kiwi note",
        content_family="chat",
        session_id="s2",
        message_id="m2",
        timestamp_ms=1712000001000,
    )
    write_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    recall = await ms.recall(query="latency kiwi", caller_id="system")
    recall_elapsed = time.perf_counter() - t1

    release.set()
    await asyncio.wait_for(worker_task, timeout=1.0)

    assert receipt["inserted"] is True
    assert write_elapsed < 0.05
    assert recall_elapsed < 0.2
    assert recall["raw_recall_count"] >= 1
    assert "latency kiwi note" in recall["context"].lower()


@pytest.mark.asyncio
async def test_write_worker_prioritizes_chat_then_document_then_artifact(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_priority")
    await _write(ms,
        content="artifact background payload",
        content_family="artifact",
        session_id="artifact-session",
        message_id="artifact-1",
        timestamp_ms=1712000001000,
    )
    await _write(ms,
        content="document payload",
        content_family="document",
        session_id="document-session",
        message_id="document-1",
        timestamp_ms=1712000002000,
    )
    await _write(ms,
        content="chat payload",
        content_family="chat",
        session_id="chat-session",
        message_id="chat-1",
        timestamp_ms=1712000003000,
    )

    seen: list[str] = []

    async def _record(entry):
        seen.append(str(entry["message_id"]))
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _record)

    processed = await ms.process_write_log_once(batch_size=3)

    assert processed == 3
    assert seen == ["chat-1", "document-1", "artifact-1"]


@pytest.mark.asyncio
async def test_write_worker_preserves_chat_chronology_without_explicit_turn_numbers(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "write_priority_chat_order")
    await _write(ms,
        content="older chat payload",
        content_family="chat",
        session_id="chat-session",
        message_id="chat-older",
        timestamp_ms=1712000001000,
    )
    await _write(ms,
        content="newer chat payload",
        content_family="chat",
        session_id="chat-session",
        message_id="chat-newer",
        timestamp_ms=1712000002000,
    )

    processed = await ms.process_write_log_once(batch_size=2)

    assert processed == 2
    chat_rows = sorted(
        [row for row in ms._raw_sessions if row.get("message_id") in {"chat-older", "chat-newer"}],
        key=lambda row: int(row.get("session_num") or 0),
    )
    assert [(row["message_id"], row["session_num"]) for row in chat_rows] == [
        ("chat-older", 1),
        ("chat-newer", 2),
    ]


@pytest.mark.asyncio
async def test_write_worker_priority_still_respects_retry_backoff(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "write_priority_backoff")
    await _write(ms,
        content="chat should back off",
        content_family="chat",
        session_id="chat-session",
        message_id="chat-failed",
        timestamp_ms=1712000003000,
    )
    await _write(ms,
        content="document ready now",
        content_family="document",
        session_id="document-session",
        message_id="document-ready",
        timestamp_ms=1712000002000,
    )
    ms._storage.mark_write_state("chat-failed", "failed", attempts_delta=1)

    seen: list[str] = []

    async def _record(entry):
        seen.append(str(entry["message_id"]))
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms, "_extract_write_log_entry", _record)

    processed = await ms.process_write_log_once(batch_size=1)

    assert processed == 1
    assert seen == ["document-ready"]
    assert ms.write_status("chat-failed")["extraction_state"] == "failed"


@pytest.mark.asyncio
async def test_write_log_durable_claim_prevents_cross_process_duplicate_extraction(tmp_path, monkeypatch):
    ms1 = MemoryServer(str(tmp_path), "write_claim_cross_process")
    for idx in range(3):
        await _write(ms1,
            content=f"claim payload {idx}",
            content_family="chat",
            session_id="claim-session",
            message_id=f"claim-{idx}",
            timestamp_ms=1712000000000 + idx,
        )
    ms2 = MemoryServer(str(tmp_path), "write_claim_cross_process")

    seen: list[str] = []

    async def _record(entry):
        seen.append(str(entry["message_id"]))
        await asyncio.sleep(0.01)
        return {"message_id": entry["message_id"]}

    monkeypatch.setattr(ms1, "_extract_write_log_entry", _record)
    monkeypatch.setattr(ms2, "_extract_write_log_entry", _record)

    processed = await asyncio.gather(
        ms1.process_write_log_once(batch_size=3),
        ms2.process_write_log_once(batch_size=3),
    )

    assert sum(processed) == 3
    assert sorted(seen) == ["claim-0", "claim-1", "claim-2"]
    assert len(seen) == len(set(seen))
    for idx in range(3):
        status = ms1.write_status(f"claim-{idx}")
        assert status["extraction_state"] == "complete"
        assert status["lease_owner"] is None
        assert status["lease_expires_at_ms"] is None


def test_write_log_claim_lease_reclaims_only_expired_in_progress(tmp_path):
    ms = MemoryServer(str(tmp_path), "write_claim_lease")
    asyncio.run(_write(ms,
        content="lease payload",
        content_family="chat",
        session_id="lease-session",
        message_id="lease-1",
        timestamp_ms=1712000000000,
    ))

    first = ms._storage.claim_write_log_entries(
        worker_id="worker-a",
        batch_size=1,
        now_ms=1000,
        lease_ms=10_000,
        retry_backoff_ms=0,
        max_attempts=3,
    )
    second = ms._storage.claim_write_log_entries(
        worker_id="worker-b",
        batch_size=1,
        now_ms=2000,
        lease_ms=10_000,
        retry_backoff_ms=0,
        max_attempts=3,
    )
    reclaimed = ms._storage.claim_write_log_entries(
        worker_id="worker-c",
        batch_size=1,
        now_ms=12_000,
        lease_ms=10_000,
        retry_backoff_ms=0,
        max_attempts=3,
    )

    assert [row["message_id"] for row in first] == ["lease-1"]
    assert second == []
    assert [row["message_id"] for row in reclaimed] == ["lease-1"]
    assert reclaimed[0]["lease_reclaimed"] is True
    assert reclaimed[0]["extraction_attempts"] == 2

    ms._storage.mark_write_state("lease-1", "complete")
    status = ms.write_status("lease-1")
    assert status["lease_owner"] is None
    assert status["lease_expires_at_ms"] is None


@pytest.mark.asyncio
async def test_full_index_build_lease_single_flight_across_servers(tmp_path, monkeypatch):
    ms1 = MemoryServer(str(tmp_path), "index_lease_singleflight")
    ms1._all_granular = [{
        "id": "f1",
        "fact": "single flight kiwi fact",
        "kind": "fact",
        "entities": ["kiwi"],
        "tags": [],
        "session": 1,
        "scope": "agent-private",
        "agent_id": "default",
        "swarm_id": "default",
        "status": "active",
    }]
    ms1._save_cache()
    ms1._mark_full_index_dirty()
    ms2 = MemoryServer(str(tmp_path), "index_lease_singleflight")

    started = asyncio.Event()
    release = asyncio.Event()
    embed_calls: list[str] = []

    async def _embed(self, texts, **kwargs):
        embed_calls.append(str(kwargs.get("label") or ""))
        if self is ms1:
            started.set()
            await release.wait()
        return np.ones((len(texts), 8), dtype=np.float32)

    async def _query(self, text, **kwargs):
        return np.ones((8,), dtype=np.float32)

    monkeypatch.setattr(MemoryServer, "_embed_texts_with_runtime_secrets", _embed)
    monkeypatch.setattr(MemoryServer, "_embed_query_with_runtime_secrets", _query)

    task = asyncio.create_task(ms1.build_index())
    await asyncio.wait_for(started.wait(), timeout=1.0)
    skipped = await ms2.build_index()
    release.set()
    built = await asyncio.wait_for(task, timeout=1.0)

    assert skipped["status"] == "skipped"
    assert skipped["index_state"] == "building"
    assert built["granular"] == 1
    assert sum(1 for label in embed_calls if label.startswith("gran-")) == 1


@pytest.mark.asyncio
async def test_index_build_failure_enters_backoff_and_scheduler_skips(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "index_backoff")
    ms._all_granular = [{
        "id": "f1",
        "fact": "backoff kiwi fact",
        "kind": "fact",
        "entities": ["kiwi"],
        "tags": [],
        "session": 1,
        "scope": "agent-private",
        "agent_id": "default",
        "swarm_id": "default",
        "status": "active",
    }]
    ms._save_cache()
    ms._mark_full_index_dirty()

    async def _boom(self, texts, **kwargs):
        raise RuntimeError("provider rate limit")

    monkeypatch.setattr(MemoryServer, "_embed_texts_with_runtime_secrets", _boom)

    with pytest.raises(RuntimeError, match="provider rate limit"):
        await ms.build_index()

    status = ms.stats()["index_status"]
    assert status["index_state"] == "backoff"
    assert status["next_index_retry_after_ms"] is not None

    tick = await ms.run_index_scheduler_once()
    assert tick["status"] == "skipped"
    assert tick["reason"] == "index_backoff"


def test_index_dirty_scheduler_debounces_and_records_dirty_during_build(tmp_path):
    ms = MemoryServer(str(tmp_path), "index_debounce")

    for offset in range(10):
        status = ms._storage.mark_index_dirty(
            now_ms=1000 + offset,
            debounce_ms=5000,
            max_delay_ms=60_000,
        )

    assert status["index_dirty"] is True
    assert status["index_dirty_since_ms"] == 1000
    assert status["last_index_dirty_ms"] == 1009
    assert status["next_index_build_after_ms"] == 6009

    lease = ms._storage.acquire_index_build_lease(
        worker_id="builder",
        snapshot_fingerprint="fp1",
        now_ms=7000,
        lease_ms=30_000,
    )
    assert lease["acquired"] is True

    dirty_during_build = ms._storage.mark_index_dirty(
        now_ms=7100,
        debounce_ms=5000,
        max_delay_ms=60_000,
    )
    assert dirty_during_build["dirty_after_build"] is True

    released = ms._storage.release_index_build_lease(
        worker_id="builder",
        success=True,
        now_ms=7200,
        debounce_ms=5000,
        max_delay_ms=60_000,
    )
    assert released["released"] is True
    assert released["index_state"] == "scheduled"
    assert released["dirty_after_build"] is False
    assert released["next_index_build_after_ms"] == 12_200


@pytest.mark.asyncio
async def test_recall_exact_fact_uses_priority_retrieval_without_full_rebuild(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "priority_recall_exact")
    ms._all_granular = [{
        "id": "f1",
        "fact": "priority kiwi code is ZX-4817",
        "kind": "fact",
        "entities": ["kiwi"],
        "tags": [],
        "session": 1,
        "scope": "agent-private",
        "agent_id": "default",
        "swarm_id": "default",
        "status": "active",
    }]
    ms._n_sessions = 1
    ms._save_cache()
    ms._mark_full_index_dirty()

    async def _canonical(self, query):
        return {
            "raw_original": query,
            "source_lang": "en",
            "canonical_en": query,
            "semantic_ready": True,
            "canonicalization_status": "identity",
            "canonicalization_error": None,
            "translation_version": "test",
        }

    async def _embed_query(self, text, **kwargs):
        return np.ones((8,), dtype=np.float32)

    async def _no_full_build(*args, **kwargs):
        raise AssertionError("recall must not launch full build_index")

    monkeypatch.setattr(MemoryServer, "_canonicalize_recall_query", _canonical)
    monkeypatch.setattr(MemoryServer, "_embed_query_with_runtime_secrets", _embed_query)
    monkeypatch.setattr(ms, "build_index", _no_full_build)

    result = await ms.recall("priority kiwi code is ZX-4817", caller_id="system")

    assert "priority kiwi code is ZX-4817" in result["context"]
    priority_trace = result["runtime_trace"]["priority_retrieval"]
    assert priority_trace["priority_retrieval_used"] is True
    assert priority_trace["priority_index_used"] is False
    assert priority_trace["priority_index_candidate_count"] == 1
    assert priority_trace["full_index_build_scheduled_after_recall"] is True


@pytest.mark.asyncio
async def test_write_rejects_non_string_ingress_metadata(tmp_path):
    ms = MemoryServer(str(tmp_path), "write_metadata_invalid")

    with pytest.raises(ValueError, match="metadata.count: expected string or list of strings"):
        await _write(ms,
            content="invalid metadata",
            content_family="chat",
            session_id="s1",
            message_id="m1",
            timestamp_ms=1712000000000,
            metadata={"count": 3},
        )

    with pytest.raises(ValueError, match="metadata.flags: lists must contain only strings"):
        await _write(ms,
            content="invalid metadata",
            content_family="chat",
            session_id="s2",
            message_id="m2",
            timestamp_ms=1712000000001,
            metadata={"flags": ["ok", 7]},
        )


@pytest.mark.asyncio
async def test_write_accepts_string_and_string_list_ingress_metadata(tmp_path):
    ms = MemoryServer(str(tmp_path), "write_metadata_valid")

    result = await _write(ms,
        content="valid metadata",
        content_family="chat",
        session_id="s1",
        message_id="m1",
        timestamp_ms=1712000000000,
        metadata={"role": "user", "tags": ["alpha", "beta"]},
    )

    assert result["inserted"] is True
    status = ms.write_status("m1")
    assert status["extraction_state"] == "pending"


@pytest.mark.asyncio
async def test_write_coerces_numeric_control_metadata_fields(tmp_path):
    ms = MemoryServer(str(tmp_path), "write_metadata_controls")

    result = await _write(ms,
        content="valid metadata",
        content_family="chat",
        session_id="s1",
        message_id="m-control",
        timestamp_ms=1712000000002,
        metadata={"turn_number": 2, "part_idx": 1, "role": "user"},
    )

    assert result["inserted"] is True
    status = ms.write_status("m-control")
    assert status["metadata"]["turn_number"] == "2"
    assert status["metadata"]["part_idx"] == "1"


@pytest.mark.asyncio
async def test_store_exact_content_dedup_blocks_across_source_ids_with_normalization(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "content_dedup_cross_source")

    first = await _store(ms, "User: Cafe\u0301 plans", 1, "2024-06-01", source_id="SRC-A")
    second = await _store(ms, "\ufeffUser: Caf\u00e9 plans", 1, "2024-06-02", source_id="SRC-B")

    assert first["status"] == "ok"
    assert second["status"] == "duplicate"
    assert second["duplicate_of"]["message_id"] == ms._raw_sessions[0]["message_id"]
    assert len(ms._raw_sessions) == 1


def test_conversation_exact_dedup_ignores_raw_only_candidate(tmp_path):
    ms = MemoryServer(str(tmp_path), "raw_only_not_duplicate")
    ms._raw_sessions = [{
        "raw_session_id": "raw-zero",
        "message_id": "msg-zero",
        "source_id": "SRC-ZERO",
        "session_num": 1,
        "status": "active",
        "format": "conversation",
        "content": "raw only duplicate text",
        "scope": "swarm-shared",
        "owner_id": "system",
        "swarm_id": "default",
    }]
    ms._index_content_entry(
        message_id="msg-zero",
        source_id="SRC-ZERO",
        session_num=1,
        stored_at="2024-06-01T00:00:00+00:00",
        scope="swarm-shared",
        owner_id="system",
        swarm_id="default",
        family="conversation",
        content="raw only duplicate text",
    )

    assert ms._find_exact_duplicate(
        content="raw only duplicate text",
        family="conversation",
        scope="swarm-shared",
        owner_id="system",
        swarm_id="default",
    ) is None


def test_conversation_exact_dedup_requires_active_semantic_evidence(tmp_path):
    ms = MemoryServer(str(tmp_path), "semantic_duplicate_links")
    base_raw = {
        "raw_session_id": "raw-active",
        "message_id": "msg-active",
        "source_id": "SRC-ACTIVE",
        "session_num": 3,
        "status": "active",
        "format": "conversation",
        "content": "semantic duplicate text",
        "scope": "swarm-shared",
        "owner_id": "system",
        "swarm_id": "default",
    }
    for fact in (
        {"id": "by-raw", "raw_session_id": "raw-active", "status": "active"},
        {"id": "by-message", "message_id": "msg-active", "status": "active"},
        {"id": "by-legacy-tuple", "source_id": "SRC-ACTIVE", "session": 3, "status": "active"},
    ):
        ms._raw_sessions = [dict(base_raw)]
        ms._all_granular = [dict(fact)]
        ms._content_dedup_index = {}
        ms._simhash_index = {}
        ms._index_content_entry(
            message_id="msg-active",
            source_id="SRC-ACTIVE",
            session_num=3,
            stored_at="2024-06-01T00:00:00+00:00",
            scope="swarm-shared",
            owner_id="system",
            swarm_id="default",
            family="conversation",
            content="semantic duplicate text",
        )
        duplicate = ms._find_exact_duplicate(
            content="semantic duplicate text",
            family="conversation",
            scope="swarm-shared",
            owner_id="system",
            swarm_id="default",
        )
        assert duplicate is not None, fact["id"]
        assert duplicate["message_id"] == "msg-active"

    ms._all_granular = [{"id": "retracted", "raw_session_id": "raw-active", "status": "retracted"}]
    assert ms._find_exact_duplicate(
        content="semantic duplicate text",
        family="conversation",
        scope="swarm-shared",
        owner_id="system",
        swarm_id="default",
    ) is None


@pytest.mark.asyncio
async def test_store_exact_content_dedup_isolated_by_acl_domain(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "content_dedup_acl")

    first = await _store(ms,
        "Private duplicate candidate",
        1,
        "2024-06-01",
        source_id="SRC-A",
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
    )
    second = await _store(ms,
        "Private duplicate candidate",
        1,
        "2024-06-01",
        source_id="SRC-B",
        agent_id="agent-b",
        swarm_id="sw1",
        scope="agent-private",
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len(ms._raw_sessions) == 2


@pytest.mark.asyncio
async def test_store_source_versioning_isolated_by_acl_domain(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "versioning_acl_conversation")

    first = await _store(ms,
        "Private source content A",
        1,
        "2024-06-01",
        source_id="SHARED-SOURCE",
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
    )
    second = await _store(ms,
        "Private source content B",
        1,
        "2024-06-02",
        source_id="SHARED-SOURCE",
        agent_id="agent-b",
        swarm_id="sw1",
        scope="agent-private",
    )

    active_rows = [rs for rs in ms._raw_sessions if rs.get("status") == "active"]
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len(active_rows) == 2
    assert len({row["artifact_id"] for row in active_rows}) == 2
    assert {row["owner_id"] for row in active_rows} == {"agent:agent-a", "agent:agent-b"}

    reloaded = MemoryServer(str(tmp_path), "versioning_acl_conversation")
    reloaded_active = [rs for rs in reloaded._raw_sessions if rs.get("status") == "active"]
    assert len(reloaded_active) == 2
    assert len({row["source_id"] for row in reloaded_active}) == 2
    assert len({fact["source_id"] for fact in reloaded._all_granular if fact.get("status") == "active"}) == 2
    assert len(reloaded._episode_corpus.get("documents", [])) == 2


@pytest.mark.asyncio
async def test_build_hybrid_context_uses_raw_session_identity_after_acl_reopen(tmp_path, monkeypatch):
    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        session_text = kwargs.get("session_text", "")
        return (
            "test_conv",
            sn,
            kwargs.get("session_date", "2024-06-01"),
            [{
                "id": "f0",
                "fact": f"Fact from {session_text}",
                "kind": "event",
                "entities": ["Alice"],
                "tags": ["test"],
                "session": sn,
            }],
            [],
        )

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        return []

    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)

    ms = MemoryServer(str(tmp_path), "hybrid_acl_context")
    await _store(ms,
        "CONTENT_A",
        1,
        "2024-06-01",
        source_id="SRC",
        agent_id="agent-a",
        swarm_id="sw1",
        scope="agent-private",
    )
    await _store(ms,
        "CONTENT_B",
        1,
        "2024-06-02",
        source_id="SRC",
        agent_id="agent-b",
        swarm_id="sw1",
        scope="agent-private",
    )

    reloaded = MemoryServer(str(tmp_path), "hybrid_acl_context")
    facts = [
        fact for fact in reloaded._all_granular
        if fact.get("owner_id") == "agent:agent-b" and fact.get("status", "active") == "active"
    ]

    context = build_hybrid_context(facts, reloaded._raw_sessions, budget=5000, raw_docs=reloaded._raw_docs)

    assert "CONTENT_B" in context
    assert "CONTENT_A" not in context


@pytest.mark.asyncio
async def test_store_near_duplicate_returns_warning_without_blocking(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "content_near_dup")
    base = "alpha beta gamma delta epsilon " * 20
    variant = base.replace("gamma", "gammx", 1)

    first = await _store(ms, base, 1, "2024-06-01", source_id="SRC-A")
    second = await _store(ms, variant, 1, "2024-06-02", source_id="SRC-B")

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["facts_extracted"] == 1
    assert second["near_duplicate_warning"]["similar_to_message_id"] == ms._raw_sessions[0]["message_id"]


@pytest.mark.asyncio
async def test_write_worker_marks_exact_duplicate_without_creating_second_raw_session(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "worker_exact_duplicate")
    original = await _store(ms, "User: normalized duplicate", 1, "2024-06-01", source_id="SRC-A")
    assert original["status"] == "ok"

    await _write(ms,
        content="\ufeffUser: normalized duplicate",
        content_family="chat",
        session_id="async-1",
        message_id="write-dup-1",
        timestamp_ms=1712000000000,
        metadata={"source_id": "SRC-B", "session_date": "2024-06-02"},
    )

    processed = await ms.process_write_log_once(batch_size=1)

    assert processed == 1
    status = ms.write_status("write-dup-1")
    assert status["extraction_state"] == "complete"
    assert status["metadata"]["duplicate_of"] == ms._raw_sessions[0]["message_id"]
    assert not any(rs.get("message_id") == "write-dup-1" for rs in ms._raw_sessions)


@pytest.mark.asyncio
async def test_write_worker_marks_near_duplicate_and_keeps_write(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "worker_near_duplicate")
    base = "alpha beta gamma delta epsilon " * 20
    variant = base.replace("gamma", "gammx", 1)

    first = await _store(ms, base, 1, "2024-06-01", source_id="SRC-A")
    assert first["status"] == "ok"

    await _write(ms,
        content=variant,
        content_family="chat",
        session_id="async-2",
        message_id="write-near-1",
        timestamp_ms=1712000001000,
        metadata={"source_id": "SRC-B", "session_date": "2024-06-03"},
    )

    processed = await ms.process_write_log_once(batch_size=1)

    assert processed == 1
    status = ms.write_status("write-near-1")
    assert status["extraction_state"] == "complete"
    assert status["metadata"]["near_duplicate_of"] == ms._raw_sessions[0]["message_id"]
    assert any(rs.get("message_id") == "write-near-1" for rs in ms._raw_sessions)


@pytest.mark.asyncio
async def test_write_worker_completes_terminal_canonicalization_failure(tmp_path, monkeypatch):
    async def _invalid_canonicalization(self, model, system, user_msg, max_tokens, sem=None):
        return {"oops": "bad-source-result"}

    monkeypatch.setattr(MemoryServer, "_call_extract_with_runtime_secrets", _invalid_canonicalization)
    ms = MemoryServer(str(tmp_path), "worker_canonicalization_failure")

    await _write(ms,
        content="Привет, это русскоязычная запись, которую надо канонизировать.",
        content_family="chat",
        session_id="s1",
        message_id="write-canon-fail-1",
        timestamp_ms=1712000000000,
        metadata={"source_id": "SRC-CANON", "session_date": "2024-06-01"},
    )

    processed = await ms.process_write_log_once(batch_size=1)
    status = ms.write_status("write-canon-fail-1")
    raw = next(rs for rs in ms._raw_sessions if rs.get("message_id") == "write-canon-fail-1")

    assert processed == 1
    assert status["extraction_state"] == "complete"
    assert status["extraction_attempts"] == 1
    assert status["metadata"]["terminal_error_code"] == "CANONICALIZATION_ERROR"
    assert status["metadata"]["canonicalization_status"] == "failed"
    assert "invalid canonical_en" in status["metadata"]["canonicalization_error"]
    assert raw["status"] == "canonicalization_failed"
    assert raw["semantic_ready"] is False
    assert raw["canonical_en"] == ""
    assert ms._all_granular == []

    assert await ms.process_write_log_once(batch_size=1) == 0


def test_short_only_store_does_not_rebuild_simhash_index_on_reopen(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    server = MemoryServer(str(tmp_path), "short_reopen_no_rebuild")
    asyncio.run(_store(server, "short text", 1, "2024-06-01", source_id="SRC-SHORT"))

    calls = 0
    original = MemoryServer._rebuild_content_dedup_indices

    def _wrapped(self, *, persist):
        nonlocal calls
        calls += 1
        return original(self, persist=persist)

    monkeypatch.setattr(MemoryServer, "_rebuild_content_dedup_indices", _wrapped)
    reopened = MemoryServer(str(tmp_path), "short_reopen_no_rebuild")

    assert reopened._content_dedup_index
    assert reopened._simhash_index == {}
    assert calls == 0


def test_store_write_through_does_not_call_save_facts(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "store_write_through_no_snapshot")

    def _boom(_payload):
        raise AssertionError("live store() should not depend on save_facts()")

    monkeypatch.setattr(ms._storage, "save_facts", _boom)

    result = asyncio.run(
        _store(ms,
            "User: I prefer SQLite for local state. Assistant: noted.",
            session_num=1,
            session_date="2026-04-05",
        )
    )

    assert result["facts_extracted"] == 3
    assert ms._raw_sessions


def test_ingest_document_write_through_does_not_call_save_facts(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "doc_write_through_no_snapshot")

    def _boom(_payload):
        raise AssertionError("live ingest_document() should not depend on save_facts()")

    monkeypatch.setattr(ms._storage, "save_facts", _boom)

    result = asyncio.run(
        _ingest_document(ms,
            content="First section. " * 300,
            source_id="DOC-SQLITE",
        )
    )

    assert result["facts_extracted"] > 0
    assert any(rs.get("source_id") == "DOC-SQLITE" for rs in ms._raw_sessions)


def test_ingest_document_source_versioning_isolated_by_acl_domain(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "versioning_acl_document")

    first = asyncio.run(
        _ingest_document(ms,
            content="Document body one. " * 120,
            source_id="DOC-SHARED",
            agent_id="agent-a",
            swarm_id="sw1",
            scope="agent-private",
        )
    )
    second = asyncio.run(
        _ingest_document(ms,
            content="Document body two. " * 120,
            source_id="DOC-SHARED",
            agent_id="agent-b",
            swarm_id="sw1",
            scope="agent-private",
        )
    )

    active_rows = [
        rs for rs in ms._raw_sessions
        if rs.get("format") == "document" and rs.get("status") == "active"
    ]
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len({row["artifact_id"] for row in active_rows}) == 2
    assert {row["owner_id"] for row in active_rows} == {"agent:agent-a", "agent:agent-b"}
    assert len({row["source_id"] for row in active_rows}) == 2

    reloaded = MemoryServer(str(tmp_path), "versioning_acl_document")
    reloaded_active = [
        rs for rs in reloaded._raw_sessions
        if rs.get("format") == "document" and rs.get("status") == "active"
    ]
    assert len(reloaded._raw_docs) == 2
    assert len({row["source_id"] for row in reloaded_active}) == 2
    assert len({fact["source_id"] for fact in reloaded._all_granular if fact.get("status") == "active"}) == 2
# ── Patches ──

def _patch_extraction(monkeypatch, n_facts=3, **tag_overrides):
    """Patch extract_session and source aggregation."""

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        return _fake_extract_result(n_facts, session=sn, **tag_overrides)

    async def mock_session_merge_stub(**kwargs):
        return ("test_conv", 1, "2024-06-01", [
            {"id": "c0", "fact": "Consolidated fact 0", "kind": "summary",
             "entities": ["Alice"], "tags": ["test"]},
        ])

    async def mock_cross_merge_stub(**kwargs):
        return ("test_conv", "alice", [
            {"id": "x0", "fact": "Cross-session fact about Alice", "kind": "profile",
             "entities": ["Alice"], "tags": ["test"]},
        ])

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        source_facts = kwargs.get("source_facts", [])
        source_id = kwargs.get("source_id", "source")
        if not source_facts:
            return []
        return [{
            "id": "xf0",
            "fact": f"Source aggregate fact for {source_id}",
            "kind": "fact",
            "entities": ["Alice"],
            "tags": ["substrate"],
            "source_ids": [f["id"] for f in source_facts],
            "metadata": {"source_aggregation": True},
        }]

    async def mock_group_document(model, source_id, title, source_date, block_dicts, grouping_config, sem):
        from src.memory import build_singleton_episodes

        return build_singleton_episodes(source_id, source_date, block_dicts), {"mode": "singleton"}, "singleton"

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr("src.memory.group_document", mock_group_document)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)


def _patch_embeddings(monkeypatch):
    """Patch embed_texts and embed_query to return random arrays.

    embed_texts and embed_query are async, so mocks must return coroutines.
    """

    async def mock_embed_texts(texts, **kwargs):
        return _rand_embs(len(texts))

    async def mock_embed_query(text, **kwargs):
        return _rand_qemb()

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)


def _patch_resolve_supersession(monkeypatch):
    """Patch resolve_supersession to be a no-op."""
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)


def _patch_all(monkeypatch, n_facts=3, **tag_overrides):
    _patch_extraction(monkeypatch, n_facts, **tag_overrides)
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)


def _pricing():
    return {
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "reasoning_per_1k": 0.0,
        "cache_read_per_1k": 0.0,
        "cache_write_per_1k": 0.0,
    }


def _planning_profiles(secret_ref: dict | None = None):
    profile = {"model": "openai/gpt-4o-mini", "pricing": _pricing()}
    if secret_ref is not None:
        profile["secret_ref"] = secret_ref
    return (
        {1: "fast", 2: "fast", 3: "fast", 4: "fast", 5: "fast"},
        {"fast": profile},
    )


@pytest.mark.asyncio
async def test_recall_is_evidence_only_and_does_not_build_payload_or_call_models(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    profiles, profile_configs = _planning_profiles()
    ms = MemoryServer(str(tmp_path), "recall_evidence_only", profiles=profiles, profile_configs=profile_configs)
    await _store(
        ms,
        "User: The release color is blue.\nAssistant: Stored.",
        session_num=1,
        session_date="2024-06-01",
        agent_id="tester",
    )

    def fail_build_payload(**kwargs):
        raise AssertionError("_build_payload must not be called by memory_recall")

    async def fail_extract(*args, **kwargs):
        raise AssertionError("_call_extract_with_runtime_secrets must not be called by English memory_recall")

    async def fail_model(*args, **kwargs):
        raise AssertionError("_call_model_with_runtime_secrets must not be called by memory_recall")

    monkeypatch.setattr(ms, "_build_payload", fail_build_payload)
    monkeypatch.setattr(ms, "_call_extract_with_runtime_secrets", fail_extract)
    monkeypatch.setattr(ms, "_call_model_with_runtime_secrets", fail_model)

    result = await ms.recall("What color is the release?", agent_id="tester")

    assert "context" in result
    for field in ("recommended_profile", "payload", "payload_meta", "_payload_secret_ref"):
        assert field not in result
    assert result["runtime_trace"]["query_language"]["source_lang"] == "en"


@pytest.mark.asyncio
async def test_recall_with_profile_configs_but_no_profile_map_does_not_error(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    _profiles, profile_configs = _planning_profiles()
    ms = MemoryServer(str(tmp_path), "recall_profile_configs_only", profiles=None, profile_configs=profile_configs)
    await _store(
        ms,
        "User: The runtime color is amber.\nAssistant: Stored.",
        session_num=1,
        session_date="2024-06-01",
        agent_id="tester",
    )

    result = await ms.recall("What color is the runtime?", agent_id="tester")

    assert "RECALL_ERROR" not in str(result.get("code") or "")
    assert "context" in result
    assert result["runtime_trace"]["evidence_context"]["finalized"] is False
    assert result["runtime_trace"]["evidence_context"]["reason"] == "no_inference_target"


@pytest.mark.asyncio
async def test_recall_non_english_query_blocks_without_extraction_call(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "recall_non_english")

    async def fail_extract(*args, **kwargs):
        raise AssertionError("_call_extract_with_runtime_secrets must not translate memory_recall")

    monkeypatch.setattr(ms, "_call_extract_with_runtime_secrets", fail_extract)

    result = await ms.recall("Что мы решили про релиз?")

    assert result["code"] == "NON_ENGLISH_QUERY"
    assert result["error"] == "memory_recall accepts English queries only; translate in the calling agent/model"
    assert result["runtime_trace"]["query_language"]["source_lang"] == "non_en"
    assert result["runtime_trace"]["query_language"]["canonicalization_status"] == "blocked_in_recall"


@pytest.mark.asyncio
async def test_plan_inference_non_english_query_returns_error_without_payload(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "plan_non_english", profiles=profiles, profile_configs=profile_configs)

    def fail_build_payload(**kwargs):
        raise AssertionError("_build_payload must not run after failed memory_recall")

    monkeypatch.setattr(ms, "_build_payload", fail_build_payload)

    result = await ms.plan_inference("Что мы решили про релиз?")

    assert result["code"] == "NON_ENGLISH_QUERY"
    assert result["error"] == "memory_recall accepts English queries only; translate in the calling agent/model"
    assert "payload" not in result
    assert "payload_meta" not in result
    assert "secret_ref" not in result


@pytest.mark.asyncio
async def test_ask_non_english_query_returns_error_without_payload(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "ask_non_english", profiles=profiles, profile_configs=profile_configs)

    def fail_build_payload(**kwargs):
        raise AssertionError("_build_payload must not run after failed memory_recall")

    async def fail_model(*args, **kwargs):
        raise AssertionError("_call_model_with_runtime_secrets must not run after failed memory_recall")

    monkeypatch.setattr(ms, "_build_payload", fail_build_payload)
    monkeypatch.setattr(ms, "_call_model_with_runtime_secrets", fail_model)

    result = await ms.ask("Что мы решили про релиз?")

    assert result["code"] == "NON_ENGLISH_QUERY"
    assert result["error"] == "memory_recall accepts English queries only; translate in the calling agent/model"
    assert "payload" not in result
    assert "payload_meta" not in result


@pytest.mark.asyncio
async def test_plan_inference_returns_payload_and_opaque_secret_ref(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "plan_inference", profiles=profiles, profile_configs=profile_configs)
    await _store(
        ms,
        "User: The launch decision is to use SQLite.\nAssistant: Stored.",
        session_num=1,
        session_date="2024-06-01",
        agent_id="tester",
    )

    plan = await ms.plan_inference("What database did we choose?", agent_id="tester")

    assert plan["recommended_profile"] == "fast"
    assert plan["payload"]["model"] == "openai/gpt-4o-mini"
    assert plan["payload_meta"]["profile_used"] == "fast"
    assert plan["secret_ref"] == {"name": "fast-runtime-secret", "scope": "system-wide"}
    assert "value" not in plan["secret_ref"]
    assert plan["reason_trace"]["prompt_type"]


@pytest.mark.asyncio
async def test_ask_and_plan_inference_use_identical_finalized_recall_context(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "ask_recall_context", profiles=profiles, profile_configs=profile_configs)
    await _store(
        ms,
        "User: The release color is blue.\nAssistant: Stored.",
        session_num=1,
        session_date="2024-06-01",
        agent_id="tester",
    )

    query = "What color is the release? Answer with only the color. Do not include any other text."
    query_features = extract_query_features(query)
    assert query_features["retrieval_target"] == "What color is the release"
    assert query_features["output_constraints"]["return_only"] is True

    recall_result = await ms.recall(query, agent_id="tester")
    recall_context = recall_result["context"]
    answer_contract = recall_result["answer_contract"]
    evidence_trace = (recall_result.get("runtime_trace") or {}).get("evidence_context") or {}
    assert evidence_trace["finalized"] is True
    assert answer_contract["prompt_key"]
    assert answer_contract["context_field"] == "context"
    assert "{context}" in answer_contract["prompt_template"]
    assert answer_contract["variables"]["question"] == query
    assert answer_contract["output_constraints"]["return_only"] is True
    for forbidden in ("payload", "payload_meta", "_payload_secret_ref", "secret_ref", "recommended_profile"):
        assert forbidden not in answer_contract
    expected_prompt = answer_contract["prompt_template"].format(
        context=recall_context,
        **answer_contract["variables"],
    )

    captured_payload: dict[str, dict] = {}

    async def fake_send_payload(payload, **_kwargs):
        captured_payload["payload"] = deepcopy(payload)
        return "The release color is blue.", False, None

    async def fixed_recall(*_args, **_kwargs):
        return deepcopy(recall_result)

    monkeypatch.setattr(ms, "_send_payload", fake_send_payload)
    monkeypatch.setattr(ms, "recall", fixed_recall)
    ask_result = await ms.ask(query, agent_id="tester")
    assert ask_result["answer"]
    payload_text = "\n".join(str(message.get("content") or "") for message in captured_payload["payload"]["messages"])
    assert payload_text == expected_prompt
    assert "Answer with only the color" in payload_text

    plan = ms._build_inference_plan_from_recall_result(
        query=query,
        recall_result=deepcopy(recall_result),
    )
    plan_text = "\n".join(str(message.get("content") or "") for message in plan["payload"]["messages"])
    assert plan_text == expected_prompt


def test_raw_recall_survives_finalized_context_packet(tmp_path):
    profiles, profile_configs = _planning_profiles()
    ms = MemoryServer(str(tmp_path), "raw_finalized", profiles=profiles, profile_configs=profile_configs)
    ms._raw_sessions = [
        {
            "message_id": "raw-apple",
            "content": "Project Alpha apple token is quartz.",
            "format": "conversation",
            "status": "active",
            "scope": "agent-private",
            "owner_id": "tester",
            "agent_id": "tester",
            "swarm_id": "default",
            "session_num": 2,
            "source_id": "conv-alpha",
        }
    ]
    result = {
        "context": "RETRIEVED FACTS:",
        "retrieved": [],
        "query_type": "lookup",
        "search_family": "conversation",
        "retrieval_families": ["conversation"],
        "runtime_trace": {"reason": "empty_visible_facts"},
    }

    merged = ms._merge_raw_recall(
        query="find apple token",
        result=result,
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )
    finalized, _trace = ms._finalize_recall_evidence_context(
        query="find apple token",
        recall_result=merged,
    )
    finalized_again, _trace = ms._finalize_recall_evidence_context(
        query="find apple token",
        recall_result=finalized,
    )

    assert "COMPLETED RAW EVIDENCE:" in finalized["context"]
    assert "Project Alpha apple token is quartz" in finalized["context"]
    assert finalized_again["context"].count("COMPLETED RAW EVIDENCE:") == 1


def test_recall_continuation_pages_same_anchor_facts_and_raw(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_pages")
    old_facts = [
        {
            "id": f"old-{idx}",
            "fact": f"Project Alpha workflow checkpoint {idx}.",
            "kind": "fact",
            "session": idx,
            "scope": "agent-private",
            "owner_id": "tester",
            "agent_id": "tester",
            "swarm_id": "default",
            "source_id": "conv-alpha",
        }
        for idx in range(1, 7)
    ]
    needle = {
        "id": "schedule-fact",
        "fact": "Project Alpha release is scheduled for 2026-04-27 at night.",
        "kind": "fact",
        "session": 7,
        "scope": "agent-private",
        "owner_id": "tester",
        "agent_id": "tester",
        "swarm_id": "default",
        "source_id": "conv-alpha",
    }
    ms._all_granular = [*old_facts, needle]
    result = {
        "context": "RETRIEVED FACTS:\n" + "\n".join(
            f"[{idx}] {fact['fact']}" for idx, fact in enumerate(old_facts[:5], 1)
        ),
        "retrieved": old_facts[:5],
        "query_type": "lookup",
        "search_family": "conversation",
        "retrieval_families": ["conversation"],
        "runtime_trace": {},
    }

    continued = ms._attach_recall_continuation(
        query="when should Project Alpha release be?",
        result=result,
        fact_filter=lambda fact: ms._acl_allows(fact, "tester", [], "user"),
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )
    page = get_more_context(
        raw_sessions=[],
        page="next",
        recall_continuation_pages=continued["_recall_continuation_pages"],
        recall_continuation_handle=continued["recall_continuation"]["handle"],
    )

    assert "schedule-fact" not in {item["id"] for item in old_facts[:5]}
    assert continued["recall_continuation"]["available"] is True
    assert "Project Alpha release is scheduled for 2026-04-27 at night" in page["result"]


def test_recall_continuation_paginates_raw_pages_until_exhausted(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_raw_pages")
    ms._raw_sessions = [
        {
            "message_id": f"raw-alpha-{idx}",
            "content": f"Project Alpha release raw detail {idx}: evidence payload {idx}.",
            "format": "conversation",
            "status": "active",
            "scope": "agent-private",
            "owner_id": "tester",
            "agent_id": "tester",
            "swarm_id": "default",
            "session_num": idx,
            "source_id": "conv-alpha",
        }
        for idx in range(1, 14)
    ]

    continued = ms._attach_recall_continuation(
        query="Project Alpha release",
        result={
            "context": "RETRIEVED FACTS:",
            "retrieved": [],
            "query_type": "lookup",
            "search_family": "conversation",
            "retrieval_families": ["conversation"],
            "runtime_trace": {},
        },
        fact_filter=lambda fact: ms._acl_allows(fact, "tester", [], "user"),
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )

    for continuation_page in continued["_recall_continuation_pages"]:
        assert "context" not in continuation_page
        for entry in continuation_page["typed_entries"]:
            assert "raw" not in entry
            assert "content" not in entry

    handle = continued["recall_continuation"]["handle"]
    pages = [
        ms.recall_continuation_page(
            continuation_handle=handle,
            page="next",
            caller_id="tester",
            caller_memberships=[],
            caller_role="user",
            swarm_id="default",
        )
        for _ in range(3)
    ]
    exhausted = ms.recall_continuation_page(
        continuation_handle=handle,
        page="next",
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )

    assert continued["recall_continuation"]["candidate_count"] == 13
    assert [page["recall_continuation"]["page"] for page in pages] == [2, 3, 4]
    assert pages[0]["context"].count("Project Alpha release raw detail") == 5
    assert pages[1]["context"].count("Project Alpha release raw detail") == 5
    assert pages[2]["context"].count("Project Alpha release raw detail") == 3
    page_texts = [page["context"] for page in pages]
    assert len(set(page_texts)) == 3
    assert "raw detail 13:" in page_texts[0]
    assert "raw detail 8:" in page_texts[1]
    assert "raw detail 3:" in page_texts[2]
    assert pages[2]["recall_continuation"]["exhausted"] is True
    assert exhausted["code"] == "RECALL_CONTINUATION_NOT_FOUND"


def test_recall_continuation_paginates_fact_candidates_after_initial_top_k(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_fact_pages")
    facts = [
        {
            "id": f"fact-{idx}",
            "fact": f"Project Alpha release checkpoint fact {idx}.",
            "kind": "fact",
            "session": idx,
            "scope": "agent-private",
            "owner_id": "tester",
            "agent_id": "tester",
            "swarm_id": "default",
            "source_id": "conv-alpha",
        }
        for idx in range(1, 16)
    ]
    ms._all_granular = facts
    initial_top_k = facts[:5]

    continued = ms._attach_recall_continuation(
        query="Project Alpha release checkpoint",
        result={
            "context": "RETRIEVED FACTS:\n" + "\n".join(fact["fact"] for fact in initial_top_k),
            "retrieved": initial_top_k,
            "query_type": "lookup",
            "search_family": "conversation",
            "retrieval_families": ["conversation"],
            "runtime_trace": {},
        },
        fact_filter=lambda fact: ms._acl_allows(fact, "tester", [], "user"),
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )

    handle = continued["recall_continuation"]["handle"]
    page2 = ms.recall_continuation_page(
        continuation_handle=handle,
        page="next",
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )
    page3 = ms.recall_continuation_page(
        continuation_handle=handle,
        page="next",
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )
    exhausted = ms.recall_continuation_page(
        continuation_handle=handle,
        page="next",
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )

    assert continued["recall_continuation"]["candidate_count"] == 10
    combined = f"{page2['context']}\n{page3['context']}"
    for fact in initial_top_k:
        assert fact["fact"] not in combined
    assert page2["context"].count("Project Alpha release checkpoint fact") == 5
    assert page3["context"].count("Project Alpha release checkpoint fact") == 5
    assert page2["context"] != page3["context"]
    assert page2["recall_continuation"]["page"] == 2
    assert page3["recall_continuation"]["page"] == 3
    assert page3["recall_continuation"]["exhausted"] is True
    assert exhausted["code"] == "RECALL_CONTINUATION_NOT_FOUND"


def test_recall_continuation_preserves_acl(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_acl")
    visible = {
        "id": "visible",
        "fact": "Project Alpha visible detail.",
        "kind": "fact",
        "session": 2,
        "scope": "agent-private",
        "owner_id": "tester",
        "agent_id": "tester",
        "swarm_id": "default",
        "source_id": "conv-alpha",
    }
    hidden = {
        "id": "hidden",
        "fact": "Project Alpha hidden secret detail.",
        "kind": "fact",
        "session": 3,
        "scope": "agent-private",
        "owner_id": "other",
        "agent_id": "other",
        "swarm_id": "default",
        "read": ["other"],
        "source_id": "conv-alpha",
    }
    ms._all_granular = [visible, hidden]

    continued = ms._attach_recall_continuation(
        query="Project Alpha detail",
        result={
            "context": "RETRIEVED FACTS:",
            "retrieved": [],
            "query_type": "lookup",
            "search_family": "conversation",
            "retrieval_families": ["conversation"],
            "runtime_trace": {},
        },
        fact_filter=lambda fact: ms._acl_allows(fact, "tester", [], "user"),
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
        raw_kind="all",
    )
    page = ms.recall_continuation_page(
        continuation_handle=continued["recall_continuation"]["handle"],
        page="next",
        caller_id="tester",
        caller_memberships=[],
        caller_role="user",
        swarm_id="default",
    )
    page_text = page["context"]

    assert "visible detail" in page_text
    assert "hidden secret" not in page_text


def test_get_more_context_tool_keeps_legacy_session_id_full_text(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_tool_legacy_session")
    ms._raw_sessions = [
        {
            "message_id": "raw-session-1",
            "content": "Full raw session text for Project Alpha.",
            "status": "active",
        }
    ]

    result = ms._execute_get_more_context_tool(
        {"session_id": 1},
        continuation_pages=[
            {
                "page": 2,
                "context": "RECALL CONTINUATION PAGE 2:\n- continuation evidence",
                "next_page": None,
                "exhausted": True,
            }
        ],
        continuation_handle="opaque-continuation",
        continuation_state={"next_page": 2},
        continuation_acl={},
        caller_id="tester",
    )

    assert "Full text of Session 1" in result["result"]
    assert "Full raw session text for Project Alpha." in result["result"]
    assert "continuation evidence" not in result["result"]


def test_recall_continuation_handles_are_unique_per_recall_and_acl_state(tmp_path):
    ms = MemoryServer(str(tmp_path), "continuation_unique_handles")
    ms._all_granular = [
        {
            "id": "alpha-a",
            "fact": "Project Alpha caller A detail.",
            "kind": "fact",
            "session": 1,
            "scope": "agent-private",
            "owner_id": "caller-a",
            "agent_id": "caller-a",
            "swarm_id": "sw-a",
            "source_id": "conv-alpha",
        },
        {
            "id": "alpha-b",
            "fact": "Project Alpha caller B detail.",
            "kind": "fact",
            "session": 1,
            "scope": "agent-private",
            "owner_id": "caller-b",
            "agent_id": "caller-b",
            "swarm_id": "sw-b",
            "source_id": "conv-alpha",
        },
    ]
    base_result = {
        "context": "RETRIEVED FACTS:",
        "retrieved": [],
        "query_type": "lookup",
        "search_family": "conversation",
        "retrieval_families": ["conversation"],
        "runtime_trace": {},
    }

    first = ms._attach_recall_continuation(
        query="Project Alpha detail",
        result=deepcopy(base_result),
        fact_filter=lambda fact: fact.get("owner_id") == "caller-a" and ms._acl_allows(fact, "caller-a", [], "user"),
        caller_id="caller-a",
        caller_memberships=[],
        caller_role="user",
        swarm_id="sw-a",
        raw_kind="all",
    )
    second = ms._attach_recall_continuation(
        query="Project Alpha detail",
        result=deepcopy(base_result),
        fact_filter=lambda fact: fact.get("owner_id") == "caller-b" and ms._acl_allows(fact, "caller-b", [], "user"),
        caller_id="caller-b",
        caller_memberships=[],
        caller_role="user",
        swarm_id="sw-b",
        raw_kind="all",
    )

    first_handle = first["recall_continuation"]["handle"]
    second_handle = second["recall_continuation"]["handle"]
    assert first_handle != second_handle

    first_page = ms.recall_continuation_page(
        continuation_handle=first_handle,
        page="next",
        caller_id="caller-a",
        caller_memberships=[],
        caller_role="user",
        swarm_id="sw-a",
    )
    second_page = ms.recall_continuation_page(
        continuation_handle=second_handle,
        page="next",
        caller_id="caller-b",
        caller_memberships=[],
        caller_role="user",
        swarm_id="sw-b",
    )

    assert "caller A detail" in first_page["context"]
    assert "caller B detail" not in first_page["context"]
    assert "caller B detail" in second_page["context"]
    assert "caller A detail" not in second_page["context"]


def test_build_inference_plan_does_not_mutate_recall_result(tmp_path):
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "plan_no_mutation", profiles=profiles, profile_configs=profile_configs)
    recall_result = {
        "context": "Context block",
        "query_type": "lookup",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "retrieved": [],
        "runtime_trace": {},
        "complexity_hint": {"score": 0.1, "level": 1},
    }
    original = deepcopy(recall_result)

    plan = ms._build_inference_plan_from_recall_result(
        query="What happened?",
        recall_result=recall_result,
    )

    assert plan["recommended_profile"] == "fast"
    assert plan["payload_meta"]["profile_used"] == "fast"
    assert recall_result == original


def test_build_payload_does_not_mutate_finalized_recall_context(tmp_path):
    profiles, profile_configs = _planning_profiles({"name": "fast-runtime-secret", "scope": "system-wide"})
    ms = MemoryServer(str(tmp_path), "payload_no_mutation", profiles=profiles, profile_configs=profile_configs)
    recall_result = {
        "context": "Finalized evidence context",
        "_context_packet": {
            "tier1": [{"text": "Finalized evidence context", "rank": 0, "source": "test"}],
            "tier2": [],
            "tier3": [],
            "tier4": [],
        },
        "query_type": "lookup",
        "recommended_prompt_type": "lookup",
        "use_tool": False,
        "retrieved": [],
        "runtime_trace": {
            "evidence_context": {
                "finalized": True,
                "context_tokens": 3,
                "memory_budget": 1000,
                "budget_exceeded": False,
                "truncation": None,
            }
        },
        "complexity_hint": {"score": 0.1, "level": 1},
    }
    original_context = deepcopy(recall_result["context"])
    original_packet = deepcopy(recall_result["_context_packet"])

    payload, payload_meta, secret_ref = ms._build_payload(
        query="What happened?",
        recall_result=recall_result,
    )

    assert payload["messages"]
    payload_text = "\n".join(str(message.get("content") or "") for message in payload["messages"])
    assert original_context in payload_text
    assert payload_text == get_inf_prompt("lookup").format(
        context=original_context,
        question="What happened?",
        speakers="User and Assistant",
        sessions_in_context=0,
        total_sessions=0,
        coverage_pct=100,
        reference_date=ms._reference_date_for_recall_result(recall_result),
    )
    assert payload_meta["profile_used"] == "fast"
    assert secret_ref == {"name": "fast-runtime-secret", "scope": "system-wide"}
    assert recall_result["context"] == original_context
    assert recall_result["_context_packet"] == original_packet
    assert "_payload_secret_ref" not in recall_result


def _patch_conversation_raw_recall_runtime(monkeypatch, extract_session_fn):
    async def _embed_texts(texts, **_kwargs):
        dim = 64
        rows = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for token in re.findall(r"[\w./:-]+", str(text).lower()):
                vec[hash(token) % dim] += 1.0
            norm = np.linalg.norm(vec)
            if norm:
                vec /= norm
            rows.append(vec)
        if not rows:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    async def _embed_query(text, **_kwargs):
        return (await _embed_texts([text]))[0]

    async def _no_source_aggregation(*_args, **_kwargs):
        return []

    async def _singleton_group_document(model, source_id, title, source_date, block_dicts, grouping_config, sem):
        from src.memory import build_singleton_episodes

        return build_singleton_episodes(source_id, source_date, block_dicts), {"mode": "singleton"}, "singleton"

    monkeypatch.setattr("src.memory.extract_session", extract_session_fn)
    monkeypatch.setattr("src.memory.group_document", _singleton_group_document)
    monkeypatch.setattr("src.memory.embed_texts", _embed_texts)
    monkeypatch.setattr("src.memory.embed_query", _embed_query)
    monkeypatch.setattr(MemoryServer, "_embed_texts_with_runtime_secrets", lambda _self, texts, **kw: _embed_texts(texts, **kw))
    monkeypatch.setattr(MemoryServer, "_embed_query_with_runtime_secrets", lambda _self, text, **kw: _embed_query(text, **kw))
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", _no_source_aggregation)
    _patch_resolve_supersession(monkeypatch)


async def _write_chat_turn(
    ms: MemoryServer,
    *,
    message_id: str,
    content: str,
    turn_number: int,
    role: str,
    session_id: str = "chat-1",
    swarm_id: str = "team-gosh",
    scope: str = "swarm-shared",
    owner_id: str = "agent:agent-a",
    read: list[str] | None = None,
    write: list[str] | None = None,
    timestamp_ms: int | None = None,
):
    read = read if read is not None else [f"swarm:{swarm_id}"]
    write = write if write is not None else [f"swarm:{swarm_id}"]
    return await ms.write(
        message_id=message_id,
        session_id=session_id,
        content=content,
        content_family="chat",
        timestamp_ms=timestamp_ms or (1712000000000 + turn_number),
        agent_id="agent-a",
        swarm_id=swarm_id,
        scope=scope,
        owner_id=owner_id,
        read=read,
        write=write,
        metadata={"role": role, "turn_number": str(turn_number)},
        caller_id=owner_id,
        caller_principal_kind="agent",
    )


async def _drain_write_log(ms: MemoryServer):
    while await ms.process_write_log_once(batch_size=16):
        pass
    ms._tiers_dirty = False


async def _recall_as_agent_b(
    ms: MemoryServer,
    query: str,
    *,
    query_type: str = "lookup",
    memberships: list[str] | None = None,
):
    return await ms.recall(
        query=query,
        agent_id="agent-b",
        swarm_id="team-gosh",
        search_family="conversation",
        token_budget=4000,
        query_type=query_type,
        kind="all",
        caller_id="agent:agent-b",
        caller_memberships=memberships if memberships is not None else ["swarm:team-gosh"],
        caller_role="agent",
    )


async def _recall_document_as_agent_b(
    ms: MemoryServer,
    query: str,
    *,
    query_type: str = "lookup",
    memberships: list[str] | None = None,
):
    return await ms.recall(
        query=query,
        agent_id="agent-b",
        swarm_id="team-gosh",
        search_family="document",
        token_budget=4000,
        query_type=query_type,
        kind="all",
        caller_id="agent:agent-b",
        caller_memberships=memberships if memberships is not None else ["swarm:team-gosh"],
        caller_role="agent",
    )


def _assert_raw_likely_trace_uses_mirror(trace: dict):
    mirror = MemoryServer.RECALL_EXTRACTION_POLICY_MIRROR
    axis_paths = list(trace.get("matched_extraction_rule_axes") or [])
    allowed_rule_ids = set()
    for axis_path in axis_paths:
        family, axis = str(axis_path).split(".", 1)
        assert family in mirror
        assert axis in mirror[family]
        allowed_rule_ids.update(mirror[family][axis]["rule_ids"])
    for rule_id in trace.get("matched_rule_ids") or []:
        assert rule_id in allowed_rule_ids


def test_recall_extraction_policy_mirror_points_to_prompt_rules():
    repo_root = Path(__file__).resolve().parents[1]
    mirror = MemoryServer.RECALL_EXTRACTION_POLICY_MIRROR
    for family_axes in mirror.values():
        for axis_meta in family_axes.values():
            source_prompt = repo_root / str(axis_meta["source_prompt"])
            assert source_prompt.exists()
            prompt_text = source_prompt.read_text(encoding="utf-8")
            for rule_id in axis_meta.get("rule_ids") or []:
                rule_text = str(rule_id)
                if rule_text.startswith("RULE "):
                    doc_number = rule_text.removeprefix("RULE ").strip()
                    assert rule_text in prompt_text or f"{doc_number}." in prompt_text
                else:
                    assert rule_text in prompt_text


def test_conversation_episode_from_raw_session_does_not_mutate_input(tmp_path):
    ms = MemoryServer(str(tmp_path), "raw_no_mutation")
    raw_session = {
        "raw_session_id": "raw-1",
        "message_id": "msg-1",
        "session_id": "session-1",
        "projection_session_num": 7,
        "metadata": {"role": "assistant", "turn_number": 3},
        "agent_id": "agent-a",
        "swarm_id": "team-gosh",
        "scope": "swarm-shared",
        "owner_id": "agent-a",
        "read": ["swarm:team-gosh"],
        "write": ["agent:agent-a"],
    }
    original = deepcopy(raw_session)

    episode = ms._conversation_episode_from_raw_session(
        raw_session=raw_session,
        source_id="session-1",
        session_num=1,
        session_date="2024-06-01",
        canonical_content="assistant: short answer",
        canonical_source={
            "raw_original": "assistant: short answer",
            "semantic_ready": True,
            "canonicalization_status": "ok",
            "canonicalization_error": None,
            "source_lang": "en",
            "translation_version": "test",
        },
    )

    assert raw_session == original
    assert episode["episode_id"].endswith("_e0001")
    assert episode["role"] == "assistant"


def test_visible_raw_episode_prefilter_is_cached_per_snapshot(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "raw_visibility_cache")
    ms._episode_corpus = {
        "documents": [
            {
                "episodes": [
                    {"episode_id": "conv-1", "source_type": "conversation", "raw_text": "alpha"},
                    {"episode_id": "doc-1", "source_type": "document", "raw_text": "beta"},
                ],
            }
        ]
    }
    calls = 0

    def _semantic_ready(record, *, fallback_text=""):
        nonlocal calls
        calls += 1
        return bool(record.get("raw_text") or fallback_text)

    monkeypatch.setattr(memory_mod, "_record_semantic_ready", _semantic_ready)
    monkeypatch.setattr(ms, "_episode_acl_allows", lambda *args, **kwargs: True)

    first = ms._iter_visible_raw_episodes(
        families={"conversation"},
        caller_id="agent:agent-b",
        caller_memberships=["swarm:team-gosh"],
        caller_role="agent",
        swarm_id=None,
    )
    second = ms._iter_visible_raw_episodes(
        families={"document"},
        caller_id="agent:agent-b",
        caller_memberships=["swarm:team-gosh"],
        caller_role="agent",
        swarm_id=None,
    )

    assert [episode["episode_id"] for episode in first] == ["conv-1"]
    assert [episode["episode_id"] for episode in second] == ["doc-1"]
    assert calls == 2


def test_raw_episode_lifecycle_visibility_uses_stable_identity(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "raw_lifecycle_identity")
    monkeypatch.setattr(ms, "_episode_acl_allows", lambda *args, **kwargs: True)
    ms._raw_sessions = [
        {
            "raw_session_id": "raw-old",
            "message_id": "same-message",
            "source_id": "chat",
            "session_num": 1,
            "artifact_id": "artifact-old",
            "version_id": "v1",
            "status": "retracted",
            "format": "conversation",
        },
        {
            "raw_session_id": "raw-new",
            "message_id": "same-message",
            "source_id": "chat",
            "session_num": 2,
            "artifact_id": "artifact-new",
            "version_id": "v2",
            "status": "active",
            "format": "conversation",
        },
    ]
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "conversation:chat",
            "episodes": [
                {
                    "episode_id": "old",
                    "source_id": "chat",
                    "source_type": "conversation",
                    "message_id": "same-message",
                    "artifact_id": "artifact-old",
                    "version_id": "v1",
                    "raw_text": "old hidden text",
                },
                {
                    "episode_id": "new",
                    "source_id": "chat",
                    "source_type": "conversation",
                    "message_id": "same-message",
                    "artifact_id": "artifact-new",
                    "version_id": "v2",
                    "raw_text": "new visible text",
                },
                {
                    "episode_id": "legacy-no-status",
                    "source_id": "legacy",
                    "source_type": "conversation",
                    "raw_text": "legacy visible text",
                },
            ],
        }]
    }

    visible = ms._iter_visible_raw_episodes(
        families={"conversation"},
        caller_id="agent:agent-b",
        caller_memberships=["agent:PUBLIC"],
        caller_role="agent",
        swarm_id=None,
    )

    assert [episode["episode_id"] for episode in visible] == ["new", "legacy-no-status"]


@pytest.mark.asyncio
async def test_retracted_adjacent_assistant_raw_is_not_visible(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "What is the access code" in text:
            facts.append({
                "id": "question",
                "fact": "The user asked for the access code.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "retracted_adjacent")
    await _write_chat_turn(ms, message_id="question", content="What is the access code?", turn_number=1, role="user")
    await _write_chat_turn(ms, message_id="answer", content="CODE-OLD.", turn_number=2, role="assistant")
    await _drain_write_log(ms)
    for raw in ms._raw_sessions:
        if raw.get("message_id") == "answer":
            raw["status"] = "retracted"
    for doc in ms._episode_corpus.get("documents", []):
        for episode in doc.get("episodes", []):
            if episode.get("message_id") == "answer":
                episode["status"] = "retracted"
    ms._bump_index_snapshot_version()

    result = await _recall_as_agent_b(ms, "what did you answer about access code?")

    assert "CODE-OLD" not in result["context"]


# ── Tests ──

def test_store_creates_cache_file(tmp_path, monkeypatch):
    """store() persists facts through the configured storage backend."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv1")

    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))

    assert ms._storage.exists
    data = ms._storage.load_facts()
    assert len(data["granular"]) == 3
    assert data["n_sessions"] == 1
    assert data["n_sessions_with_facts"] == 1


def test_store_attaches_support_spans_when_facts_as_selectors_enabled(tmp_path, monkeypatch):
    async def mock_extract_session(**kwargs):
        return (
            "conv_selectors",
            1,
            "2024-06-01",
            [
                {
                    "id": "f_01",
                    "fact": "I drive a Prius hybrid every day.",
                    "kind": "preference",
                    "entities": ["Prius"],
                    "tags": ["vehicle"],
                    "session": 1,
                }
            ],
            [],
        )

    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")

    ms = MemoryServer(str(tmp_path), "conv_selectors")
    asyncio.run(
        _store(ms,
            "User: I drive a Prius hybrid every day.\nAssistant: Nice car.",
            session_num=1,
            session_date="2024-06-01",
        )
    )

    fact = ms._all_granular[0]
    assert fact["fact_class"] == "extractive"
    assert fact["support_spans"]
    span = fact["support_spans"][0]
    assert span["source_field"] == "raw_text"
    assert span["episode_id"] == "conv_selectors_e0001"
    assert span["end"] > span["start"]


@pytest.mark.asyncio
async def test_completed_zero_fact_assistant_answer_remains_raw_retrievable(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        if "Alpha prompt" in text:
            facts = [{
                "id": "prompt",
                "fact": "Alpha prompt asks for the short code.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            }]
        else:
            facts = []
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "completed_raw_answer")
    await _write_chat_turn(ms, message_id="m-question", content="Alpha prompt asks for the short code", turn_number=1, role="user")
    await _write_chat_turn(ms, message_id="m-answer", content="ZX-91B.", turn_number=2, role="assistant")

    pending = await _recall_as_agent_b(ms, "ZX-91B")
    assert "ZX-91B" in pending["context"]
    assert pending["raw_recall_count"] == 1

    await _drain_write_log(ms)
    assert ms.write_status("m-answer")["extraction_state"] == "complete"
    assert any(raw.get("message_id") == "m-answer" and "ZX-91B" in raw.get("content", "") for raw in ms._raw_sessions)

    after_worker = await _recall_as_agent_b(ms, "ZX-91B")
    assert "COMPLETED RAW EVIDENCE:" in after_worker["context"]
    assert "ZX-91B" in after_worker["context"]
    assert after_worker["completed_raw_recall_count"] == 1
    raw_items = [
        item for item in after_worker["retrieved"]
        if item.get("raw_evidence_kind") == "completed_raw_episode"
    ]
    assert raw_items
    assert "artifact_id" not in raw_items[0]
    assert "version_id" not in raw_items[0]
    assert "raw_session_id" not in raw_items[0]

    await ms.build_index()
    after_index = await _recall_as_agent_b(ms, "ZX-91B")
    assert "ZX-91B" in after_index["context"]
    assert after_index["completed_raw_recall_count"] == 1


@pytest.mark.asyncio
async def test_question_match_includes_adjacent_zero_fact_assistant_answer(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "What is the rabbit called" in text:
            facts.append({
                "id": "rabbit-question",
                "fact": "The user asked what the rabbit is called in Nu Pogodi.",
                "kind": "event",
                "entities": ["Nu Pogodi"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "adjacent_answer")
    await _write_chat_turn(
        ms,
        message_id="rabbit-question",
        content="What is the rabbit called in Nu Pogodi?",
        turn_number=19,
        role="user",
    )
    await _write_chat_turn(
        ms,
        message_id="rabbit-answer",
        content="No personal name; only Hare.",
        turn_number=20,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what did you answer about what the rabbit is called?")
    assert "RAW CONVERSATION EVIDENCE:" in result["context"]
    assert "No personal name; only Hare." in result["context"]
    assert result["adjacent_raw_recall_count"] == 1


@pytest.mark.asyncio
async def test_adjacent_answer_lookup_uses_source_index_without_pairwise_scan(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "What is the rabbit called" in text:
            facts.append({
                "id": "rabbit-question",
                "fact": "The user asked what the rabbit is called in Nu Pogodi.",
                "kind": "event",
                "entities": ["Nu Pogodi"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "adjacent_answer_index")
    await _write_chat_turn(
        ms,
        message_id="rabbit-question",
        content="What is the rabbit called in Nu Pogodi?",
        turn_number=19,
        role="user",
        session_id="main-chat",
    )
    for idx in range(50):
        await _write_chat_turn(
            ms,
            message_id=f"decoy-answer-{idx}",
            content=f"Decoy answer {idx}.",
            turn_number=20 + idx,
            role="assistant",
            session_id=f"decoy-chat-{idx}",
        )
    await _write_chat_turn(
        ms,
        message_id="rabbit-answer",
        content="No personal name; only Hare.",
        turn_number=20,
        role="assistant",
        session_id="main-chat",
    )
    await _drain_write_log(ms)

    def _pairwise_scan_must_not_run(left, right):
        raise AssertionError(f"unexpected pairwise scan for {left=} {right=}")

    monkeypatch.setattr(ms, "_raw_entries_same_source", _pairwise_scan_must_not_run)

    result = await _recall_as_agent_b(ms, "what did you answer about what the rabbit is called?")
    assert "No personal name; only Hare." in result["context"]
    assert result["adjacent_raw_recall_count"] == 1


@pytest.mark.asyncio
async def test_non_adjacent_completed_zero_fact_answer_is_searchable_by_content(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "Unrelated user note" in text:
            facts.append({
                "id": "note",
                "fact": "The user wrote an unrelated note.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "non_adjacent_raw")
    await _write_chat_turn(ms, message_id="note", content="Unrelated user note", turn_number=1, role="user")
    await _write_chat_turn(ms, message_id="answer", content="Release label is kappa-blue.", turn_number=4, role="assistant")
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "kappa-blue")
    assert "COMPLETED RAW EVIDENCE:" in result["context"]
    assert "Release label is kappa-blue." in result["context"]


@pytest.mark.asyncio
async def test_current_query_prioritizes_new_completed_raw_answer_over_old_lexical_hit(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "old value" in text:
            facts.append({
                "id": "old-value",
                "fact": "The old value was OLD-000.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "latest_raw")
    await _write_chat_turn(ms, message_id="old", content="The old value was OLD-000.", turn_number=1, role="assistant")
    await _write_chat_turn(ms, message_id="new", content="NEW-999.", turn_number=9, role="assistant")
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "latest value written to memory", query_type="current")
    context = result["context"]
    assert "NEW-999" in context
    assert "OLD-000" in context
    raw_section = context.split("COMPLETED RAW EVIDENCE:", 1)[1]
    assert "NEW-999" in raw_section


@pytest.mark.asyncio
async def test_completed_zero_fact_raw_evidence_preserves_acl(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        return ("conv", int(kwargs.get("session_num") or 1), kwargs.get("session_date", "2024-06-01"), [], [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_acl")
    await _write_chat_turn(
        ms,
        message_id="private-answer",
        content="Private short answer omega-secret.",
        turn_number=1,
        role="assistant",
        scope="agent-private",
        read=["agent:agent-a"],
        write=["agent:agent-a"],
    )
    await _drain_write_log(ms)

    unauthorized = await _recall_as_agent_b(ms, "omega-secret", memberships=[])
    assert "omega-secret" not in unauthorized["context"]

    authorized = await ms.recall(
        query="omega-secret",
        agent_id="agent-a",
        swarm_id="team-gosh",
        search_family="conversation",
        token_budget=4000,
        query_type="lookup",
        kind="all",
        caller_id="agent:agent-a",
        caller_memberships=[],
        caller_role="agent",
    )
    assert "omega-secret" in authorized["context"]


@pytest.mark.asyncio
async def test_kind_specific_recall_does_not_use_raw_fallback(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        return ("conv", int(kwargs.get("session_num") or 1), kwargs.get("session_date", "2024-06-01"), [], [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_kind_filter")
    await _write_chat_turn(
        ms,
        message_id="raw-constraint",
        content="The deployment limit is 7 nodes.",
        turn_number=1,
        role="user",
    )
    await _drain_write_log(ms)

    kind_specific = await ms.recall(
        query="deployment limit",
        agent_id="agent-b",
        swarm_id="team-gosh",
        search_family="conversation",
        query_type="lookup",
        kind="preference",
        caller_id="agent:agent-b",
        caller_memberships=["swarm:team-gosh"],
        caller_role="agent",
    )
    assert "The deployment limit is 7 nodes." not in kind_specific["context"]
    assert kind_specific["runtime_trace"]["raw_recall"]["skipped"] == "kind_filter"

    all_kind = await _recall_as_agent_b(ms, "deployment limit")
    assert "The deployment limit is 7 nodes." in all_kind["context"]


@pytest.mark.asyncio
async def test_retract_hides_raw_and_allows_same_content_reingest(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)
    ms = MemoryServer(str(tmp_path), "retract_raw_visibility")
    first = await _store(
        ms,
        "User: The launch phrase is silver comet.",
        1,
        "2024-06-01",
        source_id="SRC-RETRACT",
        artifact_id="artifact-retract-1",
        version_id="v1",
    )
    assert first["status"] == "ok"
    before = await ms.recall("silver comet", search_family="conversation", query_type="lookup")
    assert "silver comet" in before["context"]

    retracted = await ms.retract("artifact-retract-1", caller_role="admin")
    assert retracted["status"] == "retracted"
    after = await ms.recall("silver comet", search_family="conversation", query_type="lookup")
    assert "silver comet" not in after["context"]

    second = await _store(
        ms,
        "User: The launch phrase is silver comet.",
        2,
        "2024-06-02",
        source_id="SRC-RETRACT-NEW",
        artifact_id="artifact-retract-2",
        version_id="v2",
    )
    assert second["status"] == "ok"
    assert second["facts_extracted"] == 1
    reloaded = MemoryServer(str(tmp_path), "retract_raw_visibility")
    final = await reloaded.recall("silver comet", search_family="conversation", query_type="lookup")
    assert "silver comet" in final["context"]
    assert all(
        raw.get("status") == "retracted"
        for raw in reloaded._raw_sessions
        if raw.get("artifact_id") == "artifact-retract-1"
    )


@pytest.mark.asyncio
async def test_fact_backed_completed_answer_does_not_add_duplicate_raw_block(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "Green light" in text:
            facts.append({
                "id": "green-light",
                "fact": "The assistant answered Green light.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "fact_backed_answer")
    await _write_chat_turn(ms, message_id="answer", content="Green light.", turn_number=1, role="assistant")
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "Green light")
    assert "The assistant answered Green light." in result["context"]
    assert "COMPLETED RAW EVIDENCE:" not in result["context"]
    assert result.get("completed_raw_recall_count", 0) == 0


@pytest.mark.asyncio
async def test_raw_likely_true_for_unclear_exact_value_source(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "find a number with 18 digits" in text:
            facts.append({
                "id": "number-request",
                "fact": "A number with 18 digits is being requested.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "zero_fact_short_answer")
    await _write_chat_turn(
        ms,
        message_id="number-question",
        content="find a number with 18 digits",
        turn_number=1,
        role="user",
    )
    await _write_chat_turn(
        ms,
        message_id="number-answer",
        content="123456789012345678",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "find a number with 18 digits")
    context = result["context"]
    assert "RETRIEVED FACTS:" in context
    assert "A number with 18 digits is being requested." in context
    assert "RAW CONVERSATION EVIDENCE:" in context
    assert "find a number with 18 digits" in context
    assert "assistant: 123456789012345678" in context
    assert not any("123456789012345678" in str(fact.get("fact") or "") for fact in ms._all_granular)
    trace = result["runtime_trace"]
    assert trace["raw_likely"]["fact_likelihood"] == "uncertain"
    assert trace["raw_likely"]["raw_likely"] is True
    assert "exact_value_source_unknown" in trace["raw_likely"]["uncertain_reasons"]
    _assert_raw_likely_trace_uses_mirror(trace["raw_likely"])
    assert trace["conversation_raw_window"]["enabled"] is True
    assert trace["conversation_raw_window"]["injected_episode_ids"]


@pytest.mark.asyncio
async def test_conversation_raw_recall_finds_completed_short_assistant_reply(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        return ("conv", int(kwargs.get("session_num") or 1), kwargs.get("session_date", "2024-06-01"), [], [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "completed_raw_episode_lookup")
    await _write_chat_turn(
        ms,
        message_id="answer",
        content="short answer is iris-token",
        turn_number=1,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "iris-token")
    assert "COMPLETED RAW EVIDENCE:" in result["context"]
    assert "assistant: short answer is iris-token" in result["context"]
    assert result["runtime_trace"]["raw_episode_retrieval"]["mode"] == "episode_lexical"
    assert result["runtime_trace"]["raw_episode_retrieval"]["count"] == 1


@pytest.mark.asyncio
async def test_conversation_raw_window_respects_acl(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "private question" in text:
            facts.append({
                "id": "private-question",
                "fact": "A private question was asked.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_window_acl")
    await _write_chat_turn(
        ms,
        message_id="private-question",
        content="private question",
        turn_number=1,
        role="user",
        scope="agent-private",
        read=["agent:agent-a"],
        write=["agent:agent-a"],
    )
    await _write_chat_turn(
        ms,
        message_id="private-answer",
        content="private raw answer",
        turn_number=2,
        role="assistant",
        scope="agent-private",
        read=["agent:agent-a"],
        write=["agent:agent-a"],
    )
    await _drain_write_log(ms)

    unauthorized = await _recall_as_agent_b(ms, "private question", memberships=[])
    assert "private raw answer" not in unauthorized["context"]

    authorized = await ms.recall(
        query="private question",
        agent_id="agent-a",
        swarm_id="team-gosh",
        search_family="conversation",
        token_budget=4000,
        query_type="lookup",
        kind="all",
        caller_id="agent:agent-a",
        caller_memberships=[],
        caller_role="agent",
    )
    assert "assistant: private raw answer" in authorized["context"]


@pytest.mark.asyncio
async def test_raw_likely_true_for_assistant_non_material_output(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "what is the rabbit called" in text:
            facts.append({
                "id": "rabbit-name-question",
                "fact": "The user asked what the rabbit is called.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_window_non_numeric")
    await _write_chat_turn(
        ms,
        message_id="rabbit-question",
        content="what is the rabbit called?",
        turn_number=1,
        role="user",
    )
    await _write_chat_turn(ms, message_id="rabbit-answer", content="Just Hare.", turn_number=2, role="assistant")
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what did you answer about rabbit name?")
    assert "RAW CONVERSATION EVIDENCE:" in result["context"]
    assert "assistant: Just Hare." in result["context"]
    assert result["runtime_trace"]["raw_likely"]["fact_likelihood"] == "uncertain"
    assert result["runtime_trace"]["raw_likely"]["raw_likely"] is True
    assert "assistant_output_not_material_fact" in result["runtime_trace"]["raw_likely"]["uncertain_reasons"]
    _assert_raw_likely_trace_uses_mirror(result["runtime_trace"]["raw_likely"])


@pytest.mark.asyncio
async def test_document_lookup_surfaces_raw_only_detail_without_fake_fact(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        sn = int(kwargs.get("session_num") or 1)
        return ("doc", sn, kwargs.get("session_date", "2024-06-01"), [], [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "doc_raw_only_detail")
    await _ingest_document(
        ms,
        content="Operations summary.\nThe backup code is ZX-4817.\nRotation notes are ordinary.",
        source_id="DOC-RAW-DETAIL",
        scope="swarm-shared",
        agent_id="agent-a",
        swarm_id="team-gosh",
    )

    result = await _recall_document_as_agent_b(ms, "backup code")
    assert "RAW DOCUMENT EVIDENCE:" in result["context"] or "COMPLETED RAW EVIDENCE:" in result["context"]
    assert "ZX-4817" in result["context"]
    assert not any("ZX-4817" in str(fact.get("fact") or "") for fact in ms._all_granular)
    assert result["runtime_trace"]["raw_episode_retrieval"]["count"] >= 1


@pytest.mark.asyncio
async def test_document_raw_window_is_bounded_and_traced(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "anchor topic" in text.lower():
            facts.append({
                "id": "doc-anchor",
                "fact": "The document mentions the anchor topic.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("doc", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "doc_raw_window_bounded")
    long_tail = " filler" * 1200
    await _ingest_document(
        ms,
        content=f"Anchor topic.\nThe bounded detail is Delta-73.\n{long_tail}",
        source_id="DOC-BOUNDED",
        scope="swarm-shared",
        agent_id="agent-a",
        swarm_id="team-gosh",
    )

    result = await _recall_document_as_agent_b(ms, "exact wording around anchor topic")
    trace = result["runtime_trace"]["document_raw_window"]
    assert trace["enabled"] is True
    assert trace["raw_budget_chars"] == 4000
    assert len(result["context"]) < len(long_tail) + 2000


@pytest.mark.asyncio
async def test_existing_fact_answer_remains_first_when_raw_window_has_distractor(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "answer is alpha" in text.lower():
            facts.append({
                "id": "answer-alpha",
                "fact": "The answer is Alpha.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_window_fact_first")
    await _write_chat_turn(ms, message_id="answer", content="answer is Alpha.", turn_number=1, role="user")
    await _write_chat_turn(
        ms,
        message_id="distractor",
        content="Beta is only nearby noise.",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what value did the user state as the answer?")
    context = result["context"]
    assert "The answer is Alpha." in context
    assert "RAW CONVERSATION EVIDENCE:" not in context
    assert "Beta is only nearby noise." not in context
    assert result["runtime_trace"]["raw_likely"]["raw_likely"] is False
    _assert_raw_likely_trace_uses_mirror(result["runtime_trace"]["raw_likely"])


@pytest.mark.asyncio
async def test_raw_likely_false_for_user_exact_value_axis(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "account code" in text.lower():
            facts.append({
                "id": "account-code",
                "fact": "The user's account code is 123456789012345678.",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_user_exact_value")
    await _write_chat_turn(
        ms,
        message_id="account-code",
        content="my account code is 123456789012345678",
        turn_number=1,
        role="user",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what is my account code?")
    assert "The user's account code is 123456789012345678." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.exact_values" in trace["matched_extraction_rule_axes"]
    assert "RULE 1" in trace["matched_rule_ids"]
    assert "src/prompts/extraction/conversation.md" in trace["policy_source"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_document_identifier_axis(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "doc-id-741" in text.lower():
            facts.append({
                "id": "doc-identifier",
                "fact": "The document identifier is DOC-ID-741.",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("doc", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_doc_identifier")
    await _ingest_document(
        ms,
        content="Release note.\nDocument identifier: DOC-ID-741.\nImplementation notes.",
        source_id="DOC-ID-GATE",
        scope="swarm-shared",
        agent_id="agent-a",
        swarm_id="team-gosh",
    )

    result = await _recall_document_as_agent_b(ms, "what document identifier is listed?")
    assert "The document identifier is DOC-ID-741." in result["context"]
    assert "RAW DOCUMENT EVIDENCE:" not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "document.unique_identifiers" in trace["matched_extraction_rule_axes"]
    assert "RULE 5" in trace["matched_rule_ids"]
    assert "src/prompts/extraction/document.md" in trace["policy_source"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_prompt_declared_moved_to_update(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "moved to" in text.lower():
            facts.append({
                "id": "alice-moved-place",
                "fact": "Alice moved to Boston.",
                "kind": "event",
                "entities": ["Alice", "Boston"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_moved_to_update")
    await _write_chat_turn(ms, message_id="move-place", content="Alice moved to Boston.", turn_number=1, role="user")
    await _write_chat_turn(ms, message_id="move-noise", content="Nearby raw noise.", turn_number=2, role="assistant")
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "where did Alice move?")
    assert "Alice moved to Boston." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    assert "Nearby raw noise." not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.knowledge_updates" in trace["matched_extraction_rule_axes"]
    assert "conversation.exact_values" in trace["matched_extraction_rule_axes"]
    assert "conversation.temporal_ordering" not in trace["matched_extraction_rule_axes"]
    assert "RULE 8" in trace["matched_rule_ids"]
    assert "RULE 1" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_prompt_declared_temporal_ordering(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "after" in text.lower():
            facts.append({
                "id": "after-move-event",
                "fact": "After Alice moved, Alice updated her mailing address.",
                "kind": "event",
                "entities": ["Alice"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_temporal_ordering")
    await _write_chat_turn(
        ms,
        message_id="after-move",
        content="After Alice moved, Alice updated her mailing address.",
        turn_number=1,
        role="user",
    )
    await _write_chat_turn(
        ms,
        message_id="after-noise",
        content="Nearby unrelated noise.",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what happened after Alice moved?")
    assert "After Alice moved, Alice updated her mailing address." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    assert "Nearby unrelated noise." not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.temporal_ordering" in trace["matched_extraction_rule_axes"]
    assert "RULE 7d" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_prompt_declared_event_date_update(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "moved to" in text.lower():
            facts.append({
                "id": "alice-moved-date",
                "fact": "Alice moved to Boston on March 5, 2024.",
                "kind": "event",
                "entities": ["Alice", "Boston"],
                "tags": [],
                "session": sn,
                "event_date": "2024-03-05",
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_event_date_update")
    await _write_chat_turn(
        ms,
        message_id="move-date",
        content="Alice moved to Boston on March 5, 2024.",
        turn_number=1,
        role="user",
    )
    await _write_chat_turn(
        ms,
        message_id="move-date-noise",
        content="Nearby raw date noise.",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "when did Alice move to Boston?")
    assert "Alice moved to Boston on March 5, 2024." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    assert "Nearby raw date noise." not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.exact_values" in trace["matched_extraction_rule_axes"]
    assert "conversation.knowledge_updates" in trace["matched_extraction_rule_axes"]
    assert "RULE 4" in trace["matched_rule_ids"]
    assert "RULE 8" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_prompt_declared_place_query(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "location" in text.lower():
            facts.append({
                "id": "alice-location-choice",
                "fact": "Alice chose Harbor Hall as the event location.",
                "kind": "decision",
                "entities": ["Alice", "Harbor Hall"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_place_query")
    await _write_chat_turn(
        ms,
        message_id="place-choice",
        content="Alice chose Harbor Hall as the event location.",
        turn_number=1,
        role="user",
    )
    await _write_chat_turn(
        ms,
        message_id="place-noise",
        content="Nearby unrelated noise.",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what location did Alice choose?")
    assert "Alice chose Harbor Hall as the event location." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    assert "Nearby unrelated noise." not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.exact_values" in trace["matched_extraction_rule_axes"]
    assert "conversation.named_targets" in trace["matched_extraction_rule_axes"]
    assert "RULE 1" in trace["matched_rule_ids"]
    assert "RULE 1b" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_false_for_prompt_declared_acquisition_event(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "bought" in text.lower():
            facts.append({
                "id": "alice-bought-item",
                "fact": "Alice bought a camera.",
                "kind": "event",
                "entities": ["Alice", "camera"],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_acquisition_event")
    await _write_chat_turn(ms, message_id="acquisition", content="Alice bought a camera.", turn_number=1, role="user")
    await _write_chat_turn(
        ms,
        message_id="acquisition-noise",
        content="Nearby raw purchase noise.",
        turn_number=2,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what did Alice buy?")
    assert "Alice bought a camera." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    assert "Nearby raw purchase noise." not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "high"
    assert "conversation.acquisition_events" in trace["matched_extraction_rule_axes"]
    assert "DELTA D" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


@pytest.mark.asyncio
async def test_raw_likely_medium_for_assistant_recommendation_axis(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "recommend" in text.lower():
            facts.append({
                "id": "assistant-recommendation",
                "fact": "The assistant recommended checking the release notes.",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": sn,
                "speaker": "assistant",
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_gate_assistant_material")
    await _write_chat_turn(
        ms,
        message_id="recommendation",
        content="I recommend checking the release notes.",
        turn_number=1,
        role="assistant",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what recommendation did assistant provide?")
    assert "The assistant recommended checking the release notes." in result["context"]
    assert "RAW CONVERSATION EVIDENCE:" not in result["context"]
    trace = result["runtime_trace"]["raw_likely"]
    assert trace["raw_likely"] is False
    assert trace["fact_likelihood"] == "medium"
    assert "conversation.assistant_material_facts" in trace["matched_extraction_rule_axes"]
    assert "RULE 9" in trace["matched_rule_ids"]
    _assert_raw_likely_trace_uses_mirror(trace)


def test_codebase_search_family_bypasses_conversation_document_raw_gate(tmp_path):
    ms = MemoryServer(str(tmp_path), "raw_gate_codebase")
    result = {
        "context": "RETRIEVED FACTS:\n- code fact",
        "retrieved": [],
        "search_family": "codebase",
        "retrieval_families": ["codebase"],
        "runtime_trace": {},
    }

    merged = ms._merge_raw_recall(
        query="find implementation",
        result=result,
        caller_id="agent:agent-b",
        caller_memberships=[],
        caller_role="agent",
        swarm_id="team-gosh",
    )
    assert merged["context"] == result["context"]
    assert "raw_likely" not in merged.get("runtime_trace", {})
    assert "conversation_raw_window" not in merged.get("runtime_trace", {})
    assert "document_raw_window" not in merged.get("runtime_trace", {})


@pytest.mark.asyncio
async def test_raw_window_does_not_contaminate_from_other_source(tmp_path, monkeypatch):
    async def _extract_session(**kwargs):
        text = str(kwargs.get("session_text") or "")
        sn = int(kwargs.get("session_num") or 1)
        facts = []
        if "shared query term" in text.lower() and "anchor" in text.lower():
            facts.append({
                "id": "shared-anchor",
                "fact": "The first source contains the shared query term anchor.",
                "kind": "event",
                "entities": [],
                "tags": [],
                "session": sn,
            })
        return ("conv", sn, kwargs.get("session_date", "2024-06-01"), facts, [])

    _patch_conversation_raw_recall_runtime(monkeypatch, _extract_session)
    ms = MemoryServer(str(tmp_path), "raw_window_cross_source")
    await _write_chat_turn(
        ms,
        message_id="a1",
        content="shared query term anchor",
        turn_number=1,
        role="user",
        session_id="source-a",
    )
    await _write_chat_turn(
        ms,
        message_id="a2",
        content="Answer from source A.",
        turn_number=2,
        role="assistant",
        session_id="source-a",
    )
    await _write_chat_turn(
        ms,
        message_id="b1",
        content="shared query term in source B.",
        turn_number=1,
        role="user",
        session_id="source-b",
    )
    await _write_chat_turn(
        ms,
        message_id="b2",
        content="Do not inject source B raw.",
        turn_number=2,
        role="assistant",
        session_id="source-b",
    )
    await _drain_write_log(ms)

    result = await _recall_as_agent_b(ms, "what did you answer after shared query term anchor?")
    episode_by_message = {
        str(episode.get("message_id") or ""): str(episode.get("episode_id") or "")
        for doc in ms._episode_corpus.get("documents", [])
        for episode in doc.get("episodes", [])
        if isinstance(episode, dict)
    }
    injected = set(result["runtime_trace"]["conversation_raw_window"]["injected_episode_ids"])
    assert episode_by_message["a2"] in injected
    assert episode_by_message["b2"] not in injected


def test_calendar_seeking_resolution_renders_selector_source_for_temporal_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")
    ms = MemoryServer(str(tmp_path), "conv_temporal_seek")
    raw_text = "Evan: I drove my Prius hybrid car to work on March 3, 2024."
    episode_id = "conv_temporal_seek_e0001"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "conversation:conv_temporal_seek",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "conversation",
                "source_id": "conv_temporal_seek",
                "source_date": "2024-03-03",
                "topic_key": "session",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    fact = {
        "id": "f_temporal_seek",
        "fact": "Evan drove his Prius hybrid car to work.",
        "metadata": {"episode_id": episode_id},
        "support_spans": [{
            "episode_id": episode_id,
            "source_field": "raw_text",
            "start": 0,
            "end": len(raw_text),
            "role": "primary",
        }],
    }
    ms._fact_lookup = {fact["id"]: fact}

    async def mock_resolve_calendar_seeking(*, query, candidate_facts):
        return {
            "fact": fact,
            "event": {
                "event_id": "evt_seek_1",
                "time_start": "2024-03-03",
                "time_end": "2024-03-03",
                "time_granularity": "day",
            },
            "answer": "2024",
            "trace": {"query": query},
        }

    monkeypatch.setattr(ms, "_resolve_calendar_seeking", mock_resolve_calendar_seeking)
    recall_result = {
        "context": "RETRIEVED FACTS:\n[1] (S1) Evan drove his Prius hybrid car to work.",
        "_context_packet": {"tier1": [], "tier2": [], "tier3": [], "tier4": []},
    }

    result = asyncio.run(
        ms._attach_calendar_seeking_resolution(
            query="Which year did Evan drive his car to work?",
            recall_result=recall_result,
            candidate_facts=[fact],
        )
    )

    assert "TEMPORAL EVIDENCE:" in result["context"]
    assert f"Source (raw_text, Episode {episode_id}" in result["context"]
    assert "Prius hybrid car" in result["context"]
    assert result["temporal_resolution"]["mode"] == "calendar-seeking"
    assert any(
        "Source (raw_text, Episode" in str(item.get("text", ""))
        for item in result["_context_packet"]["tier1"]
        if isinstance(item, dict)
    )


def test_calendar_answer_resolution_falls_back_to_event_source_span_for_temporal_queries(tmp_path, monkeypatch):
    monkeypatch.setenv("GOSH_FACT_SELECTORS", "1")
    ms = MemoryServer(str(tmp_path), "conv_temporal_answer")
    raw_text = "Audrey adopted Pepper, Precious, and Panda three years ago."
    episode_id = "conv_temporal_answer_e0001"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "conversation:conv_temporal_answer",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "conversation",
                "source_id": "conv_temporal_answer",
                "source_date": "2023-01-21",
                "topic_key": "session",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": raw_text,
                "provenance": {"raw_span": [0, len(raw_text)]},
            }],
        }],
    }
    fact = {
        "id": "f_temporal_answer",
        "fact": "Audrey adopted Pepper, Precious, and Panda three years ago.",
        "metadata": {"episode_id": episode_id},
    }
    ms._fact_lookup = {fact["id"]: fact}
    recall_result = {
        "context": "RETRIEVED FACTS:\n[1] (S1) Audrey adopted Pepper, Precious, and Panda three years ago.",
        "_context_packet": {"tier1": [], "tier2": [], "tier3": [], "tier4": []},
    }
    resolution = {
        "events": [{
            "event_id": "evt_answer_1",
            "time_start": "2020-01-21",
            "support_fact_ids": [fact["id"]],
            "source_span": {
                "episode_id": episode_id,
                "source_field": "raw_text",
                "start_char": 0,
                "end_char": len(raw_text),
            },
        }],
        "facts": [fact],
    }

    result = ms._attach_calendar_answer_resolution(
        query="Which year did Audrey adopt first three dogs?",
        recall_result=recall_result,
        candidate_facts=[fact],
        resolution=resolution,
    )

    assert "TEMPORAL EVIDENCE:" in result["context"]
    assert f"Source (raw_text, Episode {episode_id}" in result["context"]
    assert "three years ago" in result["context"]
    assert result["temporal_resolution"]["mode"] == "calendar-answer"
    assert any(
        "Source (raw_text, Episode" in str(item.get("text", ""))
        for item in result["_context_packet"]["tier1"]
        if isinstance(item, dict)
    )

def test_ask_returns_deterministic_exact_step_answer_without_llm(tmp_path):
    ms = MemoryServer(str(tmp_path), "ordinal_direct_answer")
    episode_id = "ORDINAL_e01"
    ms._data_dict = {}
    ms._episode_corpus = {
        "documents": [{
            "doc_id": "document:ORDINAL",
            "episodes": [{
                "episode_id": episode_id,
                "source_type": "document",
                "source_id": "ORDINAL",
                "source_date": "2026-03-01",
                "topic_key": "step 8 sql",
                "state_label": "trace",
                "currentness": "historical",
                "raw_text": (
                    "[Step 8]\n"
                    "Action: execute_snowflake_sql: SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023\n"
                    "Observation: ok"
                ),
                "provenance": {"raw_span": [0, 113]},
            }],
        }],
    }
    ms._raw_sessions = [{"session_num": 1, "session_date": "2026-03-01"}]
    ms._all_granular = [{
        "id": "f_step_8",
        "session": 1,
        "kind": "fact",
        "fact": "At step 8, the agent ran SQL over wholesale for years 2020 through 2023.",
        "metadata": {"episode_id": episode_id, "episode_source_id": "ORDINAL"},
    }]
    ms._all_cons = []
    ms._all_cross = []
    ms._fact_lookup = {fact["id"]: fact for fact in ms._all_granular}
    ms._source_records = {"ORDINAL": {"family": "document"}}
    ms._rebuild_temporal_index()

    result = asyncio.run(ms.ask("At step 8, what SQL did the agent run?"))

    assert result["answer"] == "SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023"
    assert result["profile_used"] == "deterministic:temporal_v1"
    assert result["tool_called"] is False
    assert result["runtime_trace"]["temporal_resolution"]["query_class"] == "ordinal"
def test_store_overrides_llm_session_with_caller_session(tmp_path, monkeypatch):
    """store() must persist caller session_num, not bogus model-emitted session."""
    async def _mock_call_extract(model, system, user_msg, max_tokens=8192, sem=None):
        return {
            "facts": [{
                "id": "f_01",
                "fact": "Apollo 11 landed on the Moon.",
                "session": 1969,
                "entities": ["Apollo 11"],
                "tags": ["history"],
            }],
            "temporal_links": [],
        }

    monkeypatch.setattr("src.memory.call_extract", _mock_call_extract)
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    ms = MemoryServer(str(tmp_path), "conv_session_fix")

    asyncio.run(_store(ms, "Apollo 11 landed on the Moon.", session_num=1, session_date="2024-06-01"))

    assert len(ms._all_granular) == 1
    assert ms._all_granular[0]["session"] == 1
    data = ms._storage.load_facts()
    assert data["granular"][0]["session"] == 1


def test_build_index_creates_embs(tmp_path, monkeypatch):
    """build_index() persists embeddings through the configured storage backend."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv2")

    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))
    result = asyncio.run(ms.build_index())

    loaded = ms._storage.load_embeddings()
    assert loaded is not None
    assert "gran" in loaded
    assert loaded["gran"].shape[0] == 3
    assert loaded["gran"].shape[1] == DIM
    assert result["granular"] == 3


@pytest.mark.asyncio
async def test_build_index_retries_when_snapshot_changes_during_embedding(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_build_retry")

    await _store(ms, "Hello world", session_num=1, session_date="2024-06-01")
    await ms.build_index()

    ms._emb_fingerprints = {}
    real_load_embeddings = ms._storage.load_embeddings
    monkeypatch.setattr(ms._storage, "load_embeddings", lambda: None)

    started = asyncio.Event()
    release = asyncio.Event()
    gran_calls = 0

    async def _controlled_embed(texts, **kwargs):
        nonlocal gran_calls
        if str(kwargs.get("label") or "").startswith("gran-"):
            gran_calls += 1
            if gran_calls == 1:
                started.set()
                await release.wait()
        return _rand_embs(len(texts))

    monkeypatch.setattr(ms, "_embed_texts_with_runtime_secrets", _controlled_embed)

    build_task = asyncio.create_task(ms.build_index())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    ingest_result = await _ingest_asserted_facts(
        ms,
        facts=[{
            "id": "task_fact_1",
            "fact": "Concurrent task fact for retry coverage",
            "kind": "fact",
            "entities": [],
            "tags": [],
            "session": 1,
        }],
        raw_sessions=[{
            "raw_session_id": "rs_task_1",
            "session_num": 1,
            "session_date": "2024-06-02",
            "content": "Concurrent task content",
        }],
        enrich_l0=False,
    )
    assert ingest_result["granular_added"] == 1

    release.set()
    result = await asyncio.wait_for(build_task, timeout=5.0)

    monkeypatch.setattr(ms._storage, "load_embeddings", real_load_embeddings)
    saved = real_load_embeddings()

    assert gran_calls >= 2
    assert result["granular"] == len(ms._all_granular)
    assert saved is not None
    assert saved["gran"].shape[0] == len(ms._all_granular)


@pytest.mark.asyncio
async def test_ingest_asserted_facts_survives_concurrent_index_build(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_ingest_retry")

    await _store(ms, "Hello world", session_num=1, session_date="2024-06-01")
    await ms.build_index()

    ms._emb_fingerprints = {}
    real_load_embeddings = ms._storage.load_embeddings
    monkeypatch.setattr(ms._storage, "load_embeddings", lambda: None)

    started = asyncio.Event()
    release = asyncio.Event()
    gran_calls = 0

    async def _controlled_embed(texts, **kwargs):
        nonlocal gran_calls
        if str(kwargs.get("label") or "").startswith("gran-"):
            gran_calls += 1
            if gran_calls == 1:
                started.set()
                await release.wait()
        return _rand_embs(len(texts))

    monkeypatch.setattr(ms, "_embed_texts_with_runtime_secrets", _controlled_embed)

    background_build = asyncio.create_task(ms.build_index())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    ingest_result = await asyncio.wait_for(
        _ingest_asserted_facts(
            ms,
            facts=[{
                "id": "task_fact_2",
                "fact": "Concurrent task fact must persist successfully",
                "kind": "fact",
                "entities": [],
                "tags": [],
                "session": 1,
            }],
            raw_sessions=[{
                "raw_session_id": "rs_task_2",
                "session_num": 1,
                "session_date": "2024-06-03",
                "content": "Concurrent task content two",
            }],
            enrich_l0=False,
        ),
        timeout=5.0,
    )

    release.set()
    await asyncio.wait_for(background_build, timeout=5.0)

    monkeypatch.setattr(ms._storage, "load_embeddings", real_load_embeddings)
    saved = real_load_embeddings()

    assert "error" not in ingest_result
    assert ingest_result["granular_added"] == 1
    assert any(
        f.get("fact") == "Concurrent task fact must persist successfully"
        for f in ms._all_granular
    )
    assert gran_calls >= 2
    assert saved is not None
    assert saved["gran"].shape[0] == len(ms._all_granular)


def test_recall_returns_context(tmp_path, monkeypatch):
    """recall() returns context string and metadata."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv3")

    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))
    result = asyncio.run(ms.recall("What happened?"))

    assert "context" in result
    assert "query_type" in result
    assert "retrieved" in result
    assert result["n_facts"] >= 3
    assert isinstance(result["context"], str)
    assert len(result["context"]) > 0


def test_cache_survives_restart(tmp_path, monkeypatch):
    """MemoryServer reloads persisted cache from disk on init."""
    _patch_all(monkeypatch)

    ms1 = MemoryServer(str(tmp_path), "conv4")
    asyncio.run(_store(ms1, "Hello", session_num=1, session_date="2024-06-01"))
    assert ms1.stats()["granular"] == 3

    # Create a new instance — should reload from cache
    ms2 = MemoryServer(str(tmp_path), "conv4")
    assert ms2.stats()["granular"] == 3
    assert ms2._n_sessions == 1


def test_ingest_document(tmp_path, monkeypatch):
    """ingest_document() keeps granular facts and substrate cross facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv5")

    # Two chunks (each >8000 chars) so cross-session threshold (2+ sessions) is met
    chunk1 = "First section. " * 600  # ~9000 chars
    chunk2 = "Second section. " * 600
    result = asyncio.run(_ingest_document(ms,
        content=f"{chunk1}\n\n{chunk2}",
        source_id="doc1",
    ))

    assert result["facts_extracted"] > 0
    s = ms.stats()
    assert s["granular"] > 0
    assert s["consolidated"] == 0
    assert s["cross_session"] > 0
    assert ms._all_cons == []
    for f in ms._all_cross:
        assert "scope" in f, f"cross fact missing scope: {f}"
        assert "agent_id" in f, f"cross fact missing agent_id: {f}"
        assert "swarm_id" in f, f"cross fact missing swarm_id: {f}"
        assert "created_at" in f, f"cross fact missing created_at: {f}"


def test_episode_original_text_from_blocks_uses_original_raw_span():
    original = "Header\n\n  Café trial’s line,  \nnext line\n\nTail"
    start = original.index("  Café")
    end = original.index("\n\nTail")
    blocks = {
        "DOC_b001": {
            "block_id": "DOC_b001",
            "text": "Cafe trial's line,\nnext line",
            "raw_span": [start, end],
        }
    }
    episode = {"provenance": {"block_ids": ["DOC_b001"]}}

    assert MemoryServer._episode_original_text_from_blocks(episode, blocks, original) == original[start:end]


def test_multipart_raw_docs_compose_from_ingress_raw_not_episode_text(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_multipart_raw_source", extract_model=None)
    source_id = "DOC-MULTI"
    ms._episode_corpus = {
        "documents": [
            {
                "doc_id": f"document:{source_id}",
                "episodes": [
                    {
                        "episode_id": f"{source_id}_e01",
                        "source_id": source_id,
                        "raw_original": "part one degraded episode text",
                        "raw_text": "part one normalized episode text",
                    }
                ],
            }
        ]
    }

    first = ms._store_document_original_source_text(
        source_id,
        "part one raw’s exact text",
        multipart_part_key="part-1",
        metadata={"part_idx": 1},
    )
    stored = ms._store_document_original_source_text(
        source_id,
        "part two raw’s exact text",
        multipart_part_key="part-2",
        metadata={"part_idx": 2},
    )
    replaced = ms._store_document_original_source_text(
        source_id,
        "part one updated raw’s exact text",
        multipart_part_key="part-1",
        metadata={"part_idx": 1},
    )

    assert first == "part one raw’s exact text"
    assert stored == "part one raw’s exact text\n\npart two raw’s exact text"
    assert replaced == "part one updated raw’s exact text\n\npart two raw’s exact text"
    assert ms._raw_docs[source_id] == replaced
    assert "part one raw’s exact text\n\npart one updated" not in replaced
    assert "degraded episode" not in replaced
    assert "normalized episode" not in replaced


def test_ingest_document_raw_docs_preserve_original_source_not_canonical_text(tmp_path, monkeypatch):
    _patch_all(monkeypatch, n_facts=1)

    async def _fake_canonicalize(self, block_dicts, *, model, call_extract_fn):
        canonical_blocks = []
        cursor = 0
        for block in block_dicts:
            canonical_text = str(block.get("text") or "").replace("’", "'").rstrip()
            canonical = dict(block)
            canonical["original_text"] = block.get("text")
            canonical["original_raw_span"] = list(block.get("raw_span") or [0, len(canonical_text)])
            canonical["text"] = canonical_text
            canonical["text_preview"] = canonical_text[:240]
            canonical["char_len"] = len(canonical_text)
            canonical["raw_span"] = [cursor, cursor + len(canonical_text)]
            canonical_blocks.append(canonical)
            cursor += len(canonical_text) + 2
        canonical_doc_text = "\n\n".join(str(block.get("text") or "") for block in canonical_blocks)
        return {
            "canonical_blocks": canonical_blocks,
            "canonical_doc_text": canonical_doc_text,
            "source_lang": "en",
            "semantic_ready": True,
            "canonicalization_status": "ready",
            "canonicalization_error": None,
            "translation_version": "test",
        }

    monkeypatch.setattr(MemoryServer, "_canonicalize_document_blocks_for_retrieval", _fake_canonicalize)
    ms = MemoryServer(str(tmp_path), "doc_raw_original")
    original = (
        "[Artifact 0001]\n"
        "Instruction:\n"
        "copy formatting\n\n"
        "Response:\n"
        "  Café trial’s line,  \n"
        "next line\n\n"
    )

    result = asyncio.run(_ingest_document(ms, content=original, source_id="DOC-RAW"))

    assert result["facts_extracted"] > 0
    assert ms._raw_docs["DOC-RAW"] == original
    assert any("trial’s" in str(ep.get("raw_original") or "") for doc in ms._episode_corpus["documents"] for ep in doc["episodes"])

    reloaded = MemoryServer(str(tmp_path), "doc_raw_original")
    assert reloaded._raw_docs["DOC-RAW"] == original


def test_backfill_original_raw_sources_repairs_cache_without_reextracting(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_raw_backfill", extract_model=None)
    source_id = "DOC-BACKFILL"
    original = (
        "[Artifact 0001]\n"
        "Instruction:\n"
        "copy formatting\n\n"
        "Response:\n"
        "  trial’s exact line,  \n"
        "done\n"
    )
    ms._source_records[source_id] = {
        "source_id": source_id,
        "family": "document",
        "version_id": "v1",
        "source_meta": {"logical_source_id": source_id},
    }
    ms._episode_corpus = {
        "documents": [
            {
                "doc_id": f"document:{source_id}",
                "episodes": [
                    {
                        "episode_id": "DOC-BACKFILL_e01",
                        "source_id": source_id,
                        "source_type": "document",
                        "raw_original": "trial's degraded line,\ndone",
                        "raw_text": "trial's degraded line,\ndone",
                        "artifact_span_id": f"{source_id}::artifact::0001",
                    }
                ],
            }
        ]
    }
    ms._all_granular = [{"id": "fact-1", "fact": "semantic cache stays put"}]
    ms._save_cache()
    manifest = tmp_path / "raw_manifest.json"
    manifest.write_text(
        json.dumps({"sources": [{"source_id": source_id, "original_content": original}]}),
        encoding="utf-8",
    )
    facts_before = deepcopy(ms._all_granular)

    result = ms.backfill_original_raw_sources(manifest)

    assert result["backfilled"] == [source_id]
    assert result["extraction_rerun"] is False
    assert ms._raw_docs[source_id] == original
    assert ms._all_granular == facts_before
    assert ms._source_records[source_id]["source_meta"]["raw_source_backfilled"] is True
    assert ms.validate_document_raw_sources()["ok"] is True

    reloaded = MemoryServer(str(tmp_path), "doc_raw_backfill", extract_model=None)
    assert reloaded._raw_docs[source_id] == original


def test_backfill_original_raw_sources_refuses_expected_answer_fixture(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_raw_backfill_refuse", extract_model=None)
    source_id = "DOC-ANSWER"
    ms._source_records[source_id] = {
        "source_id": source_id,
        "family": "document",
        "version_id": "v1",
        "source_meta": {"logical_source_id": source_id},
    }
    manifest = tmp_path / "answer_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": source_id,
                        "content_kind": "expected_answer",
                        "original_content": "not allowed",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ms.backfill_original_raw_sources(manifest)

    assert result["backfilled"] == []
    assert result["refused"] == [{"source_id": source_id, "code": "expected_answer_source_forbidden"}]
    assert source_id not in ms._raw_docs


def test_backfill_original_raw_sources_resolves_logical_source_id(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_raw_backfill_logical", extract_model=None)
    projected_source_id = "projected:DOC-LOGICAL"
    ms._source_records[projected_source_id] = {
        "source_id": projected_source_id,
        "family": "document",
        "version_id": "v1",
        "source_meta": {"logical_source_id": "DOC-LOGICAL"},
    }
    manifest = tmp_path / "logical_manifest.json"
    manifest.write_text(
        json.dumps({"sources": [{"source_id": "DOC-LOGICAL", "original_content": "Original raw source"}]}),
        encoding="utf-8",
    )

    result = ms.backfill_original_raw_sources(manifest)

    assert result["backfilled"] == [projected_source_id]
    assert ms._raw_docs[projected_source_id] == "Original raw source"


@pytest.mark.asyncio
async def test_admin_mcp_backfill_original_raw_sources_requires_admin_and_rebuilds_graph(tmp_path, monkeypatch):
    import src.mcp_server as mcp_mod
    from tests._auth_helpers import bootstrap_harness

    harness = bootstrap_harness(monkeypatch, tmp_path, patch_extraction=True)
    key = "admin_raw_backfill"
    source_id = "DOC-ADMIN-BACKFILL"
    original = (
        "[Artifact 0001]\n"
        "Instruction:\n"
        "copy exact raw\n\n"
        "Response:\n"
        "  admin backfill’s exact line  \n"
    )
    ms = MemoryServer(str(tmp_path), key, extract_model=None)
    ms._instance_config = {"owner_id": harness.admin_principal_id, "read": [], "write": []}
    ms._source_records[source_id] = {
        "source_id": source_id,
        "family": "document",
        "version_id": "v1",
        "source_meta": {"logical_source_id": source_id},
    }
    ms._episode_corpus = {
        "documents": [
            {
                "doc_id": f"document:{source_id}",
                "episodes": [
                    {
                        "episode_id": f"{source_id}_e01",
                        "source_id": source_id,
                        "source_type": "document",
                        "raw_text": "degraded episode text",
                        "raw_original": "degraded episode text",
                        "artifact_span_id": f"{source_id}::artifact::0001",
                    }
                ],
            }
        ]
    }
    ms._all_granular = [{"id": "fact-1", "fact": "semantic cache stays unchanged"}]
    facts_before = deepcopy(ms._all_granular)
    mcp_mod.registry.clear()
    mcp_mod.registry[key] = ms

    async def _unexpected_extract(*args, **kwargs):
        raise AssertionError("admin raw backfill must not rerun extraction")

    monkeypatch.setattr("src.memory.extract_session", _unexpected_extract)
    agent_token = harness.issue("agent:not-admin", kind="agent", token_kind="agent")

    denied = await mcp_mod.memory_admin_backfill_original_raw_sources(
        key=key,
        sources=[{"source_id": source_id, "original_content": original}],
        token=agent_token,
    )
    refused = await mcp_mod.memory_admin_backfill_original_raw_sources(
        key=key,
        sources=[{"source_id": source_id, "content_kind": "expected_answer", "original_content": original}],
        token=harness.admin_token,
    )
    result = await mcp_mod.memory_admin_backfill_original_raw_sources(
        key=key,
        sources=[{"source_id": source_id, "original_content": original}],
        token=harness.admin_token,
    )

    assert denied["code"] == "FORBIDDEN"
    assert refused["status"] == "ok"
    assert refused["refused"] == [{"source_id": source_id, "code": "expected_answer_source_forbidden"}]
    assert result["status"] == "ok"
    assert result["backfill_path"] == "memory_admin_api"
    assert result["no_new_extraction"] is True
    assert result["expected_answers_read"] is False
    assert result["extraction_rerun"] is False
    assert result["backfilled"] == [source_id]
    assert ms._raw_docs[source_id] == original
    assert ms._all_granular == facts_before
    assert ms._source_records[source_id]["source_meta"]["raw_source_backfilled"] is True
    assert ms.validate_document_raw_sources()["ok"] is True
    render_refs = [
        row
        for row in ms._container_graph.get("render_refs", [])
        if (row.get("ref_json") or {}).get("ref_type") == "document_artifact_response_text"
    ]
    assert render_refs
    assert render_refs[0]["ref_json"]["render_source"] == "raw_doc_marker_span"
    assert "admin backfill’s exact line" in render_refs[0]["ref_json"]["text"]


def test_mrcr_backfill_wrapper_only_classifies_sqlcipher_runtime_errors():
    from scripts.backfill_mrcr_cache_raw_sources import _is_sqlcipher_unavailable

    assert _is_sqlcipher_unavailable(RuntimeError("pysqlcipher3 module is not installed"))
    assert not _is_sqlcipher_unavailable(RuntimeError("schema migration failed"))


@pytest.mark.asyncio
async def test_admin_mcp_backfill_authorizes_before_loading_memory(monkeypatch):
    import src.mcp_server as mcp_mod

    def _unexpected_get_memory(_key):
        raise AssertionError("non-admin raw backfill must not load memory instance")

    monkeypatch.setattr(mcp_mod, "_get_memory", _unexpected_get_memory)

    result = await mcp_mod.memory_admin_backfill_original_raw_sources(
        key="missing-or-sensitive-key",
        sources=[],
        token="",
    )

    assert result["code"] == "AUTH_REQUIRED"


def test_generic_mcp_backfill_tool_requires_manifest_not_cache_root(tmp_path, monkeypatch, capsys):
    from scripts.backfill_raw_sources_via_mcp import main as wrapper_main

    monkeypatch.setattr(
        "sys.argv",
        [
            "backfill",
            "--cache-root",
            str(tmp_path),
            "--endpoint",
            "http://127.0.0.1:9",
            "--key",
            "current",
            "--admin-token",
            "admin-token",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        wrapper_main()

    assert exc.value.code == 2
    assert "manifest" in capsys.readouterr().err.lower()


def test_generic_mcp_backfill_wrapper_exits_nonzero_on_tool_failure(tmp_path, monkeypatch, capsys):
    import scripts.backfill_raw_sources_via_mcp as wrapper

    manifest = tmp_path / "raw_manifest.json"
    manifest.write_text(
        json.dumps({"sources": [{"source_id": "src-1", "original_content": "raw", "content_kind": "original_source"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        wrapper,
        "_tool_call",
        lambda *_args, **_kwargs: {"status": "ok", "missing": ["src-1"], "refused": [], "validation": {"ok": False}},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "backfill",
            "--manifest",
            str(manifest),
            "--endpoint",
            "http://127.0.0.1:9",
            "--key",
            "current",
            "--admin-token",
            "admin-token",
        ],
    )

    exit_code = wrapper.main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["status"] == "BACKFILL_FAILED"
    assert output["BACKFILL_PATH"] == "memory_admin_api"
    assert output["EXPECTED_ANSWERS_READ"] is False


def test_augment_commonality_facts_prefers_interest_pairs_over_event_pairs():
    facts = [
        {"id": "j_event", "fact": "Joanna took a road trip for research for her next movie.", "session": 9, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_event", "fact": "Nate thinks the road trip sounds great.", "session": 9, "speaker": "Nate", "entities": ["Nate"]},
        {"id": "j_movie", "fact": "Joanna enjoys reading, watching movies, and exploring nature, in addition to writing.", "session": 1, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_movie", "fact": "Nate's main hobbies are playing video games and watching movies.", "session": 1, "speaker": "Nate", "entities": ["Nate"]},
        {"id": "j_dessert", "fact": "Joanna tries to make dairy-free desserts just as delicious as non-dairy ones.", "session": 10, "speaker": "Joanna", "entities": ["Joanna"]},
        {"id": "n_dessert", "fact": "Nate started teaching people how to make dairy-free desserts.", "session": 10, "speaker": "Nate", "entities": ["Nate"]},
    ]

    extras = _augment_commonality_facts(
        "What kind of interests do Joanna and Nate share?",
        [],
        facts,
        limit=4,
    )

    extra_ids = [fact["id"] for fact in extras]
    assert "j_movie" in extra_ids
    assert "n_movie" in extra_ids
    assert "j_dessert" in extra_ids
    assert "n_dessert" in extra_ids
    assert "j_event" not in extra_ids
    assert "n_event" not in extra_ids


def test_conversation_structural_packet_can_use_cross_only_substrate_facts(tmp_path, monkeypatch):
    async def mock_embed_query(text, **kwargs):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)

    ms = MemoryServer(str(tmp_path), "conv_cross_only")
    ms._all_granular = []
    ms._all_cross = [
        {
            "id": "substrate_shared_root",
            "fact": "Both lost jobs and started their own businesses.",
            "kind": "fact",
            "entities": ["Jon", "Gina"],
            "source_id": "conv-30_cat1",
            "session": 19,
            "metadata": {
                "source_aggregation": True,
                "episode_id": "conv-30_cat1_e01",
                "episode_ids": ["conv-30_cat1_e01"],
            },
        }
    ]
    ms._data_dict = {
        "cross_embs": np.array([[1.0, 0.0]], dtype=np.float32),
    }

    packet = {
        "query_operator_plan": {
            "commonality": {"enabled": True},
            "list_set": {"enabled": False},
            "compare_diff": {"enabled": False},
        },
        "retrieved_episode_ids": ["conv-30_cat1_e01"],
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600}},
    }
    episode_lookup = {
        "conv-30_cat1_e01": {
            "episode_id": "conv-30_cat1_e01",
            "source_id": "conv-30_cat1",
            "source_type": "conversation",
            "raw_text": "Jon lost his job as a banker. Gina lost her Door Dash job. Both started their own businesses.",
        }
    }

    augmented_packet, retrieved = asyncio.run(
        ms._augment_conversation_structural_packet(
            query="What do Jon and Gina have in common?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert [fact["id"] for fact in retrieved] == ["substrate_shared_root"]
    assert augmented_packet["retrieved_fact_ids"] == ["substrate_shared_root"]
    assert "Both lost jobs and started their own businesses." in augmented_packet["context"]
    assert augmented_packet["actual_injected_episode_ids"] == ["conv-30_cat1_e01"]


def test_document_structural_packet_can_use_cross_only_substrate_facts(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_cross_only")
    ms._all_granular = []
    ms._all_cross = [
        {
            "id": "substrate_permit_record",
            "fact": "permit T-17 status approved date 2026-02-12 section Operations Update.",
            "kind": "fact",
            "entities": ["permit_T_17"],
            "source_id": "DOC-022",
            "session": 10,
            "metadata": {
                "source_aggregation": True,
                "episode_id": "DOC-022_e10",
                "episode_ids": ["DOC-022_e10"],
            },
        }
    ]
    ms._data_dict = {}

    packet = {
        "query_operator_plan": {
            "bounded_chain": {"enabled": True},
        },
        "retrieved_episode_ids": ["DOC-022_e10"],
        "retrieved_fact_ids": [],
        "selector_config": {"budget": 4000},
        "tuning_snapshot": {"packet": {"snippet_chars": 600, "query_specificity_bonus": 0.0}},
    }
    episode_lookup = {
        "DOC-022_e10": {
            "episode_id": "DOC-022_e10",
            "source_id": "DOC-022",
            "source_type": "document",
            "raw_text": "Permit T-17 was approved on 2026-02-12.",
        }
    }

    augmented_packet, retrieved = asyncio.run(
        ms._augment_document_structural_packet(
            query="Which permit was approved?",
            packet=packet,
            episode_lookup=episode_lookup,
            fact_filter=lambda _fact: True,
        )
    )

    assert retrieved is not None
    assert any(fact["id"] == "substrate_permit_record" for fact in retrieved)
    assert "substrate_permit_record" in augmented_packet["retrieved_fact_ids"]
    assert "Permit T-17 was approved on 2026-02-12." in augmented_packet["context"]


def test_mark_tiers_dirty_keeps_only_current_supported_doc_cross_facts(tmp_path):
    ms = MemoryServer(str(tmp_path), "doc_cross_lifecycle")
    ms._source_records["DOC-LIFE"] = {
        "family": "document",
        "artifact_id": "artifact-new",
        "version_id": "v2",
    }
    ms._all_granular = [{
        "id": "doc-fact-new",
        "fact": "Current document fact.",
        "source_id": "DOC-LIFE",
        "artifact_id": "artifact-new",
        "version_id": "v2",
        "status": "active",
    }]
    ms._all_cross = [
        {
            "id": "old-cross",
            "fact": "Old stale cross.",
            "source_id": "DOC-LIFE",
            "artifact_id": "artifact-old",
            "version_id": "v1",
            "status": "active",
            "metadata": {"source_aggregation": True, "source_id": "DOC-LIFE"},
        },
        {
            "id": "new-cross",
            "fact": "Current supported cross.",
            "source_id": "DOC-LIFE",
            "artifact_id": "artifact-new",
            "version_id": "v2",
            "status": "active",
            "metadata": {"source_aggregation": True, "source_id": "DOC-LIFE"},
        },
        {
            "id": "legacy-unversioned-cross",
            "fact": "Legacy unversioned cross.",
            "source_id": "DOC-LIFE",
            "status": "active",
            "metadata": {"source_aggregation": True, "source_id": "DOC-LIFE"},
        },
        {
            "id": "unsupported-cross",
            "fact": "Unsupported cross.",
            "source_id": "DOC-MISSING",
            "status": "active",
            "metadata": {"source_aggregation": True, "source_id": "DOC-MISSING"},
        },
        {
            "id": "asserted-cross",
            "fact": "Asserted derived cross.",
            "status": "active",
            "metadata": {"asserted_derived_tier": True},
        },
    ]

    ms._mark_tiers_dirty()

    assert [fact["id"] for fact in ms._all_cross] == ["new-cross", "asserted-cross"]


def test_reload_runtime_from_storage_drops_stale_doc_cross_facts(tmp_path):
    class StorageStub:
        exists = True

        def load_facts(self, *, internal=False):
            assert internal is True
            return {
                "granular": [{
                    "id": "doc-fact-new",
                    "fact": "Current document fact.",
                    "source_id": "DOC-RELOAD",
                    "artifact_id": "artifact-new",
                    "version_id": "v2",
                    "status": "active",
                }],
                "cons": [],
                "cross": [
                    {
                        "id": "old-cross",
                        "fact": "Old stale cross.",
                        "source_id": "DOC-RELOAD",
                        "artifact_id": "artifact-old",
                        "version_id": "v1",
                        "status": "active",
                        "metadata": {"source_aggregation": True, "source_id": "DOC-RELOAD"},
                    },
                    {
                        "id": "legacy-unversioned-cross",
                        "fact": "Legacy unversioned cross.",
                        "source_id": "DOC-RELOAD",
                        "status": "active",
                        "metadata": {"source_aggregation": True, "source_id": "DOC-RELOAD"},
                    },
                    {
                        "id": "new-cross",
                        "fact": "Current supported cross.",
                        "source_id": "DOC-RELOAD",
                        "artifact_id": "artifact-new",
                        "version_id": "v2",
                        "status": "active",
                        "metadata": {"source_aggregation": True, "source_id": "DOC-RELOAD"},
                    },
                    {
                        "id": "asserted-cross",
                        "fact": "Asserted derived cross.",
                        "status": "active",
                        "metadata": {"asserted_derived_tier": True},
                    },
                ],
                "tlinks": [],
                "raw_sessions": [],
                "raw_docs": {},
                "episode_corpus": {"documents": []},
                "container_graph": {},
                "source_records": {
                    "DOC-RELOAD": {
                        "family": "document",
                        "artifact_id": "artifact-new",
                        "version_id": "v2",
                    }
                },
                "n_sessions": 1,
                "n_sessions_with_facts": 1,
            }

    ms = MemoryServer(str(tmp_path), "doc_cross_reload")
    ms._storage = StorageStub()

    ms._reload_runtime_from_storage()

    assert [fact["id"] for fact in ms._all_cross] == ["new-cross", "asserted-cross"]


@pytest.mark.asyncio
async def test_temporal_recall_prefers_semantic_temporal_fact_over_conflicting_granular(tmp_path, monkeypatch):
    ms = MemoryServer(str(tmp_path), "episode_temporal_preference")
    episode_id = "conv-42_e01"
    source_id = "conv-42"
    ms._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [{
                "episode_id": episode_id,
                "source_id": source_id,
                "source_type": "conversation",
                "raw_text": (
                    'Joanna: I first watched "Eternal Sunshine of the Spotless Mind" around 3 years ago.'
                ),
                "topic_key": "session_1",
                "state_label": "session",
                "currentness": "unknown",
            }],
        }],
    }
    ms._all_granular = [
        {
            "id": "g_wrong_2020",
            "fact": "Joanna first watched the movie around 2020.",
            "kind": "fact",
            "entities": ["Joanna"],
            "source_id": source_id,
            "event_date": "2020",
            "metadata": {"episode_id": episode_id, "episode_source_id": source_id},
        }
    ]
    ms._all_cons = []
    ms._all_cross = [
        {
            "id": "substrate_2019",
            "fact": "Joanna first watched Eternal Sunshine of the Spotless Mind in 2019.",
            "kind": "fact",
            "entities": ["Joanna", "Eternal Sunshine of the Spotless Mind"],
            "source_id": source_id,
            "metadata": {
                "semantic_class": "temporal_semantics",
                "source_aggregation": True,
                "episode_id": episode_id,
                "episode_ids": [episode_id],
                "episode_source_id": source_id,
                "resolved_year": 2019,
            },
        }
    ]
    ms._data_dict = {
        "atomic_embs": np.array([[1.0, 0.0]], dtype=float),
        "cons_embs": np.zeros((0, 2), dtype=float),
        "cross_embs": np.array([[1.0, 0.0]], dtype=float),
        "fact_lookup": {
            "g_wrong_2020": ms._all_granular[0],
            "substrate_2019": ms._all_cross[0],
        },
    }
    ms._fact_lookup = dict(ms._data_dict["fact_lookup"])

    async def _fake_embed_query(_text, model=None, provider=None):
        return np.array([1.0, 0.0], dtype=float)

    monkeypatch.setattr("src.memory.embed_query", _fake_embed_query)

    result = await ms.recall('When did Joanna first watch "Eternal Sunshine of the Spotless Mind"?')

    retrieved_ids = [fact["id"] for fact in result["retrieved"]]
    assert "substrate_2019" in retrieved_ids
    assert "g_wrong_2020" not in retrieved_ids
    assert "2020" not in result["context"]
def test_context_for_passes_swarm_id(tmp_path, monkeypatch):
    """Bug A regression: context_for must pass swarm_id to recall."""
    _patch_all(monkeypatch, scope="swarm-shared", swarm_id="sw1")
    ms = MemoryServer(str(tmp_path), "conv_cfor", scope="swarm-shared", swarm_id="sw1")
    asyncio.run(_store(ms, "Test", session_num=1, session_date="2024-06-01"))

    # context_for must accept and forward swarm_id
    result = asyncio.run(ms.context_for("query", agent_id="a1", swarm_id="sw1"))
    assert "context" in result


def test_concurrent_store_no_corruption(tmp_path, monkeypatch):
    """Multiple concurrent store() calls don't corrupt data."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv10")

    async def _concurrent():
        tasks = [
            _store(ms, f"Message {i}", session_num=i + 1, session_date="2024-06-01")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(_concurrent())

    # All 5 stores returned 3 facts each
    assert all(r["facts_extracted"] == 3 for r in results)
    # Total facts = 5 * 3 = 15
    assert ms.stats()["granular"] == 15
    assert ms._n_sessions == 5
    # Persisted snapshot remains readable through the storage backend
    data = ms._storage.load_facts()
    assert len(data["granular"]) == 15


# ── 3-tier guarantee tests ──


def test_tiers_dirty_set_on_store(tmp_path, monkeypatch):
    """store() must set _tiers_dirty = True."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_dirty1")

    assert ms._tiers_dirty is False
    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True


def test_tiers_dirty_set_on_ingest_document(tmp_path, monkeypatch):
    """ingest_document() must set _tiers_dirty = True."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_dirty3")

    assert ms._tiers_dirty is False
    chunk1 = "First section. " * 600
    chunk2 = "Second section. " * 600
    asyncio.run(_ingest_document(ms,
        content=f"{chunk1}\n\n{chunk2}",
        source_id="doc1",
    ))
    assert ms._tiers_dirty is True


def test_build_index_rebuilds_tiers(tmp_path, monkeypatch):
    """build_index() must rebuild substrate cross facts when dirty."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_rebuild1")

    # Store 3 facts
    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True

    # build_index should rebuild tiers and clear the flag
    result = asyncio.run(ms.build_index())
    assert ms._tiers_dirty is False
    assert result["granular"] == 3
    assert result["consolidated"] == 0
    assert result["cross_session"] >= 1


def test_build_index_namespaces_derived_tier_ids(tmp_path, monkeypatch):
    """Derived tiers must never reuse granular extractor IDs."""
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)

    async def mock_extract_session(**kwargs):
        sn = kwargs.get("session_num", 1)
        return _fake_extract_result(3, session=sn)

    async def mock_extract_source_aggregation_facts(self, **kwargs):
        return [{
            "id": "s1_f_01",
            "fact": "Derived duplicate id",
            "kind": "fact",
            "entities": ["Alice"],
            "tags": ["substrate"],
            "source_ids": ["s1_f_01"],
            "metadata": {"source_aggregation": True},
        }]

    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_extract_source_aggregation_facts)

    ms = MemoryServer(str(tmp_path), "conv_collision")
    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))

    result = asyncio.run(ms.build_index())
    assert result["cross_session"] >= 1
    assert ms._all_granular[0]["id"] != ms._all_cross[0]["id"]
    assert ms._all_cross[0]["id"].startswith("substrate_")


def test_build_index_rebuilds_dirty_tiers(tmp_path, monkeypatch):
    """build_index() rebuilds dirty derived tiers and clears the dirty flag."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_flush1")

    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))
    assert ms._tiers_dirty is True

    result = asyncio.run(ms.build_index())
    assert ms._tiers_dirty is False
    assert result["granular"] >= 1
    assert "cross_session" in result


def test_rebuild_tiers_replaces_not_appends(tmp_path, monkeypatch):
    """_rebuild_tiers() replaces substrate cross, not appends."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_nodup")

    asyncio.run(_store(ms, "Hello world", session_num=1, session_date="2024-06-01"))

    # Rebuild twice — cons/cross counts should stay the same, not double
    asyncio.run(ms._rebuild_tiers())
    cons_after_first = len(ms._all_cons)
    cross_after_first = len(ms._all_cross)

    asyncio.run(ms._rebuild_tiers())
    assert len(ms._all_cons) == 0 == cons_after_first
    assert len(ms._all_cross) == cross_after_first


def test_rebuild_tiers_respects_scope_boundaries(tmp_path, monkeypatch):
    """_rebuild_tiers() preserves identity fields on substrate cross facts."""
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_scope1")

    # Store facts from two different agents with different scopes
    asyncio.run(_store(ms, "Agent A data", session_num=1, session_date="2024-06-01",
                         agent_id="agent-A", scope="agent-private"))
    asyncio.run(_store(ms, "Agent B data", session_num=2, session_date="2024-06-01",
                         agent_id="agent-B", scope="agent-private"))

    asyncio.run(ms._rebuild_tiers())

    assert ms._all_cons == []
    assert len({f["id"] for f in ms._all_cross}) == len(ms._all_cross)
    for f in ms._all_cross:
        assert "agent_id" in f, f"cross fact missing agent_id: {f}"
        assert "swarm_id" in f, f"cross fact missing swarm_id: {f}"
        assert "scope" in f, f"cross fact missing scope: {f}"


def test_store_write_through_creates_root_write_log_without_snapshot_dependency(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_store_root")

    def _boom(_data):
        raise AssertionError("sync store() should not depend on save_facts for ingress truth")

    monkeypatch.setattr(ms._storage, "save_facts", _boom)

    result = asyncio.run(
        _store(ms,
            "Store root write-log path",
            session_num=1,
            session_date="2024-06-01",
            message_id="root-store-1",
            agent_id="agent-a",
            swarm_id="swarm-a",
            scope="swarm-shared",
        )
    )

    assert result["facts_extracted"] == 3
    status = ms.write_status("root-store-1")
    assert status is not None
    assert status["extraction_state"] == "complete"
    entries = [entry for entry in ms._storage.list_write_log_entries(states=["complete"]) if entry["message_id"] == "root-store-1"]
    assert len(entries) == 1
    assert entries[0]["agent_id"] == "agent-a"
    assert entries[0]["swarm_id"] == "swarm-a"
    assert entries[0]["scope"] == "swarm-shared"
    assert entries[0]["content_family"] == "conversation"
    assert len([rs for rs in ms._raw_sessions if rs.get("message_id") == "root-store-1"]) == 1


def test_ingest_document_write_through_uses_single_root_row_and_preserves_acl(tmp_path, monkeypatch):
    _patch_all(monkeypatch)
    ms = MemoryServer(str(tmp_path), "conv_doc_root")

    def _boom(_data):
        raise AssertionError("sync ingest_document() should not depend on save_facts for ingress truth")

    monkeypatch.setattr(ms._storage, "save_facts", _boom)

    result = asyncio.run(
        _ingest_document(ms,
            content="Document body. " * 200,
            source_id="doc-root-1",
            message_id="doc-root-msg-1",
            agent_id="agent-a",
            swarm_id="swarm-a",
            scope="agent-private",
            metadata={"origin": "test"},
        )
    )

    assert result["facts_extracted"] > 0
    status = ms.write_status("doc-root-msg-1")
    assert status is not None
    assert status["extraction_state"] == "complete"
    entries = [entry for entry in ms._storage.list_write_log_entries(states=["complete"]) if entry["message_id"] == "doc-root-msg-1"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["agent_id"] == "agent-a"
    assert entry["swarm_id"] == "swarm-a"
    assert entry["scope"] == "agent-private"
    assert entry["owner_id"] == "agent:agent-a"
    assert entry["read"] == []
    assert entry["write"] == []
    assert entry["content_family"] == "document"
    with ms._storage._connect() as conn:
        raw_doc_row = conn.execute(
            "SELECT message_id FROM raw_docs WHERE source_id = ?",
            ("doc-root-1",),
        ).fetchone()
        extra_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM write_log WHERE message_id = ?",
            ("rawdoc:doc-root-1",),
        ).fetchone()
    assert raw_doc_row is not None
    assert raw_doc_row["message_id"] == "doc-root-msg-1"
    assert int(extra_rows["n"]) == 0


def _valid_fact_flag(flag_id: str = "flag_test") -> dict:
    return {
        "flag_id": flag_id,
        "producer": "unified_source_extractor",
        "category": "extraction",
        "severity": "high",
        "status": "resolved",
        "message": "support_fact_ids referenced an unknown fact and was repaired",
        "code": "ref.unknown_id",
        "path": "events[0].support_fact_ids[0]",
        "object_id": "event_flagged",
        "resolution": "repaired",
        "repair_attempted": True,
        "details": {"raw_code": "unknown_fact_reference"},
    }


@pytest.mark.asyncio
async def test_store_persists_block_extraction_report_on_raw_session(tmp_path, monkeypatch):
    async def mock_extract_session(**kwargs):
        assert kwargs.get("return_report") is True
        return (
            "conv_diag",
            1,
            "2024-06-01",
            [{
                "id": "f_diag",
                "fact": "Jon started his own business.",
                "kind": "event",
                "entities": ["Jon"],
                "tags": ["test"],
                "session": 1,
            }],
            [],
            {
                "report_id": "report_extract",
                "report_kind": "extraction",
                "producer": "block_extractor",
                "status": "partial",
                "entries": [{
                    "entry_id": "entry_extract",
                    "target_path": "facts[0]",
                    "status": "dropped",
                    "repair_attempted": True,
                    "issue": {
                        "normalized_code": "shape.list_item_type_error",
                        "raw_code": "invalid_fact_item_type",
                        "message": "bad fact item",
                    },
                    "details": None,
                }],
                "summary": {"entry_count": 1},
            },
        )

    async def mock_source_agg(self, **kwargs):
        return []

    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_source_agg)

    ms = MemoryServer(str(tmp_path), "store_extraction_report")
    result = await _store(ms, "User: Jon started his own business.", session_num=1, session_date="2024-06-01")

    assert result["status"] == "ok"
    artifact = ms._raw_sessions[0]["extraction_report"]
    assert artifact["producer"] == "block_extractor"
    assert artifact["entries"][0]["target_path"] == "facts[0]"


@pytest.mark.asyncio
async def test_source_aggregation_report_is_materialized_to_runtime_artifacts(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    ms = MemoryServer(str(tmp_path), "source_agg_report")

    source_id = "conv-30_cat1"
    ms._raw_sessions = [{
        "raw_session_id": "rs1",
        "session_num": 1,
        "session_date": "2024-06-01",
        "content": "Jon started his own business.",
        "source_id": source_id,
        "logical_source_id": source_id,
        "status": "active",
    }]
    ms._source_records[source_id] = {
        "family": "conversation",
        "source_meta": {},
    }
    ms._episode_corpus = {
        "documents": [{
            "doc_id": f"conversation:{source_id}",
            "episodes": [{
                "episode_id": f"{source_id}_e01",
                "source_id": source_id,
                "source_type": "conversation",
                "source_date": "2024-06-01",
                "topic_key": "session_1",
                "state_label": "session",
                "currentness": "unknown",
                "raw_text": "Jon started his own business.",
                "metadata": {},
            }],
        }],
    }

    async def mock_extract_source_aggregation(**kwargs):
        return {
            "validation": {
                "aggregation_status": "failed",
                "accepted_layers": [],
                "dropped_layers": ["event_layer"],
                "failure_reasons": ["repair.exhausted"],
            },
            "derived_facts": [],
            "source_aggregation_report": {
                "report_id": "report_source_agg",
                "report_kind": "source_aggregation",
                "producer": "unified_source_extractor",
                "status": "failed",
                "entries": [{
                    "entry_id": "entry_source_agg",
                    "target_path": "events[0]",
                    "status": "dropped",
                    "repair_attempted": True,
                    "issue": {
                        "normalized_code": "repair.exhausted",
                        "raw_code": "target_repair_exhausted",
                        "message": "could not repair event",
                    },
                    "details": None,
                }],
                "summary": {"entry_count": 1},
            },
        }

    monkeypatch.setattr("src.memory.extract_source_aggregation", mock_extract_source_aggregation)

    derived = await ms._extract_source_aggregation_facts(
        source_id=source_id,
        source_kind="conversation",
        source_facts=[{
            "id": "fact_1",
            "fact": "Jon started his own business.",
            "kind": "event",
            "entities": ["Jon"],
            "metadata": {"episode_id": f"{source_id}_e01", "episode_source_id": source_id},
        }],
        source_date="2024-06-01",
        model="qwen/qwen3-32b",
        call_extract_fn=None,
        agent_id="default",
    )

    assert derived == []
    raw_artifact = ms._raw_sessions[0]["source_aggregation_report"]
    assert raw_artifact["producer"] == "unified_source_extractor"
    assert raw_artifact["entries"][0]["target_path"] == "events[0]"
    source_artifact = ms._source_records[source_id]["source_meta"]["source_aggregation_report"]
    assert source_artifact["entries"][0]["issue"]["normalized_code"] == "repair.exhausted"


def test_reload_normalizes_legacy_report_fields_and_legacy_object_flags(tmp_path):
    ms = MemoryServer(str(tmp_path), "legacy_report_normalization")
    payload = ms._storage.load_facts(internal=True)
    payload["raw_sessions"] = [{
        "message_id": "raw:legacy-1",
        "raw_session_id": "legacy-1",
        "session_num": 1,
        "session_date": "2024-06-01",
        "content": "legacy session",
        "speakers": "User and Assistant",
        "stored_at": None,
        "format": "conversation",
        "source_id": "legacy-source",
        "artifact_id": None,
        "version_id": None,
        "content_hash": None,
        "status": "active",
        "agent_id": None,
        "swarm_id": None,
        "scope": None,
        "owner_id": None,
        "read": [],
        "write": [],
        "extraction_diagnostics": {
            "producer": "block_extractor",
            "diagnostics": [{
                "target_path": "facts[0]",
                "status": "dropped",
                "repair_attempted": True,
                "issue": {
                    "normalized_code": "shape.list_item_type_error",
                    "raw_code": "invalid_fact_item_type",
                    "message": "legacy bad fact",
                },
            }],
        },
    }]
    payload["source_records"] = {
        "legacy-source": {
            "family": "conversation",
            "owner_id": None,
            "read": [],
            "write": [],
            "artifact_id": None,
            "version_id": None,
            "content_hash": None,
            "metadata": {},
            "target": [],
            "source_meta": {
                "source_aggregation_diagnostics": {
                    "producer": "unified_source_extractor",
                    "diagnostics": [{
                        "target_path": "events[0]",
                        "status": "dropped",
                        "repair_attempted": True,
                        "issue": {
                            "normalized_code": "repair.exhausted",
                            "raw_code": "target_repair_exhausted",
                            "message": "legacy source agg failure",
                        },
                    }],
                },
                "flags": [{
                    "flag_id": "flag_legacy",
                    "origin": "extraction",
                    "producer": "unified_source_extractor",
                    "normalized_code": "ref.unknown_id",
                    "raw_code": "unknown_fact_reference",
                    "severity": "high",
                    "status": "resolved",
                    "message": "legacy extraction flag",
                    "path": "events[0].support_fact_ids[0]",
                    "object_id": "event_flagged",
                    "resolution": "repaired",
                    "repair_attempted": True,
                }],
            },
            "created_at": None,
            "updated_at": None,
        },
    }
    ms._storage.save_facts(payload)

    ms2 = MemoryServer(str(tmp_path), "legacy_report_normalization")
    assert ms2._raw_sessions[0]["extraction_report"]["entries"][0]["target_path"] == "facts[0]"
    source_record = ms2._source_records["legacy-source"]
    assert source_record["source_aggregation_report"]["entries"][0]["target_path"] == "events[0]"
    assert source_record["flags"][0]["category"] == "extraction"
    assert source_record["flags"][0]["code"] == "ref.unknown_id"


@pytest.mark.asyncio
async def test_store_accepts_extracted_fact_with_top_level_flags(tmp_path, monkeypatch):
    async def mock_extract_session(**kwargs):
        return (
            "conv_flags",
            1,
            "2024-06-01",
            [{
                "id": "f_flagged",
                "fact": "Jon started his own business.",
                "kind": "event",
                "entities": ["Jon"],
                "tags": ["test"],
                "session": 1,
                "flags": [_valid_fact_flag()],
            }],
            [],
        )

    async def mock_source_agg(self, **kwargs):
        return []

    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    monkeypatch.setattr("src.memory.extract_session", mock_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", mock_source_agg)

    ms = MemoryServer(str(tmp_path), "store_fact_flags")
    result = await _store(ms, "User: Jon started his own business.", session_num=1, session_date="2024-06-01")

    assert result["status"] == "ok"
    assert ms._all_granular[0]["flags"][0]["flag_id"] == "flag_test"


@pytest.mark.asyncio
async def test_ingest_asserted_facts_accepts_top_level_flags(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    ms = MemoryServer(str(tmp_path), "asserted_fact_flags")

    result = await _ingest_asserted_facts(
        ms,
        facts=[{
            "id": "f1",
            "fact": "Imported fact",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
            "flags": [_valid_fact_flag()],
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "Imported session",
        }],
        enrich_l0=False,
    )

    assert result["granular_added"] == 1
    assert ms._all_granular[0]["flags"][0]["flag_id"] == "flag_test"


@pytest.mark.asyncio
async def test_top_level_flags_survive_storage_and_load(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    ms = MemoryServer(str(tmp_path), "persist_fact_flags")

    await _ingest_asserted_facts(
        ms,
        facts=[{
            "id": "f1",
            "fact": "Imported fact",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
            "flags": [_valid_fact_flag("flag_persist")],
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "Imported session",
        }],
        enrich_l0=False,
    )

    ms2 = MemoryServer(str(tmp_path), "persist_fact_flags")
    assert ms2._all_granular[0]["flags"][0]["flag_id"] == "flag_persist"


@pytest.mark.asyncio
async def test_fact_flags_and_metadata_flags_can_coexist_without_changing_metadata_contract(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    ms = MemoryServer(str(tmp_path), "fact_and_metadata_flags")

    result = await _ingest_asserted_facts(
        ms,
        facts=[{
            "id": "f1",
            "fact": "Imported fact",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
            "flags": [_valid_fact_flag()],
            "metadata": {"flags": ["legacy-flag"]},
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "Imported session",
        }],
        enrich_l0=False,
    )

    assert result["granular_added"] == 1
    assert ms._all_granular[0]["flags"][0]["flag_id"] == "flag_test"
    assert ms._all_granular[0]["metadata"]["flags"] == ["legacy-flag"]


@pytest.mark.asyncio
async def test_invalid_fact_flags_schema_is_rejected_deterministically(tmp_path, monkeypatch):
    _patch_embeddings(monkeypatch)
    _patch_resolve_supersession(monkeypatch)
    ms = MemoryServer(str(tmp_path), "invalid_fact_flags")

    result = await _ingest_asserted_facts(
        ms,
        facts=[{
            "id": "f1",
            "fact": "Imported fact",
            "kind": "event",
            "entities": [],
            "tags": [],
            "session": 1,
            "flags": "bad-flags",
        }],
        raw_sessions=[{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "Imported session",
        }],
        enrich_l0=False,
    )

    assert result["code"] == "VALIDATION_ERROR"
    assert "flags must be a list" in result["error"]


@pytest.mark.asyncio
async def test_set_config_accepts_local_cli_profile_config(tmp_path):
    ms = MemoryServer(str(tmp_path), "local_cli_config_valid")

    await ms.set_config({
        "schema_version": 1,
        "embedding_model": "text-embedding-3-small",
        "librarian_profile": "fast",
        "profiles": {1: "fast"},
        "profile_configs": {
            "fast": {
                "backend": "local_cli",
                "model": "local/my-cli",
                "cli_bin": "/abs/path/to/my-cli",
                "cli_args_prefix": ["run"],
                "context_window": 200000,
                "max_output_tokens": 4096,
                "temperature": 0,
            }
        },
        "retrieval": {"search_family": "auto", "default_token_budget": 4000},
    })

    assert ms.get_config()["profile_configs"]["fast"]["backend"] == "local_cli"


@pytest.mark.asyncio
async def test_set_config_rejects_local_cli_profile_without_cli_bin(tmp_path):
    ms = MemoryServer(str(tmp_path), "local_cli_config_invalid")

    with pytest.raises(ValueError, match="cli_bin"):
        await ms.set_config({
            "schema_version": 1,
            "embedding_model": "text-embedding-3-small",
            "librarian_profile": "fast",
            "profiles": {1: "fast"},
            "profile_configs": {
                "fast": {
                    "backend": "local_cli",
                    "model": "local/my-cli",
                    "cli_args_prefix": ["run"],
                    "context_window": 200000,
                    "max_output_tokens": 4096,
                    "temperature": 0,
                }
            },
            "retrieval": {"search_family": "auto", "default_token_budget": 4000},
        })


@pytest.mark.asyncio
async def test_store_uses_local_cli_extraction_without_secret_ref(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)
    captured = {}

    def _fake_run_local_cli(prompt, cli_bin, cli_args_prefix):
        captured["prompt"] = prompt
        captured["cli_bin"] = cli_bin
        captured["cli_args_prefix"] = cli_args_prefix
        return json.dumps({
            "facts": [{
                "id": "f_01",
                "fact": "CLI extracted fact",
                "kind": "fact",
                "entities": [],
                "tags": [],
            }],
            "temporal_links": [],
        })

    monkeypatch.setattr("src.memory.run_local_cli", _fake_run_local_cli)
    ms = MemoryServer(str(tmp_path), "local_cli_extract")
    await ms.set_config({
        "schema_version": 1,
        "embedding_model": "text-embedding-3-small",
        "librarian_profile": "fast",
        "profiles": {1: "fast"},
        "profile_configs": {
            "fast": {
                "backend": "local_cli",
                "model": "local/my-cli",
                "cli_bin": "/abs/path/to/my-cli",
                "cli_args_prefix": ["run"],
                "context_window": 200000,
                "max_output_tokens": 4096,
                "temperature": 0,
            }
        },
        "retrieval": {"search_family": "auto", "default_token_budget": 4000},
    })

    result = await _store(ms, "Narrative content for local CLI extraction.", session_num=1, session_date="2024-06-01")

    assert result["status"] == "ok"
    assert result["facts_extracted"] >= 1
    assert ms._all_granular[0]["fact"] == "CLI extracted fact"
    assert captured["cli_bin"] == "/abs/path/to/my-cli"
    assert captured["cli_args_prefix"] == ["run"]
    assert captured["prompt"].startswith("SYSTEM:\n")


@pytest.mark.asyncio
async def test_store_local_cli_timeout_returns_explicit_failure_without_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)

    def _fake_run_local_cli(prompt, cli_bin, cli_args_prefix):
        raise LocalCliTimeoutError("local_cli subprocess timed out (timeout_secs=0.05)")

    monkeypatch.setattr("src.memory.run_local_cli", _fake_run_local_cli)
    ms = MemoryServer(str(tmp_path), "local_cli_extract_timeout")
    await ms.set_config({
        "schema_version": 1,
        "embedding_model": "text-embedding-3-small",
        "librarian_profile": "fast",
        "profiles": {1: "fast"},
        "profile_configs": {
            "fast": {
                "backend": "local_cli",
                "model": "local/my-cli",
                "cli_bin": "/abs/path/to/my-cli",
                "cli_args_prefix": ["run"],
                "context_window": 200000,
                "max_output_tokens": 4096,
                "temperature": 0,
            }
        },
        "retrieval": {"search_family": "auto", "default_token_budget": 4000},
    })

    mrcr_style_content = (
        "Alice: I left the keys in the red bowl by the door.\n"
        "Bob: Right, and the spare is taped under the kitchen table.\n"
        "Alice: Also remember that Carol borrowed the blue umbrella yesterday.\n"
        "Bob: I wrote that on the whiteboard next to the grocery list.\n"
        "Alice: Good, because tomorrow Dan needs the umbrella for the train station pickup."
    )

    result = await asyncio.wait_for(
        _store(ms, mrcr_style_content, session_num=1, session_date="2024-06-01"),
        timeout=1.0,
    )

    assert result["code"] == "LOCAL_CLI_TIMEOUT"
    assert result["status"] == "extraction_failed"
    assert result["facts_extracted"] == 0
    assert "timed out" in result["error"]
    assert ms._all_granular == []
    assert ms._raw_sessions[0]["status"] == "extraction_failed"
    assert ms._raw_sessions[0]["extraction_error_code"] == "LOCAL_CLI_TIMEOUT"


@pytest.mark.asyncio
async def test_write_log_local_cli_timeout_does_not_mark_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("src.memory.resolve_supersession", lambda facts, lookup: None)

    def _fake_run_local_cli(prompt, cli_bin, cli_args_prefix):
        raise LocalCliTimeoutError("local_cli subprocess timed out (timeout_secs=0.05)")

    monkeypatch.setattr("src.memory.run_local_cli", _fake_run_local_cli)
    ms = MemoryServer(str(tmp_path), "local_cli_write_timeout")
    await ms.set_config({
        "schema_version": 1,
        "embedding_model": "text-embedding-3-small",
        "librarian_profile": "fast",
        "profiles": {1: "fast"},
        "profile_configs": {
            "fast": {
                "backend": "local_cli",
                "model": "local/my-cli",
                "cli_bin": "/abs/path/to/my-cli",
                "cli_args_prefix": ["run"],
                "context_window": 200000,
                "max_output_tokens": 4096,
                "temperature": 0,
            }
        },
        "retrieval": {"search_family": "auto", "default_token_budget": 4000},
    })

    receipt = await _write(
        ms,
        message_id="local-cli-timeout-msg",
        session_id="sess-1",
        content="Alice: The backup key is behind the picture frame.\nBob: Carol needs it tomorrow morning.",
        content_family="chat",
        timestamp_ms=1712000000000,
    )
    assert receipt["extraction_state"] == "pending"

    monkeypatch.setattr(ms, "_should_retry_write_entry", lambda entry, now_ms: True)

    for _ in range(3):
        processed = await ms.process_write_log_once(batch_size=1)
        assert processed == 0

    status = ms.write_status("local-cli-timeout-msg")
    assert status["extraction_state"] == "failed"
    assert ms._all_granular == []
    assert any(rs.get("status") == "extraction_failed" for rs in ms._raw_sessions)


# ──────────────────────────────────────────────────────────────────────────
# _answer_core_for_grounding — citation/explanation tail stripping
# ──────────────────────────────────────────────────────────────────────────


def test_answer_core_strips_trailing_sources_block():
    # Regression: a correct answer with a "Sources: [1] (S3) ..." trailing
    # block was previously rejected by the strict-all token gate because
    # the citation tokens (sources, explicitly, S3) inflated the answer
    # token set. The helper must strip that tail.
    answer = (
        "John signed with the Minnesota Wolves on 21 May 2023. "
        "Sources: [1] (S1) explicitly states John signed with the Minnesota Wolves."
    )
    core = memory_mod._answer_core_for_grounding(answer)
    assert "Minnesota Wolves" in core
    assert "Sources" not in core
    assert "[1]" not in core
    assert "(S1)" not in core
    assert "explicitly states" not in core


def test_answer_core_strips_parenthesized_evidence_tail():
    # Regression: positive answer with "(Evidence: [1][2] explicitly state...)"
    # parenthesized tail.
    answer = (
        "Evan got a new Prius after his old Prius broke down. "
        "(Evidence: [1][2] explicitly state he repaired/sold the old Prius.)"
    )
    core = memory_mod._answer_core_for_grounding(answer)
    assert "new Prius" in core
    assert "Evidence" not in core
    assert "[1]" not in core
    assert "[2]" not in core


def test_answer_core_strips_according_to_prefix_clause():
    # "According to ..." prefix clause should be stripped, leaving the
    # substantive claim. Strip to the first comma/period only — do not
    # remove the whole sentence.
    answer = "According to the retrieved facts, the answer is Mary."
    core = memory_mod._answer_core_for_grounding(answer)
    assert "Mary" in core
    assert "According to" not in core
    assert "retrieved facts" not in core


def test_answer_core_preserves_plain_answer():
    # Non-citation text must pass through unchanged so we do not mask real
    # hallucinations.
    answer = "Caroline went to the LGBTQ support group on May 7, 2023."
    core = memory_mod._answer_core_for_grounding(answer)
    assert core == answer


def test_answer_core_handles_empty_input():
    assert memory_mod._answer_core_for_grounding("") == ""
    assert memory_mod._answer_core_for_grounding(None) == ""  # type: ignore[arg-type]


def test_answer_core_strips_meta_explanation_tail():
    # Trailing meta-explanation clauses like "As noted in fact [3], this..."
    # should be removed.
    answer = "Maria donated her old car. As noted in fact [3], this happened in December 2023."
    core = memory_mod._answer_core_for_grounding(answer)
    assert "old car" in core
    assert "As noted in fact" not in core


# ──────────────────────────────────────────────────────────────────────────
# _normalize_grounded_answer — gate integration via _answer_core
# ──────────────────────────────────────────────────────────────────────────


def _make_recall_with_facts(*fact_texts: str) -> dict:
    return {
        "retrieved": [{"fact": text, "id": f"f{i}"} for i, text in enumerate(fact_texts)],
    }


def test_grounded_gate_keeps_correct_answer_with_sources_tail(tmp_path):
    # Regression: the audit on full1986 found correct answers like
    # "John signed with the Minnesota Wolves on 21 May 2023. Sources: [1]
    # (S1) explicitly states ..." were rejected as Not mentioned because
    # the strict ``all(token in grounding_text)`` rule treated citation
    # tokens (sources, explicitly, S1) as ungrounded. With the
    # answer-core strip in place, the gate must keep the answer.
    ms = MemoryServer(str(tmp_path), "grounded_gate_sources_tail")
    recall = _make_recall_with_facts(
        "John signed with the Minnesota Wolves on 2023-05-21.",
        "John attended a press conference about the Minnesota Wolves signing.",
    )
    answer = (
        "John signed with the Minnesota Wolves on 21 May 2023. "
        "Sources: [1] (S1) explicitly states John signed."
    )
    out = ms._normalize_grounded_answer("What team did John sign with?", answer, recall)
    assert "Minnesota Wolves" in out
    assert out != "Not mentioned in the provided context."


def test_grounded_gate_keeps_correct_answer_with_evidence_parens(tmp_path):
    # Regression: positive answer with "(Evidence: [1][2] ...)" tail.
    ms = MemoryServer(str(tmp_path), "grounded_gate_evidence_parens")
    recall = _make_recall_with_facts(
        "Evan got a new Prius after his old Prius broke down.",
        "Evan repaired and sold the old Prius before getting a new Prius.",
    )
    answer = (
        "Evan got a new Prius after his old Prius broke down. "
        "(Evidence: [1][2] explicitly state he repaired/sold the old Prius.)"
    )
    out = ms._normalize_grounded_answer("What car did Evan get?", answer, recall)
    assert "Prius" in out
    assert out != "Not mentioned in the provided context."


def test_grounded_gate_rejects_unsupported_invented_answer(tmp_path):
    # Negative regression: a fabricated answer with no grounded tokens
    # in the recall must still be rejected. The gate must not weaken into
    # "accept any answer" when citation stripping is enabled.
    ms = MemoryServer(str(tmp_path), "grounded_gate_invented")
    recall = _make_recall_with_facts(
        "John signed with the Minnesota Wolves on 2023-05-21.",
    )
    # Invented answer with completely different content.
    answer = "John signed with the Atlantis Octopuses on 21 May 2023."
    out = ms._normalize_grounded_answer("What team did John sign with?", answer, recall)
    # Octopuses/Atlantis are not in the recall — gate must NM.
    assert out == "Not mentioned in the provided context."


def test_grounded_gate_preserves_explicit_not_mentioned(tmp_path):
    # Negative regression: explicit "Not mentioned" with no grounded
    # candidate must remain "Not mentioned" — fix must not flip negative
    # answers into positive.
    ms = MemoryServer(str(tmp_path), "grounded_gate_explicit_nm")
    recall = _make_recall_with_facts(
        "John signed with the Minnesota Wolves on 2023-05-21.",
    )
    # Note: the question is unrelated to the recall (no Mary in it).
    answer = "Not mentioned. The retrieved facts do not specify Mary's role."
    out = ms._normalize_grounded_answer("What is Mary's role?", answer, recall)
    assert out == "Not mentioned in the provided context."
