# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from src.memory import MemoryServer
from src.normalizer import acl_domain_key
import src.storage as storage_mod
from src.storage import (
    IngressWriteStorageBackend,
    JSONNPZStorage,
    SQLiteStorageBackend,
    StorageBackend,
    make_storage,
    migrate_jsonnpz_to_sqlite,
)


def _sqlite_unique_indexes(conn, table_name: str) -> set[tuple[str, ...]]:
    indexes = conn.execute(f"PRAGMA index_list({table_name})").fetchall()
    unique_cols: set[tuple[str, ...]] = set()
    for row in indexes:
        if int(row["unique"] or 0) != 1:
            continue
        if str(row["origin"] or "") == "pk":
            continue
        info = conn.execute(f"PRAGMA index_info({row['name']})").fetchall()
        unique_cols.add(tuple(col["name"] for col in sorted(info, key=lambda item: int(item["seqno"] or 0))))
    return unique_cols


def _create_legacy_sqlite_v1(path: Path) -> tuple[np.ndarray, np.ndarray]:
    gran = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cross = np.array([[5.0, 6.0]], dtype=np.float32)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE meta (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE write_log (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                swarm_id TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'shared',
                owner_id TEXT,
                scope TEXT,
                read_json TEXT,
                write_json TEXT,
                content_family TEXT NOT NULL,
                content_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                extraction_state TEXT NOT NULL DEFAULT 'pending',
                extraction_attempts INTEGER NOT NULL DEFAULT 0,
                last_extraction_attempt_ms INTEGER,
                sort_order INTEGER NOT NULL UNIQUE
            );
            CREATE TABLE raw_sessions (
                raw_session_id TEXT PRIMARY KEY,
                session_num INTEGER,
                message_id TEXT NOT NULL UNIQUE REFERENCES write_log(message_id),
                source_id TEXT,
                format TEXT,
                session_date TEXT,
                speakers TEXT,
                stored_at TEXT,
                artifact_id TEXT,
                version_id TEXT,
                content_hash TEXT,
                owner_id TEXT,
                scope TEXT,
                agent_id TEXT,
                swarm_id TEXT,
                read_json TEXT,
                write_json TEXT,
                target_json TEXT,
                metadata_json TEXT,
                source_meta_json TEXT,
                status TEXT,
                sort_order INTEGER NOT NULL UNIQUE
            );
            CREATE TABLE raw_docs (
                source_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL UNIQUE REFERENCES write_log(message_id),
                metadata_json TEXT,
                sort_order INTEGER NOT NULL UNIQUE
            );
            CREATE TABLE facts (
                tier TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                fact_id TEXT,
                kind TEXT,
                session_num INTEGER,
                source_id TEXT,
                agent_id TEXT,
                swarm_id TEXT,
                scope TEXT,
                owner_id TEXT,
                status TEXT,
                created_at TEXT,
                event_date TEXT,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (tier, sort_order)
            );
            CREATE TABLE embeddings (
                tier TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY (tier, sort_order)
            );
            CREATE TABLE episode_corpus (
                doc_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                episode_json TEXT NOT NULL,
                PRIMARY KEY (doc_id, episode_id),
                UNIQUE (doc_id, sort_order)
            );
            CREATE TABLE temporal_links (
                sort_order INTEGER PRIMARY KEY,
                link_json TEXT NOT NULL
            );
            CREATE TABLE source_records (
                source_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                owner_id TEXT,
                read_json TEXT,
                write_json TEXT,
                artifact_id TEXT,
                version_id TEXT,
                content_hash TEXT,
                metadata_json TEXT,
                target_json TEXT,
                source_meta_json TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE state_json (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO meta(name, value_json) VALUES(?, ?)",
            [
                ("schema_version", json.dumps(1)),
                ("storage_backend", json.dumps("sqlite")),
            ],
        )
        conn.executemany(
            """
            INSERT INTO write_log(
                message_id, session_id, agent_id, swarm_id, visibility, owner_id, scope,
                read_json, write_json, content_family, content_text, metadata_json,
                timestamp_ms, extraction_state, extraction_attempts, last_extraction_attempt_ms,
                sort_order
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "m1",
                    "s1",
                    "agent-a",
                    "swarm-a",
                    "shared",
                    "agent:agent-a",
                    "swarm-shared",
                    json.dumps(["swarm:swarm-a"]),
                    json.dumps(["swarm:swarm-a"]),
                    "conversation",
                    "hello there",
                    json.dumps({"role": "user"}),
                    1712000000000,
                    "complete",
                    1,
                    1712000001000,
                    0,
                ),
                (
                    "rawdoc:doc-1",
                    "doc:doc-1",
                    "agent-a",
                    "swarm-a",
                    "shared",
                    "agent:agent-a",
                    "swarm-shared",
                    json.dumps(["swarm:swarm-a"]),
                    json.dumps(["swarm:swarm-a"]),
                    "document",
                    "Part A\n\nPart B",
                    json.dumps({"origin": "legacy"}),
                    1712000002000,
                    "complete",
                    1,
                    1712000003000,
                    1,
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_session_id, session_num, message_id, source_id, format, session_date,
                speakers, stored_at, artifact_id, version_id, content_hash, owner_id,
                scope, agent_id, swarm_id, read_json, write_json, target_json,
                metadata_json, source_meta_json, status, sort_order
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rs1",
                1,
                "m1",
                "legacy",
                "conversation",
                "2024-06-01",
                "User and Assistant",
                "2024-06-01T00:00:00+00:00",
                "art-1",
                "ver-1",
                "sha256:abc",
                "agent:agent-a",
                "swarm-shared",
                "agent-a",
                "swarm-a",
                json.dumps(["swarm:swarm-a"]),
                json.dumps(["swarm:swarm-a"]),
                json.dumps([]),
                json.dumps({"k": "v"}),
                json.dumps({"origin": "legacy"}),
                "active",
                0,
            ),
        )
        conn.execute(
            "INSERT INTO raw_docs(source_id, message_id, metadata_json, sort_order) VALUES(?, ?, ?, ?)",
            ("doc-1", "rawdoc:doc-1", json.dumps({"origin": "legacy"}), 0),
        )
        conn.executemany(
            """
            INSERT INTO facts(
                tier, sort_order, fact_id, kind, session_num, source_id, agent_id, swarm_id,
                scope, owner_id, status, created_at, event_date, payload_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "granular",
                    0,
                    "g1",
                    "event",
                    1,
                    "legacy",
                    "agent-a",
                    "swarm-a",
                    "swarm-shared",
                    "agent:agent-a",
                    "active",
                    "2024-06-01T00:00:00+00:00",
                    "2024-06-01",
                    json.dumps({"id": "g1", "fact": "hello", "kind": "event", "session": 1}),
                ),
                (
                    "granular",
                    1,
                    "g2",
                    "event",
                    1,
                    "legacy",
                    "agent-a",
                    "swarm-a",
                    "swarm-shared",
                    "agent:agent-a",
                    "active",
                    "2024-06-01T00:00:01+00:00",
                    "2024-06-01",
                    json.dumps({"id": "g2", "fact": "world", "kind": "event", "session": 1}),
                ),
                (
                    "cross",
                    0,
                    "x1",
                    "fact",
                    1,
                    "doc-1",
                    "agent-a",
                    "swarm-a",
                    "swarm-shared",
                    "agent:agent-a",
                    "active",
                    "2024-06-01T00:00:02+00:00",
                    None,
                    json.dumps({"id": "x1", "fact": "cross", "kind": "fact"}),
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO embeddings(tier, sort_order, dim, dtype, vector_blob) VALUES(?, ?, ?, ?, ?)",
            [
                ("gran", 0, 2, "float32", gran[0].tobytes()),
                ("gran", 1, 2, "float32", gran[1].tobytes()),
                ("cross", 0, 2, "float32", cross[0].tobytes()),
            ],
        )
        conn.executemany(
            "INSERT INTO episode_corpus(doc_id, episode_id, sort_order, episode_json) VALUES(?, ?, ?, ?)",
            [
                ("document:doc-1", "e1", 0, json.dumps({"episode_id": "e1", "raw_text": "Part A"})),
                ("document:doc-1", "e2", 1, json.dumps({"episode_id": "e2", "raw_text": "Part B"})),
            ],
        )
        conn.execute("INSERT INTO temporal_links(sort_order, link_json) VALUES(?, ?)", (0, json.dumps({"kind": "before"})))
        conn.execute(
            """
            INSERT INTO source_records(
                source_id, family, owner_id, read_json, write_json, artifact_id, version_id,
                content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "doc-1",
                "document",
                "agent:agent-a",
                json.dumps(["swarm:swarm-a"]),
                json.dumps(["swarm:swarm-a"]),
                "art-1",
                "ver-1",
                "sha256:def",
                json.dumps({"origin": "legacy"}),
                json.dumps([]),
                json.dumps({"channel": "import"}),
                "2024-06-01T00:00:00+00:00",
                "2024-06-01T00:00:00+00:00",
            ),
        )
        conn.executemany(
            "INSERT INTO state_json(name, value_json) VALUES(?, ?)",
            [
                ("n_sessions", json.dumps(1)),
                ("_episode_doc_order", json.dumps(["document:doc-1"])),
                ("_episode_corpus_layout", json.dumps({"documents": ["document:doc-1"]})),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return gran, cross


def test_json_npz_storage_implements_protocol(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "test")
    assert isinstance(s, StorageBackend)
    assert not isinstance(s, IngressWriteStorageBackend)


def test_storage_not_exists_initially(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "new_key")
    assert not s.exists


def test_save_and_load_facts(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    data = {"granular": [{"fact": "x"}], "n_sessions": 1, "secrets": []}
    s.save_facts(data)
    loaded = s.load_facts()
    assert loaded["granular"][0]["fact"] == "x"
    assert loaded["n_sessions"] == 1


def test_missing_keys_return_defaults(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    (tmp_path / "k.json").write_text("{}")
    loaded = s.load_facts()
    assert loaded.get("granular", []) == []


def test_exists_after_save(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    s.save_facts({"granular": []})
    assert s.exists


def test_save_and_load_embeddings(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    gran = np.random.rand(5, 32).astype(np.float32)
    cons = np.random.rand(2, 32).astype(np.float32)
    cross = np.zeros((0, 32), dtype=np.float32)
    s.save_embeddings(gran, cons, cross)
    loaded = s.load_embeddings()
    assert loaded is not None
    np.testing.assert_allclose(loaded["gran"], gran)
    np.testing.assert_allclose(loaded["cons"], cons)


def test_load_embeddings_none_when_missing(tmp_path):
    s = JSONNPZStorage(str(tmp_path), "k")
    assert s.load_embeddings() is None


def test_make_storage_returns_sqlite_for_new_keys(tmp_path):
    s = make_storage(str(tmp_path), "k")
    assert isinstance(s, SQLiteStorageBackend)
    assert isinstance(s, IngressWriteStorageBackend)


def test_memory_server_uses_storage_backend(tmp_path):
    """MemoryServer must load/save via storage backend."""
    from src.memory import MemoryServer
    server = MemoryServer(data_dir=str(tmp_path), key="storage_test")
    assert hasattr(server, "_storage")
    assert isinstance(server._storage, StorageBackend)


def test_custom_storage_injected(tmp_path):
    """MemoryServer accepts custom storage via constructor injection."""
    from src.memory import MemoryServer
    storage = SQLiteStorageBackend(str(tmp_path), "custom_key")
    server = MemoryServer(data_dir=str(tmp_path), key="any_key", storage=storage)
    assert server._storage is storage


def test_storage_roundtrip_via_memory_server(tmp_path):
    """Data written by MemoryServer is readable by a new instance."""
    from src.memory import MemoryServer
    s1 = MemoryServer(data_dir=str(tmp_path), key="rt")
    s1._all_granular = [{"fact": "test fact", "kind": "fact"}]
    s1._n_sessions = 1
    s1._save_cache()

    s2 = MemoryServer(data_dir=str(tmp_path), key="rt")
    assert len(s2._all_granular) == 1
    assert s2._all_granular[0]["fact"] == "test fact"


def test_sqlite_storage_roundtrip_facts_and_embeddings(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_rt")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1}],
        "cons": [{"id": "c1", "fact": "summary", "kind": "summary"}],
        "cross": [],
        "tlinks": [{"kind": "before"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "default",
            "swarm_id": "default",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
        }],
        "raw_docs": {"doc-1": "# Doc"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc"}]}]},
        "source_records": {"doc-1": {"family": "document", "metadata": {}, "target": [], "source_meta": {}, "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}},
        "n_sessions": 1,
    }
    storage.save_facts(facts)
    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "hello"
    assert loaded["tlinks"][0]["kind"] == "before"
    assert loaded["raw_sessions"][0]["message_id"] == "m1"
    assert loaded["episode_corpus"]["documents"][0]["episodes"][0]["episode_id"] == "e1"

    gran = np.random.rand(1, 8).astype(np.float32)
    cons = np.random.rand(1, 8).astype(np.float32)
    cross = np.zeros((0, 8), dtype=np.float32)
    storage.save_embeddings(gran, cons, cross)
    embs = storage.load_embeddings()
    np.testing.assert_allclose(embs["gran"], gran)
    np.testing.assert_allclose(embs["cons"], cons)


def test_sqlite_new_db_uses_spec_grade_schema_and_meta(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_schema_v2")

    with storage._connect() as conn:
        assert storage._table_pk_columns(conn, "facts") == ["tier", "fact_id"]
        assert storage._table_pk_columns(conn, "embeddings") == ["tier", "fact_id"]
        assert storage._table_pk_columns(conn, "raw_sessions") == ["session_num"]
        assert ("tier", "sort_order") in _sqlite_unique_indexes(conn, "facts")
        assert ("tier", "sort_order") in _sqlite_unique_indexes(conn, "embeddings")
        assert ("message_id",) not in _sqlite_unique_indexes(conn, "raw_sessions")
        raw_session_cols = {row["name"]: row for row in storage._table_info(conn, "raw_sessions")}
        raw_doc_cols = {row["name"]: row for row in storage._table_info(conn, "raw_docs")}
        assert int(raw_session_cols["message_id"]["notnull"]) == 1
        assert int(raw_doc_cols["message_id"]["notnull"]) == 1
        meta = storage._read_meta(conn)

    assert meta["schema_version"] == 4
    assert meta["storage_backend"] == "sqlite"
    assert "migrated_from" in meta
    assert meta["migrated_from"] is None
    assert isinstance(meta["migration_completed_at"], str)
    assert meta["migration_completed_at"]


def test_sqlite_save_facts_snapshot_path_does_not_call_append_write_log(tmp_path, monkeypatch):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_snapshot_role_split")

    def _boom(**_kwargs):
        raise AssertionError("save_facts should not use append_write_log ingress API")

    monkeypatch.setattr(storage, "append_write_log", _boom)
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "default",
            "swarm_id": "default",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
        }],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
        "n_sessions": 1,
    }

    storage.save_facts(facts)

    loaded = storage.load_facts()
    assert loaded["raw_sessions"][0]["message_id"] == "m1"



def test_sqlite_save_facts_preserves_rowids_for_unchanged_snapshot(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_stable_rows")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1}],
        "cons": [],
        "cross": [],
        "tlinks": [{"kind": "before"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "default",
            "swarm_id": "default",
            "scope": "swarm-shared",
            "owner_id": "system",
            "read": ["agent:PUBLIC"],
            "write": ["agent:PUBLIC"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
        }],
        "raw_docs": {"doc-1": "# Doc"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc"}]}]},
        "source_records": {"doc-1": {"family": "document", "metadata": {}, "target": [], "source_meta": {}, "read": ["agent:PUBLIC"], "write": ["agent:PUBLIC"]}},
        "n_sessions": 1,
    }
    storage.save_facts(facts)
    with storage._connect() as conn:
        first = {
            "raw_sessions": conn.execute("SELECT rowid FROM raw_sessions WHERE raw_session_id = ?", ("rs1",)).fetchone()[0],
            "facts": conn.execute("SELECT rowid FROM facts WHERE tier = ? AND sort_order = 0", ("granular",)).fetchone()[0],
            "episode_corpus": conn.execute("SELECT rowid FROM episode_corpus WHERE doc_id = ? AND episode_id = ?", ("document:doc-1", "e1")).fetchone()[0],
            "source_records": conn.execute("SELECT rowid FROM source_records WHERE source_id = ?", ("doc-1",)).fetchone()[0],
            "state_json": conn.execute("SELECT rowid FROM state_json WHERE name = ?", ("n_sessions",)).fetchone()[0],
        }
    storage.save_facts(facts)
    with storage._connect() as conn:
        second = {
            "raw_sessions": conn.execute("SELECT rowid FROM raw_sessions WHERE raw_session_id = ?", ("rs1",)).fetchone()[0],
            "facts": conn.execute("SELECT rowid FROM facts WHERE tier = ? AND sort_order = 0", ("granular",)).fetchone()[0],
            "episode_corpus": conn.execute("SELECT rowid FROM episode_corpus WHERE doc_id = ? AND episode_id = ?", ("document:doc-1", "e1")).fetchone()[0],
            "source_records": conn.execute("SELECT rowid FROM source_records WHERE source_id = ?", ("doc-1",)).fetchone()[0],
            "state_json": conn.execute("SELECT rowid FROM state_json WHERE name = ?", ("n_sessions",)).fetchone()[0],
        }
    assert second == first

def test_sqlite_save_facts_updates_projections_without_on_conflict_syntax(tmp_path):
    class _RejectOnConflictConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            if "ON CONFLICT" in str(sql).upper():
                raise sqlite3.OperationalError('near "ON": syntax error')
            return self._inner.execute(sql, params)

        def executemany(self, sql, seq_of_params):
            if "ON CONFLICT" in str(sql).upper():
                raise sqlite3.OperationalError('near "ON": syntax error')
            return self._inner.executemany(sql, seq_of_params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_sqlcipher_upsert_compat")
    first = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event", "session": 1, "owner_id": "agent:alice"}],
        "cons": [],
        "cross": [],
        "tlinks": [{"kind": "before"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "alice",
            "swarm_id": "alpha",
            "scope": "swarm-shared",
            "owner_id": "agent:alice",
            "read": ["swarm:alpha"],
            "write": ["swarm:alpha"],
            "stored_at": "2024-06-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1",
            "status": "active",
            "metadata": {"rev": 1},
            "source_meta": {"origin": "first"},
        }],
        "raw_docs": {"doc-1": "# Doc"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc"}]}]},
        "source_records": {"doc-1": {
            "family": "document",
            "owner_id": "agent:alice",
            "metadata": {"rev": 1},
            "target": ["agent:alice"],
            "source_meta": {"origin": "first"},
            "read": ["agent:alice"],
            "write": ["agent:alice"],
        }},
        "n_sessions": 1,
    }
    second = {
        "granular": [{"id": "g1", "fact": "updated hello", "kind": "event", "session": 1, "owner_id": "agent:alice"}],
        "cons": [],
        "cross": [],
        "tlinks": [{"kind": "after"}],
        "raw_sessions": [{
            "message_id": "m1",
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2024-06-02",
            "content": "updated hello",
            "speakers": "User and Assistant",
            "agent_id": "alice",
            "swarm_id": "alpha",
            "scope": "swarm-shared",
            "owner_id": "agent:alice",
            "read": ["agent:alice", "swarm:alpha"],
            "write": ["agent:alice"],
            "stored_at": "2024-06-02T00:00:00+00:00",
            "format": "conversation",
            "source_id": "chat-1-updated",
            "status": "superseded",
            "metadata": {"rev": 2},
            "source_meta": {"origin": "second"},
        }],
        "raw_docs": {"doc-1": "# Doc updated"},
        "episode_corpus": {"documents": [{"doc_id": "document:doc-1", "episodes": [{"episode_id": "e1", "raw_text": "# Doc updated"}]}]},
        "source_records": {"doc-1": {
            "family": "document",
            "owner_id": "agent:alice",
            "metadata": {"rev": 2},
            "target": ["agent:alice", "swarm:alpha"],
            "source_meta": {"origin": "second"},
            "read": ["agent:alice", "swarm:alpha"],
            "write": ["agent:alice"],
        }},
        "n_sessions": 1,
    }

    storage.save_facts(first)
    with storage._connections_lock:
        storage._connections[threading.get_ident()] = _RejectOnConflictConnection(
            storage._connections[threading.get_ident()]
        )

    storage.save_facts(second)

    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "updated hello"
    assert loaded["tlinks"][0]["kind"] == "after"

    with storage._connect() as conn:
        raw_session = conn.execute(
            "SELECT source_id, status, metadata_json, source_meta_json FROM raw_sessions WHERE session_num = ?",
            (1,),
        ).fetchone()
        raw_doc = conn.execute(
            "SELECT metadata_json FROM raw_docs WHERE source_id = ?",
            ("doc-1",),
        ).fetchone()
        fact_row = conn.execute(
            "SELECT payload_json FROM facts WHERE tier = ? AND fact_id = ?",
            ("granular", "g1"),
        ).fetchone()
        source_record = conn.execute(
            "SELECT metadata_json, source_meta_json, read_json, write_json FROM source_records WHERE source_id = ?",
            ("doc-1",),
        ).fetchone()

    assert raw_session["source_id"] == "chat-1-updated"
    assert raw_session["status"] == "superseded"
    assert json.loads(raw_session["metadata_json"]) == {"rev": 2}
    assert json.loads(raw_session["source_meta_json"]) == {"source_meta": {"origin": "second"}}
    assert json.loads(raw_doc["metadata_json"]) == {"rev": 2}
    assert json.loads(fact_row["payload_json"])["fact"] == "updated hello"
    assert json.loads(source_record["metadata_json"]) == {"rev": 2}
    assert json.loads(source_record["source_meta_json"]) == {"origin": "second"}
    assert json.loads(source_record["read_json"]) == ["agent:alice", "swarm:alpha"]
    assert json.loads(source_record["write_json"]) == ["agent:alice"]


def test_sqlite_container_graph_upsert_avoids_on_conflict_syntax(tmp_path):
    class _RejectOnConflictConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, params=()):
            if "ON CONFLICT" in str(sql).upper():
                raise sqlite3.OperationalError('near "ON": syntax error')
            return self._inner.execute(sql, params)

        def executemany(self, sql, seq_of_params):
            if "ON CONFLICT" in str(sql).upper():
                raise sqlite3.OperationalError('near "ON": syntax error')
            return self._inner.executemany(sql, seq_of_params)

        def executescript(self, sql):
            if "ON CONFLICT" in str(sql).upper():
                raise sqlite3.OperationalError('near "ON": syntax error')
            return self._inner.executescript(sql)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_container_graph_upsert_compat")
    with storage._connect() as conn:
        storage._ensure_container_graph_tables(conn)
        wrapped = _RejectOnConflictConnection(conn)
        storage._upsert_container_graph_rows(
            wrapped,
            "container_graph_revisions",
            [{
                "container_graph_revision_id": "rev-1",
                "source_id": "doc-1",
                "logical_source_id": "logical-doc-1",
                "family": "document",
                "revision_id": "source-rev-1",
                "revision_scope": "source_revision",
                "adapter_name": "document_artifact_adapter",
                "adapter_version": "v1",
                "input_fingerprint": "input-1",
                "graph_fingerprint": "graph-1",
                "status": "active",
            }],
        )
        storage._upsert_container_graph_rows(
            wrapped,
            "container_graph_revisions",
            [{
                "container_graph_revision_id": "rev-1",
                "source_id": "doc-1",
                "logical_source_id": "logical-doc-1",
                "family": "document",
                "revision_id": "source-rev-1",
                "revision_scope": "source_revision",
                "adapter_name": "document_artifact_adapter",
                "adapter_version": "v2",
                "input_fingerprint": "input-1",
                "graph_fingerprint": "graph-2",
                "status": "active",
            }],
        )
        row = conn.execute(
            "SELECT adapter_version, graph_fingerprint FROM container_graph_revisions WHERE container_graph_revision_id = ?",
            ("rev-1",),
        ).fetchone()

    assert row["adapter_version"] == "v2"
    assert row["graph_fingerprint"] == "graph-2"


def test_sqlite_save_facts_creates_missing_container_graph_tables_for_legacy_cache(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_legacy_container_graph_migration")
    with storage._connect() as conn:
        for table_name in storage_mod.CONTAINER_GRAPH_SNAPSHOT_KEYS.values():
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()

    storage.save_facts({
        "granular": [],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
        "container_graph": {
            "graph_revisions": [{
                "container_graph_revision_id": "rev-legacy",
                "source_id": "doc-legacy",
                "family": "document",
                "revision_id": "source-rev-legacy",
                "revision_scope": "source_revision",
                "adapter_name": "document_artifact_adapter",
                "adapter_version": "v1",
                "input_fingerprint": "input-legacy",
                "graph_fingerprint": "graph-legacy",
                "status": "active",
            }]
        },
    })

    loaded = storage.load_facts()
    assert loaded["container_graph"]["graph_revisions"][0]["container_graph_revision_id"] == "rev-legacy"


def test_sqlite_storage_serializes_list_event_date_hot_column(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_event_date_list")
    facts = {
        "granular": [{"id": "g1", "fact": "dated", "kind": "event", "event_date": ["2024-01-01", "2024-01-02"]}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
    }

    storage.save_facts(facts)

    loaded = storage.load_facts()
    assert loaded["granular"][0]["event_date"] == ["2024-01-01", "2024-01-02"]

    with storage._connect() as conn:
        row = conn.execute("SELECT event_date FROM facts WHERE tier = ? AND fact_id = ?", ("granular", "g1")).fetchone()
    assert row[0] == json.dumps(["2024-01-01", "2024-01-02"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def test_migrate_jsonnpz_to_sqlite_preserves_snapshot_and_backs_up_legacy(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "migrate")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": [{"doc_id": "d1", "episodes": [{"episode_id": "e1"}]}]},
        "n_sessions": 1,
    }
    legacy.save_facts(facts)
    (tmp_path / "migrate_corpus.json").write_text(json.dumps(facts["episode_corpus"]))
    gran = np.random.rand(1, 4).astype(np.float32)
    legacy.save_embeddings(gran, np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32))

    result = migrate_jsonnpz_to_sqlite(str(tmp_path), "migrate")
    assert result["migrated"] is True
    assert (tmp_path / "migrate.sqlite3").exists()
    assert (tmp_path / "migrate.json.bak").exists()
    assert (tmp_path / "migrate_embs.npz.bak").exists()
    assert (tmp_path / "migrate_corpus.json.bak").exists()

    storage = SQLiteStorageBackend(str(tmp_path), "migrate")
    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "hello"
    assert loaded["episode_corpus"]["documents"][0]["episodes"][0]["episode_id"] == "e1"
    embs = storage.load_embeddings()
    np.testing.assert_allclose(embs["gran"], gran)
    with storage._connect() as conn:
        meta = storage._read_meta(conn)
    assert meta["migrated_from"] == "jsonnpz"
    assert isinstance(meta["migration_completed_at"], str)
    assert meta["migration_completed_at"]


def test_migrate_jsonnpz_to_sqlite_accepts_legacy_snapshot_shape(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "legacy_shape")
    facts = {
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [{
            "raw_session_id": "rs1",
            "session_num": 1,
            "session_date": "2026-01-01",
            "content": "hello",
            "speakers": "User and Assistant",
            "agent_id": "agent-a",
            "swarm_id": "swarm-a",
            "scope": "swarm-shared",
            "owner_id": "agent:agent-a",
            "read": ["swarm:swarm-a"],
            "write": ["swarm:swarm-a"],
            "stored_at": "2026-01-01T00:00:00+00:00",
            "format": "conversation",
            "source_id": "legacy_shape",
            "artifact_id": "art-1",
            "version_id": "ver-1",
            "content_hash": "sha256:abc",
            "status": "active",
            "episode_id": "e1",
            "ingest_transport": "text",
        }],
        "raw_docs": {},
        "episode_corpus": {"documents": [{"doc_id": "legacy_shape", "episodes": [{"episode_id": "e1", "raw_text": "hello"}]}]},
        "source_records": {
            "legacy_shape": {
                "source_id": "legacy_shape",
                "family": "document",
                "scope_id": "legacy_shape",
                "owner_id": "agent:agent-a",
                "read": ["swarm:swarm-a"],
                "write": ["swarm:swarm-a"],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "artifact_id": "art-1",
                "version_id": "ver-1",
                "content_hash": "sha256:abc",
                "metadata": {"bench": "ama"},
                "source_meta": {"ingest_transport": "text"},
            }
        },
        "n_sessions": 1,
    }
    legacy.save_facts(facts)
    (tmp_path / "legacy_shape_corpus.json").write_text(json.dumps(facts["episode_corpus"]))

    result = migrate_jsonnpz_to_sqlite(str(tmp_path), "legacy_shape")

    assert result["migrated"] is True
    storage = SQLiteStorageBackend(str(tmp_path), "legacy_shape")
    loaded = storage.load_facts()
    assert loaded["raw_sessions"][0]["message_id"] == "raw:rs1"
    assert loaded["source_records"]["legacy_shape"]["family"] == "document"
    assert loaded["source_records"]["legacy_shape"]["target"] == []


def test_migrate_jsonnpz_to_sqlite_imports_legacy_after_empty_sqlite_bootstrap(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "bootstrap_then_migrate")
    legacy.save_facts(
        {
            "granular": [{"id": "g1", "fact": "legacy bootstrap", "kind": "event"}],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [],
            "raw_docs": {},
            "episode_corpus": {"documents": []},
            "source_records": {},
            "n_sessions": 1,
        }
    )

    server = MemoryServer(str(tmp_path), "bootstrap_then_migrate")
    assert server._all_granular == []

    result = migrate_jsonnpz_to_sqlite(str(tmp_path), "bootstrap_then_migrate")

    assert result["migrated"] is True
    assert result.get("already_migrated") is not True

    storage = SQLiteStorageBackend(str(tmp_path), "bootstrap_then_migrate")
    loaded = storage.load_facts()
    assert loaded["granular"][0]["fact"] == "legacy bootstrap"


def test_sqlite_backend_reuses_same_thread_connection(tmp_path, monkeypatch):
    calls = []
    orig_connect = storage_mod.sqlite3.connect

    def _wrapped_connect(*args, **kwargs):
        calls.append(1)
        return orig_connect(*args, **kwargs)

    monkeypatch.setattr(storage_mod.sqlite3, "connect", _wrapped_connect)
    storage = SQLiteStorageBackend(str(tmp_path), "write_log_reuse")
    assert len(calls) == 1

    storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    storage.get_write_status("m1")
    storage.list_write_log_entries(states=["pending"], order="asc")
    storage.mark_write_state("m1", "complete")

    assert len(calls) == 1
    storage.close()


def test_sqlite_auto_migrates_legacy_v1_schema_preserving_payloads_and_embeddings(tmp_path):
    db_path = tmp_path / "legacy_v1.sqlite3"
    gran, cross = _create_legacy_sqlite_v1(db_path)

    storage = SQLiteStorageBackend(str(tmp_path), "legacy_v1")
    loaded = storage.load_facts()
    embs = storage.load_embeddings()

    assert [fact["id"] for fact in loaded["granular"]] == ["g1", "g2"]
    assert [fact["fact"] for fact in loaded["granular"]] == ["hello", "world"]
    assert loaded["raw_sessions"][0]["message_id"] == "m1"
    assert loaded["raw_sessions"][0]["raw_session_id"] == "rs1"
    assert loaded["raw_docs"]["doc-1"] == "Part A\n\nPart B"
    assert loaded["source_records"]["doc-1"]["metadata"] == {"origin": "legacy"}
    np.testing.assert_allclose(embs["gran"], gran)
    np.testing.assert_allclose(embs["cross"], cross)

    with storage._connect() as conn:
        assert storage._table_pk_columns(conn, "facts") == ["tier", "fact_id"]
        assert storage._table_pk_columns(conn, "embeddings") == ["tier", "fact_id"]
        assert storage._table_pk_columns(conn, "raw_sessions") == ["session_num"]
        assert ("message_id",) not in _sqlite_unique_indexes(conn, "raw_sessions")
        meta = storage._read_meta(conn)

    assert meta["schema_version"] == 4
    assert meta["storage_backend"] == "sqlite"
    assert meta["migrated_from"] == "sqlite_v1"
    assert isinstance(meta["migration_completed_at"], str)
    assert meta["migration_completed_at"]


def test_sqlite_load_raw_docs_uses_write_log_original_not_episode_rebuild(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "raw_docs_original_source")
    original = "[Artifact 0001]\nResponse:\n  trial’s exact line,  \n"
    storage.save_facts(
        {
            "granular": [],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [],
            "raw_docs": {"DOC": original},
            "episode_corpus": {
                "documents": [
                    {
                        "doc_id": "document:DOC",
                        "episodes": [
                            {
                                "episode_id": "DOC_e01",
                                "source_id": "DOC",
                                "source_type": "document",
                                "raw_text": "trial's degraded line,",
                                "raw_original": "trial's degraded line,",
                            }
                        ],
                    }
                ]
            },
            "source_records": {"DOC": {"family": "document", "version_id": "v1"}},
        }
    )

    loaded = storage.load_facts(internal=True)

    assert loaded["raw_docs"]["DOC"] == original


def test_sqlite_load_document_raw_sessions_preserves_write_log_original(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "raw_sessions_original_source")
    original = "[Artifact 0001]\nResponse:\n  trial’s exact line,  \n"
    storage.save_facts(
        {
            "granular": [],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [
                {
                    "message_id": "rawdoc:DOC:v1",
                    "raw_session_id": "DOC_e01_rs",
                    "session_num": 1,
                    "session_date": "2024-06-01",
                    "content": original,
                    "speakers": "Document",
                    "agent_id": "default",
                    "swarm_id": "default",
                    "scope": "swarm-shared",
                    "owner_id": "system",
                    "read": ["agent:PUBLIC"],
                    "write": ["agent:PUBLIC"],
                    "stored_at": "2024-06-01T00:00:00+00:00",
                    "format": "document",
                    "source_id": "DOC",
                    "status": "active",
                    "source_meta": {"episode_id": "DOC_e01"},
                }
            ],
            "raw_docs": {"DOC": original},
            "episode_corpus": {
                "documents": [
                    {
                        "doc_id": "document:DOC",
                        "episodes": [
                            {
                                "episode_id": "DOC_e01",
                                "source_id": "DOC",
                                "source_type": "document",
                                "raw_text": "trial's degraded line,",
                                "raw_original": "trial's degraded line,",
                            }
                        ],
                    }
                ]
            },
            "source_records": {"DOC": {"family": "document", "version_id": "v1"}},
        }
    )

    loaded = storage.load_facts(internal=True)

    assert loaded["raw_sessions"][0]["content"] == original


def test_sqlite_raw_doc_delta_writes_original_raw_write_log_row(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "raw_doc_delta_original")
    original = "[Artifact 0001]\nResponse:\n  exact source’s trailing spaces  \n"

    storage.persist_projection_delta(
        raw_doc_upserts=[
            {
                "source_id": "DOC",
                "message_id": "rawdoc:DOC",
                "content_text": original,
                "metadata": {"raw_source_provenance": "original_source"},
            }
        ]
    )

    loaded = storage.load_facts(internal=True)

    assert loaded["raw_docs"]["DOC"] == original


def test_sqlite_reopen_of_upgraded_db_is_safe_noop(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "sqlite_reopen_noop")
    storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    with storage._connect() as conn:
        before_meta = storage._read_meta(conn)
        before_count = int(conn.execute("SELECT COUNT(*) AS n FROM write_log").fetchone()["n"])
    storage.close()

    reopened = SQLiteStorageBackend(str(tmp_path), "sqlite_reopen_noop")
    with reopened._connect() as conn:
        after_meta = reopened._read_meta(conn)
        after_count = int(conn.execute("SELECT COUNT(*) AS n FROM write_log").fetchone()["n"])

    assert after_count == before_count == 1
    assert after_meta == before_meta


def test_sqlite_load_facts_compat_hides_projection_source_tokens(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "compat_projection_tokens")
    storage.save_facts(
        {
            "granular": [
                {
                    "id": "f1",
                    "fact": "fact A",
                    "kind": "event",
                    "source_id": "SRC",
                    "session": 1,
                    "status": "active",
                },
                {
                    "id": "f2",
                    "fact": "fact B",
                    "kind": "event",
                    "source_id": "SRC@@deadbeef0001",
                    "session": 1,
                    "status": "active",
                    "metadata": {
                        "document_source": "DOC@@deadbeef0002",
                        "episode_source_id": "SRC@@deadbeef0001",
                        "episode_id": "SRC@@deadbeef0001_e01",
                    },
                },
                {
                    "id": "DOC@@deadbeef0002_e01_f1",
                    "fact": "doc fact",
                    "kind": "event",
                    "source_id": "DOC@@deadbeef0002",
                    "session": 1,
                    "status": "active",
                    "metadata": {
                        "document_source": "DOC@@deadbeef0002",
                        "episode_source_id": "DOC@@deadbeef0002",
                        "episode_id": "DOC@@deadbeef0002_e01",
                    },
                },
            ],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [
                {
                    "raw_session_id": "rs1",
                    "message_id": "raw:rs1",
                    "session_num": 1,
                    "content": "A",
                    "session_date": "2024-06-01",
                    "speakers": "User",
                    "owner_id": "agent:agent-a",
                    "scope": "agent-private",
                    "agent_id": "agent-a",
                    "swarm_id": "sw1",
                    "read": [],
                    "write": [],
                    "stored_at": "2024-06-01T00:00:00+00:00",
                    "format": "conversation",
                    "source_id": "SRC",
                    "status": "active",
                },
                {
                    "raw_session_id": "rs2",
                    "message_id": "raw:rs2",
                    "session_num": 1,
                    "projection_session_num": 2,
                    "content": "B",
                    "session_date": "2024-06-02",
                    "speakers": "User",
                    "owner_id": "agent:agent-b",
                    "scope": "agent-private",
                    "agent_id": "agent-b",
                    "swarm_id": "sw1",
                    "read": [],
                    "write": [],
                    "stored_at": "2024-06-02T00:00:00+00:00",
                    "format": "conversation",
                    "source_id": "SRC@@deadbeef0001",
                    "logical_source_id": "SRC",
                    "status": "active",
                },
            ],
            "raw_docs": {
                "DOC": "Document A",
                "DOC@@deadbeef0002": "Document B",
            },
            "episode_corpus": {
                "documents": [
                    {
                        "doc_id": "document:DOC",
                        "episodes": [{"episode_id": "DOC_e01", "source_id": "DOC", "raw_text": "Document A"}],
                    },
                    {
                        "doc_id": "document:DOC@@deadbeef0002",
                        "episodes": [{"episode_id": "DOC@@deadbeef0002_e01", "source_id": "DOC@@deadbeef0002", "raw_text": "Document B"}],
                    },
                ]
            },
            "source_records": {
                "SRC": {
                    "family": "conversation",
                    "owner_id": "agent:agent-a",
                    "read": [],
                    "write": [],
                    "metadata": {},
                    "target": [],
                    "source_meta": {"logical_source_id": "SRC"},
                },
                "SRC@@deadbeef0001": {
                    "family": "conversation",
                    "owner_id": "agent:agent-b",
                    "read": [],
                    "write": [],
                    "metadata": {},
                    "target": [],
                    "source_meta": {"logical_source_id": "SRC"},
                },
                "DOC": {
                    "family": "document",
                    "owner_id": "agent:agent-a",
                    "read": [],
                    "write": [],
                    "metadata": {},
                    "target": [],
                    "source_meta": {"logical_source_id": "DOC"},
                },
                "DOC@@deadbeef0002": {
                    "family": "document",
                    "owner_id": "agent:agent-b",
                    "read": [],
                    "write": [],
                    "metadata": {},
                    "target": [],
                    "source_meta": {"logical_source_id": "DOC"},
                },
            },
            "n_sessions": 2,
        }
    )

    compat = storage.load_facts()
    exact = storage.load_facts(internal=True)

    assert {row["source_id"] for row in compat["raw_sessions"]} == {"SRC"}
    assert {fact["source_id"] for fact in compat["granular"]} == {"SRC", "DOC"}
    assert all("@@" not in fact["id"] for fact in compat["granular"])
    assert compat["granular"][1]["metadata"]["document_source"] == "DOC"
    assert compat["granular"][1]["metadata"]["episode_source_id"] == "SRC"
    assert compat["granular"][1]["metadata"]["episode_id"] == "SRC_e01"
    assert compat["granular"][2]["id"] == "DOC_e01_f1"
    assert compat["granular"][2]["metadata"]["document_source"] == "DOC"
    assert compat["granular"][2]["metadata"]["episode_source_id"] == "DOC"
    assert compat["granular"][2]["metadata"]["episode_id"] == "DOC_e01"
    assert set(compat["raw_docs"].keys()) == {"DOC"}
    assert set(compat["source_records"].keys()) == {"SRC", "DOC"}
    assert len(compat["raw_doc_entries"]) == 2
    assert len(compat["source_record_entries"]) == 4
    compat_docs = compat["episode_corpus"]["documents"]
    assert [doc["doc_id"] for doc in compat_docs] == ["document:DOC"]
    compat_episode_ids = [ep["episode_id"] for ep in compat_docs[0]["episodes"]]
    assert len(compat_episode_ids) == 2
    assert len(set(compat_episode_ids)) == 2
    assert all("@@" not in episode_id for episode_id in compat_episode_ids)
    assert any("@@" in row["source_id"] for row in exact["raw_sessions"])
    assert any("@@" in key for key in exact["raw_docs"])
    assert any("@@" in key for key in exact["source_records"])


def test_sqlite_write_log_idempotent_append_and_status(tmp_path):
    storage = SQLiteStorageBackend(str(tmp_path), "write_log")
    first = storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    second = storage.append_write_log(
        message_id="m1",
        session_id="s1",
        agent_id="agent-a",
        swarm_id="swarm-a",
        visibility="shared",
        owner_id="agent:agent-a",
        scope="swarm-shared",
        read=["swarm:swarm-a"],
        write=["swarm:swarm-a"],
        content_family="chat",
        content_text="hello",
        metadata={"role": "user"},
        timestamp_ms=1712000000000,
    )
    assert first["inserted"] is True
    assert second["inserted"] is False
    status = storage.get_write_status("m1")
    assert status["extraction_state"] == "pending"


def _create_encrypted_legacy_store(tmp_path: Path, key: str) -> bytes:
    enc_key = bytes(range(32))
    legacy = JSONNPZStorage(str(tmp_path), key, encryption_key=enc_key)
    legacy.save_facts({
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
        "n_sessions": 1,
    })
    legacy.save_embeddings(
        np.ones((1, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
    )
    return enc_key


def test_validate_migration_encryption_requirements_rejects_plaintext_downgrade(tmp_path):
    _create_encrypted_legacy_store(tmp_path, "enc_plaintext")

    with pytest.raises(RuntimeError, match="plaintext downgrade is not allowed"):
        storage_mod._validate_migration_encryption_requirements(
            data_dir_path=tmp_path,
            key="enc_plaintext",
            encryption_key=None,
        )


def test_validate_migration_encryption_requirements_requires_sqlcipher(tmp_path, monkeypatch):
    enc_key = _create_encrypted_legacy_store(tmp_path, "enc_sqlcipher")
    monkeypatch.setattr(storage_mod, "_sqlcipher_available", lambda: False)

    with pytest.raises(RuntimeError, match="SQLCipher"):
        storage_mod._validate_migration_encryption_requirements(
            data_dir_path=tmp_path,
            key="enc_sqlcipher",
            encryption_key=enc_key,
        )


def test_validate_migration_encryption_requirements_accepts_allowed_decision_path(tmp_path, monkeypatch):
    enc_key = _create_encrypted_legacy_store(tmp_path, "enc_ok")
    monkeypatch.setattr(storage_mod, "_sqlcipher_available", lambda: True)

    assert storage_mod._validate_migration_encryption_requirements(
        data_dir_path=tmp_path,
        key="enc_ok",
        encryption_key=enc_key,
    ) is True


def test_migrate_jsonnpz_to_sqlite_fails_explicitly_for_encrypted_legacy_without_sqlcipher(tmp_path, monkeypatch):
    enc_key = _create_encrypted_legacy_store(tmp_path, "enc_migrate")
    monkeypatch.setenv("GOSH_MEMORY_ENCRYPTION_KEY", enc_key.hex())
    monkeypatch.setattr(storage_mod, "_sqlcipher_available", lambda: False)

    with pytest.raises(RuntimeError, match="SQLCipher"):
        migrate_jsonnpz_to_sqlite(str(tmp_path), "enc_migrate")


def test_migrate_jsonnpz_to_sqlite_is_safe_noop_when_already_migrated(tmp_path):
    legacy = JSONNPZStorage(str(tmp_path), "rerun_key")
    legacy.save_facts({
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
        "n_sessions": 1,
    })

    first = migrate_jsonnpz_to_sqlite(str(tmp_path), "rerun_key")
    second = migrate_jsonnpz_to_sqlite(str(tmp_path), "rerun_key")

    assert first["migrated"] is True
    assert second["migrated"] is False
    assert second["verified"] is True
    assert second["already_migrated"] is True
    assert (tmp_path / "rerun_key.sqlite3").exists()
    assert (tmp_path / "rerun_key.json.bak").exists()


def test_migrate_jsonnpz_to_sqlite_rolls_back_without_backups_on_verification_failure(tmp_path, monkeypatch):
    legacy = JSONNPZStorage(str(tmp_path), "rollback_key")
    legacy.save_facts({
        "granular": [{"id": "g1", "fact": "hello", "kind": "event"}],
        "cons": [],
        "cross": [],
        "tlinks": [],
        "raw_sessions": [],
        "raw_docs": {},
        "episode_corpus": {"documents": []},
        "n_sessions": 1,
    })

    monkeypatch.setattr(storage_mod, "_verify_sqlite_payload", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verify boom")))

    with pytest.raises(RuntimeError, match="verify boom"):
        migrate_jsonnpz_to_sqlite(str(tmp_path), "rollback_key")

    assert (tmp_path / "rollback_key.json").exists()
    assert not (tmp_path / "rollback_key.json.bak").exists()
    assert not (tmp_path / "rollback_key.sqlite3").exists()
    assert not (tmp_path / "rollback_key.sqlite3.tmp").exists()


def test_sqlite_secret_store_schema_and_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    storage = SQLiteStorageBackend(str(tmp_path), "secret_store")

    result = storage.upsert_secret(
        name="API_KEY",
        value="sk-123",
        created_by_principal_id="agent:alice",
        owner_id="agent:alice",
        scope="agent-private",
        agent_id="alice",
        swarm_id=None,
        read=[],
        write=[],
        metadata={"provider": "test"},
    )
    listed = storage.list_secret_rows()
    fetched = storage.get_secret_row(name="API_KEY", acl_domain_key=result["acl_domain_key"], include_value=True)

    assert result["stored"] is True
    assert listed == [
        {
                "secret_id": result["secret_id"],
                "name": "API_KEY",
                "value_encoding": "utf-8",
                "acl_domain_key": result["acl_domain_key"],
            "created_by_principal_id": "agent:alice",
            "owner_id": "agent:alice",
            "scope": "agent-private",
            "agent_id": "alice",
            "swarm_id": None,
            "read": [],
            "write": [],
            "created_at": listed[0]["created_at"],
            "updated_at": listed[0]["updated_at"],
            "metadata": {"provider": "test"},
        }
    ]
    assert fetched["value"] == "sk-123"

    with storage._connect() as conn:
        assert ("name", "acl_domain_key") in _sqlite_unique_indexes(conn, "secrets")


def test_sqlite_secret_store_duplicate_create_does_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    storage = SQLiteStorageBackend(str(tmp_path), "secret_duplicate")

    first = storage.upsert_secret(
        name="API_KEY",
        value="sk-123",
        created_by_principal_id="agent:alice",
        owner_id="agent:alice",
        scope="agent-private",
        agent_id="alice",
        swarm_id=None,
        read=[],
        write=[],
        metadata={"version": 1},
    )
    duplicate = storage.upsert_secret(
        name="API_KEY",
        value="sk-456",
        created_by_principal_id="agent:bob",
        owner_id="agent:alice",
        scope="agent-private",
        agent_id="alice",
        swarm_id=None,
        read=[],
        write=[],
        metadata={"version": 2},
    )
    fetched = storage.get_secret_row(name="API_KEY", acl_domain_key=first["acl_domain_key"], include_value=True)

    assert first["stored"] is True
    assert duplicate == {
        "stored": False,
        "code": "SECRET_ALREADY_EXISTS",
        "secret_id": first["secret_id"],
        "acl_domain_key": first["acl_domain_key"],
    }
    assert fetched["value"] == "sk-123"
    assert fetched["created_by_principal_id"] == "agent:alice"
    assert fetched["metadata"] == {"version": 1}


def test_sqlite_secret_store_plaintext_backend_fails_closed_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", raising=False)
    storage = SQLiteStorageBackend(str(tmp_path), "secret_plaintext_forbidden")

    with pytest.raises(RuntimeError, match="Secret storage requires encrypted SQLite"):
        storage.upsert_secret(
            name="API_KEY",
            value="sk-123",
            created_by_principal_id="agent:alice",
            owner_id="agent:alice",
            scope="agent-private",
            agent_id="alice",
            swarm_id=None,
            read=[],
            write=[],
        )


def test_sqlite_secret_store_migrates_legacy_state_json_idempotently(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    storage = SQLiteStorageBackend(str(tmp_path), "secret_legacy")
    with storage._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO state_json(name, value_json) VALUES(?, ?)",
            (
                "secrets",
                json.dumps(
                    [
                        {
                            "name": "LEGACY_KEY",
                            "value": "xyz",
                            "agent_id": "alice",
                            "swarm_id": "alpha",
                            "scope": "swarm-shared",
                            "stored_at": "2026-04-08T00:00:00+00:00",
                        }
                    ]
                ),
            ),
        )
        conn.commit()
    storage.close()

    reopened = SQLiteStorageBackend(str(tmp_path), "secret_legacy")
    rows = reopened.list_secret_rows(include_values=True)
    with reopened._connect() as conn:
        legacy_row = conn.execute("SELECT value_json FROM state_json WHERE name = ?", ("secrets",)).fetchone()
    reopened.close()

    reopened_again = SQLiteStorageBackend(str(tmp_path), "secret_legacy")
    rows_again = reopened_again.list_secret_rows(include_values=True)

    assert legacy_row is None
    assert rows == rows_again
    assert rows[0]["name"] == "LEGACY_KEY"
    assert rows[0]["value"] == "xyz"
    assert rows[0]["acl_domain_key"] == acl_domain_key("agent:alice", ["swarm:alpha"], ["swarm:alpha"])
    assert rows[0]["created_by_principal_id"] == "agent:alice"


def test_sqlite_secret_store_schema_backfills_created_by_principal_id_from_owner_id(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    storage = SQLiteStorageBackend(str(tmp_path), "secret_backfill")
    first = storage.upsert_secret(
        name="API_KEY",
        value="sk-123",
        created_by_principal_id="agent:alice",
        owner_id="agent:alice",
        scope="agent-private",
        agent_id="alice",
        swarm_id=None,
        read=[],
        write=[],
    )
    storage.close()

    db_path = tmp_path / "secret_backfill.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE secrets_legacy (
                secret_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                value_encoding TEXT NOT NULL,
                acl_domain_key TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                agent_id TEXT,
                swarm_id TEXT,
                read_json TEXT NOT NULL,
                write_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            );
            INSERT INTO secrets_legacy(
                secret_id, name, value_blob, value_encoding, acl_domain_key, owner_id, scope,
                agent_id, swarm_id, read_json, write_json, created_at, updated_at, metadata_json
            )
            SELECT
                secret_id, name, value_blob, value_encoding, acl_domain_key, owner_id, scope,
                agent_id, swarm_id, read_json, write_json, created_at, updated_at, metadata_json
            FROM secrets;
            DROP TABLE secrets;
            ALTER TABLE secrets_legacy RENAME TO secrets;
            DELETE FROM meta WHERE name = 'schema_version';
            INSERT INTO meta(name, value_json) VALUES('schema_version', '3');
            """
        )
        conn.commit()
    finally:
        conn.close()

    reopened = SQLiteStorageBackend(str(tmp_path), "secret_backfill")
    fetched = reopened.get_secret_row(name="API_KEY", acl_domain_key=first["acl_domain_key"], include_value=True)

    assert fetched["secret_id"] == first["secret_id"]
    assert fetched["created_by_principal_id"] == "agent:alice"
    assert fetched["value"] == "sk-123"


def test_migrate_jsonnpz_to_sqlite_moves_legacy_secrets_to_secret_table(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")
    legacy = JSONNPZStorage(str(tmp_path), "secret_jsonnpz")
    legacy.save_facts(
        {
            "granular": [],
            "cons": [],
            "cross": [],
            "tlinks": [],
            "raw_sessions": [],
            "raw_docs": {},
            "episode_corpus": {"documents": []},
            "n_sessions": 0,
            "secrets": [
                {
                    "name": "JSON_KEY",
                    "value": "abc",
                    "agent_id": "alice",
                    "swarm_id": "alpha",
                    "scope": "swarm-shared",
                }
            ],
        }
    )

    migrate_jsonnpz_to_sqlite(str(tmp_path), "secret_jsonnpz")
    storage = SQLiteStorageBackend(str(tmp_path), "secret_jsonnpz")
    rows = storage.list_secret_rows(include_values=True)
    payload = storage.load_facts(internal=True)

    assert rows[0]["name"] == "JSON_KEY"
    assert rows[0]["value"] == "abc"
    assert rows[0]["created_by_principal_id"] == "agent:alice"
    assert "secrets" not in payload
