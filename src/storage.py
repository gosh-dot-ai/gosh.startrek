# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import io
import json
import logging
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .episodes import load_episode_corpus
from .normalizer import acl_domain_key, dedup_domain_key

_resource: Any
try:
    import resource as _resource
except ImportError:  # pragma: no cover - non-Unix platforms
    _resource = None

log = logging.getLogger(__name__)
_SQLCIPHER_MEMLOCK_DIAGNOSTIC_EMITTED = False
_SQLCIPHER_MEMLOCK_DIAGNOSTIC_LOCK = threading.Lock()


@runtime_checkable
class StorageBackend(Protocol):
    """Snapshot/projection persistence interface used by MemoryServer."""

    def load_facts(self, *, internal: bool = False) -> dict:
        ...

    def save_facts(self, data: dict) -> None:
        ...

    def load_embeddings(self) -> dict | None:
        ...

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        ...

    @property
    def exists(self) -> bool:
        ...


@runtime_checkable
class IngressWriteStorageBackend(Protocol):
    """Ingress-truth write-log interface for async/raw write paths."""

    def append_write_log(
        self,
        *,
        message_id: str,
        session_id: str,
        agent_id: str,
        swarm_id: str,
        visibility: str,
        owner_id: str | None,
        scope: str | None,
        read: list[str] | None,
        write: list[str] | None,
        content_family: str,
        content_text: str,
        metadata: dict | None,
        timestamp_ms: int,
    ) -> dict:
        ...

    def get_write_status(self, message_id: str) -> dict | None:
        ...

    def list_write_log_entries(
        self,
        *,
        states: list[str] | None = None,
        swarm_id: str | None = None,
        order: str = "asc",
    ) -> list[dict]:
        ...

    def mark_write_state(self, message_id: str, state: str, *, attempts_delta: int = 0) -> None:
        ...

    def merge_write_log_metadata(self, message_id: str, patch: dict[str, Any]) -> None:
        ...

    def claim_write_log_entries(
        self,
        *,
        worker_id: str,
        batch_size: int,
        now_ms: int,
        lease_ms: int,
        retry_backoff_ms: int,
        max_attempts: int,
    ) -> list[dict]:
        ...

    def mark_index_dirty(
        self,
        *,
        now_ms: int,
        debounce_ms: int,
        max_delay_ms: int,
    ) -> dict[str, Any]:
        ...

    def read_index_status(self, *, now_ms: int | None = None) -> dict[str, Any]:
        ...

    def acquire_index_build_lease(
        self,
        *,
        worker_id: str,
        snapshot_fingerprint: str,
        now_ms: int,
        lease_ms: int,
    ) -> dict[str, Any]:
        ...

    def release_index_build_lease(
        self,
        *,
        worker_id: str,
        success: bool,
        now_ms: int,
        debounce_ms: int,
        max_delay_ms: int,
        retry_after_ms: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        ...


@runtime_checkable
class ProjectionWriteThroughStorageBackend(Protocol):
    """Incremental projection persistence used by SQLite sync write-through paths."""

    def persist_projection_delta(
        self,
        *,
        raw_session_upserts: list[dict] | None = None,
        raw_session_deletes: list[int] | None = None,
        raw_doc_upserts: list[dict] | None = None,
        fact_upserts: dict[str, list[dict]] | None = None,
        fact_deletes: list[tuple[str, str]] | None = None,
        episode_doc_replacements: dict[str, list[dict]] | None = None,
        temporal_link_appends: list[dict] | None = None,
        source_record_upserts: dict[str, dict] | None = None,
        state_values: dict[str, Any] | None = None,
        episode_corpus: dict | None = None,
        container_graph_revision_upserts: list[dict] | None = None,
        container_upserts: list[dict] | None = None,
        container_relation_upserts: list[dict] | None = None,
        container_anchor_upserts: list[dict] | None = None,
        container_evidence_upserts: list[dict] | None = None,
        container_ref_upserts: list[dict] | None = None,
        container_ref_lookup_upserts: list[dict] | None = None,
        container_ref_range_upserts: list[dict] | None = None,
        container_render_ref_upserts: list[dict] | None = None,
        container_contract_upserts: list[dict] | None = None,
        container_artifact_upserts: list[dict] | None = None,
        container_state_upserts: list[dict] | None = None,
        container_deletes: list[str] | None = None,
        replace_container_graph: bool = False,
        complete_message_ids: list[str] | None = None,
    ) -> None:
        ...


@runtime_checkable
class SecretStorageBackend(Protocol):
    """Dedicated persisted secret storage interface."""

    def secret_storage_allowed(self) -> bool:
        ...

    def list_secret_rows(
        self,
        *,
        acl_domain_key: str | None = None,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        ...

    def get_secret_row(
        self,
        *,
        name: str,
        acl_domain_key: str,
        include_value: bool = True,
    ) -> dict[str, Any] | None:
        ...

    def upsert_secret(
        self,
        *,
        name: str,
        value: str,
        created_by_principal_id: str,
        owner_id: str,
        scope: str,
        agent_id: str | None,
        swarm_id: str | None,
        read: list[str],
        write: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def delete_secret(self, *, name: str, acl_domain_key: str) -> bool:
        ...


MAGIC = b"GME1"
SQLITE_SCHEMA_VERSION = 4
INTERNAL_EPISODE_DOC_ORDER = "_episode_doc_order"
INTERNAL_EPISODE_CORPUS_LAYOUT = "_episode_corpus_layout"
STATE_JSON_KEYS = {
    "n_sessions",
    "n_sessions_with_facts",
    "_emb_fingerprints",
    "_dedup_index",
    "_content_dedup_index",
    "_git_dedup_index",
    "_simhash_index",
    "metadata_schema",
    "instance_config",
    "scope_record",
    "profiles",
    "profile_configs",
    "memory_config",
    INTERNAL_EPISODE_DOC_ORDER,
    INTERNAL_EPISODE_CORPUS_LAYOUT,
}
CONTAINER_GRAPH_TABLE_SPECS = {
    "container_graph_revisions": {
        "id": "container_graph_revision_id",
        "json": {"profile_ids_json", "coverage_report_ids_json"},
    },
    "containers": {
        "id": "container_id",
        "json": {"traits_json", "order_key_json", "span_refs_json", "episode_ids_json", "render_ref_json", "read_json", "write_json"},
    },
    "container_relations": {
        "id": "relation_id",
        "json": {"order_key_json", "traits_json"},
    },
    "container_anchors": {
        "id": "anchor_id",
        "json": {"origin_ref_json"},
    },
    "container_evidence": {
        "id": "evidence_id",
        "json": {"evidence_ref_json", "trace_json"},
    },
    "container_render_refs": {
        "id": "render_ref_id",
        "json": {"ref_json"},
    },
    "container_state": {
        "id": "state_id",
        "json": {"state_reason_json"},
    },
    "container_refs": {
        "id": "ref_id",
        "json": {"ref_json"},
    },
    "container_ref_lookup": {
        "id": "lookup_id",
        "json": set(),
    },
    "container_ref_ranges": {
        "id": "range_id",
        "json": set(),
    },
    "container_contracts": {
        "id": "contract_id",
        "json": {"payload_json"},
    },
    "container_artifacts": {
        "id": "artifact_id",
        "json": {"container_graph_revision_ids_json", "families_json", "profile_ids_json", "payload_json"},
    },
}
CONTAINER_GRAPH_SNAPSHOT_KEYS = {
    "graph_revisions": "container_graph_revisions",
    "containers": "containers",
    "relations": "container_relations",
    "anchors": "container_anchors",
    "evidence": "container_evidence",
    "render_refs": "container_render_refs",
    "state": "container_state",
    "refs": "container_refs",
    "ref_lookup": "container_ref_lookup",
    "ref_ranges": "container_ref_ranges",
    "contracts": "container_contracts",
    "artifacts": "container_artifacts",
}
PLAINTEXT_SECRET_STORAGE_ENV = "GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS"  # noqa: S105
SECRET_STORAGE_POLICY_ERROR = (
    "Secret storage requires encrypted SQLite or "
    f"{PLAINTEXT_SECRET_STORAGE_ENV}=1 for explicit dev/test plaintext mode."
)


def _path_has_magic_prefix(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _sqlcipher_available() -> bool:
    return importlib.util.find_spec("pysqlcipher3") is not None


def _execute_sqlcipher_pragma_if_supported(conn: Any, statement: str) -> None:
    try:
        conn.execute(statement)
    except Exception:
        # SQLCipher logging pragmas were added after older SQLCipher releases.
        # They are optional noise-control settings; unsupported builds should
        # still open encrypted databases normally.
        return


def _maybe_log_sqlcipher_memlock_runtime_diagnostic() -> None:
    global _SQLCIPHER_MEMLOCK_DIAGNOSTIC_EMITTED
    if _resource is None:
        return
    try:
        soft, hard = _resource.getrlimit(_resource.RLIMIT_MEMLOCK)
    except Exception:
        return
    if soft == _resource.RLIM_INFINITY or soft >= 1024 * 1024:
        return
    with _SQLCIPHER_MEMLOCK_DIAGNOSTIC_LOCK:
        if _SQLCIPHER_MEMLOCK_DIAGNOSTIC_EMITTED:
            return
        _SQLCIPHER_MEMLOCK_DIAGNOSTIC_EMITTED = True
    log.warning(
        "SQLCipher secure-memory locking may be limited by process/container "
        "RLIMIT_MEMLOCK (soft=%s hard=%s). If native sqlcipher_mlock warnings "
        "appear, fix the runtime with an adequate memlock limit and, where "
        "required, CAP_IPC_LOCK. SQLCipher memory security remains enabled.",
        soft,
        hard,
    )


def _configure_sqlcipher_connection_logging(conn: Any) -> None:
    # These PRAGMAs only control SQLCipher's internal log output. They do not
    # disable encryption, key derivation, HMAC checks, or memory security.
    _execute_sqlcipher_pragma_if_supported(conn, "PRAGMA cipher_log_level = NONE")
    _execute_sqlcipher_pragma_if_supported(conn, "PRAGMA cipher_log_source = NONE")
    _maybe_log_sqlcipher_memlock_runtime_diagnostic()


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_loads(raw: str | None, default: Any) -> Any:
    if raw in (None, ""):
        return default
    assert raw is not None
    try:
        return json.loads(raw)
    except Exception:
        return default


def _source_record_meta_payload(record: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict((record or {}).get("source_meta") or {})
    for key in ("flags", "extraction_report", "source_aggregation_report"):
        if isinstance(record, dict) and key in record:
            payload[key] = record[key]
    return payload


def _sqlite_text_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return _json_dumps(value)
    return value


def _timestamp_ms(value: Any, fallback: int) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.isdigit():
            return int(text)
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    return fallback


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret_value_blob(value: str) -> bytes:
    return str(value).encode("utf-8")


def _decode_secret_value(row: dict[str, Any]) -> str:
    encoding = str(row.get("value_encoding") or "utf-8")
    blob = row.get("value_blob")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise RuntimeError("Secret storage row missing value blob")
    payload = bytes(blob)
    if encoding != "utf-8":
        raise RuntimeError(f"Unsupported secret value_encoding: {encoding}")
    return payload.decode("utf-8")


def _secret_id_for(name: str, acl_domain_key: str) -> str:
    seed = f"{acl_domain_key}\x1f{name}".encode()
    digest = hashlib.sha256(seed).hexdigest()[:32]
    return f"secret:{digest}"


def _secret_acl_domain_key(owner_id: str, read: list[str], write: list[str]) -> str:
    return acl_domain_key(owner_id, read, write)


def _normalize_secret_scope(scope: str | None) -> str:
    normalized = str(scope or "").strip()
    if normalized not in {"agent-private", "swarm-shared", "system-wide"}:
        raise RuntimeError(f"Legacy secret migration failed: invalid scope {scope!r}")
    return normalized


def _normalize_secret_acl_principals(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("Legacy secret migration failed: ACL principals must be list[str]")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise RuntimeError("Legacy secret migration failed: ACL principals must be list[str]")
        item = raw.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _parse_optional_datetime(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


class SQLiteAuthorityStorage:
    """Server-wide persisted principal + swarm authority store."""

    AUTHORITY_SCHEMA_VERSION = 2

    def __init__(
        self,
        data_dir: str,
        encryption_key: bytes | None = None,
        db_path: str | Path | None = None,
    ):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = Path(db_path) if db_path is not None else (self._dir / "_authority.sqlite3")
        self._encryption_key = encryption_key
        self._sqlite_mod, self._sqlcipher = self._resolve_sqlite_module(encryption_key)
        self._connections: dict[int, Any] = {}
        self._connections_lock = threading.Lock()
        atexit.register(self.close)
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.exists()

    def _resolve_sqlite_module(self, encryption_key: bytes | None):
        if encryption_key is None:
            return sqlite3, False
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Encrypted SQLite mode requires SQLCipher (pysqlcipher3). "
                "Authority storage cannot run without SQLCipher when encryption is enabled."
            ) from exc
        return sqlcipher_sqlite, True

    def _open_connection(self):
        conn = self._sqlite_mod.connect(str(self._path), timeout=30, check_same_thread=False)
        if self._sqlcipher:
            _configure_sqlcipher_connection_logging(conn)
            conn.execute(f"PRAGMA key = \"x'{self._encryption_key.hex()}'\"")
        conn.row_factory = self._sqlite_mod.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _connection_for_thread(self):
        tid = threading.get_ident()
        with self._connections_lock:
            conn = self._connections.get(tid)
            if conn is not None:
                return conn
            conn = self._open_connection()
            self._connections[tid] = conn
            return conn

    @contextmanager
    def _connect(self):
        conn = self._connection_for_thread()
        try:
            yield conn
        except Exception:
            try:
                if getattr(conn, "in_transaction", False):
                    conn.rollback()
            except Exception:
                pass
            raise

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS authority_meta (
                    name TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS principals (
                    principal_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    display_name TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT,
                    created_by TEXT,
                    metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS principal_tokens (
                    token_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL REFERENCES principals(principal_id),
                    token_hash BLOB NOT NULL UNIQUE,
                    token_kind TEXT NOT NULL,
                    description TEXT,
                    issued_at TEXT,
                    issued_by TEXT,
                    expires_at TEXT,
                    revoked_at TEXT,
                    revoked_by TEXT,
                    last_used_at TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS principal_tokens_principal_idx
                    ON principal_tokens(principal_id);
                CREATE INDEX IF NOT EXISTS principal_tokens_active_idx
                    ON principal_tokens(token_kind, principal_id)
                    WHERE revoked_at IS NULL;

                CREATE TABLE IF NOT EXISTS swarms (
                    swarm_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    owner_principal_id TEXT NOT NULL REFERENCES principals(principal_id),
                    status TEXT NOT NULL,
                    created_at TEXT,
                    created_by TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS swarms_owner_idx
                    ON swarms(owner_principal_id);

                CREATE TABLE IF NOT EXISTS swarm_memberships (
                    membership_id TEXT PRIMARY KEY,
                    swarm_id TEXT NOT NULL REFERENCES swarms(swarm_id),
                    principal_id TEXT NOT NULL REFERENCES principals(principal_id),
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    granted_at TEXT,
                    granted_by TEXT,
                    revoked_at TEXT,
                    revoked_by TEXT,
                    expires_at TEXT,
                    metadata_json TEXT
                );
                CREATE INDEX IF NOT EXISTS swarm_memberships_principal_idx
                    ON swarm_memberships(principal_id, status);
                CREATE INDEX IF NOT EXISTS swarm_memberships_swarm_idx
                    ON swarm_memberships(swarm_id, status);
                CREATE UNIQUE INDEX IF NOT EXISTS swarm_memberships_active_unique
                    ON swarm_memberships(swarm_id, principal_id)
                    WHERE status = 'active';
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO authority_meta(name, value_json) VALUES(?, ?)",
                ("schema_version", _json_dumps(self.AUTHORITY_SCHEMA_VERSION)),
            )
            conn.commit()

    def bootstrap_state_get(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM authority_meta WHERE name = ?",
                ("bootstrap_state",),
            ).fetchone()
        if row is None:
            return None
        value = _json_loads(row["value_json"], None)
        return value if isinstance(value, dict) else None

    def principal_get(self, principal_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT principal_id, kind, display_name, status, created_at, created_by, metadata_json
                FROM principals
                WHERE principal_id = ?
                """,
                (principal_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "principal_id": row["principal_id"],
            "kind": row["kind"],
            "display_name": row["display_name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def principal_upsert(
        self,
        *,
        principal_id: str,
        kind: str,
        display_name: str | None,
        status: str,
        created_at: str,
        created_by: str | None,
        metadata: dict | None,
    ) -> dict:
        metadata_json = _json_dumps(metadata or {})
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT principal_id FROM principals WHERE principal_id = ?",
                (principal_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO principals(
                        principal_id, kind, display_name, status, created_at, created_by, metadata_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (principal_id, kind, display_name, status, created_at, created_by, metadata_json),
                )
            else:
                conn.execute(
                    """
                    UPDATE principals
                    SET kind = ?, display_name = ?, status = ?, metadata_json = ?
                    WHERE principal_id = ?
                    """,
                    (kind, display_name, status, metadata_json, principal_id),
                )
            conn.commit()
        return self.principal_get(principal_id) or {}

    def principal_set_status(self, principal_id: str, status: str) -> dict | None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE principals SET status = ? WHERE principal_id = ?",
                (status, principal_id),
            )
            conn.commit()
        return self.principal_get(principal_id)

    def principal_set_metadata(self, principal_id: str, metadata: dict[str, Any]) -> dict | None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE principals SET metadata_json = ? WHERE principal_id = ?",
                (_json_dumps(metadata or {}), principal_id),
            )
            conn.commit()
        return self.principal_get(principal_id)

    def token_insert(
        self,
        *,
        token_id: str,
        principal_id: str,
        token_hash: bytes,
        token_kind: str,
        description: str | None,
        issued_at: str,
        issued_by: str | None,
        expires_at: str | None,
        metadata: dict | None,
    ) -> dict:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO principal_tokens(
                    token_id, principal_id, token_hash, token_kind, description,
                    issued_at, issued_by, expires_at, revoked_at, revoked_by,
                    last_used_at, metadata_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    token_id,
                    principal_id,
                    sqlite3.Binary(token_hash),
                    token_kind,
                    description,
                    issued_at,
                    issued_by,
                    expires_at,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.commit()
        return self.token_get(token_id) or {}

    def bootstrap_admin_once(
        self,
        *,
        principal_id: str,
        kind: str,
        display_name: str | None,
        description: str | None,
        expires_at: str | None,
        metadata: dict[str, Any] | None,
        token_id: str,
        token_hash: bytes,
        issued_at: str,
        issued_by: str,
    ) -> dict[str, Any]:
        metadata_payload = metadata or {}
        metadata_json = _json_dumps(metadata_payload)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            state_row = conn.execute(
                "SELECT value_json FROM authority_meta WHERE name = ?",
                ("bootstrap_state",),
            ).fetchone()
            if state_row is not None:
                state = _json_loads(state_row["value_json"], {})
                if isinstance(state, dict) and str(state.get("bootstrapped_at") or "").strip():
                    raise RuntimeError(_json_dumps(state))
            existing_admin_state = self._historical_admin_bootstrap_state(conn)
            if existing_admin_state is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO authority_meta(name, value_json) VALUES(?, ?)",
                    ("bootstrap_state", _json_dumps(existing_admin_state)),
                )
                conn.commit()
                raise RuntimeError(_json_dumps(existing_admin_state))

            principal_row = conn.execute(
                """
                SELECT principal_id, status
                FROM principals
                WHERE principal_id = ?
                """,
                (principal_id,),
            ).fetchone()
            if principal_row is None:
                conn.execute(
                    """
                    INSERT INTO principals(
                        principal_id, kind, display_name, status, created_at, created_by, metadata_json
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (principal_id, kind, display_name, "active", issued_at, issued_by, metadata_json),
                )
            elif str(principal_row["status"] or "") != "active":
                raise PermissionError("cannot bootstrap disabled principal")

            conn.execute(
                """
                INSERT INTO principal_tokens(
                    token_id, principal_id, token_hash, token_kind, description,
                    issued_at, issued_by, expires_at, revoked_at, revoked_by,
                    last_used_at, metadata_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    token_id,
                    principal_id,
                    sqlite3.Binary(token_hash),
                    "admin",
                    description,
                    issued_at,
                    issued_by,
                    expires_at,
                    metadata_json,
                ),
            )

            bootstrap_state = {
                "bootstrapped_at": issued_at,
                "bootstrapped_principal_id": principal_id,
                "bootstrapped_token_id": token_id,
            }
            conn.execute(
                "INSERT OR REPLACE INTO authority_meta(name, value_json) VALUES(?, ?)",
                ("bootstrap_state", _json_dumps(bootstrap_state)),
            )
            conn.commit()

        return {
            "bootstrap_state": bootstrap_state,
            "principal": self.principal_get(principal_id) or {},
            "token": self.token_get(token_id) or {},
        }

    @staticmethod
    def _historical_admin_bootstrap_state(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT pt.token_id, pt.principal_id, pt.issued_at
            FROM principal_tokens pt
            WHERE pt.token_kind = 'admin'
            ORDER BY pt.issued_at ASC, pt.token_id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return {
            "bootstrapped_at": row["issued_at"] or _utcnow_iso(),
            "bootstrapped_principal_id": row["principal_id"],
            "bootstrapped_token_id": row["token_id"],
        }

    def token_get(self, token_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT token_id, principal_id, token_kind, description, issued_at, issued_by,
                       expires_at, revoked_at, revoked_by, last_used_at, metadata_json
                FROM principal_tokens
                WHERE token_id = ?
                """,
                (token_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "token_id": row["token_id"],
            "principal_id": row["principal_id"],
            "token_kind": row["token_kind"],
            "description": row["description"],
            "issued_at": row["issued_at"],
            "issued_by": row["issued_by"],
            "expires_at": row["expires_at"],
            "revoked_at": row["revoked_at"],
            "revoked_by": row["revoked_by"],
            "last_used_at": row["last_used_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def token_revoke(self, token_id: str, *, revoked_at: str, revoked_by: str | None) -> dict | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE principal_tokens
                SET revoked_at = ?, revoked_by = ?
                WHERE token_id = ?
                """,
                (revoked_at, revoked_by, token_id),
            )
            conn.commit()
        return self.token_get(token_id)

    def token_list(self, *, principal_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if principal_id is None:
                rows = conn.execute(
                    """
                    SELECT token_id, principal_id, token_kind, description, issued_at, issued_by,
                           expires_at, revoked_at, revoked_by, last_used_at, metadata_json
                    FROM principal_tokens
                    ORDER BY issued_at ASC, token_id ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT token_id, principal_id, token_kind, description, issued_at, issued_by,
                           expires_at, revoked_at, revoked_by, last_used_at, metadata_json
                    FROM principal_tokens
                    WHERE principal_id = ?
                    ORDER BY issued_at ASC, token_id ASC
                    """,
                    (principal_id,),
                ).fetchall()
        return [
            {
                "token_id": row["token_id"],
                "principal_id": row["principal_id"],
                "token_kind": row["token_kind"],
                "description": row["description"],
                "issued_at": row["issued_at"],
                "issued_by": row["issued_by"],
                "expires_at": row["expires_at"],
                "revoked_at": row["revoked_at"],
                "revoked_by": row["revoked_by"],
                "last_used_at": row["last_used_at"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def resolve_token_hash(self, token_hash: bytes) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT pt.token_id, pt.principal_id, pt.token_kind, pt.description, pt.issued_at,
                       pt.issued_by, pt.expires_at, pt.revoked_at, pt.revoked_by, pt.last_used_at,
                       pt.metadata_json, p.kind AS principal_kind, p.display_name, p.status AS principal_status
                FROM principal_tokens pt
                JOIN principals p ON p.principal_id = pt.principal_id
                WHERE pt.token_hash = ?
                """,
                (sqlite3.Binary(token_hash),),
            ).fetchone()
            if row is None:
                return None
            now = _utcnow_iso()
            conn.execute(
                "UPDATE principal_tokens SET last_used_at = ? WHERE token_id = ?",
                (now, row["token_id"]),
            )
            conn.commit()
        return {
            "token_id": row["token_id"],
            "principal_id": row["principal_id"],
            "token_kind": row["token_kind"],
            "description": row["description"],
            "issued_at": row["issued_at"],
            "issued_by": row["issued_by"],
            "expires_at": row["expires_at"],
            "revoked_at": row["revoked_at"],
            "revoked_by": row["revoked_by"],
            "last_used_at": now,
            "metadata": _json_loads(row["metadata_json"], {}),
            "principal_kind": row["principal_kind"],
            "display_name": row["display_name"],
            "principal_status": row["principal_status"],
        }

    def admin_bootstrap_history_exists(self) -> bool:
        with self._connect() as conn:
            return self._historical_admin_bootstrap_state(conn) is not None

    def swarm_get(self, swarm_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT swarm_id, display_name, owner_principal_id, status, created_at, created_by, metadata_json
                FROM swarms
                WHERE swarm_id = ?
                """,
                (swarm_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "swarm_id": row["swarm_id"],
            "display_name": row["display_name"],
            "owner_principal_id": row["owner_principal_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def swarm_insert(
        self,
        *,
        swarm_id: str,
        display_name: str | None,
        owner_principal_id: str,
        status: str,
        created_at: str,
        created_by: str | None,
        metadata: dict | None,
    ) -> dict:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO swarms(
                    swarm_id, display_name, owner_principal_id, status,
                    created_at, created_by, metadata_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    swarm_id,
                    display_name,
                    owner_principal_id,
                    status,
                    created_at,
                    created_by,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.commit()
        return self.swarm_get(swarm_id) or {}

    def swarm_list(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT swarm_id, display_name, owner_principal_id, status, created_at, created_by, metadata_json
                FROM swarms
                ORDER BY created_at ASC, swarm_id ASC
                """
            ).fetchall()
        return [
            {
                "swarm_id": row["swarm_id"],
                "display_name": row["display_name"],
                "owner_principal_id": row["owner_principal_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def membership_active_row(self, *, swarm_id: str, principal_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                       revoked_at, revoked_by, expires_at, metadata_json
                FROM swarm_memberships
                WHERE swarm_id = ? AND principal_id = ? AND status = 'active'
                ORDER BY granted_at DESC, membership_id DESC
                LIMIT 1
                """,
                (swarm_id, principal_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "membership_id": row["membership_id"],
            "swarm_id": row["swarm_id"],
            "principal_id": row["principal_id"],
            "role": row["role"],
            "status": row["status"],
            "granted_at": row["granted_at"],
            "granted_by": row["granted_by"],
            "revoked_at": row["revoked_at"],
            "revoked_by": row["revoked_by"],
            "expires_at": row["expires_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def membership_insert(
        self,
        *,
        membership_id: str,
        swarm_id: str,
        principal_id: str,
        role: str,
        status: str,
        granted_at: str,
        granted_by: str | None,
        expires_at: str | None,
        metadata: dict | None,
    ) -> dict:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO swarm_memberships(
                    membership_id, swarm_id, principal_id, role, status,
                    granted_at, granted_by, revoked_at, revoked_by, expires_at, metadata_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    membership_id,
                    swarm_id,
                    principal_id,
                    role,
                    status,
                    granted_at,
                    granted_by,
                    expires_at,
                    _json_dumps(metadata or {}),
                ),
            )
            conn.commit()
        return self.membership_active_row(swarm_id=swarm_id, principal_id=principal_id) or {}

    def membership_revoke(
        self,
        *,
        membership_id: str,
        revoked_at: str,
        revoked_by: str | None,
    ) -> dict | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE swarm_memberships
                SET status = 'revoked', revoked_at = ?, revoked_by = ?
                WHERE membership_id = ? AND status = 'active'
                """,
                (revoked_at, revoked_by, membership_id),
            )
            row = conn.execute(
                """
                SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                       revoked_at, revoked_by, expires_at, metadata_json
                FROM swarm_memberships
                WHERE membership_id = ?
                """,
                (membership_id,),
            ).fetchone()
            conn.commit()
        if row is None:
            return None
        return {
            "membership_id": row["membership_id"],
            "swarm_id": row["swarm_id"],
            "principal_id": row["principal_id"],
            "role": row["role"],
            "status": row["status"],
            "granted_at": row["granted_at"],
            "granted_by": row["granted_by"],
            "revoked_at": row["revoked_at"],
            "revoked_by": row["revoked_by"],
            "expires_at": row["expires_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def membership_list(
        self,
        *,
        swarm_id: str | None = None,
        principal_id: str | None = None,
        include_revoked: bool = False,
    ) -> list[dict]:
        with self._connect() as conn:
            if swarm_id is None and principal_id is None and include_revoked:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    ORDER BY granted_at ASC, membership_id ASC
                    """
                ).fetchall()
            elif swarm_id is None and principal_id is None:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE status = 'active'
                    ORDER BY granted_at ASC, membership_id ASC
                    """
                ).fetchall()
            elif swarm_id is not None and principal_id is None and include_revoked:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE swarm_id = ?
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (swarm_id,),
                ).fetchall()
            elif swarm_id is not None and principal_id is None:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE swarm_id = ? AND status = 'active'
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (swarm_id,),
                ).fetchall()
            elif swarm_id is None and principal_id is not None and include_revoked:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE principal_id = ?
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (principal_id,),
                ).fetchall()
            elif swarm_id is None and principal_id is not None:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE principal_id = ? AND status = 'active'
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (principal_id,),
                ).fetchall()
            elif include_revoked:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE swarm_id = ? AND principal_id = ?
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (swarm_id, principal_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT membership_id, swarm_id, principal_id, role, status, granted_at, granted_by,
                           revoked_at, revoked_by, expires_at, metadata_json
                    FROM swarm_memberships
                    WHERE swarm_id = ? AND principal_id = ? AND status = 'active'
                    ORDER BY granted_at ASC, membership_id ASC
                    """,
                    (swarm_id, principal_id),
                ).fetchall()
        return [
            {
                "membership_id": row["membership_id"],
                "swarm_id": row["swarm_id"],
                "principal_id": row["principal_id"],
                "role": row["role"],
                "status": row["status"],
                "granted_at": row["granted_at"],
                "granted_by": row["granted_by"],
                "revoked_at": row["revoked_at"],
                "revoked_by": row["revoked_by"],
                "expires_at": row["expires_at"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]


class JSONNPZStorage:
    """Explicit legacy snapshot backend kept only for migration/research paths."""

    def __init__(self, data_dir: str, key: str, encryption_key: bytes | None = None):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self._dir / f"{key}.json"
        self._embs_path = self._dir / f"{key}_embs.npz"
        self._encryption_key = encryption_key

    def _encrypt(self, data: bytes) -> bytes:
        if self._encryption_key is None:
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            return data
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._encryption_key)
        ct = aesgcm.encrypt(nonce, data, None)
        return MAGIC + nonce + ct

    def _decrypt(self, data: bytes) -> bytes:
        if not data.startswith(MAGIC):
            return data
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise RuntimeError("Encrypted file but cryptography package not installed") from exc
        if self._encryption_key is None:
            raise RuntimeError("Encrypted file but no encryption key provided")
        nonce = data[4:16]
        ct = data[16:]
        aesgcm = AESGCM(self._encryption_key)
        return aesgcm.decrypt(nonce, ct, None)

    @property
    def exists(self) -> bool:
        return self._json_path.exists()

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def embs_path(self) -> Path:
        return self._embs_path

    def load_facts(self, *, internal: bool = False) -> dict:
        if not self._json_path.exists():
            return {}
        raw = self._json_path.read_bytes()
        decrypted = self._decrypt(raw)
        return json.loads(decrypted.decode("utf-8"))

    def save_facts(self, data: dict) -> None:
        plaintext = json.dumps(data).encode("utf-8")
        output = self._encrypt(plaintext)
        fd, tmp_path = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(output)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, str(self._json_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_embeddings(self) -> dict | None:
        if not self._embs_path.exists():
            return None
        raw = self._embs_path.read_bytes()
        decrypted = self._decrypt(raw)
        if raw.startswith(MAGIC):
            buf = io.BytesIO(decrypted)
            loaded = np.load(buf)
        else:
            loaded = np.load(self._embs_path)
        return {k: loaded[k] for k in loaded.files}

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, gran=gran, cons=cons, cross=cross)
        plaintext = buf.getvalue()
        output = self._encrypt(plaintext)
        if output is plaintext and self._encryption_key is None:
            np.savez_compressed(self._embs_path, gran=gran, cons=cons, cross=cross)
        else:
            self._embs_path.write_bytes(output)


class SQLiteStorageBackend:
    """SQLite-backed single durable store for one memory key."""

    def __init__(
        self,
        data_dir: str,
        key: str,
        encryption_key: bytes | None = None,
        db_path: str | Path | None = None,
    ):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._path = Path(db_path) if db_path is not None else (self._dir / f"{key}.sqlite3")
        self._encryption_key = encryption_key
        self._sqlite_mod, self._sqlcipher = self._resolve_sqlite_module(encryption_key)
        self._connections: dict[int, Any] = {}
        self._connections_lock = threading.Lock()
        atexit.register(self.close)
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def exists(self) -> bool:
        return self._path.exists()

    def _resolve_sqlite_module(self, encryption_key: bytes | None):
        if encryption_key is None:
            return sqlite3, False
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher_sqlite  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Encrypted SQLite mode requires SQLCipher (pysqlcipher3). "
                "Migration must abort when SQLCipher is unavailable."
            ) from exc
        return sqlcipher_sqlite, True

    def _open_connection(self):
        conn = self._sqlite_mod.connect(str(self._path), timeout=30, check_same_thread=False)
        if self._sqlcipher:
            _configure_sqlcipher_connection_logging(conn)
            conn.execute(f"PRAGMA key = \"x'{self._encryption_key.hex()}'\"")
        conn.row_factory = self._sqlite_mod.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _connection_for_thread(self):
        tid = threading.get_ident()
        with self._connections_lock:
            conn = self._connections.get(tid)
            if conn is not None:
                return conn
            conn = self._open_connection()
            self._connections[tid] = conn
            return conn

    @contextmanager
    def _connect(self):
        conn = self._connection_for_thread()
        try:
            yield conn
        except Exception:
            try:
                if getattr(conn, "in_transaction", False):
                    conn.rollback()
            except Exception:
                pass
            raise

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass

    def _table_exists(self, conn, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _has_user_tables(self, conn) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        return row is not None

    def _table_info(self, conn, table_name: str) -> list[dict[str, Any]]:
        # Table names come from fixed internal schema identifiers.
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()  # nosec B608
        return [dict(row) for row in rows]

    def _table_pk_columns(self, conn, table_name: str) -> list[str]:
        cols = self._table_info(conn, table_name)
        pk_cols = [col for col in cols if int(col.get("pk") or 0) > 0]
        pk_cols.sort(key=lambda col: int(col.get("pk") or 0))
        return [str(col["name"]) for col in pk_cols]

    def _unique_index_columns(self, conn, table_name: str) -> set[tuple[str, ...]]:
        # Table and index names come from fixed internal schema identifiers.
        index_rows = conn.execute(f"PRAGMA index_list({table_name})").fetchall()  # nosec B608
        unique_cols: set[tuple[str, ...]] = set()
        for row in index_rows:
            if int(row["unique"] or 0) != 1:
                continue
            if str(row["origin"] or "") == "pk":
                continue
            index_name = str(row["name"])
            info_rows = conn.execute(f"PRAGMA index_info({index_name})").fetchall()  # nosec B608
            cols = tuple(str(info["name"]) for info in sorted(info_rows, key=lambda item: int(item["seqno"] or 0)))
            if cols:
                unique_cols.add(cols)
        return unique_cols

    def _read_meta(self, conn, *, table_name: str = "meta") -> dict[str, Any]:
        if not self._table_exists(conn, table_name):
            return {}
        rows = conn.execute(f"SELECT name, value_json FROM {table_name}").fetchall()  # noqa: S608  # nosec B608
        return {str(row["name"]): _json_loads(row["value_json"], None) for row in rows}

    def _set_meta(self, conn, name: str, value: Any, *, table_name: str = "meta") -> None:
        conn.execute(
            f"INSERT OR REPLACE INTO {table_name}(name, value_json) VALUES(?, ?)",  # nosec B608
            (name, _json_dumps(value)),
        )

    def _schema_script(self, *, suffix: str = "", if_not_exists: bool = False) -> str:
        create_clause = " IF NOT EXISTS" if if_not_exists else ""

        def t(name: str) -> str:
            return f"{name}{suffix}"

        def i(name: str) -> str:
            return f"{name}{suffix}"

        return f"""
            CREATE TABLE{create_clause} {t('meta')} (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            CREATE TABLE{create_clause} {t('write_log')} (
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
                lease_owner TEXT,
                lease_expires_at_ms INTEGER,
                sort_order INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX{create_clause} {i('idx_wl_state')} ON {t('write_log')}(extraction_state, timestamp_ms);
            CREATE INDEX{create_clause} {i('idx_wl_claim')} ON {t('write_log')}(extraction_state, lease_expires_at_ms, sort_order);
            CREATE INDEX{create_clause} {i('idx_wl_swarm')} ON {t('write_log')}(swarm_id, timestamp_ms);
            CREATE INDEX{create_clause} {i('idx_wl_session')} ON {t('write_log')}(session_id, timestamp_ms);

            CREATE TABLE{create_clause} {t('raw_sessions')} (
                session_num INTEGER PRIMARY KEY,
                message_id TEXT NOT NULL REFERENCES {t('write_log')}(message_id),
                raw_session_id TEXT UNIQUE,
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

            CREATE TABLE{create_clause} {t('raw_docs')} (
                source_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL UNIQUE REFERENCES {t('write_log')}(message_id),
                metadata_json TEXT,
                sort_order INTEGER NOT NULL UNIQUE
            );

            CREATE TABLE{create_clause} {t('facts')} (
                tier TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
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
                PRIMARY KEY (tier, fact_id),
                UNIQUE (tier, sort_order)
            );

            CREATE TABLE{create_clause} {t('embeddings')} (
                tier TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY (tier, fact_id),
                UNIQUE (tier, sort_order)
            );

            CREATE TABLE{create_clause} {t('episode_corpus')} (
                doc_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                episode_json TEXT NOT NULL,
                PRIMARY KEY (doc_id, episode_id),
                UNIQUE (doc_id, sort_order)
            );

            CREATE TABLE{create_clause} {t('temporal_links')} (
                sort_order INTEGER PRIMARY KEY,
                link_json TEXT NOT NULL
            );

            CREATE TABLE{create_clause} {t('source_records')} (
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

            CREATE TABLE{create_clause} {t('secrets')} (
                secret_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                value_encoding TEXT NOT NULL,
                acl_domain_key TEXT NOT NULL,
                created_by_principal_id TEXT NOT NULL,
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
            CREATE UNIQUE INDEX{create_clause} {i('secrets_name_domain_unique')}
                ON {t('secrets')}(name, acl_domain_key);
            CREATE INDEX{create_clause} {i('secrets_domain_idx')}
                ON {t('secrets')}(acl_domain_key, name);
            CREATE INDEX{create_clause} {i('secrets_owner_idx')}
                ON {t('secrets')}(owner_id, name);
            CREATE INDEX{create_clause} {i('secrets_scope_idx')}
                ON {t('secrets')}(scope, swarm_id, owner_id);

            CREATE TABLE{create_clause} {t('state_json')} (
                name TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );

            CREATE TABLE{create_clause} {t('index_status')} (
                name TEXT PRIMARY KEY,
                index_dirty INTEGER NOT NULL DEFAULT 0,
                index_dirty_since_ms INTEGER,
                last_index_dirty_ms INTEGER,
                next_index_build_after_ms INTEGER,
                next_index_retry_after_ms INTEGER,
                index_build_lease_owner TEXT,
                index_build_lease_expires_at_ms INTEGER,
                last_index_build_started_ms INTEGER,
                last_index_build_completed_ms INTEGER,
                last_index_build_error TEXT,
                last_index_build_error_count INTEGER NOT NULL DEFAULT 0,
                dirty_after_build INTEGER NOT NULL DEFAULT 0,
                snapshot_fingerprint TEXT,
                updated_at_ms INTEGER NOT NULL DEFAULT 0
            );
        """

    def _schema_is_v2(self, conn) -> bool:
        required_tables = {
            "meta",
            "write_log",
            "raw_sessions",
            "raw_docs",
            "facts",
            "embeddings",
            "episode_corpus",
            "temporal_links",
            "source_records",
            "secrets",
            "state_json",
        }
        if not all(self._table_exists(conn, table_name) for table_name in required_tables):
            return False

        facts_pk = self._table_pk_columns(conn, "facts")
        embeddings_pk = self._table_pk_columns(conn, "embeddings")
        raw_sessions_pk = self._table_pk_columns(conn, "raw_sessions")
        if facts_pk != ["tier", "fact_id"]:
            return False
        if embeddings_pk != ["tier", "fact_id"]:
            return False
        if raw_sessions_pk != ["session_num"]:
            return False

        if ("tier", "sort_order") not in self._unique_index_columns(conn, "facts"):
            return False
        if ("tier", "sort_order") not in self._unique_index_columns(conn, "embeddings"):
            return False

        raw_session_cols = {col["name"]: col for col in self._table_info(conn, "raw_sessions")}
        raw_doc_cols = {col["name"]: col for col in self._table_info(conn, "raw_docs")}
        if "raw_session_id" not in raw_session_cols:
            return False
        if int(raw_session_cols.get("message_id", {}).get("notnull") or 0) != 1:
            return False
        if int(raw_doc_cols.get("message_id", {}).get("notnull") or 0) != 1:
            return False
        if ("message_id",) in self._unique_index_columns(conn, "raw_sessions"):
            return False
        secret_cols = {col["name"]: col for col in self._table_info(conn, "secrets")}
        required_secret_cols = {
            "secret_id",
            "name",
            "value_blob",
            "value_encoding",
            "acl_domain_key",
            "created_by_principal_id",
            "owner_id",
            "scope",
            "read_json",
            "write_json",
            "created_at",
            "updated_at",
        }
        if not required_secret_cols.issubset(secret_cols):
            return False
        if ("name", "acl_domain_key") not in self._unique_index_columns(conn, "secrets"):
            return False
        return True

    def _ensure_meta_defaults(self, conn) -> None:
        now = datetime.now(timezone.utc).isoformat()
        current = self._read_meta(conn)
        self._set_meta(conn, "schema_version", SQLITE_SCHEMA_VERSION)
        self._set_meta(conn, "storage_backend", "sqlite")
        if "migrated_from" not in current:
            self._set_meta(conn, "migrated_from", None)
        if "migration_completed_at" not in current:
            self._set_meta(conn, "migration_completed_at", now)

    def secret_storage_allowed(self) -> bool:
        return bool(self._sqlcipher or _env_truthy(PLAINTEXT_SECRET_STORAGE_ENV))

    def _require_secret_storage_allowed(self) -> None:
        if not self.secret_storage_allowed():
            raise RuntimeError(SECRET_STORAGE_POLICY_ERROR)

    def _ensure_secret_table(self, conn) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS secrets (
                secret_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                value_blob BLOB NOT NULL,
                value_encoding TEXT NOT NULL,
                acl_domain_key TEXT NOT NULL,
                created_by_principal_id TEXT NOT NULL,
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
            CREATE UNIQUE INDEX IF NOT EXISTS secrets_name_domain_unique
                ON secrets(name, acl_domain_key);
            CREATE INDEX IF NOT EXISTS secrets_domain_idx
                ON secrets(acl_domain_key, name);
            CREATE INDEX IF NOT EXISTS secrets_owner_idx
                ON secrets(owner_id, name);
            CREATE INDEX IF NOT EXISTS secrets_scope_idx
                ON secrets(scope, swarm_id, owner_id);
            """
        )

    def _canonical_secret_row(
        self,
        *,
        name: str,
        value: str,
        created_by_principal_id: str,
        owner_id: str,
        scope: str,
        agent_id: str | None,
        swarm_id: str | None,
        read: list[str],
        write: list[str],
        created_at: str | None = None,
        updated_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("secret name must be non-empty")
        normalized_scope = _normalize_secret_scope(scope)
        normalized_swarm = str(swarm_id or "").strip() or None
        normalized_read = _normalize_secret_acl_principals(read)
        normalized_write = _normalize_secret_acl_principals(write)
        acl_domain = _secret_acl_domain_key(str(owner_id), normalized_read, normalized_write)
        return {
            "secret_id": _secret_id_for(normalized_name, acl_domain),
            "name": normalized_name,
            "value_blob": _secret_value_blob(value),
            "value_encoding": "utf-8",
            "acl_domain_key": acl_domain,
            "created_by_principal_id": str(created_by_principal_id),
            "owner_id": str(owner_id),
            "scope": normalized_scope,
            "agent_id": str(agent_id).strip() if str(agent_id or "").strip() else None,
            "swarm_id": normalized_swarm,
            "read_json": _json_dumps(normalized_read),
            "write_json": _json_dumps(normalized_write),
            "created_at": created_at or _utcnow_iso(),
            "updated_at": updated_at or _utcnow_iso(),
            "metadata_json": _json_dumps(metadata or {}),
        }

    def _insert_secret_row(self, conn, row: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO secrets(
                secret_id, name, value_blob, value_encoding, acl_domain_key, created_by_principal_id, owner_id,
                scope, agent_id, swarm_id, read_json, write_json,
                created_at, updated_at, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["secret_id"],
                row["name"],
                row["value_blob"],
                row["value_encoding"],
                row["acl_domain_key"],
                row["created_by_principal_id"],
                row["owner_id"],
                row["scope"],
                row["agent_id"],
                row["swarm_id"],
                row["read_json"],
                row["write_json"],
                row["created_at"],
                row["updated_at"],
                row["metadata_json"],
            ),
        )

    def _normalize_legacy_secret_rows(self, raw_value: Any) -> list[dict[str, Any]]:
        if raw_value in (None, ""):
            return []
        if not isinstance(raw_value, list):
            raise RuntimeError("Legacy secret migration failed: secrets state must be a list")
        normalized: list[dict[str, Any]] = []
        for entry in raw_value:
            if not isinstance(entry, dict):
                raise RuntimeError("Legacy secret migration failed: secret entry must be an object")
            name = str(entry.get("name") or "").strip()
            value = entry.get("value")
            if not name or not isinstance(value, str):
                raise RuntimeError("Legacy secret migration failed: secret name/value missing")
            scope = _normalize_secret_scope(entry.get("scope"))
            agent_id = str(entry.get("agent_id") or "").strip() or None
            swarm_id = str(entry.get("swarm_id") or "").strip() or None
            owner_id = entry.get("owner_id")
            if not isinstance(owner_id, str) or not owner_id.strip():
                if scope == "system-wide":
                    owner_id = "system"
                elif agent_id:
                    owner_id = f"agent:{agent_id}"
                else:
                    raise RuntimeError("Legacy secret migration failed: owner_id missing")
            read = entry.get("read")
            write = entry.get("write")
            if read is None or write is None:
                if scope == "system-wide":
                    read = ["agent:PUBLIC"]
                    write = ["agent:PUBLIC"]
                elif scope == "swarm-shared":
                    if not swarm_id or swarm_id == "default":
                        raise RuntimeError("Legacy secret migration failed: swarm-shared secret missing swarm_id")
                    grant = f"swarm:{swarm_id}"
                    read = [grant]
                    write = [grant]
                else:
                    read = []
                    write = []
            created_at = str(entry.get("created_at") or entry.get("stored_at") or _utcnow_iso())
            updated_at = str(entry.get("updated_at") or entry.get("stored_at") or created_at)
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            normalized.append(
                self._canonical_secret_row(
                    name=name,
                    value=value,
                    created_by_principal_id=str(owner_id),
                    owner_id=str(owner_id),
                    scope=scope,
                    agent_id=agent_id,
                    swarm_id=swarm_id,
                    read=_normalize_secret_acl_principals(read),
                    write=_normalize_secret_acl_principals(write),
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=metadata,
                )
            )
        return normalized

    def _migrate_legacy_secret_state(self, conn) -> None:
        secret_state = conn.execute(
            "SELECT value_json FROM state_json WHERE name = ?",
            ("secrets",),
        ).fetchone()
        if secret_state is None:
            return
        rows = self._normalize_legacy_secret_rows(_json_loads(secret_state["value_json"], None))
        if rows:
            self._require_secret_storage_allowed()
        for row in rows:
            self._insert_secret_row(conn, row)
        conn.execute("DELETE FROM state_json WHERE name = ?", ("secrets",))

    def _rebuild_secret_acl_domains(self, conn) -> None:
        if not self._table_exists(conn, "secrets"):
            return
        rows = conn.execute(
            """
            SELECT secret_id, name, acl_domain_key, owner_id, read_json, write_json
            FROM secrets
            ORDER BY name, acl_domain_key
            """
        ).fetchall()
        for row in rows:
            normalized_read = _normalize_secret_acl_principals(_json_loads(row["read_json"], []))
            normalized_write = _normalize_secret_acl_principals(_json_loads(row["write_json"], []))
            expected_domain = _secret_acl_domain_key(str(row["owner_id"]), normalized_read, normalized_write)
            expected_secret_id = _secret_id_for(str(row["name"]), expected_domain)
            if expected_domain == str(row["acl_domain_key"]) and expected_secret_id == str(row["secret_id"]):
                continue
            existing = conn.execute(
                """
                SELECT secret_id
                FROM secrets
                WHERE name = ? AND acl_domain_key = ? AND secret_id != ?
                """,
                (str(row["name"]), expected_domain, str(row["secret_id"])),
            ).fetchone()
            if existing is not None:
                raise RuntimeError(
                    "Secret ACL migration failed: duplicate canonical secret domain for "
                    f"name={row['name']!r}"
                )
            conn.execute(
                """
                UPDATE secrets
                SET secret_id = ?, acl_domain_key = ?, read_json = ?, write_json = ?
                WHERE secret_id = ?
                """,
                (
                    expected_secret_id,
                    expected_domain,
                    _json_dumps(normalized_read),
                    _json_dumps(normalized_write),
                    str(row["secret_id"]),
                ),
            )

    def _migrate_schema_to_v2(self, conn) -> None:
        old_meta = self._read_meta(conn)
        migrated_from = old_meta.get("migrated_from")
        if migrated_from is None:
            migrated_from = "sqlite_v1"
        migration_completed_at = datetime.now(timezone.utc).isoformat()

        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executescript(self._schema_script(suffix="_new"))

            if self._table_exists(conn, "meta"):
                conn.execute("INSERT INTO meta_new(name, value_json) SELECT name, value_json FROM meta")

            if self._table_exists(conn, "write_log"):
                conn.execute(
                    """
                    INSERT INTO write_log_new(
                        message_id, session_id, agent_id, swarm_id, visibility,
                        owner_id, scope, read_json, write_json, content_family,
                        content_text, metadata_json, timestamp_ms, extraction_state,
                        extraction_attempts, last_extraction_attempt_ms, sort_order
                    )
                    SELECT
                        message_id, session_id, agent_id, swarm_id, visibility,
                        owner_id, scope, read_json, write_json, content_family,
                        content_text, metadata_json, timestamp_ms, extraction_state,
                        extraction_attempts, last_extraction_attempt_ms, sort_order
                    FROM write_log
                    ORDER BY sort_order, timestamp_ms
                    """
                )

            fact_id_by_sort: dict[tuple[str, int], str] = {}

            if self._table_exists(conn, "raw_sessions"):
                session_rows = conn.execute(
                    """
                    SELECT raw_session_id, session_num, message_id, source_id, format, session_date,
                           speakers, stored_at, artifact_id, version_id, content_hash, owner_id,
                           scope, agent_id, swarm_id, read_json, write_json, target_json,
                           metadata_json, source_meta_json, status, sort_order
                    FROM raw_sessions
                    ORDER BY sort_order
                    """
                ).fetchall()
                used_session_nums: set[int] = set()
                next_session_num = 1
                session_batch: list[tuple[Any, ...]] = []
                for idx, row in enumerate(session_rows):
                    current = row["session_num"]
                    if isinstance(current, int) and current > 0 and current not in used_session_nums:
                        session_num = int(current)
                    else:
                        session_num = idx + 1
                        while session_num in used_session_nums:
                            session_num += 1
                        next_session_num = max(next_session_num, session_num + 1)
                    used_session_nums.add(session_num)
                    session_batch.append(
                        (
                            session_num,
                            row["message_id"],
                            row["raw_session_id"],
                            row["source_id"],
                            row["format"],
                            row["session_date"],
                            row["speakers"],
                            row["stored_at"],
                            row["artifact_id"],
                            row["version_id"],
                            row["content_hash"],
                            row["owner_id"],
                            row["scope"],
                            row["agent_id"],
                            row["swarm_id"],
                            row["read_json"],
                            row["write_json"],
                            row["target_json"],
                            row["metadata_json"],
                            row["source_meta_json"],
                            row["status"],
                            row["sort_order"],
                        )
                    )
                if session_batch:
                    conn.executemany(
                        """
                        INSERT INTO raw_sessions_new(
                            session_num, message_id, raw_session_id, source_id, format, session_date,
                            speakers, stored_at, artifact_id, version_id, content_hash, owner_id,
                            scope, agent_id, swarm_id, read_json, write_json, target_json,
                            metadata_json, source_meta_json, status, sort_order
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        session_batch,
                    )

            if self._table_exists(conn, "raw_docs"):
                conn.execute(
                    """
                    INSERT INTO raw_docs_new(source_id, message_id, metadata_json, sort_order)
                    SELECT source_id, message_id, metadata_json, sort_order
                    FROM raw_docs
                    ORDER BY sort_order
                    """
                )

            if self._table_exists(conn, "facts"):
                fact_rows = conn.execute(
                    """
                    SELECT tier, sort_order, fact_id, kind, session_num, source_id, agent_id, swarm_id,
                           scope, owner_id, status, created_at, event_date, payload_json
                    FROM facts
                    ORDER BY tier, sort_order
                    """
                ).fetchall()
                seen_fact_ids: dict[str, set[str]] = {}
                fact_batch: list[tuple[Any, ...]] = []
                for row in fact_rows:
                    tier = str(row["tier"] or "")
                    seen = seen_fact_ids.setdefault(tier, set())
                    fact_id = str(row["fact_id"] or "").strip()
                    if not fact_id:
                        payload = _json_loads(row["payload_json"], {})
                        if isinstance(payload, dict):
                            fact_id = str(payload.get("id") or "").strip()
                    if not fact_id:
                        fact_id = f"migrated:{tier}:{int(row['sort_order'] or 0)}"
                    if fact_id in seen:
                        fact_id = f"{fact_id}__{int(row['sort_order'] or 0)}"
                    seen.add(fact_id)
                    sort_order = int(row["sort_order"] or 0)
                    fact_id_by_sort[(tier, sort_order)] = fact_id
                    fact_batch.append(
                        (
                            tier,
                            fact_id,
                            sort_order,
                            row["kind"],
                            row["session_num"],
                            row["source_id"],
                            row["agent_id"],
                            row["swarm_id"],
                            row["scope"],
                            row["owner_id"],
                            row["status"],
                            row["created_at"],
                            row["event_date"],
                            row["payload_json"],
                        )
                    )
                if fact_batch:
                    conn.executemany(
                        """
                        INSERT INTO facts_new(
                            tier, fact_id, sort_order, kind, session_num, source_id, agent_id,
                            swarm_id, scope, owner_id, status, created_at, event_date, payload_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        fact_batch,
                    )

            if self._table_exists(conn, "embeddings"):
                emb_rows = conn.execute(
                    """
                    SELECT tier, sort_order, dim, dtype, vector_blob
                    FROM embeddings
                    ORDER BY tier, sort_order
                    """
                ).fetchall()
                embedding_batch: list[tuple[Any, ...]] = []
                fact_tier_map = {"gran": "granular", "cons": "cons", "cross": "cross"}
                for row in emb_rows:
                    emb_tier = str(row["tier"] or "")
                    fact_tier = fact_tier_map.get(emb_tier, emb_tier)
                    sort_order = int(row["sort_order"] or 0)
                    mapped_fact_id = fact_id_by_sort.get((fact_tier, sort_order))
                    if not mapped_fact_id:
                        raise RuntimeError(
                            f"SQLite schema migration failed: missing fact mapping for embedding {emb_tier}:{sort_order}"
                        )
                    embedding_batch.append(
                        (
                            emb_tier,
                            mapped_fact_id,
                            sort_order,
                            row["dim"],
                            row["dtype"],
                            row["vector_blob"],
                        )
                    )
                if embedding_batch:
                    conn.executemany(
                        """
                        INSERT INTO embeddings_new(tier, fact_id, sort_order, dim, dtype, vector_blob)
                        VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        embedding_batch,
                    )

            if self._table_exists(conn, "episode_corpus"):
                conn.execute(
                    """
                    INSERT INTO episode_corpus_new(doc_id, episode_id, sort_order, episode_json)
                    SELECT doc_id, episode_id, sort_order, episode_json
                    FROM episode_corpus
                    ORDER BY doc_id, sort_order
                    """
                )

            if self._table_exists(conn, "temporal_links"):
                conn.execute(
                    """
                    INSERT INTO temporal_links_new(sort_order, link_json)
                    SELECT sort_order, link_json
                    FROM temporal_links
                    ORDER BY sort_order
                    """
                )

            if self._table_exists(conn, "source_records"):
                conn.execute(
                    """
                    INSERT INTO source_records_new(
                        source_id, family, owner_id, read_json, write_json, artifact_id, version_id,
                        content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at
                    )
                    SELECT
                        source_id, family, owner_id, read_json, write_json, artifact_id, version_id,
                        content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at
                    FROM source_records
                    ORDER BY source_id
                    """
                )

            if self._table_exists(conn, "secrets"):
                secret_cols = {col["name"] for col in self._table_info(conn, "secrets")}
                if "created_by_principal_id" in secret_cols:
                    created_by_select = "created_by_principal_id"
                else:
                    created_by_select = "owner_id"
                conn.execute(
                    f"""
                    INSERT INTO secrets_new(
                        secret_id, name, value_blob, value_encoding, acl_domain_key, created_by_principal_id, owner_id,
                        scope, agent_id, swarm_id, read_json, write_json,
                        created_at, updated_at, metadata_json
                    )
                    SELECT
                        secret_id, name, value_blob, value_encoding, acl_domain_key, {created_by_select}, owner_id,
                        scope, agent_id, swarm_id, read_json, write_json,
                        created_at, updated_at, metadata_json
                    FROM secrets
                    ORDER BY name, acl_domain_key
                    """  # noqa: S608  # nosec B608
                )

            if self._table_exists(conn, "state_json"):
                conn.execute(
                    """
                    INSERT INTO state_json_new(name, value_json)
                    SELECT name, value_json
                    FROM state_json
                    ORDER BY name
                    """
                )

            for table_name in (
                "write_log",
                "raw_sessions",
                "raw_docs",
                "facts",
                "embeddings",
                "episode_corpus",
                "temporal_links",
                "source_records",
                "secrets",
                "state_json",
            ):
                if not self._table_exists(conn, table_name):
                    continue
                before = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()["n"])  # noqa: S608  # nosec B608
                after = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}_new").fetchone()["n"])  # noqa: S608  # nosec B608
                if before != after:
                    raise RuntimeError(
                        f"SQLite schema migration failed: row-count mismatch for {table_name} ({before} != {after})"
                    )

            for table_name in (
                "raw_docs",
                "raw_sessions",
                "embeddings",
                "facts",
                "episode_corpus",
                "temporal_links",
                "source_records",
                "secrets",
                "state_json",
                "write_log",
                "meta",
            ):
                if self._table_exists(conn, table_name):
                    conn.execute(f"DROP TABLE {table_name}")  # nosec B608

            for table_name in (
                "meta",
                "write_log",
                "raw_sessions",
                "raw_docs",
                "facts",
                "embeddings",
                "episode_corpus",
                "temporal_links",
                "source_records",
                "secrets",
                "state_json",
            ):
                conn.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")  # nosec B608

            self._set_meta(conn, "schema_version", SQLITE_SCHEMA_VERSION)
            self._set_meta(conn, "storage_backend", "sqlite")
            self._set_meta(conn, "migrated_from", migrated_from)
            self._set_meta(conn, "migration_completed_at", migration_completed_at)
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    def _ensure_container_graph_tables(self, conn) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS container_graph_revisions (
                container_graph_revision_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                logical_source_id TEXT, family TEXT NOT NULL, revision_id TEXT NOT NULL,
                revision_scope TEXT NOT NULL DEFAULT 'source_revision', content_revision_id TEXT,
                external_revision_id TEXT, adapter_name TEXT NOT NULL, adapter_version TEXT NOT NULL,
                inference_version TEXT, input_fingerprint TEXT NOT NULL, graph_fingerprint TEXT,
                profile_ids_json TEXT NOT NULL DEFAULT '[]', coverage_report_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL, parent_graph_revision_id TEXT, created_at TEXT, completed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS containers (
                container_id TEXT PRIMARY KEY, id_origin TEXT NOT NULL DEFAULT 'stable_hash',
                id_schema_version TEXT NOT NULL, identity_scope TEXT NOT NULL, source_id TEXT NOT NULL,
                logical_source_id TEXT, family TEXT NOT NULL, revision_id TEXT NOT NULL,
                revision_scope TEXT NOT NULL DEFAULT 'source_revision', content_revision_id TEXT,
                external_revision_id TEXT, container_graph_revision_id TEXT NOT NULL,
                kind_ns TEXT NOT NULL, kind TEXT NOT NULL, kind_fq TEXT NOT NULL, kind_version TEXT,
                traits_json TEXT NOT NULL DEFAULT '{}', order_key_json TEXT NOT NULL,
                order_basis TEXT NOT NULL, order_scope TEXT NOT NULL, order_scope_id TEXT NOT NULL,
                span_refs_json TEXT NOT NULL, episode_ids_json TEXT NOT NULL DEFAULT '[]',
                primary_render_ref_id TEXT, primary_render_ref_fingerprint TEXT, render_ref_json TEXT NOT NULL,
                status TEXT NOT NULL, supersedes_container_id TEXT, superseded_by_container_id TEXT,
                boundary_score REAL, kind_score REAL, render_score REAL,
                acl_inherit_source INTEGER NOT NULL DEFAULT 1, owner_id TEXT, scope TEXT,
                agent_id TEXT, swarm_id TEXT, read_json TEXT, write_json TEXT, acl_source_id TEXT,
                acl_policy TEXT NOT NULL DEFAULT 'inherit_source_record', adapter_name TEXT NOT NULL,
                adapter_version TEXT NOT NULL, inference_version TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_relations (
                relation_id TEXT PRIMARY KEY, container_graph_revision_id TEXT NOT NULL,
                src_container_id TEXT NOT NULL, dst_container_id TEXT NOT NULL,
                src_container_graph_revision_id TEXT, dst_container_graph_revision_id TEXT,
                relation_ns TEXT NOT NULL, relation_kind TEXT NOT NULL, order_key_json TEXT,
                traits_json TEXT NOT NULL DEFAULT '{}', relation_score REAL, status TEXT NOT NULL,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_anchors (
                anchor_id TEXT PRIMARY KEY, container_id TEXT NOT NULL, container_graph_revision_id TEXT NOT NULL,
                anchor_kind TEXT NOT NULL, anchor_value TEXT NOT NULL, anchor_norm TEXT, anchor_hash TEXT,
                anchor_lang TEXT, origin_kind TEXT NOT NULL, origin_ref_json TEXT NOT NULL, origin_id TEXT,
                origin_tier TEXT, origin_source_id TEXT, origin_revision_id TEXT, origin_partition_kind TEXT,
                origin_partition_id TEXT, role TEXT, weight REAL NOT NULL DEFAULT 1.0, anchor_score REAL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS container_evidence (
                evidence_id TEXT PRIMARY KEY, container_graph_revision_id TEXT NOT NULL,
                subject_type TEXT NOT NULL, subject_id TEXT NOT NULL, evidence_kind TEXT NOT NULL,
                evidence_ref_json TEXT NOT NULL, role TEXT NOT NULL, score_name TEXT, score REAL,
                status TEXT NOT NULL DEFAULT 'active', trace_json TEXT NOT NULL DEFAULT '{}', created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_render_refs (
                render_ref_id TEXT PRIMARY KEY, container_id TEXT NOT NULL, container_graph_revision_id TEXT NOT NULL,
                render_kind TEXT NOT NULL, render_mode TEXT NOT NULL, ref_json TEXT NOT NULL,
                ref_fingerprint TEXT NOT NULL, fidelity TEXT NOT NULL, language TEXT, format TEXT,
                token_estimate INTEGER, status TEXT NOT NULL DEFAULT 'active', created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_state (
                state_id TEXT PRIMARY KEY, container_id TEXT NOT NULL, container_graph_revision_id TEXT NOT NULL,
                state_kind TEXT NOT NULL, state_scope_kind TEXT NOT NULL, state_scope_id TEXT NOT NULL,
                state_subject_kind TEXT, state_subject_id TEXT, state_conflict_group_id TEXT, confidence REAL,
                valid_from TEXT, valid_until TEXT, superseded_by_container_id TEXT, supersedes_container_id TEXT,
                current_state INTEGER NOT NULL DEFAULT 0, state_reason_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_refs (
                ref_id TEXT PRIMARY KEY, container_graph_revision_id TEXT NOT NULL, container_id TEXT,
                ref_role TEXT NOT NULL, ref_type TEXT NOT NULL, ref_json TEXT NOT NULL,
                ref_fingerprint TEXT NOT NULL, source_id TEXT, logical_source_id TEXT, revision_id TEXT,
                coverage_required INTEGER NOT NULL DEFAULT 0, coverage_status TEXT NOT NULL DEFAULT 'mapped',
                status TEXT NOT NULL DEFAULT 'active', created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_ref_lookup (
                lookup_id TEXT PRIMARY KEY, ref_id TEXT NOT NULL, lookup_ns TEXT NOT NULL, lookup_key TEXT NOT NULL,
                value_text TEXT, value_int INTEGER, value_hash TEXT, status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS container_ref_ranges (
                range_id TEXT PRIMARY KEY, ref_id TEXT NOT NULL, range_kind TEXT NOT NULL, range_scope TEXT NOT NULL,
                start_value INTEGER NOT NULL, end_value INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS container_contracts (
                contract_id TEXT PRIMARY KEY, contract_kind TEXT NOT NULL, payload_schema_id TEXT NOT NULL,
                family TEXT, profile_id TEXT, profile_version TEXT, contract_version TEXT NOT NULL,
                subject_kind TEXT, subject_id TEXT, payload_json TEXT NOT NULL, payload_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS container_artifacts (
                artifact_id TEXT PRIMARY KEY, artifact_kind TEXT NOT NULL, artifact_schema_version TEXT NOT NULL,
                container_graph_revision_id TEXT, container_graph_revision_ids_json TEXT NOT NULL DEFAULT '[]',
                families_json TEXT NOT NULL DEFAULT '[]', profile_ids_json TEXT NOT NULL DEFAULT '[]',
                query_id TEXT, profile_id TEXT, subject_type TEXT, subject_id TEXT,
                payload_json TEXT NOT NULL, payload_fingerprint TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_containers_source_rev_kind
                ON containers(source_id, revision_id, family, kind_ns, kind, status);
            CREATE INDEX IF NOT EXISTS idx_containers_order
                ON containers(source_id, revision_id, order_basis, order_scope, order_scope_id);
            CREATE INDEX IF NOT EXISTS idx_container_refs_fingerprint
                ON container_refs(container_graph_revision_id, ref_role, ref_type, ref_fingerprint, status);
            CREATE INDEX IF NOT EXISTS idx_container_ref_lookup_text
                ON container_ref_lookup(lookup_ns, lookup_key, value_text, status);
            CREATE INDEX IF NOT EXISTS idx_container_render_refs_container
                ON container_render_refs(container_id, render_kind, render_mode, status);
            """
        )

    def _ensure_runtime_coordination_schema(self, conn) -> None:
        """Add durable coordination fields without rewriting existing SQLite DBs."""

        if self._table_exists(conn, "write_log"):
            write_log_cols = {col["name"] for col in self._table_info(conn, "write_log")}
            if "lease_owner" not in write_log_cols:
                conn.execute("ALTER TABLE write_log ADD COLUMN lease_owner TEXT")
            if "lease_expires_at_ms" not in write_log_cols:
                conn.execute("ALTER TABLE write_log ADD COLUMN lease_expires_at_ms INTEGER")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_wl_claim
                ON write_log(extraction_state, lease_expires_at_ms, sort_order)
                """
            )

        if not self._table_exists(conn, "index_status"):
            conn.execute(
                """
                CREATE TABLE index_status (
                    name TEXT PRIMARY KEY,
                    index_dirty INTEGER NOT NULL DEFAULT 0,
                    index_dirty_since_ms INTEGER,
                    last_index_dirty_ms INTEGER,
                    next_index_build_after_ms INTEGER,
                    next_index_retry_after_ms INTEGER,
                    index_build_lease_owner TEXT,
                    index_build_lease_expires_at_ms INTEGER,
                    last_index_build_started_ms INTEGER,
                    last_index_build_completed_ms INTEGER,
                    last_index_build_error TEXT,
                    last_index_build_error_count INTEGER NOT NULL DEFAULT 0,
                    dirty_after_build INTEGER NOT NULL DEFAULT 0,
                    snapshot_fingerprint TEXT,
                    updated_at_ms INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            if not self._has_user_tables(conn):
                conn.executescript(self._schema_script())
                self._ensure_container_graph_tables(conn)
                self._ensure_runtime_coordination_schema(conn)
                self._ensure_meta_defaults(conn)
                conn.commit()
                return
            if not self._schema_is_v2(conn):
                self._migrate_schema_to_v2(conn)
            self._ensure_container_graph_tables(conn)
            self._ensure_secret_table(conn)
            self._ensure_runtime_coordination_schema(conn)
            self._ensure_meta_defaults(conn)
            self._migrate_legacy_secret_state(conn)
            self._rebuild_secret_acl_domains(conn)
            conn.commit()

    def _legacy_message_id_for_session(self, raw_session: dict, fallback_idx: int | None = None) -> str:
        message_id = str(raw_session.get("message_id") or "").strip()
        if message_id:
            return message_id
        raw_session_id = str(raw_session.get("raw_session_id") or "").strip()
        if raw_session_id:
            return f"raw:{raw_session_id}"
        source_id = str(raw_session.get("source_id") or "").strip()
        session_num = raw_session.get("session_num")
        if source_id and session_num is not None:
            return f"legacy:{source_id}:{session_num}"
        if session_num is not None:
            return f"legacy:{session_num}"
        content = str(raw_session.get("content") or "")
        session_date = str(raw_session.get("session_date") or "")
        if content or session_date:
            seed = f"{source_id}|{session_num}|{session_date}|{content}"
            digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()
            return f"legacy:auto:{digest[:16]}"
        if fallback_idx is not None:
            return f"legacy:row:{fallback_idx}"
        raise ValueError("raw_session missing stable identity for message_id derivation")

    def _legacy_message_id_for_doc(self, source_id: str) -> str:
        return f"rawdoc:{source_id}"

    def _canonical_session_num(self, raw_session: dict, *, fallback_idx: int | None = None) -> int:
        projection_session_num = raw_session.get("projection_session_num")
        if isinstance(projection_session_num, int) and projection_session_num > 0:
            return projection_session_num
        if (
            isinstance(projection_session_num, str)
            and projection_session_num.isdigit()
            and int(projection_session_num) > 0
        ):
            return int(projection_session_num)
        session_num = raw_session.get("session_num")
        if isinstance(session_num, int) and session_num > 0:
            return session_num
        if isinstance(session_num, str) and session_num.isdigit() and int(session_num) > 0:
            return int(session_num)
        if fallback_idx is not None:
            return fallback_idx + 1
        raise ValueError("raw_session missing canonical session_num")

    def _raw_session_identity(self, raw_session: dict, fallback_idx: int | None = None) -> str:
        raw_session_id = str(raw_session.get("raw_session_id") or "").strip()
        if raw_session_id:
            return raw_session_id
        message_id = str(raw_session.get("message_id") or "").strip()
        if message_id:
            return message_id
        source_id = str(raw_session.get("source_id") or "").strip()
        session_num = raw_session.get("session_num")
        if source_id and session_num is not None:
            return f"legacy:{source_id}:{session_num}"
        if session_num is not None:
            return f"legacy:{session_num}"
        content = str(raw_session.get("content") or "")
        session_date = str(raw_session.get("session_date") or "")
        if content or session_date:
            seed = f"{source_id}|{session_num}|{session_date}|{content}"
            digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()
            return f"legacy:auto:{digest[:16]}"
        if fallback_idx is not None:
            return f"legacy:row:{fallback_idx}"
        raise ValueError("raw_session missing stable identity")

    def _canonical_fact_id(self, fact: dict, *, tier: str, fallback_sort: int) -> str:
        fact_id = str(fact.get("id") or "").strip()
        if fact_id:
            return fact_id
        return f"compat:{tier}:{fallback_sort}"

    def _episode_corpus_layout(self, corpus: dict) -> dict[str, list[str]]:
        layout: dict[str, list[str]] = {}
        if not isinstance(corpus, dict):
            return {"documents": []}
        for group_name, docs in corpus.items():
            if not isinstance(docs, list):
                continue
            doc_ids = [
                str(doc.get("doc_id") or "")
                for doc in docs
                if isinstance(doc, dict) and str(doc.get("doc_id") or "")
            ]
            if doc_ids:
                layout[str(group_name)] = doc_ids
        layout.setdefault("documents", [])
        return layout

    def _sync_rows(
        self,
        conn,
        *,
        select_sql: str,
        desired_rows: list[tuple],
        key_len: int,
        delete_sql: str,
        insert_sql: str,
        select_params: tuple[Any, ...] = (),
    ) -> None:
        existing_rows = [tuple(row) for row in conn.execute(select_sql, select_params).fetchall()]
        existing = {row[:key_len]: row for row in existing_rows}
        desired = {row[:key_len]: row for row in desired_rows}

        delete_keys = [key for key in existing if key not in desired or existing[key] != desired[key]]
        if delete_keys:
            conn.executemany(
                delete_sql,
                [key if isinstance(key, tuple) else (key,) for key in delete_keys],
            )

        changed_rows = [row for row in desired_rows if existing.get(row[:key_len]) != row]
        if changed_rows:
            conn.executemany(insert_sql, changed_rows)

    def _upsert_write_log_snapshot_row(
        self,
        conn,
        *,
        message_id: str,
        session_id: str,
        agent_id: str,
        swarm_id: str,
        visibility: str,
        owner_id: str | None,
        scope: str | None,
        read: list[str] | None,
        write: list[str] | None,
        content_family: str,
        content_text: str,
        metadata: dict | None,
        timestamp_ms: int,
        extraction_state: str,
        sort_order: int,
    ) -> None:
        row = (
            message_id,
            session_id,
            agent_id,
            swarm_id,
            visibility,
            owner_id,
            scope,
            _json_dumps(read or []),
            _json_dumps(write or []),
            content_family,
            content_text,
            _json_dumps(metadata or {}),
            timestamp_ms,
            extraction_state,
            sort_order,
        )
        updated = conn.execute(
            """
            UPDATE write_log
            SET session_id = ?,
                agent_id = ?,
                swarm_id = ?,
                visibility = ?,
                owner_id = ?,
                scope = ?,
                read_json = ?,
                write_json = ?,
                content_family = ?,
                content_text = ?,
                metadata_json = ?,
                timestamp_ms = ?,
                extraction_state = ?,
                sort_order = ?
            WHERE message_id = ?
            """,
            row[1:] + (message_id,),
        )
        if int(updated.rowcount or 0) == 0:
            conn.execute(
                """
                INSERT INTO write_log(
                    message_id, session_id, agent_id, swarm_id, visibility,
                    owner_id, scope, read_json, write_json, content_family,
                    content_text, metadata_json, timestamp_ms, extraction_state,
                    extraction_attempts, last_extraction_attempt_ms, sort_order
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
                """,
                row,
            )

    def _fact_row_payload(
        self,
        fact: dict,
        *,
        tier: str,
        fallback_sort: int,
        fact_id: str | None = None,
    ) -> tuple:
        payload = dict(fact)
        payload.pop("_temporal_links", None)
        canonical_fact_id = fact_id or self._canonical_fact_id(fact, tier=tier, fallback_sort=fallback_sort)
        if canonical_fact_id:
            payload["id"] = canonical_fact_id
        return (
            canonical_fact_id,
            fact.get("kind"),
            fact.get("session") if fact.get("session") is not None else fact.get("session_num"),
            fact.get("source_id"),
            fact.get("agent_id"),
            fact.get("swarm_id"),
            fact.get("scope"),
            fact.get("owner_id"),
            fact.get("status"),
            fact.get("created_at"),
            _sqlite_text_scalar(fact.get("event_date")),
            _json_dumps(payload),
        )

    def _fact_rows_for_tier(self, tier_name: str, facts: list[dict]) -> list[tuple]:
        rows: list[tuple] = []
        seen: set[str] = set()
        for idx, fact in enumerate(facts):
            fact_id = self._canonical_fact_id(fact, tier=tier_name, fallback_sort=idx)
            if fact_id in seen:
                fact_id = f"{fact_id}__{idx}"
            seen.add(fact_id)
            rows.append((tier_name, *self._fact_row_payload(fact, tier=tier_name, fallback_sort=idx, fact_id=fact_id), idx))
        return rows

    def _load_existing_write_log_rows_for_snapshot(self, conn) -> dict[str, dict[str, Any]]:
        return {
            str(row["message_id"]): {
                "session_id": row["session_id"],
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "visibility": row["visibility"],
                "owner_id": row["owner_id"],
                "scope": row["scope"],
                "read_json": row["read_json"],
                "write_json": row["write_json"],
                "content_family": row["content_family"],
                "content_text": row["content_text"],
                "metadata_json": row["metadata_json"],
                "timestamp_ms": int(row["timestamp_ms"] or 0),
                "extraction_state": row["extraction_state"],
                "sort_order": int(row["sort_order"] or 0),
            }
            for row in conn.execute(
                """
                SELECT message_id, session_id, agent_id, swarm_id, visibility, owner_id, scope,
                       read_json, write_json, content_family, content_text, metadata_json,
                       timestamp_ms, extraction_state, sort_order
                FROM write_log
                """
            ).fetchall()
        }

    def _sync_write_log_from_snapshot(
        self,
        conn,
        *,
        raw_sessions: list[dict],
        raw_docs: dict[str, str],
        source_records_by_id: dict[str, dict],
    ) -> set[str]:
        """Synchronize write_log rows from the persisted snapshot state."""
        existing_write_log = self._load_existing_write_log_rows_for_snapshot(conn)
        next_write_log_sort = max((row["sort_order"] for row in existing_write_log.values()), default=-1) + 1
        desired_write_log_ids: set[str] = set()

        for idx, raw_session in enumerate(raw_sessions):
            message_id = self._legacy_message_id_for_session(raw_session, fallback_idx=idx)
            desired_write_log_ids.add(message_id)
            session_num = raw_session.get("session_num")
            session_id = str(
                raw_session.get("session_key")
                or raw_session.get("raw_session_id")
                or raw_session.get("message_id")
                or f"session:{session_num}"
            )
            extraction_state = "pending" if raw_session.get("status") == "pending_extraction" else "complete"
            visibility = "private" if raw_session.get("scope") == "agent-private" else "shared"
            ts = _timestamp_ms(raw_session.get("stored_at") or raw_session.get("session_date"), idx)
            existing = existing_write_log.get(message_id)
            sort_order = existing["sort_order"] if existing is not None else next_write_log_sort
            if existing is None:
                next_write_log_sort += 1
            desired_tuple = (
                session_id,
                str(raw_session.get("agent_id") or "default"),
                str(raw_session.get("swarm_id") or "default"),
                visibility,
                raw_session.get("owner_id"),
                raw_session.get("scope"),
                _json_dumps(raw_session.get("read") or []),
                _json_dumps(raw_session.get("write") or []),
                str(raw_session.get("format") or "conversation"),
                str(raw_session.get("content") or ""),
                _json_dumps(raw_session.get("metadata") or {}),
                ts,
                extraction_state,
                sort_order,
            )
            existing_tuple = None
            if existing is not None:
                existing_tuple = (
                    existing["session_id"],
                    existing["agent_id"],
                    existing["swarm_id"],
                    existing["visibility"],
                    existing["owner_id"],
                    existing["scope"],
                    existing["read_json"],
                    existing["write_json"],
                    existing["content_family"],
                    existing["content_text"],
                    existing["metadata_json"],
                    existing["timestamp_ms"],
                    existing["extraction_state"],
                    existing["sort_order"],
                )
            if existing_tuple != desired_tuple:
                self._upsert_write_log_snapshot_row(
                    conn,
                    message_id=message_id,
                    session_id=session_id,
                    agent_id=str(raw_session.get("agent_id") or "default"),
                    swarm_id=str(raw_session.get("swarm_id") or "default"),
                    visibility=visibility,
                    owner_id=raw_session.get("owner_id"),
                    scope=raw_session.get("scope"),
                    read=list(raw_session.get("read") or []),
                    write=list(raw_session.get("write") or []),
                    content_family=str(raw_session.get("format") or "conversation"),
                    content_text=str(raw_session.get("content") or ""),
                    metadata=raw_session.get("metadata") or {},
                    timestamp_ms=ts,
                    extraction_state=extraction_state,
                    sort_order=sort_order,
                )

        for idx, (source_id, raw_text) in enumerate(raw_docs.items()):
            record = source_records_by_id.get(source_id) or {}
            message_id = self._legacy_message_id_for_doc(str(source_id))
            desired_write_log_ids.add(message_id)
            existing = existing_write_log.get(message_id)
            sort_order = existing["sort_order"] if existing is not None else next_write_log_sort
            if existing is None:
                next_write_log_sort += 1
            desired_tuple = (
                f"doc:{source_id}",
                str(record.get("owner_id") or "default"),
                "default",
                "shared",
                record.get("owner_id"),
                None,
                _json_dumps(record.get("read") or []),
                _json_dumps(record.get("write") or []),
                str(record.get("family") or "document"),
                str(raw_text or ""),
                _json_dumps(record.get("metadata") or {}),
                1_000_000 + idx,
                "complete",
                sort_order,
            )
            existing_tuple = None
            if existing is not None:
                existing_tuple = (
                    existing["session_id"],
                    existing["agent_id"],
                    existing["swarm_id"],
                    existing["visibility"],
                    existing["owner_id"],
                    existing["scope"],
                    existing["read_json"],
                    existing["write_json"],
                    existing["content_family"],
                    existing["content_text"],
                    existing["metadata_json"],
                    existing["timestamp_ms"],
                    existing["extraction_state"],
                    existing["sort_order"],
                )
            if existing_tuple != desired_tuple:
                self._upsert_write_log_snapshot_row(
                    conn,
                    message_id=message_id,
                    session_id=f"doc:{source_id}",
                    agent_id=str(record.get("owner_id") or "default"),
                    swarm_id="default",
                    visibility="shared",
                    owner_id=record.get("owner_id"),
                    scope=None,
                    read=record.get("read") or [],
                    write=record.get("write") or [],
                    content_family=str(record.get("family") or "document"),
                    content_text=str(raw_text or ""),
                    metadata=record.get("metadata") or {},
                    timestamp_ms=1_000_000 + idx,
                    extraction_state="complete",
                    sort_order=sort_order,
                )

        return desired_write_log_ids

    def _prune_write_log_from_snapshot(self, conn, *, desired_write_log_ids: set[str]) -> None:
        if desired_write_log_ids:
            placeholders = ",".join("?" for _ in desired_write_log_ids)
            # Placeholder count is derived internally; values stay parameterized.
            delete_sql = f"DELETE FROM write_log WHERE extraction_state = 'complete' AND message_id NOT IN ({placeholders})"  # noqa: S608  # nosec B608
            conn.execute(
                delete_sql,
                tuple(desired_write_log_ids),
            )
        else:
            conn.execute("DELETE FROM write_log WHERE extraction_state = 'complete'")

    def _sync_raw_session_projection_from_snapshot(self, conn, *, raw_sessions: list[dict]) -> None:
        def _status_rank(status: Any) -> int:
            normalized = str(status or "active")
            if normalized == "active":
                return 3
            if normalized == "superseded":
                return 2
            if normalized == "retracted":
                return 1
            return 0

        selected_rows: dict[int, tuple[tuple, int, int]] = {}
        for idx, raw_session in enumerate(raw_sessions):
            session_num = self._canonical_session_num(raw_session, fallback_idx=idx)
            message_id = self._legacy_message_id_for_session(raw_session, fallback_idx=idx)
            source_id = str(raw_session.get("source_id") or self._key)
            source_meta = {
                key: value
                for key, value in raw_session.items()
                if key not in {
                    "message_id", "raw_session_id", "session_num", "session_date", "content", "speakers",
                    "agent_id", "swarm_id", "scope", "owner_id", "read", "write",
                    "stored_at", "format", "source_id", "artifact_id",
                    "version_id", "content_hash", "status", "metadata",
                    "target",
                }
            }
            row = (
                session_num,
                message_id,
                self._raw_session_identity(raw_session, fallback_idx=idx),
                source_id,
                raw_session.get("format"),
                raw_session.get("session_date"),
                raw_session.get("speakers"),
                raw_session.get("stored_at"),
                raw_session.get("artifact_id"),
                raw_session.get("version_id"),
                raw_session.get("content_hash"),
                raw_session.get("owner_id"),
                raw_session.get("scope"),
                raw_session.get("agent_id"),
                raw_session.get("swarm_id"),
                _json_dumps(raw_session.get("read") or []),
                _json_dumps(raw_session.get("write") or []),
                _json_dumps(raw_session.get("target") or []),
                _json_dumps(raw_session.get("metadata") or {}),
                _json_dumps(source_meta),
                raw_session.get("status"),
                idx,
            )
            existing = selected_rows.get(session_num)
            candidate = (row, _status_rank(raw_session.get("status")), idx)
            if existing is None or candidate[1] > existing[1] or (candidate[1] == existing[1] and idx >= existing[2]):
                selected_rows[session_num] = candidate

        raw_session_rows = [
            row[:-1] + (position,)
            for position, (_session_num, (row, _rank, _idx)) in enumerate(sorted(selected_rows.items(), key=lambda item: item[0]))
        ]
        self._sync_rows(
            conn,
            select_sql=(
                "SELECT session_num, message_id, raw_session_id, source_id, format, session_date, "
                "speakers, stored_at, artifact_id, version_id, content_hash, owner_id, scope, "
                "agent_id, swarm_id, read_json, write_json, target_json, metadata_json, "
                "source_meta_json, status, sort_order FROM raw_sessions ORDER BY session_num"
            ),
            desired_rows=raw_session_rows,
            key_len=1,
            delete_sql="DELETE FROM raw_sessions WHERE session_num = ?",
            insert_sql=(
                "INSERT INTO raw_sessions("
                "session_num, message_id, raw_session_id, source_id, format, session_date, "
                "speakers, stored_at, artifact_id, version_id, content_hash, owner_id, scope, "
                "agent_id, swarm_id, read_json, write_json, target_json, metadata_json, "
                "source_meta_json, status, sort_order"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
        )

    def _sync_raw_doc_projection_from_snapshot(
        self,
        conn,
        *,
        raw_docs: dict[str, str],
        source_records_by_id: dict[str, dict],
    ) -> None:
        raw_doc_rows = [
            (
                str(source_id),
                self._legacy_message_id_for_doc(str(source_id)),
                _json_dumps((source_records_by_id.get(source_id) or {}).get("metadata") or {}),
                idx,
            )
            for idx, (source_id, _raw_text) in enumerate(raw_docs.items())
        ]
        self._sync_rows(
            conn,
            select_sql="SELECT source_id, message_id, metadata_json, sort_order FROM raw_docs ORDER BY sort_order",
            desired_rows=raw_doc_rows,
            key_len=1,
            delete_sql="DELETE FROM raw_docs WHERE source_id = ?",
            insert_sql="INSERT INTO raw_docs(source_id, message_id, metadata_json, sort_order) VALUES(?, ?, ?, ?)",
        )

    def _sync_fact_projection_from_snapshot(
        self,
        conn,
        *,
        granular: list[dict],
        cons: list[dict],
        cross: list[dict],
    ) -> None:
        for tier_name, tier_facts in (("granular", granular), ("cons", cons), ("cross", cross)):
            fact_rows = self._fact_rows_for_tier(tier_name, tier_facts)
            self._sync_rows(
                conn,
                select_sql=(
                    "SELECT tier, fact_id, kind, session_num, source_id, "
                    "agent_id, swarm_id, scope, owner_id, status, created_at, event_date, payload_json "
                    ", sort_order FROM facts WHERE tier = ? ORDER BY sort_order"
                ),
                select_params=(tier_name,),
                desired_rows=fact_rows,
                key_len=2,
                delete_sql="DELETE FROM facts WHERE tier = ? AND fact_id = ?",
                insert_sql=(
                    "INSERT INTO facts("
                    "tier, fact_id, kind, session_num, source_id, agent_id, swarm_id, "
                    "scope, owner_id, status, created_at, event_date, payload_json"
                    ", sort_order) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                ),
            )

    def _sync_temporal_link_projection_from_snapshot(self, conn, *, temporal_links: list[dict]) -> None:
        temporal_rows = [
            (idx, _json_dumps(link))
            for idx, link in enumerate(temporal_links)
        ]
        self._sync_rows(
            conn,
            select_sql="SELECT sort_order, link_json FROM temporal_links ORDER BY sort_order",
            desired_rows=temporal_rows,
            key_len=1,
            delete_sql="DELETE FROM temporal_links WHERE sort_order = ?",
            insert_sql="INSERT INTO temporal_links(sort_order, link_json) VALUES(?, ?)",
        )

    def _sync_episode_corpus_projection_from_snapshot(
        self,
        conn,
        *,
        episode_corpus: dict,
    ) -> tuple[list[str], dict[str, list[str]]]:
        corpus_layout = self._episode_corpus_layout(episode_corpus)
        doc_order = [doc_id for doc_ids in corpus_layout.values() for doc_id in doc_ids]
        episode_rows = []
        for docs in (
            docs
            for docs in episode_corpus.values()
            if isinstance(episode_corpus, dict) and isinstance(docs, list)
        ):
            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                doc_id = str(doc.get("doc_id") or "")
                if not doc_id:
                    continue
                for idx, episode in enumerate(doc.get("episodes", []) or []):
                    episode_rows.append((doc_id, str(episode.get("episode_id") or ""), idx, _json_dumps(episode)))
        self._sync_rows(
            conn,
            select_sql="SELECT doc_id, episode_id, sort_order, episode_json FROM episode_corpus ORDER BY doc_id, sort_order",
            desired_rows=episode_rows,
            key_len=2,
            delete_sql="DELETE FROM episode_corpus WHERE doc_id = ? AND episode_id = ?",
            insert_sql="INSERT INTO episode_corpus(doc_id, episode_id, sort_order, episode_json) VALUES(?, ?, ?, ?)",
        )
        return doc_order, corpus_layout

    def _sync_source_record_projection_from_snapshot(self, conn, *, source_records_by_id: dict[str, dict]) -> None:
        source_rows = [
            (
                str(source_id),
                record.get("family") or "unknown",
                record.get("owner_id"),
                _json_dumps(record.get("read") or []),
                _json_dumps(record.get("write") or []),
                record.get("artifact_id"),
                record.get("version_id"),
                record.get("content_hash"),
                _json_dumps(record.get("metadata") or {}),
                _json_dumps(record.get("target") or []),
                _json_dumps(_source_record_meta_payload(record)),
                record.get("created_at"),
                record.get("updated_at"),
            )
            for source_id, record in source_records_by_id.items()
        ]
        self._sync_rows(
            conn,
            select_sql=(
                "SELECT source_id, family, owner_id, read_json, write_json, artifact_id, version_id, "
                "content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at "
                "FROM source_records ORDER BY source_id"
            ),
            desired_rows=source_rows,
            key_len=1,
            delete_sql="DELETE FROM source_records WHERE source_id = ?",
            insert_sql=(
                "INSERT INTO source_records("
                "source_id, family, owner_id, read_json, write_json, artifact_id, version_id, "
                "content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
        )

    def _container_graph_db_row(self, conn, table_name: str, row: dict[str, Any]) -> tuple[Any, ...]:
        json_columns = CONTAINER_GRAPH_TABLE_SPECS[table_name]["json"]
        values = []
        for column in self._container_graph_columns(conn, table_name):
            value = row.get(column)
            if column in json_columns:
                value = _json_dumps(value if value is not None else ([] if column.endswith("_ids_json") else {}))
            values.append(value)
        return tuple(values)

    def _container_graph_columns(self, conn, table_name: str) -> list[str]:
        return [str(col["name"]) for col in self._table_info(conn, table_name)]

    def _sync_container_graph_table_from_snapshot(self, conn, table_name: str, rows: list[dict]) -> None:
        id_column = str(CONTAINER_GRAPH_TABLE_SPECS[table_name]["id"])
        columns = self._container_graph_columns(conn, table_name)
        desired_rows = [
            self._container_graph_db_row(conn, table_name, row)
            for row in rows
            if str(row.get(id_column) or "")
        ]
        column_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        self._sync_rows(
            conn,
            select_sql=f"SELECT {column_sql} FROM {table_name} ORDER BY {id_column}",  # noqa: S608  # nosec B608
            desired_rows=desired_rows,
            key_len=1,
            delete_sql=f"DELETE FROM {table_name} WHERE {id_column} = ?",  # noqa: S608  # nosec B608
            insert_sql=f"INSERT INTO {table_name}({column_sql}) VALUES({placeholders})",  # nosec B608
        )

    def _sync_container_graph_projection_from_snapshot(self, conn, *, container_graph: dict) -> None:
        self._ensure_container_graph_tables(conn)
        graph = container_graph if isinstance(container_graph, dict) else {}
        for snapshot_key, table_name in CONTAINER_GRAPH_SNAPSHOT_KEYS.items():
            self._sync_container_graph_table_from_snapshot(
                conn,
                table_name,
                graph.get(snapshot_key) or [],
            )

    def _load_container_graph_table(self, conn, table_name: str) -> list[dict]:
        if not self._table_exists(conn, table_name):
            return []
        json_columns = CONTAINER_GRAPH_TABLE_SPECS[table_name]["json"]
        id_column = str(CONTAINER_GRAPH_TABLE_SPECS[table_name]["id"])
        rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY {id_column}").fetchall()  # noqa: S608  # nosec B608
        loaded = []
        for row in rows:
            item = dict(row)
            for column in json_columns:
                if column in item:
                    item[column] = _json_loads(item[column], [] if column.endswith("_ids_json") else {})
            loaded.append(item)
        return loaded

    def _load_container_graph(self, conn) -> dict:
        return {
            snapshot_key: self._load_container_graph_table(conn, table_name)
            for snapshot_key, table_name in CONTAINER_GRAPH_SNAPSHOT_KEYS.items()
        }

    def _upsert_container_graph_rows(self, conn, table_name: str, rows: list[dict]) -> None:
        if not rows:
            return
        id_column = str(CONTAINER_GRAPH_TABLE_SPECS[table_name]["id"])
        columns = self._container_graph_columns(conn, table_name)
        non_id_columns = [column for column in columns if column != id_column]
        update_sql = ", ".join(f"{column} = ?" for column in non_id_columns)
        insert_columns_sql = ", ".join(columns)
        insert_placeholders_sql = ", ".join("?" for _ in columns)
        for row in rows:
            db_row = self._container_graph_db_row(conn, table_name, row)
            row_by_column = dict(zip(columns, db_row, strict=False))
            updated = conn.execute(
                f"UPDATE {table_name} SET {update_sql} WHERE {id_column} = ?",  # noqa: S608  # nosec B608
                tuple(row_by_column[column] for column in non_id_columns) + (row_by_column[id_column],),
            )
            if int(updated.rowcount or 0) == 0:
                conn.execute(
                    f"INSERT INTO {table_name}({insert_columns_sql}) VALUES({insert_placeholders_sql})",  # nosec B608
                    db_row,
                )

    def _sync_legacy_secret_state_from_snapshot(self, conn, *, data: dict) -> None:
        if "secrets" not in data:
            return
        rows = self._normalize_legacy_secret_rows(data.get("secrets"))
        if rows:
            self._require_secret_storage_allowed()
        for row in rows:
            self._insert_secret_row(conn, row)
        conn.execute("DELETE FROM state_json WHERE name = ?", ("secrets",))

    def _sync_state_json_projection_from_snapshot(
        self,
        conn,
        *,
        data: dict,
        doc_order: list[str],
        corpus_layout: dict[str, list[str]],
    ) -> None:
        state_values = {name: data.get(name) for name in STATE_JSON_KEYS if name in data}
        state_values[INTERNAL_EPISODE_DOC_ORDER] = doc_order
        state_values[INTERNAL_EPISODE_CORPUS_LAYOUT] = corpus_layout
        state_rows = [(name, _json_dumps(value)) for name, value in state_values.items()]
        self._sync_rows(
            conn,
            select_sql="SELECT name, value_json FROM state_json ORDER BY name",
            desired_rows=state_rows,
            key_len=1,
            delete_sql="DELETE FROM state_json WHERE name = ?",
            insert_sql="INSERT INTO state_json(name, value_json) VALUES(?, ?)",
        )

    # ── Snapshot/projection persistence operations ──

    def save_facts(self, data: dict) -> None:
        """Persist the current snapshot/projection view.

        This persists the current snapshot view. Live ingress write paths still
        establish new truth through append_write_log() and projection deltas.
        """
        granular = data.get("granular", []) or []
        cons = data.get("cons", []) or []
        cross = data.get("cross", []) or []
        raw_sessions = data.get("raw_sessions", []) or []
        raw_docs = data.get("raw_docs", {}) or {}
        episode_corpus = data.get("episode_corpus", {"documents": []}) or {"documents": []}
        temporal_links = data.get("tlinks", []) or []
        source_records = data.get("source_records", {}) or {}
        source_records_by_id = source_records if isinstance(source_records, dict) else {}
        container_graph = data.get("container_graph") or {}

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._sync_legacy_secret_state_from_snapshot(conn, data=data)
            desired_write_log_ids = self._sync_write_log_from_snapshot(
                conn,
                raw_sessions=raw_sessions,
                raw_docs=raw_docs,
                source_records_by_id=source_records_by_id,
            )
            self._sync_raw_session_projection_from_snapshot(conn, raw_sessions=raw_sessions)
            self._sync_raw_doc_projection_from_snapshot(
                conn,
                raw_docs=raw_docs,
                source_records_by_id=source_records_by_id,
            )
            self._sync_fact_projection_from_snapshot(
                conn,
                granular=granular,
                cons=cons,
                cross=cross,
            )
            self._sync_temporal_link_projection_from_snapshot(conn, temporal_links=temporal_links)
            doc_order, corpus_layout = self._sync_episode_corpus_projection_from_snapshot(
                conn,
                episode_corpus=episode_corpus,
            )
            self._sync_source_record_projection_from_snapshot(conn, source_records_by_id=source_records_by_id)
            self._sync_container_graph_projection_from_snapshot(conn, container_graph=container_graph)
            self._sync_state_json_projection_from_snapshot(
                conn,
                data=data,
                doc_order=doc_order,
                corpus_layout=corpus_layout,
            )
            self._prune_write_log_from_snapshot(conn, desired_write_log_ids=desired_write_log_ids)
            conn.commit()

    def _load_tier(self, conn, tier: str) -> list[dict]:
        rows = conn.execute(
            "SELECT payload_json FROM facts WHERE tier = ? ORDER BY sort_order",
            (tier,),
        ).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def _load_raw_sessions(self, conn) -> list[dict]:
        rows = conn.execute(
            """
            SELECT rs.*, wl.content_text
            FROM raw_sessions rs
            JOIN write_log wl ON wl.message_id = rs.message_id
            ORDER BY rs.session_num, rs.sort_order
            """
        ).fetchall()
        sessions: list[dict] = []
        for row in rows:
            message_id = str(row["message_id"] or "")
            raw_session_id = message_id.removeprefix("raw:") if message_id.startswith("raw:") else message_id
            source_meta = _json_loads(row["source_meta_json"], {})
            content_text = str(row["content_text"] or "")
            session = {
                "message_id": message_id,
                "raw_session_id": row["raw_session_id"] or raw_session_id,
                "session_num": source_meta.get("logical_session_num", row["session_num"]),
                "session_date": row["session_date"],
                "content": content_text,
                "speakers": row["speakers"],
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "scope": row["scope"],
                "owner_id": row["owner_id"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "stored_at": row["stored_at"],
                "format": row["format"],
                "source_id": row["source_id"],
                "artifact_id": row["artifact_id"],
                "version_id": row["version_id"],
                "content_hash": row["content_hash"],
                "status": row["status"],
            }
            logical_source_id = str(source_meta.get("logical_source_id") or "").strip()
            if logical_source_id:
                session["logical_source_id"] = logical_source_id
            if source_meta.get("logical_session_num") is not None:
                session["projection_session_num"] = row["session_num"]
            metadata = _json_loads(row["metadata_json"], {})
            if metadata:
                session["metadata"] = metadata
            target = _json_loads(row["target_json"], [])
            if target:
                session["target"] = target
            if source_meta:
                session.update(source_meta)
            sessions.append(session)
        return sessions

    def _load_raw_docs(self, conn) -> dict[str, str]:
        rows = conn.execute(
            """
            SELECT rd.source_id, wl.content_text
            FROM raw_docs rd
            JOIN write_log wl ON wl.message_id = rd.message_id
            ORDER BY rd.sort_order
            """
        ).fetchall()
        raw_docs: dict[str, str] = {}
        for row in rows:
            source_id = str(row["source_id"] or "")
            raw_docs[source_id] = str(row["content_text"] or "")
        return raw_docs

    def backfill_raw_doc_content(
        self,
        *,
        source_id: str,
        content_text: str,
        metadata: dict | None = None,
        message_id: str | None = None,
    ) -> dict:
        """Write trusted original document raw text without rerunning extraction."""

        source_id = str(source_id or "").strip()
        if not source_id:
            raise ValueError("source_id is required")
        content_text = str(content_text or "")
        if not content_text:
            raise ValueError("content_text is required")
        message_id = str(message_id or self._legacy_message_id_for_doc(source_id))
        metadata_json = _json_dumps(metadata or {})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            raw_row = conn.execute(
                "SELECT message_id, sort_order FROM raw_docs WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            if raw_row is not None:
                message_id = str(raw_row["message_id"] or message_id)
                raw_doc_sort_order = int(raw_row["sort_order"])
            else:
                raw_doc_sort_order = self._next_sort_order(conn, "raw_docs")
            write_row = conn.execute(
                """
                SELECT session_id, agent_id, swarm_id, visibility, owner_id, scope, read_json,
                       write_json, content_family, metadata_json, timestamp_ms, extraction_state,
                       sort_order
                FROM write_log WHERE message_id = ?
                """,
                (message_id,),
            ).fetchone()
            write_existing = dict(write_row) if write_row is not None else {}
            write_sort_order = int(write_existing["sort_order"]) if write_existing else self._next_sort_order(conn, "write_log")
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            merged_metadata = _json_loads(write_existing.get("metadata_json"), {}) if write_existing else {}
            merged_metadata.update(metadata or {})
            self._upsert_write_log_snapshot_row(
                conn,
                message_id=message_id,
                session_id=str(write_existing.get("session_id") or f"doc:{source_id}"),
                agent_id=str(write_existing.get("agent_id") or "system"),
                swarm_id=str(write_existing.get("swarm_id") or "default"),
                visibility=str(write_existing.get("visibility") or "shared"),
                owner_id=write_existing.get("owner_id"),
                scope=write_existing.get("scope"),
                read=_json_loads(write_existing.get("read_json"), []) if write_existing else [],
                write=_json_loads(write_existing.get("write_json"), []) if write_existing else [],
                content_family=str(write_existing.get("content_family") or "document"),
                content_text=content_text,
                metadata=merged_metadata,
                timestamp_ms=int(write_existing.get("timestamp_ms") or now_ms),
                extraction_state=str(write_existing.get("extraction_state") or "complete"),
                sort_order=write_sort_order,
            )
            existing = conn.execute(
                "UPDATE raw_docs SET message_id = ?, metadata_json = ?, sort_order = ? WHERE source_id = ?",
                (message_id, metadata_json, raw_doc_sort_order, source_id),
            )
            if int(existing.rowcount or 0) == 0:
                conn.execute(
                    "INSERT INTO raw_docs(source_id, message_id, metadata_json, sort_order) VALUES(?, ?, ?, ?)",
                    (source_id, message_id, metadata_json, raw_doc_sort_order),
                )
            conn.commit()
        return {"source_id": source_id, "message_id": message_id, "content_hash": hashlib.sha256(content_text.encode("utf-8")).hexdigest()}

    def validate_raw_doc_projection(self) -> list[dict]:
        """Validate raw_docs rows resolve to non-empty original write_log content."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rd.source_id, rd.message_id, wl.content_text
                FROM raw_docs rd
                LEFT JOIN write_log wl ON wl.message_id = rd.message_id
                ORDER BY rd.sort_order
                """
            ).fetchall()
        errors: list[dict] = []
        for row in rows:
            if not str(row["content_text"] or ""):
                errors.append(
                    {
                        "code": "RAW_SOURCE_MISSING",
                        "source_id": str(row["source_id"] or ""),
                        "message_id": str(row["message_id"] or ""),
                    }
                )
        return errors

    def _load_episode_corpus(self, conn) -> dict:
        layout_row = conn.execute(
            "SELECT value_json FROM state_json WHERE name = ?",
            (INTERNAL_EPISODE_CORPUS_LAYOUT,),
        ).fetchone()
        corpus_layout = _json_loads(layout_row["value_json"] if layout_row else None, {})
        doc_order_row = conn.execute(
            "SELECT value_json FROM state_json WHERE name = ?",
            (INTERNAL_EPISODE_DOC_ORDER,),
        ).fetchone()
        doc_order = _json_loads(doc_order_row["value_json"] if doc_order_row else None, [])
        rows = conn.execute(
            "SELECT doc_id, episode_json FROM episode_corpus ORDER BY doc_id, sort_order"
        ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row["doc_id"]), []).append(_json_loads(row["episode_json"], {}))
        if not isinstance(corpus_layout, dict) or not corpus_layout:
            ordered_doc_ids = [doc_id for doc_id in doc_order if doc_id in grouped]
            ordered_doc_ids.extend(doc_id for doc_id in grouped if doc_id not in ordered_doc_ids)
            return {
                "documents": [
                    {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
                    for doc_id in ordered_doc_ids
                ]
            }

        seen_doc_ids: list[str] = []
        corpus: dict[str, list[dict]] = {}
        for group_name, group_doc_ids in corpus_layout.items():
            if not isinstance(group_doc_ids, list):
                continue
            ordered_group_doc_ids = [doc_id for doc_id in group_doc_ids if doc_id in grouped]
            if group_name == "documents":
                ordered_group_doc_ids.extend(
                    doc_id
                    for doc_id in doc_order
                    if doc_id in grouped and doc_id not in ordered_group_doc_ids and doc_id not in seen_doc_ids
                )
            if not ordered_group_doc_ids:
                corpus[str(group_name)] = []
                continue
            seen_doc_ids.extend(doc_id for doc_id in ordered_group_doc_ids if doc_id not in seen_doc_ids)
            corpus[str(group_name)] = [
                {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
                for doc_id in ordered_group_doc_ids
            ]

        remaining_doc_ids = [doc_id for doc_id in doc_order if doc_id in grouped and doc_id not in seen_doc_ids]
        remaining_doc_ids.extend(doc_id for doc_id in grouped if doc_id not in seen_doc_ids and doc_id not in remaining_doc_ids)
        documents = corpus.setdefault("documents", [])
        documents.extend(
            {"doc_id": doc_id, "episodes": grouped.get(doc_id, [])}
            for doc_id in remaining_doc_ids
        )
        return corpus

    def _logical_source_id_from_record(self, source_id: str, record: dict | None) -> str:
        record = record or {}
        source_meta = record.get("source_meta") or {}
        logical = str(source_meta.get("logical_source_id") or "").strip()
        return logical or str(source_id or "").strip()

    def _compat_projection_map(self, payload: dict) -> dict[str, str]:
        projection_to_logical: dict[str, str] = {}
        for source_id, record in (payload.get("source_records") or {}).items():
            projection_to_logical[str(source_id)] = self._logical_source_id_from_record(str(source_id), record)
        for raw_session in (payload.get("raw_sessions") or []):
            projection_source_id = str(raw_session.get("source_id") or "").strip()
            logical_source_id = str(raw_session.get("logical_source_id") or projection_source_id).strip()
            if projection_source_id:
                projection_to_logical[projection_source_id] = logical_source_id or projection_source_id
        return projection_to_logical

    @staticmethod
    def _logicalize_doc_id(doc_id: str, projection_to_logical: dict[str, str]) -> str:
        prefix, sep, suffix = str(doc_id or "").partition(":")
        if not sep:
            return projection_to_logical.get(str(doc_id or ""), str(doc_id or ""))
        return f"{prefix}:{projection_to_logical.get(suffix, suffix)}"

    @staticmethod
    def _episode_source_key(source_id: str) -> str:
        key = "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in str(source_id or "")
        )
        key = key.strip("._-")
        return key or "source"

    def _logicalize_fact_id(
        self,
        fact_id: str,
        *,
        projection_source_id: str,
        logical_source_id: str,
    ) -> str:
        fact_id = str(fact_id or "").strip()
        if not fact_id:
            return fact_id
        projection_source_id = str(projection_source_id or "").strip()
        logical_source_id = str(logical_source_id or projection_source_id).strip()
        if not projection_source_id or projection_source_id == logical_source_id:
            return fact_id
        projection_key = self._episode_source_key(projection_source_id)
        logical_key = self._episode_source_key(logical_source_id)
        if fact_id == projection_source_id:
            return logical_source_id
        if fact_id.startswith(f"{projection_source_id}_"):
            return f"{logical_source_id}{fact_id[len(projection_source_id):]}"
        raw_substrate_prefix = f"substrate_{projection_source_id}"
        if fact_id == raw_substrate_prefix:
            return f"substrate_{logical_source_id}"
        if fact_id.startswith(f"{raw_substrate_prefix}_"):
            return f"substrate_{logical_source_id}_{fact_id[len(raw_substrate_prefix) + 1:]}"
        substrate_prefix = f"substrate_{projection_key}"
        if fact_id == substrate_prefix:
            return f"substrate_{logical_key}"
        if fact_id.startswith(f"{substrate_prefix}_"):
            return f"substrate_{logical_key}_{fact_id[len(substrate_prefix) + 1:]}"
        if fact_id.startswith(f"{projection_key}_"):
            return f"{logical_key}_{fact_id[len(projection_key) + 1:]}"
        return fact_id

    def _logicalize_episode_id(
        self,
        episode_id: str,
        *,
        projection_source_id: str,
        logical_source_id: str,
    ) -> str:
        episode_id = str(episode_id or "").strip()
        if not episode_id:
            return episode_id
        projection_source_id = str(projection_source_id or "").strip()
        logical_source_id = str(logical_source_id or projection_source_id).strip()
        if not projection_source_id or projection_source_id == logical_source_id:
            return episode_id
        if episode_id == projection_source_id:
            return logical_source_id
        if episode_id.startswith(f"{projection_source_id}_"):
            return f"{logical_source_id}{episode_id[len(projection_source_id):]}"
        projection_key = self._episode_source_key(projection_source_id)
        logical_key = self._episode_source_key(logical_source_id)
        if episode_id == projection_key:
            return logical_key
        if episode_id.startswith(f"{projection_key}_"):
            return f"{logical_key}{episode_id[len(projection_key):]}"
        return episode_id

    def _compat_load_payload(self, payload: dict) -> dict:
        projection_to_logical = self._compat_projection_map(payload)
        compat = dict(payload)

        fact_id_map: dict[tuple[str, str], str] = {}
        fact_ref_map: dict[str, str] = {}
        seen_fact_ids: set[str] = set()
        for tier_name in ("granular", "cons", "cross"):
            for fact in payload.get(tier_name) or []:
                original_id = str(fact.get("id") or "").strip()
                if not original_id:
                    continue
                projection_source_id = str(fact.get("source_id") or "").strip()
                logical_source_id = projection_to_logical.get(projection_source_id, projection_source_id)
                candidate_id = self._logicalize_fact_id(
                    original_id,
                    projection_source_id=projection_source_id,
                    logical_source_id=logical_source_id,
                )
                unique_id = candidate_id
                suffix = 2
                while unique_id in seen_fact_ids:
                    unique_id = f"{candidate_id}__compat{suffix}"
                    suffix += 1
                seen_fact_ids.add(unique_id)
                fact_id_map[(tier_name, original_id)] = unique_id
                fact_ref_map.setdefault(original_id, unique_id)

        compat_sessions: list[dict] = []
        for raw_session in payload.get("raw_sessions") or []:
            item = dict(raw_session)
            projection_source_id = str(item.get("source_id") or "").strip()
            logical_source_id = projection_to_logical.get(
                projection_source_id,
                str(item.get("logical_source_id") or projection_source_id).strip(),
            )
            if logical_source_id:
                item["source_id"] = logical_source_id
            item.pop("logical_source_id", None)
            item.pop("projection_session_num", None)
            compat_sessions.append(item)
        compat["raw_sessions"] = compat_sessions

        def _compat_fact_list(tier_name: str, facts: list[dict]) -> list[dict]:
            normalized: list[dict] = []
            for fact in facts or []:
                item = dict(fact)
                original_id = str(item.get("id") or "").strip()
                if original_id:
                    item["id"] = fact_id_map.get((tier_name, original_id), original_id)
                projection_source_id = str(item.get("source_id") or "").strip()
                if projection_source_id:
                    item["source_id"] = projection_to_logical.get(projection_source_id, projection_source_id)
                if isinstance(item.get("source_ids"), list):
                    item["source_ids"] = [
                        fact_ref_map.get(str(value or "").strip(), str(value or "").strip())
                        for value in item.get("source_ids") or []
                        if str(value or "").strip()
                    ]
                metadata = item.get("metadata")
                if isinstance(metadata, dict):
                    metadata = dict(metadata)
                    for key in ("document_source", "episode_source_id", "logical_source_id"):
                        value = str(metadata.get(key) or "").strip()
                        if value:
                            metadata[key] = projection_to_logical.get(value, value)
                    episode_id = str(metadata.get("episode_id") or "").strip()
                    if episode_id:
                        metadata["episode_id"] = self._logicalize_episode_id(
                            episode_id,
                            projection_source_id=projection_source_id,
                            logical_source_id=projection_to_logical.get(projection_source_id, projection_source_id),
                        )
                    if isinstance(metadata.get("episode_ids"), list):
                        metadata["episode_ids"] = [
                            self._logicalize_episode_id(
                                str(value or "").strip(),
                                projection_source_id=projection_source_id,
                                logical_source_id=projection_to_logical.get(projection_source_id, projection_source_id),
                            )
                            for value in metadata.get("episode_ids") or []
                            if str(value or "").strip()
                        ]
                    item["metadata"] = metadata
                normalized.append(item)
            return normalized

        compat["granular"] = _compat_fact_list("granular", payload.get("granular") or [])
        compat["cons"] = _compat_fact_list("cons", payload.get("cons") or [])
        compat["cross"] = _compat_fact_list("cross", payload.get("cross") or [])
        compat["tlinks"] = [
            {
                **dict(link or {}),
                "before": fact_ref_map.get(str((link or {}).get("before") or "").strip(), str((link or {}).get("before") or "").strip()),
                "after": fact_ref_map.get(str((link or {}).get("after") or "").strip(), str((link or {}).get("after") or "").strip()),
            }
            for link in (payload.get("tlinks") or [])
        ]

        source_record_entries: list[dict] = []
        compat_source_records: dict[str, dict] = {}
        source_record_versions: dict[str, str] = {}
        for projection_source_id, record in (payload.get("source_records") or {}).items():
            logical_source_id = projection_to_logical.get(str(projection_source_id), str(projection_source_id))
            item = dict(record or {})
            source_meta = dict(item.get("source_meta") or {})
            source_meta["logical_source_id"] = logical_source_id
            item["source_meta"] = source_meta
            source_record_entries.append({"source_id": logical_source_id, **item})
            updated_at = str(item.get("updated_at") or item.get("created_at") or "")
            if logical_source_id not in compat_source_records or updated_at >= source_record_versions.get(logical_source_id, ""):
                compat_source_records[logical_source_id] = item
                source_record_versions[logical_source_id] = updated_at
        compat["source_records"] = compat_source_records
        compat["source_record_entries"] = source_record_entries

        raw_doc_entries: list[dict] = []
        compat_raw_docs: dict[str, str] = {}
        raw_doc_versions: dict[str, str] = {}
        for projection_source_id, raw_text in (payload.get("raw_docs") or {}).items():
            logical_source_id = projection_to_logical.get(str(projection_source_id), str(projection_source_id))
            record = (payload.get("source_records") or {}).get(str(projection_source_id)) or {}
            updated_at = str(record.get("updated_at") or record.get("created_at") or "")
            raw_doc_entries.append(
                {
                    "source_id": logical_source_id,
                    "content": str(raw_text or ""),
                    "owner_id": record.get("owner_id"),
                    "read": list(record.get("read") or []),
                    "write": list(record.get("write") or []),
                }
            )
            if logical_source_id not in compat_raw_docs or updated_at >= raw_doc_versions.get(logical_source_id, ""):
                compat_raw_docs[logical_source_id] = str(raw_text or "")
                raw_doc_versions[logical_source_id] = updated_at
        compat["raw_docs"] = compat_raw_docs
        compat["raw_doc_entries"] = raw_doc_entries

        corpus = payload.get("episode_corpus") or {}
        compat_corpus: dict[str, list[dict]] = {}
        for group_name, docs in corpus.items():
            compat_docs: list[dict] = []
            compat_doc_index: dict[str, dict] = {}
            compat_episode_ids: dict[str, set[str]] = {}
            for doc in docs or []:
                item = dict(doc)
                logical_doc_id = self._logicalize_doc_id(str(item.get("doc_id") or ""), projection_to_logical)
                item["doc_id"] = logical_doc_id
                if logical_doc_id in compat_doc_index:
                    compat_item = compat_doc_index[logical_doc_id]
                else:
                    compat_item = dict(item)
                    compat_item["episodes"] = []
                    compat_doc_index[logical_doc_id] = compat_item
                    compat_docs.append(compat_item)
                    compat_episode_ids[logical_doc_id] = set()
                for episode in item.get("episodes") or []:
                    episode_item = dict(episode)
                    projection_source_id = str(episode_item.get("source_id") or "").strip()
                    logical_source_id = projection_to_logical.get(projection_source_id, projection_source_id)
                    if logical_source_id:
                        episode_item["source_id"] = logical_source_id
                    original_episode_id = str(episode_item.get("episode_id") or "").strip()
                    logical_episode_id = self._logicalize_episode_id(
                        original_episode_id,
                        projection_source_id=projection_source_id,
                        logical_source_id=logical_source_id,
                    )
                    unique_episode_id = logical_episode_id
                    suffix = 2
                    while unique_episode_id in compat_episode_ids[logical_doc_id]:
                        unique_episode_id = f"{logical_episode_id}__compat{suffix}"
                        suffix += 1
                    compat_episode_ids[logical_doc_id].add(unique_episode_id)
                    if unique_episode_id:
                        episode_item["episode_id"] = unique_episode_id
                    compat_item["episodes"].append(episode_item)
            compat_corpus[str(group_name)] = compat_docs
        compat["episode_corpus"] = compat_corpus
        return compat

    def load_facts(self, *, internal: bool = False) -> dict:
        if not self.exists:
            return {}
        with self._connect() as conn:
            state_rows = conn.execute("SELECT name, value_json FROM state_json").fetchall()
            state = {str(row["name"]): _json_loads(row["value_json"], None) for row in state_rows}
            payload: dict[str, Any] = {
                "granular": self._load_tier(conn, "granular"),
                "cons": self._load_tier(conn, "cons"),
                "cross": self._load_tier(conn, "cross"),
                "tlinks": [
                    _json_loads(row["link_json"], {})
                    for row in conn.execute("SELECT link_json FROM temporal_links ORDER BY sort_order").fetchall()
                ],
                "raw_sessions": self._load_raw_sessions(conn),
                "raw_docs": self._load_raw_docs(conn),
                "episode_corpus": self._load_episode_corpus(conn),
                "container_graph": self._load_container_graph(conn),
                "source_records": {},
            }
            for row in conn.execute("SELECT * FROM source_records ORDER BY source_id").fetchall():
                payload["source_records"][row["source_id"]] = {
                    "family": row["family"],
                    "owner_id": row["owner_id"],
                    "read": _json_loads(row["read_json"], []),
                    "write": _json_loads(row["write_json"], []),
                    "artifact_id": row["artifact_id"],
                    "version_id": row["version_id"],
                    "content_hash": row["content_hash"],
                    "metadata": _json_loads(row["metadata_json"], {}),
                    "target": _json_loads(row["target_json"], []),
                    "source_meta": _json_loads(row["source_meta_json"], {}),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            for key in STATE_JSON_KEYS:
                if key in {INTERNAL_EPISODE_DOC_ORDER, INTERNAL_EPISODE_CORPUS_LAYOUT}:
                    continue
                if key in state:
                    payload[key] = state[key]
            if internal:
                return payload
            return self._compat_load_payload(payload)

    def load_embeddings(self) -> dict | None:
        if not self.exists:
            return None
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"])
            if count == 0:
                return None
            result: dict[str, np.ndarray] = {}
            for tier_key in ("gran", "cons", "cross"):
                rows = conn.execute(
                    "SELECT dim, dtype, vector_blob FROM embeddings WHERE tier = ? ORDER BY sort_order",
                    (tier_key,),
                ).fetchall()
                if not rows:
                    result[tier_key] = np.zeros((0, 3072), dtype=np.float32)
                    continue
                dim = int(rows[0]["dim"])
                dtype = np.dtype(rows[0]["dtype"])
                matrix = np.zeros((len(rows), dim), dtype=dtype)
                for idx, row in enumerate(rows):
                    matrix[idx] = np.frombuffer(row["vector_blob"], dtype=dtype, count=dim)
                result[tier_key] = matrix
            return result

    def save_embeddings(self, gran: np.ndarray, cons: np.ndarray, cross: np.ndarray) -> None:
        tier_map = {
            "gran": ("granular", gran),
            "cons": ("cons", cons),
            "cross": ("cross", cross),
        }
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM embeddings")
            for tier_key, (fact_tier, matrix) in tier_map.items():
                fact_rows = conn.execute(
                    "SELECT fact_id, sort_order FROM facts WHERE tier = ? ORDER BY sort_order",
                    (fact_tier,),
                ).fetchall()
                if len(fact_rows) != len(matrix):
                    raise ValueError(
                        f"Embedding count mismatch for {tier_key}: {len(matrix)} vectors, {len(fact_rows)} facts"
                    )
                batch: list[tuple[Any, ...]] = []
                for idx, row in enumerate(fact_rows):
                    vector = np.asarray(matrix[idx])
                    batch.append(
                        (
                            tier_key,
                            row["fact_id"],
                            row["sort_order"],
                            int(vector.shape[0]),
                            str(vector.dtype),
                            vector.tobytes(),
                        )
                    )
                    if len(batch) >= 1000:
                        conn.executemany(
                            "INSERT INTO embeddings(tier, fact_id, sort_order, dim, dtype, vector_blob) VALUES(?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch.clear()
                if batch:
                    conn.executemany(
                        "INSERT INTO embeddings(tier, fact_id, sort_order, dim, dtype, vector_blob) VALUES(?, ?, ?, ?, ?, ?)",
                        batch,
                    )
            conn.commit()

    # ── Dedicated secret storage ──

    @staticmethod
    def _secret_row_from_db(row: Any, *, include_value: bool = False) -> dict[str, Any]:
        record = {
            "secret_id": row["secret_id"],
            "name": row["name"],
            "value_encoding": row["value_encoding"],
            "acl_domain_key": row["acl_domain_key"],
            "created_by_principal_id": row["created_by_principal_id"],
            "owner_id": row["owner_id"],
            "scope": row["scope"],
            "agent_id": row["agent_id"],
            "swarm_id": row["swarm_id"],
            "read": _json_loads(row["read_json"], []),
            "write": _json_loads(row["write_json"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }
        if include_value:
            record["value"] = _decode_secret_value(record | {"value_blob": row["value_blob"]})
        return record

    def list_secret_rows(
        self,
        *,
        acl_domain_key: str | None = None,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        self._require_secret_storage_allowed()
        with self._connect() as conn:
            if acl_domain_key is None:
                rows = conn.execute(
                    """
                    SELECT secret_id, name, value_blob, value_encoding, acl_domain_key, created_by_principal_id, owner_id, scope,
                           agent_id, swarm_id, read_json, write_json, created_at, updated_at, metadata_json
                    FROM secrets
                    ORDER BY acl_domain_key, name
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT secret_id, name, value_blob, value_encoding, acl_domain_key, created_by_principal_id, owner_id, scope,
                           agent_id, swarm_id, read_json, write_json, created_at, updated_at, metadata_json
                    FROM secrets
                    WHERE acl_domain_key = ?
                    ORDER BY name
                    """,
                    (acl_domain_key,),
                ).fetchall()
        return [self._secret_row_from_db(row, include_value=include_values) for row in rows]

    def get_secret_row(
        self,
        *,
        name: str,
        acl_domain_key: str,
        include_value: bool = True,
    ) -> dict[str, Any] | None:
        self._require_secret_storage_allowed()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT secret_id, name, value_blob, value_encoding, acl_domain_key, created_by_principal_id, owner_id, scope,
                       agent_id, swarm_id, read_json, write_json, created_at, updated_at, metadata_json
                FROM secrets
                WHERE name = ? AND acl_domain_key = ?
                """,
                (name, acl_domain_key),
            ).fetchone()
        if row is None:
            return None
        return self._secret_row_from_db(row, include_value=include_value)

    def upsert_secret(
        self,
        *,
        name: str,
        value: str,
        created_by_principal_id: str,
        owner_id: str,
        scope: str,
        agent_id: str | None,
        swarm_id: str | None,
        read: list[str],
        write: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_secret_storage_allowed()
        row = self._canonical_secret_row(
            name=name,
            value=value,
            created_by_principal_id=created_by_principal_id,
            owner_id=owner_id,
            scope=scope,
            agent_id=agent_id,
            swarm_id=swarm_id,
            read=list(read),
            write=list(write),
            metadata=metadata,
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT secret_id, acl_domain_key FROM secrets WHERE name = ? AND acl_domain_key = ?",
                (row["name"], row["acl_domain_key"]),
            ).fetchone()
            if existing is not None:
                conn.rollback()
                return {
                    "stored": False,
                    "code": "SECRET_ALREADY_EXISTS",
                    "secret_id": str(existing["secret_id"]),
                    "acl_domain_key": str(existing["acl_domain_key"]),
                }
            self._insert_secret_row(conn, row)
            conn.commit()
        return {
            "stored": True,
            "secret_id": row["secret_id"],
            "acl_domain_key": row["acl_domain_key"],
        }

    def delete_secret(self, *, name: str, acl_domain_key: str) -> bool:
        self._require_secret_storage_allowed()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "DELETE FROM secrets WHERE name = ? AND acl_domain_key = ?",
                (name, acl_domain_key),
            )
            deleted = int(cur.rowcount or 0) > 0
            conn.commit()
        return deleted

    def _next_sort_order(self, conn, table_name: str, *, tier: str | None = None) -> int:
        if tier is None:
            row = conn.execute(
                f"SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM {table_name}"  # noqa: S608  # nosec B608
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM {table_name} WHERE tier = ?",  # noqa: S608  # nosec B608
                (tier,),
            ).fetchone()
        return int(row["next_sort"] or 0)

    def _upsert_raw_session_projection_row(self, conn, raw_session: dict) -> None:
        session_num = self._canonical_session_num(raw_session)
        message_id = str(raw_session.get("message_id") or self._legacy_message_id_for_session(raw_session))
        by_session = conn.execute(
            "SELECT sort_order FROM raw_sessions WHERE session_num = ?",
            (session_num,),
        ).fetchone()
        sort_order = int(by_session["sort_order"]) if by_session is not None else self._next_sort_order(conn, "raw_sessions")
        source_id = str(raw_session.get("source_id") or self._key)
        source_meta = {
            key: value
            for key, value in raw_session.items()
            if key not in {
                "message_id", "raw_session_id", "session_num", "session_date", "content", "speakers",
                "agent_id", "swarm_id", "scope", "owner_id", "read", "write",
                "stored_at", "format", "source_id", "artifact_id",
                "version_id", "content_hash", "status", "metadata",
                "target",
            }
        }
        row = (
            session_num,
            message_id,
            self._raw_session_identity(raw_session),
            source_id,
            raw_session.get("format"),
            raw_session.get("session_date"),
            raw_session.get("speakers"),
            raw_session.get("stored_at"),
            raw_session.get("artifact_id"),
            raw_session.get("version_id"),
            raw_session.get("content_hash"),
            raw_session.get("owner_id"),
            raw_session.get("scope"),
            raw_session.get("agent_id"),
            raw_session.get("swarm_id"),
            _json_dumps(raw_session.get("read") or []),
            _json_dumps(raw_session.get("write") or []),
            _json_dumps(raw_session.get("target") or []),
            _json_dumps(raw_session.get("metadata") or {}),
            _json_dumps(source_meta),
            raw_session.get("status"),
            sort_order,
        )
        updated = conn.execute(
            """
            UPDATE raw_sessions
            SET message_id = ?,
                raw_session_id = ?,
                source_id = ?,
                format = ?,
                session_date = ?,
                speakers = ?,
                stored_at = ?,
                artifact_id = ?,
                version_id = ?,
                content_hash = ?,
                owner_id = ?,
                scope = ?,
                agent_id = ?,
                swarm_id = ?,
                read_json = ?,
                write_json = ?,
                target_json = ?,
                metadata_json = ?,
                source_meta_json = ?,
                status = ?,
                sort_order = ?
            WHERE session_num = ?
            """,
            row[1:] + (session_num,),
        )
        if int(updated.rowcount or 0) == 0:
            conn.execute(
                """
                INSERT INTO raw_sessions(
                    session_num, message_id, raw_session_id, source_id, format, session_date,
                    speakers, stored_at, artifact_id, version_id, content_hash, owner_id, scope,
                    agent_id, swarm_id, read_json, write_json, target_json, metadata_json,
                    source_meta_json, status, sort_order
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def _upsert_raw_doc_projection_row(self, conn, row: dict) -> None:
        source_id = str(row.get("source_id") or "")
        existing = conn.execute(
            "SELECT sort_order FROM raw_docs WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        sort_order = int(existing["sort_order"]) if existing is not None else self._next_sort_order(conn, "raw_docs")
        message_id = str(row.get("message_id") or self._legacy_message_id_for_doc(source_id))
        metadata = dict(row.get("metadata") or {})
        metadata_json = _json_dumps(metadata)
        if "content_text" in row:
            content_text = str(row.get("content_text") or "")
            if content_text:
                write_row = conn.execute(
                    """
                    SELECT session_id, agent_id, swarm_id, visibility, owner_id, scope, read_json,
                           write_json, content_family, metadata_json, timestamp_ms, extraction_state,
                           sort_order
                    FROM write_log WHERE message_id = ?
                    """,
                    (message_id,),
                ).fetchone()
                write_existing = dict(write_row) if write_row is not None else {}
                write_sort_order = (
                    int(write_existing["sort_order"])
                    if write_existing
                    else self._next_sort_order(conn, "write_log")
                )
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                merged_metadata = _json_loads(write_existing.get("metadata_json"), {}) if write_existing else {}
                merged_metadata.update(metadata)
                self._upsert_write_log_snapshot_row(
                    conn,
                    message_id=message_id,
                    session_id=str(write_existing.get("session_id") or f"doc:{source_id}"),
                    agent_id=str(write_existing.get("agent_id") or "system"),
                    swarm_id=str(write_existing.get("swarm_id") or "default"),
                    visibility=str(write_existing.get("visibility") or "shared"),
                    owner_id=write_existing.get("owner_id"),
                    scope=write_existing.get("scope"),
                    read=_json_loads(write_existing.get("read_json"), []) if write_existing else [],
                    write=_json_loads(write_existing.get("write_json"), []) if write_existing else [],
                    content_family=str(write_existing.get("content_family") or "document"),
                    content_text=content_text,
                    metadata=merged_metadata,
                    timestamp_ms=int(write_existing.get("timestamp_ms") or now_ms),
                    extraction_state=str(write_existing.get("extraction_state") or "complete"),
                    sort_order=write_sort_order,
                )
        updated = conn.execute(
            "UPDATE raw_docs SET message_id = ?, metadata_json = ?, sort_order = ? WHERE source_id = ?",
            (message_id, metadata_json, sort_order, source_id),
        )
        if int(updated.rowcount or 0) == 0:
            conn.execute(
                "INSERT INTO raw_docs(source_id, message_id, metadata_json, sort_order) VALUES(?, ?, ?, ?)",
                (source_id, message_id, metadata_json, sort_order),
            )

    def _delete_fact_rows(self, conn, fact_keys: list[tuple[str, str]]) -> None:
        if not fact_keys:
            return
        conn.executemany(
            "DELETE FROM facts WHERE tier = ? AND fact_id = ?",
            fact_keys,
        )
        emb_rows = []
        emb_tier_map = {"granular": "gran", "cons": "cons", "cross": "cross"}
        for fact_tier, fact_id in fact_keys:
            emb_rows.append((emb_tier_map.get(fact_tier, fact_tier), fact_id))
        conn.executemany(
            "DELETE FROM embeddings WHERE tier = ? AND fact_id = ?",
            emb_rows,
        )

    def _upsert_fact_projection_rows(self, conn, tier_name: str, facts: list[dict]) -> None:
        if not facts:
            return
        ordered: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for idx, fact in enumerate(facts):
            fact_id = self._canonical_fact_id(fact, tier=tier_name, fallback_sort=idx)
            if fact_id in seen:
                fact_id = f"{fact_id}__{idx}"
            seen.add(fact_id)
            ordered.append((fact_id, fact))
        existing: dict[str, int] = {}
        if ordered:
            for start in range(0, len(ordered), 500):
                chunk = ordered[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT fact_id, sort_order FROM facts WHERE tier = ? AND fact_id IN ({placeholders})",  # noqa: S608  # nosec B608
                    (tier_name, *(fact_id for fact_id, _ in chunk)),
                ).fetchall()
                existing.update({str(row["fact_id"]): int(row["sort_order"] or 0) for row in rows})
        next_sort = self._next_sort_order(conn, "facts", tier=tier_name)
        for idx, (fact_id, fact) in enumerate(ordered):
            sort_order = existing.get(fact_id)
            if sort_order is None:
                sort_order = next_sort
                next_sort += 1
            row = (tier_name, *self._fact_row_payload(fact, tier=tier_name, fallback_sort=idx, fact_id=fact_id), sort_order)
            updated = conn.execute(
                """
                UPDATE facts
                SET kind = ?,
                    session_num = ?,
                    source_id = ?,
                    agent_id = ?,
                    swarm_id = ?,
                    scope = ?,
                    owner_id = ?,
                    status = ?,
                    created_at = ?,
                    event_date = ?,
                    payload_json = ?,
                    sort_order = ?
                WHERE tier = ? AND fact_id = ?
                """,
                row[2:] + (row[0], row[1]),
            )
            if int(updated.rowcount or 0) == 0:
                conn.execute(
                    """
                    INSERT INTO facts(
                        tier, fact_id, kind, session_num, source_id, agent_id, swarm_id, scope,
                        owner_id, status, created_at, event_date, payload_json, sort_order
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )

    def _replace_episode_document_rows(self, conn, doc_id: str, episodes: list[dict]) -> None:
        conn.execute("DELETE FROM episode_corpus WHERE doc_id = ?", (doc_id,))
        rows = [
            (doc_id, str(episode.get("episode_id") or ""), idx, _json_dumps(episode))
            for idx, episode in enumerate(episodes)
            if str(episode.get("episode_id") or "")
        ]
        if rows:
            conn.executemany(
                "INSERT INTO episode_corpus(doc_id, episode_id, sort_order, episode_json) VALUES(?, ?, ?, ?)",
                rows,
            )

    def _append_temporal_link_rows(self, conn, links: list[dict]) -> None:
        if not links:
            return
        next_sort = self._next_sort_order(conn, "temporal_links")
        rows = []
        for idx, link in enumerate(links):
            rows.append((next_sort + idx, _json_dumps(link)))
        conn.executemany(
            "INSERT INTO temporal_links(sort_order, link_json) VALUES(?, ?)",
            rows,
        )

    def _upsert_source_record_rows(self, conn, source_records: dict[str, dict]) -> None:
        if not source_records:
            return
        rows = [
            (
                str(source_id),
                record.get("family") or "unknown",
                record.get("owner_id"),
                _json_dumps(record.get("read") or []),
                _json_dumps(record.get("write") or []),
                record.get("artifact_id"),
                record.get("version_id"),
                record.get("content_hash"),
                _json_dumps(record.get("metadata") or {}),
                _json_dumps(record.get("target") or []),
                _json_dumps(_source_record_meta_payload(record)),
                record.get("created_at"),
                record.get("updated_at"),
            )
            for source_id, record in source_records.items()
        ]
        for row in rows:
            updated = conn.execute(
                """
                UPDATE source_records
                SET family = ?,
                    owner_id = ?,
                    read_json = ?,
                    write_json = ?,
                    artifact_id = ?,
                    version_id = ?,
                    content_hash = ?,
                    metadata_json = ?,
                    target_json = ?,
                    source_meta_json = ?,
                    created_at = ?,
                    updated_at = ?
                WHERE source_id = ?
                """,
                row[1:] + (row[0],),
            )
            if int(updated.rowcount or 0) == 0:
                conn.execute(
                    """
                    INSERT INTO source_records(
                        source_id, family, owner_id, read_json, write_json, artifact_id, version_id,
                        content_hash, metadata_json, target_json, source_meta_json, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )

    def _upsert_state_json_values(self, conn, *, state_values: dict[str, Any], episode_corpus: dict | None = None) -> None:
        values = {
            name: state_values[name]
            for name in STATE_JSON_KEYS
            if name in state_values and name not in {INTERNAL_EPISODE_DOC_ORDER, INTERNAL_EPISODE_CORPUS_LAYOUT}
        }
        if episode_corpus is not None:
            corpus_layout = self._episode_corpus_layout(episode_corpus)
            doc_order = [doc_id for doc_ids in corpus_layout.values() for doc_id in doc_ids]
            values[INTERNAL_EPISODE_DOC_ORDER] = doc_order
            values[INTERNAL_EPISODE_CORPUS_LAYOUT] = corpus_layout
        rows = [(name, _json_dumps(value)) for name, value in values.items()]
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO state_json(name, value_json) VALUES(?, ?)",
                rows,
            )

    # ── Incremental projection persistence for SQLite sync write-through ──

    def persist_projection_delta(
        self,
        *,
        raw_session_upserts: list[dict] | None = None,
        raw_session_deletes: list[int] | None = None,
        raw_doc_upserts: list[dict] | None = None,
        fact_upserts: dict[str, list[dict]] | None = None,
        fact_deletes: list[tuple[str, str]] | None = None,
        episode_doc_replacements: dict[str, list[dict]] | None = None,
        temporal_link_appends: list[dict] | None = None,
        source_record_upserts: dict[str, dict] | None = None,
        state_values: dict[str, Any] | None = None,
        episode_corpus: dict | None = None,
        container_graph_revision_upserts: list[dict] | None = None,
        container_upserts: list[dict] | None = None,
        container_relation_upserts: list[dict] | None = None,
        container_anchor_upserts: list[dict] | None = None,
        container_evidence_upserts: list[dict] | None = None,
        container_ref_upserts: list[dict] | None = None,
        container_ref_lookup_upserts: list[dict] | None = None,
        container_ref_range_upserts: list[dict] | None = None,
        container_render_ref_upserts: list[dict] | None = None,
        container_contract_upserts: list[dict] | None = None,
        container_artifact_upserts: list[dict] | None = None,
        container_state_upserts: list[dict] | None = None,
        container_deletes: list[str] | None = None,
        replace_container_graph: bool = False,
        complete_message_ids: list[str] | None = None,
    ) -> None:
        raw_session_upserts = raw_session_upserts or []
        raw_session_deletes = raw_session_deletes or []
        raw_doc_upserts = raw_doc_upserts or []
        fact_upserts = fact_upserts or {}
        fact_deletes = fact_deletes or []
        episode_doc_replacements = episode_doc_replacements or {}
        temporal_link_appends = temporal_link_appends or []
        source_record_upserts = source_record_upserts or {}
        state_values = state_values or {}
        container_deletes = container_deletes or []
        complete_message_ids = complete_message_ids or []

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if raw_session_deletes:
                conn.executemany(
                    "DELETE FROM raw_sessions WHERE session_num = ?",
                    [(session_num,) for session_num in raw_session_deletes],
                )
            self._delete_fact_rows(conn, fact_deletes)
            for raw_session in raw_session_upserts:
                self._upsert_raw_session_projection_row(conn, raw_session)
            for raw_doc in raw_doc_upserts:
                self._upsert_raw_doc_projection_row(conn, raw_doc)
            for tier_name, facts in fact_upserts.items():
                self._upsert_fact_projection_rows(conn, tier_name, facts)
            for doc_id, episodes in episode_doc_replacements.items():
                self._replace_episode_document_rows(conn, doc_id, episodes)
            self._append_temporal_link_rows(conn, temporal_link_appends)
            self._upsert_source_record_rows(conn, source_record_upserts)
            if replace_container_graph:
                for table_name in reversed(CONTAINER_GRAPH_SNAPSHOT_KEYS.values()):
                    conn.execute(f"DELETE FROM {table_name}")  # noqa: S608  # nosec B608
            if container_deletes:
                conn.executemany("DELETE FROM containers WHERE container_id = ?", [(value,) for value in container_deletes])
            self._upsert_container_graph_rows(conn, "container_graph_revisions", container_graph_revision_upserts or [])
            self._upsert_container_graph_rows(conn, "containers", container_upserts or [])
            self._upsert_container_graph_rows(conn, "container_relations", container_relation_upserts or [])
            self._upsert_container_graph_rows(conn, "container_anchors", container_anchor_upserts or [])
            self._upsert_container_graph_rows(conn, "container_evidence", container_evidence_upserts or [])
            self._upsert_container_graph_rows(conn, "container_refs", container_ref_upserts or [])
            self._upsert_container_graph_rows(conn, "container_ref_lookup", container_ref_lookup_upserts or [])
            self._upsert_container_graph_rows(conn, "container_ref_ranges", container_ref_range_upserts or [])
            self._upsert_container_graph_rows(conn, "container_render_refs", container_render_ref_upserts or [])
            self._upsert_container_graph_rows(conn, "container_contracts", container_contract_upserts or [])
            self._upsert_container_graph_rows(conn, "container_artifacts", container_artifact_upserts or [])
            self._upsert_container_graph_rows(conn, "container_state", container_state_upserts or [])
            self._upsert_state_json_values(conn, state_values=state_values, episode_corpus=episode_corpus)
            if complete_message_ids:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                conn.executemany(
                    """
                    UPDATE write_log
                    SET extraction_state = 'complete',
                        last_extraction_attempt_ms = ?
                    WHERE message_id = ?
                    """,
                    [(now_ms, message_id) for message_id in complete_message_ids],
                )
            conn.commit()

    # ── Ingress truth operations ──

    def append_write_log(
        self,
        *,
        message_id: str,
        session_id: str,
        agent_id: str,
        swarm_id: str,
        visibility: str,
        owner_id: str | None,
        scope: str | None,
        read: list[str] | None,
        write: list[str] | None,
        content_family: str,
        content_text: str,
        metadata: dict | None,
        timestamp_ms: int,
    ) -> dict:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT message_id, extraction_state FROM write_log WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return {
                    "message_id": message_id,
                    "extraction_state": existing["extraction_state"],
                    "inserted": False,
                }
            sort_order = int(
                conn.execute("SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort FROM write_log").fetchone()["next_sort"]
            )
            conn.execute(
                """
                INSERT INTO write_log(
                    message_id, session_id, agent_id, swarm_id, visibility,
                    owner_id, scope, read_json, write_json, content_family,
                    content_text, metadata_json, timestamp_ms, extraction_state,
                    extraction_attempts, last_extraction_attempt_ms, sort_order
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?)
                """,
                (
                    message_id,
                    session_id,
                    agent_id,
                    swarm_id,
                    visibility,
                    owner_id,
                    scope,
                    _json_dumps(read or []),
                    _json_dumps(write or []),
                    content_family,
                    content_text,
                    _json_dumps(metadata or {}),
                    int(timestamp_ms),
                    sort_order,
                ),
            )
            conn.commit()
            return {"message_id": message_id, "extraction_state": "pending", "inserted": True}

    def get_write_status(self, message_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT message_id, extraction_state, extraction_attempts, last_extraction_attempt_ms, "
                "owner_id, scope, read_json, write_json, agent_id, swarm_id, metadata_json, "
                "lease_owner, lease_expires_at_ms "
                "FROM write_log WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "message_id": row["message_id"],
                "extraction_state": row["extraction_state"],
                "extraction_attempts": int(row["extraction_attempts"] or 0),
                "last_extraction_attempt_ms": row["last_extraction_attempt_ms"],
                "owner_id": row["owner_id"],
                "scope": row["scope"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "lease_owner": row["lease_owner"],
                "lease_expires_at_ms": row["lease_expires_at_ms"],
            }

    def list_write_log_entries(
        self,
        *,
        states: list[str] | None = None,
        swarm_id: str | None = None,
        order: str = "asc",
    ) -> list[dict]:
        states = states or ["pending", "in_progress", "failed"]
        clauses = [f"extraction_state IN ({','.join('?' for _ in states)})"]
        params: list[Any] = list(states)
        if swarm_id is not None:
            clauses.append("swarm_id = ?")
            params.append(swarm_id)
        order_sql = "ASC" if order.lower() == "asc" else "DESC"
        # Clauses come from fixed internal strings; order_sql is enum-validated.
        sql = "SELECT message_id, session_id, agent_id, swarm_id, visibility, owner_id, scope, read_json, write_json, content_family, content_text, metadata_json, timestamp_ms, extraction_state, extraction_attempts, last_extraction_attempt_ms, lease_owner, lease_expires_at_ms, sort_order FROM write_log WHERE " + " AND ".join(clauses) + f" ORDER BY sort_order {order_sql}, timestamp_ms {order_sql}"  # noqa: S608  # nosec B608
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "message_id": row["message_id"],
                "session_id": row["session_id"],
                "agent_id": row["agent_id"],
                "swarm_id": row["swarm_id"],
                "visibility": row["visibility"],
                "owner_id": row["owner_id"],
                "scope": row["scope"],
                "read": _json_loads(row["read_json"], []),
                "write": _json_loads(row["write_json"], []),
                "content_family": row["content_family"],
                "content": row["content_text"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "timestamp_ms": int(row["timestamp_ms"] or 0),
                "extraction_state": row["extraction_state"],
                "extraction_attempts": int(row["extraction_attempts"] or 0),
                "last_extraction_attempt_ms": row["last_extraction_attempt_ms"],
                "lease_owner": row["lease_owner"],
                "lease_expires_at_ms": row["lease_expires_at_ms"],
                "sort_order": int(row["sort_order"] or 0),
            }
            for row in rows
        ]

    def mark_write_state(self, message_id: str, state: str, *, attempts_delta: int = 0) -> None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        clear_lease = state != "in_progress"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE write_log
                SET extraction_state = ?,
                    extraction_attempts = extraction_attempts + ?,
                    last_extraction_attempt_ms = ?,
                    lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
                    lease_expires_at_ms = CASE WHEN ? THEN NULL ELSE lease_expires_at_ms END
                WHERE message_id = ?
                """,
                (state, attempts_delta, now_ms, 1 if clear_lease else 0, 1 if clear_lease else 0, message_id),
            )
            conn.commit()

    @staticmethod
    def _write_log_claim_where_sql() -> str:
        return """
            (
                extraction_state = 'pending'
                OR (
                    extraction_state = 'failed'
                    AND extraction_attempts < ?
                    AND (
                        last_extraction_attempt_ms IS NULL
                        OR last_extraction_attempt_ms <= ?
                    )
                )
                OR (
                    extraction_state = 'in_progress'
                    AND lease_expires_at_ms IS NOT NULL
                    AND lease_expires_at_ms <= ?
                )
            )
            AND (
                lease_owner IS NULL
                OR lease_expires_at_ms IS NULL
                OR lease_expires_at_ms <= ?
            )
        """

    def claim_write_log_entries(
        self,
        *,
        worker_id: str,
        batch_size: int,
        now_ms: int,
        lease_ms: int,
        retry_backoff_ms: int,
        max_attempts: int,
    ) -> list[dict]:
        if batch_size <= 0:
            return []
        retry_ready_ms = int(now_ms) - max(0, int(retry_backoff_ms))
        lease_until_ms = int(now_ms) + max(1, int(lease_ms))
        claim_where_sql = self._write_log_claim_where_sql()
        claim_where_params = (
            int(max_attempts),
            retry_ready_ms,
            int(now_ms),
            int(now_ms),
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT message_id, session_id, agent_id, swarm_id, visibility,
                       owner_id, scope, read_json, write_json, content_family,
                       content_text, metadata_json, timestamp_ms, extraction_state,
                       extraction_attempts, last_extraction_attempt_ms,
                       lease_owner, lease_expires_at_ms, sort_order
                FROM write_log
                WHERE {claim_where_sql}
                ORDER BY sort_order ASC, timestamp_ms ASC
                LIMIT ?
                """,  # noqa: S608  # nosec B608
                (*claim_where_params, int(batch_size)),
            ).fetchall()
            claimed: list[dict[str, Any]] = []
            for row in rows:
                previous_state = str(row["extraction_state"] or "")
                previous_lease_expired = (
                    previous_state == "in_progress"
                    and row["lease_expires_at_ms"] is not None
                    and int(row["lease_expires_at_ms"]) <= int(now_ms)
                )
                cur = conn.execute(
                    f"""
                    UPDATE write_log
                    SET extraction_state = 'in_progress',
                        extraction_attempts = extraction_attempts + 1,
                        last_extraction_attempt_ms = ?,
                        lease_owner = ?,
                        lease_expires_at_ms = ?
                    WHERE message_id = ?
                      AND {claim_where_sql}
                    """,  # noqa: S608  # nosec B608
                    (
                        int(now_ms),
                        worker_id,
                        lease_until_ms,
                        row["message_id"],
                        *claim_where_params,
                    ),
                )
                if cur.rowcount != 1:
                    continue
                claimed.append({
                    "message_id": row["message_id"],
                    "session_id": row["session_id"],
                    "agent_id": row["agent_id"],
                    "swarm_id": row["swarm_id"],
                    "visibility": row["visibility"],
                    "owner_id": row["owner_id"],
                    "scope": row["scope"],
                    "read": _json_loads(row["read_json"], []),
                    "write": _json_loads(row["write_json"], []),
                    "content_family": row["content_family"],
                    "content": row["content_text"],
                    "metadata": _json_loads(row["metadata_json"], {}),
                    "timestamp_ms": int(row["timestamp_ms"] or 0),
                    "extraction_state": "in_progress",
                    "previous_extraction_state": previous_state,
                    "extraction_attempts": int(row["extraction_attempts"] or 0) + 1,
                    "last_extraction_attempt_ms": int(now_ms),
                    "lease_owner": worker_id,
                    "lease_expires_at_ms": lease_until_ms,
                    "lease_reclaimed": previous_lease_expired,
                    "sort_order": int(row["sort_order"] or 0),
                })
            conn.commit()
        return claimed

    def merge_write_log_metadata(self, message_id: str, patch: dict[str, Any]) -> None:
        if not patch:
            return
        with self._connect() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM write_log WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return
            merged = _json_loads(row["metadata_json"], {})
            if not isinstance(merged, dict):
                merged = {}
            merged.update(patch)
            conn.execute(
                "UPDATE write_log SET metadata_json = ? WHERE message_id = ?",
                (_json_dumps(merged), message_id),
            )
            conn.commit()

    @staticmethod
    def _index_status_from_row(row: Any | None, *, now_ms: int | None = None) -> dict[str, Any]:
        status: dict[str, Any]
        if row is None:
            status = {
                "index_dirty": False,
                "index_dirty_since_ms": None,
                "last_index_dirty_ms": None,
                "next_index_build_after_ms": None,
                "next_index_retry_after_ms": None,
                "index_build_lease_owner": None,
                "index_build_lease_expires_at_ms": None,
                "last_index_build_started_ms": None,
                "last_index_build_completed_ms": None,
                "last_index_build_error": None,
                "last_index_build_error_count": 0,
                "dirty_after_build": False,
                "snapshot_fingerprint": None,
                "updated_at_ms": None,
            }
        else:
            status = {
                "index_dirty": bool(row["index_dirty"]),
                "index_dirty_since_ms": row["index_dirty_since_ms"],
                "last_index_dirty_ms": row["last_index_dirty_ms"],
                "next_index_build_after_ms": row["next_index_build_after_ms"],
                "next_index_retry_after_ms": row["next_index_retry_after_ms"],
                "index_build_lease_owner": row["index_build_lease_owner"],
                "index_build_lease_expires_at_ms": row["index_build_lease_expires_at_ms"],
                "last_index_build_started_ms": row["last_index_build_started_ms"],
                "last_index_build_completed_ms": row["last_index_build_completed_ms"],
                "last_index_build_error": row["last_index_build_error"],
                "last_index_build_error_count": int(row["last_index_build_error_count"] or 0),
                "dirty_after_build": bool(row["dirty_after_build"]),
                "snapshot_fingerprint": row["snapshot_fingerprint"],
                "updated_at_ms": row["updated_at_ms"],
            }
        now = int(now_ms if now_ms is not None else datetime.now(timezone.utc).timestamp() * 1000)
        lease_live = (
            status["index_build_lease_owner"] is not None
            and status["index_build_lease_expires_at_ms"] is not None
            and int(status["index_build_lease_expires_at_ms"]) > now
        )
        retry_live = (
            status["next_index_retry_after_ms"] is not None
            and int(status["next_index_retry_after_ms"]) > now
        )
        state: str
        if lease_live:
            state = "building"
        elif retry_live:
            state = "backoff"
        elif status["index_dirty"] and status["next_index_build_after_ms"] is not None and int(status["next_index_build_after_ms"]) > now:
            state = "scheduled"
        elif status["index_dirty"]:
            state = "dirty"
        elif status["last_index_build_completed_ms"] is not None:
            state = "ready"
        elif status["last_index_build_error"]:
            state = "failed"
        else:
            state = "missing"
        status["index_state"] = state
        status["index_build_lease_live"] = lease_live
        return status

    def _read_index_status_row(self, conn) -> Any | None:
        return conn.execute("SELECT * FROM index_status WHERE name = 'default'").fetchone()

    def _upsert_index_status(self, conn, values: dict[str, Any]) -> None:
        row = self._read_index_status_row(conn)
        base = self._index_status_from_row(row)
        base.update(values)
        conn.execute(
            """
            INSERT OR REPLACE INTO index_status(
                name, index_dirty, index_dirty_since_ms, last_index_dirty_ms,
                next_index_build_after_ms, next_index_retry_after_ms,
                index_build_lease_owner, index_build_lease_expires_at_ms,
                last_index_build_started_ms, last_index_build_completed_ms,
                last_index_build_error, last_index_build_error_count,
                dirty_after_build, snapshot_fingerprint, updated_at_ms
            ) VALUES('default', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1 if base.get("index_dirty") else 0,
                base.get("index_dirty_since_ms"),
                base.get("last_index_dirty_ms"),
                base.get("next_index_build_after_ms"),
                base.get("next_index_retry_after_ms"),
                base.get("index_build_lease_owner"),
                base.get("index_build_lease_expires_at_ms"),
                base.get("last_index_build_started_ms"),
                base.get("last_index_build_completed_ms"),
                base.get("last_index_build_error"),
                int(base.get("last_index_build_error_count") or 0),
                1 if base.get("dirty_after_build") else 0,
                base.get("snapshot_fingerprint"),
                base.get("updated_at_ms"),
            ),
        )

    def read_index_status(self, *, now_ms: int | None = None) -> dict[str, Any]:
        with self._connect() as conn:
            return self._index_status_from_row(self._read_index_status_row(conn), now_ms=now_ms)

    def mark_index_dirty(
        self,
        *,
        now_ms: int,
        debounce_ms: int,
        max_delay_ms: int,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._index_status_from_row(self._read_index_status_row(conn), now_ms=now_ms)
            dirty_since = current.get("index_dirty_since_ms") if current.get("index_dirty") else None
            if dirty_since is None:
                dirty_since = int(now_ms)
            latest_allowed_ms = int(dirty_since) + max(0, int(max_delay_ms))
            next_after = min(int(now_ms) + max(0, int(debounce_ms)), latest_allowed_ms)
            dirty_after_build = bool(current.get("dirty_after_build"))
            if current.get("index_state") == "building":
                dirty_after_build = True
            self._upsert_index_status(conn, {
                "index_dirty": True,
                "index_dirty_since_ms": int(dirty_since),
                "last_index_dirty_ms": int(now_ms),
                "next_index_build_after_ms": next_after,
                "dirty_after_build": dirty_after_build,
                "updated_at_ms": int(now_ms),
            })
            conn.commit()
            return self.read_index_status(now_ms=now_ms)

    def acquire_index_build_lease(
        self,
        *,
        worker_id: str,
        snapshot_fingerprint: str,
        now_ms: int,
        lease_ms: int,
    ) -> dict[str, Any]:
        lease_until_ms = int(now_ms) + max(1, int(lease_ms))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._index_status_from_row(self._read_index_status_row(conn), now_ms=now_ms)
            if current.get("index_build_lease_live"):
                conn.commit()
                return {"acquired": False, **current}
            if current.get("next_index_retry_after_ms") is not None and int(current["next_index_retry_after_ms"]) > int(now_ms):
                conn.commit()
                return {"acquired": False, **current}
            self._upsert_index_status(conn, {
                "index_build_lease_owner": worker_id,
                "index_build_lease_expires_at_ms": lease_until_ms,
                "last_index_build_started_ms": int(now_ms),
                "snapshot_fingerprint": snapshot_fingerprint,
                "updated_at_ms": int(now_ms),
            })
            conn.commit()
        return {"acquired": True, **self.read_index_status(now_ms=now_ms)}

    def release_index_build_lease(
        self,
        *,
        worker_id: str,
        success: bool,
        now_ms: int,
        debounce_ms: int,
        max_delay_ms: int,
        retry_after_ms: int | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._index_status_from_row(self._read_index_status_row(conn), now_ms=now_ms)
            if current.get("index_build_lease_owner") != worker_id:
                conn.commit()
                return {"released": False, **current}
            if success:
                if current.get("dirty_after_build"):
                    dirty_since = current.get("index_dirty_since_ms") or int(now_ms)
                    latest_allowed_ms = int(dirty_since) + max(0, int(max_delay_ms))
                    next_after = min(int(now_ms) + max(0, int(debounce_ms)), latest_allowed_ms)
                    next_values = {
                        "index_dirty": True,
                        "index_dirty_since_ms": int(dirty_since),
                        "next_index_build_after_ms": next_after,
                        "next_index_retry_after_ms": None,
                    }
                else:
                    next_values = {
                        "index_dirty": False,
                        "index_dirty_since_ms": None,
                        "next_index_build_after_ms": None,
                        "next_index_retry_after_ms": None,
                    }
                self._upsert_index_status(conn, {
                    **next_values,
                    "index_build_lease_owner": None,
                    "index_build_lease_expires_at_ms": None,
                    "last_index_build_completed_ms": int(now_ms),
                    "last_index_build_error": None,
                    "last_index_build_error_count": 0,
                    "dirty_after_build": False,
                    "updated_at_ms": int(now_ms),
                })
            else:
                error_count = int(current.get("last_index_build_error_count") or 0) + 1
                self._upsert_index_status(conn, {
                    "index_dirty": True,
                    "index_build_lease_owner": None,
                    "index_build_lease_expires_at_ms": None,
                    "next_index_retry_after_ms": int(now_ms) + int(retry_after_ms or 0),
                    "last_index_build_error": str(error or "index build failed")[:1000],
                    "last_index_build_error_count": error_count,
                    "dirty_after_build": False,
                    "updated_at_ms": int(now_ms),
                })
            conn.commit()
        return {"released": True, **self.read_index_status(now_ms=now_ms)}


def _legacy_storage_paths(data_dir_path: Path, key: str) -> list[Path]:
    return [
        data_dir_path / f"{key}.json",
        data_dir_path / f"{key}_embs.npz",
        data_dir_path / f"{key}_corpus.json",
        data_dir_path / f"{key}_temporal.json",
    ]


def _migration_encryption_key_from_env() -> bytes | None:
    enc_key_hex = os.environ.get("GOSH_MEMORY_ENCRYPTION_KEY")
    return bytes.fromhex(enc_key_hex) if enc_key_hex else None


def _legacy_storage_is_encrypted(data_dir_path: Path, key: str) -> bool:
    return any(_path_has_magic_prefix(path) for path in _legacy_storage_paths(data_dir_path, key))


def _validate_migration_encryption_requirements(
    *,
    data_dir_path: Path,
    key: str,
    encryption_key: bytes | None,
) -> bool:
    encrypted_legacy = _legacy_storage_is_encrypted(data_dir_path, key)
    if not encrypted_legacy:
        return False
    if encryption_key is None:
        raise RuntimeError(
            "Encrypted legacy storage requires GOSH_MEMORY_ENCRYPTION_KEY; "
            "plaintext downgrade is not allowed."
        )
    if not _sqlcipher_available():
        raise RuntimeError(
            "Encrypted SQLite mode requires SQLCipher (pysqlcipher3). "
            "Migration must abort when SQLCipher is unavailable."
        )
    return True


def _normalize_snapshot_for_migration(sqlite_backend: SQLiteStorageBackend, payload: dict) -> dict:
    normalized = dict(payload)
    normalized.pop("secrets", None)
    normalized.setdefault("granular", [])
    normalized.setdefault("cons", [])
    normalized.setdefault("cross", [])
    normalized.setdefault("tlinks", [])
    normalized.setdefault("raw_sessions", [])
    normalized.setdefault("raw_docs", {})
    normalized.setdefault("episode_corpus", {"documents": []})
    normalized.setdefault("source_records", {})
    normalized.setdefault("container_graph", {key: [] for key in CONTAINER_GRAPH_SNAPSHOT_KEYS})

    normalized_raw_sessions: list[dict] = []
    for idx, raw_session in enumerate(normalized["raw_sessions"]):
        item = dict(raw_session)
        if not str(item.get("message_id") or "").strip():
            item["message_id"] = sqlite_backend._legacy_message_id_for_session(
                item,
                fallback_idx=idx,
            )
        normalized_raw_sessions.append(item)
    normalized["raw_sessions"] = normalized_raw_sessions

    normalized_source_records: dict[str, dict] = {}
    for source_id, record in (normalized["source_records"] or {}).items():
        item = dict(record)
        item.pop("source_id", None)
        item.pop("scope_id", None)
        item.setdefault("target", [])
        normalized_source_records[str(source_id)] = item
    normalized["source_records"] = normalized_source_records
    return normalized


def _verify_sqlite_payload(
    sqlite_backend: SQLiteStorageBackend,
    *,
    expected_snapshot: dict | None = None,
    expected_embeddings: dict | None = None,
) -> None:
    if expected_snapshot is not None:
        loaded_snapshot = sqlite_backend.load_facts(internal=True)
        if _normalize_snapshot_for_migration(sqlite_backend, loaded_snapshot) != _normalize_snapshot_for_migration(
            sqlite_backend,
            expected_snapshot,
        ):
            raise RuntimeError("SQLite migration verification failed: facts snapshot mismatch")
        expected_secrets = expected_snapshot.get("secrets") if isinstance(expected_snapshot, dict) else None
        if expected_secrets is not None:
            expected_rows = sqlite_backend._normalize_legacy_secret_rows(expected_secrets)
            loaded_rows = sqlite_backend.list_secret_rows(include_values=True)
            expected_comp = [
                {
                    "name": row["name"],
                    "value": _decode_secret_value(row),
                    "acl_domain_key": row["acl_domain_key"],
                    "owner_id": row["owner_id"],
                    "scope": row["scope"],
                    "agent_id": row["agent_id"],
                    "swarm_id": row["swarm_id"],
                    "read": _json_loads(row["read_json"], []),
                    "write": _json_loads(row["write_json"], []),
                    "metadata": _json_loads(row["metadata_json"], {}),
                }
                for row in expected_rows
            ]
            loaded_comp = [
                {
                    "name": row["name"],
                    "value": row["value"],
                    "acl_domain_key": row["acl_domain_key"],
                    "owner_id": row["owner_id"],
                    "scope": row["scope"],
                    "agent_id": row["agent_id"],
                    "swarm_id": row["swarm_id"],
                    "read": row["read"],
                    "write": row["write"],
                    "metadata": row.get("metadata") or {},
                }
                for row in loaded_rows
            ]
            if expected_comp != loaded_comp:
                raise RuntimeError("SQLite migration verification failed: secret store mismatch")

    loaded_embeddings = sqlite_backend.load_embeddings()
    if expected_embeddings is None:
        return
    if loaded_embeddings is None:
        raise RuntimeError("SQLite migration verification failed: expected embeddings missing")
    for tier in ("gran", "cons", "cross"):
        expected = expected_embeddings.get(tier)
        got = loaded_embeddings.get(tier)
        if expected is None and got is None:
            continue
        if expected is None or got is None:
            raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")
        if len(expected) == 0 and len(got) == 0:
            continue
        if expected.shape != got.shape:
            raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")
        if not np.array_equal(expected, got):
            raise RuntimeError(f"SQLite migration verification failed: embedding mismatch for {tier}")


def migrate_jsonnpz_to_sqlite(data_dir: str, key: str) -> dict:
    enc_key = _migration_encryption_key_from_env()
    data_dir_path = Path(data_dir)
    _validate_migration_encryption_requirements(
        data_dir_path=data_dir_path,
        key=key,
        encryption_key=enc_key,
    )
    legacy = JSONNPZStorage(data_dir, key, encryption_key=enc_key)
    legacy_exists = legacy.exists
    final_path = data_dir_path / f"{key}.sqlite3"
    reuse_existing_sqlite = False
    if final_path.exists():
        sqlite_backend = SQLiteStorageBackend(data_dir, key, encryption_key=enc_key, db_path=final_path)
        try:
            existing_snapshot = sqlite_backend.load_facts(internal=True)
            existing_embeddings = sqlite_backend.load_embeddings()
        finally:
            sqlite_backend.close()

        sqlite_has_payload = any(
            [
                bool(existing_snapshot.get("granular")),
                bool(existing_snapshot.get("cons")),
                bool(existing_snapshot.get("cross")),
                bool(existing_snapshot.get("tlinks")),
                bool(existing_snapshot.get("raw_sessions")),
                bool(existing_snapshot.get("raw_docs")),
                bool(existing_snapshot.get("episode_corpus", {}).get("documents")),
                bool(existing_snapshot.get("source_records")),
                bool(existing_snapshot.get("secrets")),
            ]
        )
        sqlite_has_embeddings = existing_embeddings is not None and any(
            len(existing_embeddings.get(tier, [])) > 0 for tier in ("gran", "cons", "cross")
        )

        if not legacy_exists or sqlite_has_payload or sqlite_has_embeddings:
            return {
                "key": key,
                "sqlite_path": str(final_path),
                "migrated": False,
                "verified": True,
                "already_migrated": True,
                "embeddings": existing_embeddings is not None,
            }

        reuse_existing_sqlite = True

    if not legacy_exists:
        raise FileNotFoundError(f"Legacy JSON storage not found for key={key!r}")

    snapshot = legacy.load_facts()
    corpus_path = data_dir_path / f"{key}_corpus.json"
    if corpus_path.exists():
        snapshot["episode_corpus"] = load_episode_corpus(corpus_path, strict=False)
    embeddings = legacy.load_embeddings()

    target_path = final_path if reuse_existing_sqlite else (data_dir_path / f"{key}.sqlite3.tmp")
    if target_path != final_path and target_path.exists():
        target_path.unlink()
    sqlite_backend = SQLiteStorageBackend(data_dir, key, encryption_key=enc_key, db_path=target_path)
    try:
        sqlite_backend.save_facts(snapshot)
        if embeddings is not None:
            sqlite_backend.save_embeddings(
                embeddings.get("gran", np.zeros((0, 3072), dtype=np.float32)),
                embeddings.get("cons", np.zeros((0, 3072), dtype=np.float32)),
                embeddings.get("cross", np.zeros((0, 3072), dtype=np.float32)),
            )
        _verify_sqlite_payload(
            sqlite_backend,
            expected_snapshot=snapshot,
            expected_embeddings=embeddings,
        )
        with sqlite_backend._connect() as conn:
            sqlite_backend._set_meta(conn, "migrated_from", "jsonnpz")
            sqlite_backend._set_meta(conn, "migration_completed_at", datetime.now(timezone.utc).isoformat())
            conn.commit()
    except Exception:
        if target_path != final_path and target_path.exists():
            target_path.unlink()
        raise
    finally:
        sqlite_backend.close()

    if target_path != final_path:
        os.replace(target_path, final_path)
    for path in _legacy_storage_paths(data_dir_path, key):
        if path.exists():
            bak = path.with_name(path.name + ".bak")
            if bak.exists():
                bak.unlink()
            os.replace(path, bak)

    return {
        "key": key,
        "sqlite_path": str(final_path),
        "migrated": True,
        "embeddings": embeddings is not None,
    }


def make_storage(data_dir: str, key: str) -> StorageBackend:
    """Return the primary runtime storage backend.

    Normal runtime storage is always SQLite. Legacy JSON/NPZ access must go
    through an explicit migration/research path instead of implicit fallback.
    """
    enc_key_hex = os.environ.get("GOSH_MEMORY_ENCRYPTION_KEY")
    enc_key = bytes.fromhex(enc_key_hex) if enc_key_hex else None
    return SQLiteStorageBackend(data_dir, key, encryption_key=enc_key)
