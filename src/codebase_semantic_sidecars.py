#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic


def _sanitize_ref_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-._/" else "_" for ch in value)
    token = token.strip("./")
    return token or "unknown"


def _payload_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class CodebaseSemanticSidecarStore:
    """Persist semantic sidecars inside the normal memory data dir."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "codebase_semantic_sidecars"
        self.root.mkdir(parents=True, exist_ok=True)

    def persist_sidecar(self, sidecar: dict[str, Any]) -> dict[str, Any]:
        payload = sidecar.get("payload")
        if payload is None:
            raise ValueError(f"sidecar {sidecar.get('sidecar_id', '?')} missing payload")
        raw = _payload_bytes(payload)
        content_hash = hashlib.sha256(raw).hexdigest()
        repo_token = _sanitize_ref_token(str(sidecar.get("repo_id") or "repo"))
        revision_token = _sanitize_ref_token(str(sidecar.get("revision") or "revision"))
        sidecar_id = str(sidecar.get("sidecar_id") or "sidecar")
        rel_path = Path(repo_token) / revision_token / f"{sidecar_id}.json"
        abs_path = self.root / rel_path
        write_json_atomic(abs_path, payload, ensure_ascii=False, indent=2)

        persisted = deepcopy(sidecar)
        persisted["storage_ref"] = str(abs_path.relative_to(self.data_dir))
        persisted["content_hash"] = content_hash
        persisted["byte_size"] = len(raw)
        persisted.pop("payload", None)
        return persisted

    def hydrate_sidecar(self, sidecar_ref: dict[str, Any]) -> dict[str, Any]:
        storage_ref = str(sidecar_ref.get("storage_ref") or "").strip()
        if not storage_ref:
            raise ValueError("sidecar_ref.storage_ref is required")
        path = self.data_dir / storage_ref
        if not path.exists():
            raise FileNotFoundError(f"sidecar payload not found: {storage_ref}")
        return json.loads(path.read_text(encoding="utf-8"))
