#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    target = Path(path)
    _atomic_write_bytes(target, text.encode(encoding))
    return target


def write_json_atomic(path: str | Path, payload, *, ensure_ascii: bool = False, indent: int = 2) -> Path:
    target = Path(path)
    data = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    _atomic_write_bytes(target, data.encode("utf-8"))
    return target


def append_jsonl(path: str | Path, row: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def create_run_dir(root: str | Path, label: str) -> Path:
    root = Path(root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in label).strip("._")
    base = f"{ts}_{safe_label}" if safe_label else ts
    candidate = runs_dir / base
    idx = 1
    while candidate.exists():
        candidate = runs_dir / f"{base}_{idx:02d}"
        idx += 1
    candidate.mkdir(parents=True, exist_ok=False)
    append_jsonl(root / "run_manifest.jsonl", {"run_dir": str(candidate), "label": label, "created_utc": ts})
    return candidate


def latest_run_dir(root: str | Path) -> Path | None:
    runs_dir = Path(root) / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    return candidates[-1] if candidates else None
