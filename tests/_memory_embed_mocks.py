# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import hashlib

import numpy as np


DIM = 3072


def _embed_vector(text: str) -> np.ndarray:
    """Deterministic unit-length-ish vector derived from input text."""
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def patch_memory_embeddings(monkeypatch) -> None:
    async def mock_embed_texts(texts, **kwargs):
        return np.stack([_embed_vector(str(text)) for text in texts]).astype(np.float32)

    async def mock_embed_query(text, **kwargs):
        return _embed_vector(str(text)).astype(np.float32)

    monkeypatch.setattr("src.memory.embed_texts", mock_embed_texts)
    monkeypatch.setattr("src.memory.embed_query", mock_embed_query)
