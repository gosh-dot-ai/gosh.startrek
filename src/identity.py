# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import hashlib
from uuid import uuid4

from .normalizer import content_hash_normalized as _content_hash_normalized
from .normalizer import normalize_text


def _generate_artifact_id() -> str:
    """Generate a unique artifact ID."""
    return "art_" + uuid4().hex[:10]


def _generate_version_id() -> str:
    """Generate a unique version ID."""
    return "ver_" + uuid4().hex[:10]


def content_hash_text(text: str, family: str | None = None) -> str:
    """Hash text content after family-aware normalization."""
    normalized = normalize_text(text, family=family)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash_normalized(text: str) -> str:
    """Hash text with the full content-aware normalization pipeline."""
    return _content_hash_normalized(text)


def content_hash_bytes(data: bytes) -> str:
    """Hash raw bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash_git(blob_sha: str) -> str:
    """Wrap a git blob SHA as a content hash."""
    return "git:" + blob_sha
