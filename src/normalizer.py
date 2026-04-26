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
import re
import unicodedata

CONTENT_AWARE_FAMILIES = {"conversation", "document"}
_CONTENT_AWARE_ALIASES = {"chat": "conversation"}

_SAFE_INVISIBLES = {
    "\u200b",  # zero-width space
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
    "\u200e",  # LTR mark
    "\u200f",  # RTL mark
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi embedding/override
    "\u2066", "\u2067", "\u2068", "\u2069",  # bidi isolates
}

_SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2026": "...",
    }
)


def _family_key(family: str | None) -> str | None:
    if family is None:
        return None
    normalized = str(family).strip().lower()
    return _CONTENT_AWARE_ALIASES.get(normalized, normalized)


def _cleanup_encoding(raw: str) -> str:
    cleaned = raw.lstrip("\ufeff")
    return "".join(
        "\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch
        for ch in cleaned
    )


def _remove_safe_invisibles(text: str) -> str:
    if not text:
        return text
    return "".join(ch for ch in text if ch not in _SAFE_INVISIBLES)


def _strip_trailing_whitespace_preserving_fences(text: str) -> str:
    lines = text.split("\n")
    in_fence = False
    normalized_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            normalized_lines.append(line)
            in_fence = not in_fence
            continue
        normalized_lines.append(line if in_fence else line.rstrip())
    return "\n".join(normalized_lines)


def _compact_blank_lines_preserving_fences(text: str) -> str:
    lines = text.split("\n")
    in_fence = False
    blank_run = 0
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if line.lstrip().startswith("```"):
            blank_run = 0
            output.append(line)
            in_fence = not in_fence
            continue
        if in_fence:
            output.append(line)
            continue
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                output.append("")
            continue
        blank_run = 0
        output.append(line)
    while output and output[0].strip() == "":
        output.pop(0)
    while output and output[-1].strip() == "":
        output.pop()
    return "\n".join(output)


def _normalize_smart_punctuation(text: str) -> str:
    if not text:
        return text
    normalized = text.translate(_SMART_PUNCT_TRANSLATION)
    normalized = re.sub(r"\s*—\s*", " -- ", normalized)
    normalized = re.sub(r"(?<!\d)–|–(?!\d)", "-", normalized)
    return normalized


def normalize_text(raw: str, family: str | None = None) -> str:
    """Normalize text deterministically.

    Universal steps run for all families. Content-aware steps run only for
    conversation/document inputs; ``chat`` is treated as conversation.
    """
    text = _cleanup_encoding(str(raw or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _remove_safe_invisibles(text)

    family_key = _family_key(family)
    if family_key not in CONTENT_AWARE_FAMILIES:
        return text

    text = _strip_trailing_whitespace_preserving_fences(text)
    text = _compact_blank_lines_preserving_fences(text)
    text = _normalize_smart_punctuation(text)
    return text


def content_hash_normalized(text: str) -> str:
    """Hash text with the full content-aware normalization pipeline."""
    normalized = normalize_text(text, family="conversation")
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_ngrams(text: str, ngram: int) -> list[str]:
    tokens = [token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if token]
    if not tokens:
        fallback = text.strip().lower()
        return [fallback] if fallback else []
    if len(tokens) < max(1, int(ngram)):
        return [" ".join(tokens)]
    n = max(1, int(ngram))
    return [" ".join(tokens[idx: idx + n]) for idx in range(len(tokens) - n + 1)]


def simhash(text: str, ngram: int = 3) -> int:
    """Compute a deterministic 64-bit SimHash over word n-grams."""
    features = _word_ngrams(text, ngram)
    if not features:
        return 0
    weights = [0] * 64
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        for bit in range(64):
            mask = 1 << bit
            weights[bit] += 1 if value & mask else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two 64-bit integers."""
    return int(a ^ b).bit_count()


def dedup_domain_key(scope: str, owner_id: str | None, swarm_id: str | None) -> str:
    """Compute the ACL-local dedup domain key."""
    normalized_scope = str(scope or "").strip().lower()
    if normalized_scope == "system-wide":
        return "system"
    if normalized_scope == "swarm-shared":
        return f"swarm:{swarm_id or 'default'}"
    if normalized_scope == "agent-private":
        return str(owner_id or "system")
    if owner_id:
        return str(owner_id)
    if swarm_id:
        return f"swarm:{swarm_id}"
    return "system"


def acl_domain_key(
    owner_id: str | None,
    read: list[str] | None,
    write: list[str] | None,
) -> str:
    """Compute a stable domain key from the canonical ACL triple only."""
    payload = {
        "owner_id": str(owner_id or "system"),
        "read": sorted(str(item) for item in (read or [])),
        "write": sorted(str(item) for item in (write or [])),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"acl:{digest}"
