#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import re
from collections import defaultdict
from pathlib import Path as _Path

from .extraction import (
    MAX_DOC_CHUNK_CHARS,
    extract_session_via_routing,
)
from .extraction import (
    chunk_document as _chunk_document_impl,
)
from .extraction import (
    detect_format as _detect_format_impl,
)
from .extraction import (
    format_session as _format_session_impl,
)
from .extraction import (
    preprocess_json_conv as _preprocess_json_conv_impl,
)
from .object_reports import normalize_legacy_report_fields
from .prompt_safety import render_data_block, render_kv_block

# ── Prompt loader ──

_PROMPT_DIR = _Path(__file__).parent / "prompts" / "extraction"


def _load_extraction_prompt(name: str) -> str:
    """Load extraction prompt from .md file."""
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Extraction prompt '{name}' not found at {path}. "
            f"Expected src/prompts/extraction/{name}.md"
        )
    return path.read_text(encoding="utf-8")


# ── Prompts (loaded from src/prompts/extraction/*.md) ──

EXTRACTION_PROMPT_FALLBACK = _load_extraction_prompt("legacy")
EXTRACTION_PROMPT = EXTRACTION_PROMPT_FALLBACK
EXTRACTION_PROMPT_CONVERSATION = _load_extraction_prompt("conversation")
EXTRACTION_PROMPT_AGENT_TRACE = _load_extraction_prompt("agent_trace")
EXTRACTION_PROMPT_DOCUMENT = _load_extraction_prompt("document")
EXTRACTION_PROMPT_FACT_LIST = _load_extraction_prompt("fact_list")
EXTRACTION_PROMPT_NARRATIVE = _load_extraction_prompt("narrative")
ENGLISH_CANONICAL_SOURCE_PROMPT = _load_extraction_prompt("english_canonical_source")
ENGLISH_CANONICAL_QUERY_PROMPT = _load_extraction_prompt("english_canonical_query")

ENGLISH_CANONICAL_TRANSLATION_VERSION = "english_canonical_v1"

SUPPORTED_EXTRACTION_FORMATS = frozenset({
    "CONVERSATION",
    "DOCUMENT",
    "AGENT_TRACE",
    "JSON_CONV",
    "WEB_DOM",
    "GAME_BOARD",
    "CODE_TRACE",
    "FACT_LIST",
    "NARRATIVE",
})

_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_NON_LATIN_SCRIPT_RANGES = (
    (0x0400, 0x04FF),  # Cyrillic
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0x0590, 0x05FF),  # Hebrew
    (0x0370, 0x03FF),  # Greek
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
    (0x0530, 0x058F),  # Armenian
    (0x10A0, 0x10FF),  # Georgian
    (0x2D00, 0x2D2F),  # Georgian Supplement
)
_GREEK_RANGES = (
    (0x0370, 0x03FF),
)
def _is_greek_alpha(ch: str) -> bool:
    if not ch or not ch.isalpha():
        return False
    codepoint = ord(ch)
    return any(start <= codepoint <= end for start, end in _GREEK_RANGES)


def _is_isolated_greek_symbol(text: str, idx: int) -> bool:
    ch = text[idx]
    if not _is_greek_alpha(ch):
        return False
    prev_is_greek_alpha = idx > 0 and _is_greek_alpha(text[idx - 1])
    next_is_greek_alpha = idx + 1 < len(text) and _is_greek_alpha(text[idx + 1])
    return not prev_is_greek_alpha and not next_is_greek_alpha


def _contains_obvious_non_latin_script(text: str) -> bool:
    for idx, ch in enumerate(text):
        if not ch.isalpha():
            continue
        if _is_isolated_greek_symbol(text, idx):
            continue
        codepoint = ord(ch)
        for start, end in _NON_LATIN_SCRIPT_RANGES:
            if start <= codepoint <= end:
                return True
    return False


def _is_ascii_english_with_ignorable_symbols(text: str) -> bool:
    if not _ASCII_ALPHA_RE.search(text):
        return False
    for idx, ch in enumerate(text):
        if not ch.isalpha():
            continue
        if ord(ch) < 128:
            continue
        if _is_isolated_greek_symbol(text, idx):
            continue
        return False
    return True


def detect_source_language(text: str) -> str:
    """Best-effort language gate for English-only semantic canonicalization.

    Safe local fast-path only:
    - no text / no alphabetic chars -> und
    - all alphabetic chars ASCII -> en
    - known non-Latin scripts -> non_en
    - Latin text with any non-ASCII alphabetic chars -> ambiguous
    """
    raw = str(text or "").strip()
    if not raw:
        return "und"
    alpha_chars = [ch for ch in raw if ch.isalpha()]
    if not alpha_chars:
        return "und"
    if _contains_obvious_non_latin_script(raw):
        return "non_en"
    if _is_ascii_english_with_ignorable_symbols(raw):
        return "en"
    return "ambiguous"


def needs_english_canonicalization(text: str) -> bool:
    """Return True when the text should be normalized into canonical English."""
    return detect_source_language(text) not in {"en", "und"}


def _normalized_canonical_result(
    original_text: str,
    source_lang: str,
    canonical_text: str | None,
) -> dict[str, str | bool | None]:
    canonical_en = str(canonical_text or "").strip() or original_text
    return {
        "raw_original": original_text,
        "source_lang": source_lang or "und",
        "canonical_en": canonical_en,
        "semantic_ready": True,
        "canonicalization_status": "ready",
        "canonicalization_error": None,
        "translation_version": ENGLISH_CANONICAL_TRANSLATION_VERSION,
    }


def _failed_canonical_result(
    original_text: str,
    source_lang: str,
    error: str,
) -> dict[str, str | bool | None]:
    return {
        "raw_original": original_text,
        "source_lang": source_lang or "und",
        "canonical_en": "",
        "semantic_ready": False,
        "canonicalization_status": "failed",
        "canonicalization_error": error,
        "translation_version": ENGLISH_CANONICAL_TRANSLATION_VERSION,
    }


def _validated_english_canonical_text(text: str | None) -> str | None:
    canonical = re.sub(r"\s+", " ", str(text or "")).strip()
    if not canonical:
        return None
    if _contains_obvious_non_latin_script(canonical):
        return None
    lang = detect_source_language(canonical)
    if lang in {"en", "und", "ambiguous"}:
        return canonical
    return None


def _english_like_raw_fallback(text: str | None) -> str | None:
    """Accept raw text when it is overwhelmingly ASCII-English already.

    This protects large English documents from failing ingest because the
    canonicalization model returned malformed JSON/fields. The fallback stays
    conservative: we only trust the raw source when ASCII alphabetic content
    massively dominates non-ASCII alphabetic content.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    ascii_alpha = 0
    non_ascii_alpha = 0
    for ch in raw:
        if not ch.isalpha():
            continue
        if ch.isascii():
            ascii_alpha += 1
        else:
            non_ascii_alpha += 1
    if ascii_alpha < 128:
        return None
    if ascii_alpha < max(1, non_ascii_alpha) * 20:
        return None
    return raw


def _render_source_text_payload(text: str) -> str:
    return render_data_block("SOURCE_TEXT", text)


def _render_query_text_payload(text: str) -> str:
    return render_data_block("QUERY_TEXT", text)


def _render_session_payload(
    *,
    fmt: str,
    session_num: int,
    session_date: str,
    speakers: str,
    conv_id: str,
    source_text: str,
) -> str:
    metadata = {
        "format": fmt,
        "session_num": session_num,
        "session_date": session_date or "unknown",
        "speakers": speakers or "unknown",
        "conversation_id": conv_id or "unknown",
    }
    return "\n\n".join(
        [
            render_kv_block("SESSION_METADATA", metadata),
            render_data_block("SOURCE_TEXT", source_text),
        ]
    )


async def canonicalize_source_to_english(
    text: str,
    *,
    model: str | None,
    call_extract_fn,
) -> dict[str, str | bool | None]:
    """Produce the canonical English semantic representation for one source."""
    original_text = str(text or "")
    source_lang = detect_source_language(original_text)
    if not original_text.strip() or not needs_english_canonicalization(original_text):
        return _normalized_canonical_result(original_text, source_lang, original_text)
    fallback = _english_like_raw_fallback(original_text)
    if fallback is not None:
        return _normalized_canonical_result(original_text, "en", fallback)
    if model in (None, ""):
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english canonicalization model unavailable",
        )

    try:
        result = await call_extract_fn(
            model,
            ENGLISH_CANONICAL_SOURCE_PROMPT,
            _render_source_text_payload(original_text),
            max_tokens=4096,
        )
    except Exception as exc:
        return _failed_canonical_result(
            original_text,
            source_lang,
            f"english source canonicalization failed: {exc.__class__.__name__}",
        )

    if not isinstance(result, dict):
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english source canonicalization returned non-object result",
        )

    canonical_en = _validated_english_canonical_text(result.get("canonical_en"))
    if canonical_en is None:
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english source canonicalization returned invalid canonical_en",
        )

    return _normalized_canonical_result(
        original_text,
        str(result.get("source_lang") or source_lang or "und").strip().lower(),
        canonical_en,
    )


async def canonicalize_query_to_english(
    text: str,
    *,
    model: str | None,
    call_extract_fn,
) -> dict[str, str | bool | None]:
    """Normalize a retrieval query into canonical English."""
    original_text = str(text or "").strip()
    source_lang = detect_source_language(original_text)
    if not original_text or not needs_english_canonicalization(original_text):
        return _normalized_canonical_result(original_text, source_lang, original_text)
    if model in (None, ""):
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english query canonicalization model unavailable",
        )

    try:
        result = await call_extract_fn(
            model,
            ENGLISH_CANONICAL_QUERY_PROMPT,
            _render_query_text_payload(original_text),
            max_tokens=1024,
        )
    except Exception as exc:
        return _failed_canonical_result(
            original_text,
            source_lang,
            f"english query canonicalization failed: {exc.__class__.__name__}",
        )

    if not isinstance(result, dict):
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english query canonicalization returned non-object result",
        )

    canonical_en = _validated_english_canonical_text(result.get("canonical_en"))
    if canonical_en is None:
        return _failed_canonical_result(
            original_text,
            source_lang,
            "english query canonicalization returned invalid canonical_en",
        )

    return _normalized_canonical_result(
        original_text,
        str(result.get("source_lang") or source_lang or "und").strip().lower(),
        canonical_en,
    )


def normalize_content_format(fmt: str | None) -> str | None:
    """Normalize and validate an explicit extraction format override."""
    if fmt is None:
        return None
    normalized = str(fmt).strip().upper()
    if not normalized:
        raise ValueError("content_format must be non-empty when provided")
    if normalized not in SUPPORTED_EXTRACTION_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_EXTRACTION_FORMATS))
        raise ValueError(
            f"unsupported content_format: {fmt!r}. Supported formats: {supported}"
        )
    return normalized


def format_session(session_turns, session_num):
    """Format conversation turns into readable text."""
    return _format_session_impl(session_turns, session_num)


# ── Extraction ──

async def extract_session(session_text, session_num, session_date, conv_id,
                          speakers, model, call_extract_fn, fmt=None,
                          block_prompt_overrides=None, return_report: bool = False,
                          return_diagnostics: bool | None = None):
    """Extract atomic facts + temporal links from a single session.

    Format-aware prompt selection.  When *fmt* is ``None`` the format is
    auto-detected via :func:`detect_format`.

    Args:
        call_extract_fn: async fn(model, system, user_msg, max_tokens) -> dict
        fmt: explicit format override (None = auto-detect)
    Returns:
        (conv_id, session_num, session_date, facts, temporal_links)
    """
    result = await extract_session_via_routing(
        session_text=session_text,
        session_num=session_num,
        session_date=session_date,
        conv_id=conv_id,
        speakers=speakers,
        model=model,
        call_extract_fn=call_extract_fn,
        fmt=fmt,
        block_prompt_overrides=block_prompt_overrides,
        return_report=return_report,
        return_diagnostics=return_diagnostics,
        prompt_bundle={
            "fallback": EXTRACTION_PROMPT_FALLBACK,
            "conversation": EXTRACTION_PROMPT_CONVERSATION,
            "agent_trace": EXTRACTION_PROMPT_AGENT_TRACE,
            "document": EXTRACTION_PROMPT_DOCUMENT,
            "fact_list": EXTRACTION_PROMPT_FACT_LIST,
            "narrative": EXTRACTION_PROMPT_NARRATIVE,
        },
    )
    if isinstance(result, (tuple, list)) and len(result) == 6:
        conv_id, session_num, session_date, facts, temporal_links, report = result
        if isinstance(report, list):
            report = normalize_legacy_report_fields(
                {"diagnostics": report},
                producer="librarian",
                report_kind="extraction",
            )
        elif isinstance(report, dict):
            report = normalize_legacy_report_fields(
                report,
                producer="librarian",
                report_kind="extraction",
            ) or report
        return conv_id, session_num, session_date, facts, temporal_links, report
    return result

# ── Supersession Resolution ──

def resolve_supersession(all_facts, fact_lookup):
    """Link facts with supersedes_topic to existing facts via text overlap.

    For each new fact with supersedes_topic set, find the best matching
    existing fact and create bidirectional links:
      new_fact["supersedes"] = old_fact_id
      old_fact["status"] = "superseded"
      old_fact["superseded_by"] = new_fact_id

    Uses simple token overlap (no external deps). Called after all
    extraction tiers are built and fact_lookup is populated.
    """
    # Index facts by entity for fast lookup
    from collections import defaultdict
    entity_index = defaultdict(list)
    for f in all_facts:
        for e in f.get("entities", []):
            if isinstance(e, str):
                entity_index[e.lower()].append(f)

    def _tokenize(text):
        return set(text.lower().split())

    linked = 0
    for f in all_facts:
        topic = f.get("supersedes_topic")
        if not topic:
            continue
        # Guard: LLM sometimes returns list instead of string
        if isinstance(topic, list):
            topic = " ".join(str(t) for t in topic)
        if not isinstance(topic, str):
            continue

        f_id = f.get("id")
        if not f_id:
            continue

        topic_tokens = _tokenize(topic)
        # Candidates: facts sharing at least one entity, from earlier sessions
        f_session = f.get("session", 999)
        candidates = []
        for e in f.get("entities", []):
            for c in entity_index.get(e.lower(), []):
                c_id = c.get("id")
                if c_id and c_id != f_id and c.get("session", 999) < f_session:
                    candidates.append(c)

        if not candidates:
            # Broaden: any fact with topic token overlap
            for c in all_facts:
                c_id = c.get("id")
                if not c_id or c_id == f_id:
                    continue
                if c.get("session", 999) >= f_session:
                    continue
                fact_tokens = _tokenize(c.get("fact", ""))
                if topic_tokens & fact_tokens:
                    candidates.append(c)

        if not candidates:
            continue

        # Score by token overlap with supersedes_topic
        best, best_score = None, 0
        for c in candidates:
            fact_tokens = _tokenize(c.get("fact", ""))
            score = len(topic_tokens & fact_tokens)
            if score > best_score:
                best, best_score = c, score

        if best and best_score > 0:
            f["supersedes"] = best.get("id", "")
            best["status"] = "superseded"
            best["superseded_by"] = f_id
            # Phase 1B: mark superseded fact as outdated
            meta = best.setdefault("metadata", {})
            meta["version_status"] = "outdated"
            meta["version_superseded_by"] = f.get("supersedes_topic", "")
            linked += 1

    return linked


# ═══════════════════════════════════════════════════════════════════════════
# Format Detection (deterministic)
# Moved from multibench/sprint23d/run_23d.py and extended with prompt-aware
# FACT_LIST / NARRATIVE routing.
# ═══════════════════════════════════════════════════════════════════════════

_CONVERSATION_MARKERS = (
    re.compile(r'^\s*(user|assistant|system|tool)\s*:', re.I | re.M),
    re.compile(r'^\[D\d+:[^\]]*\]', re.I | re.M),
)
_DOCUMENT_MARKERS = (
    re.compile(r'^#{1,3}\s', re.M),
    re.compile(r'^---$', re.M),
    re.compile(r'^\*\*[A-Z][^*]+\*\*:', re.M),
)
_FACT_LIST_LINE = re.compile(r'^\s*(\d+[\.\)]\s+|[-*•]\s+)')
_NARRATIVE_SEQUENCE = re.compile(
    r'\b(then|after|before|later|when|eventually|meanwhile)\b', re.I)
_NARRATIVE_THIRD_PERSON = re.compile(
    r'\b(he|she|they|his|her|their)\b', re.I)


def _has_conversation_markers(text: str) -> bool:
    return any(p.search(text) for p in _CONVERSATION_MARKERS)


def _has_document_markers(text: str) -> bool:
    return any(p.search(text) for p in _DOCUMENT_MARKERS)


def _is_fact_list(text: str) -> bool:
    non_empty = [line for line in text.splitlines() if line.strip()]
    if len(non_empty) < 4:
        return False
    if _has_conversation_markers(text) or _has_document_markers(text):
        return False
    numbered = sum(1 for line in non_empty if _FACT_LIST_LINE.match(line))
    return (numbered / len(non_empty)) > 0.6


def _is_narrative(text: str) -> bool:
    if _has_conversation_markers(text) or _has_document_markers(text):
        return False
    if _is_fact_list(text):
        return False
    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(paragraphs) < 2 and len(text) <= 800:
        return False
    narrative_signals = sum([
        bool(re.search(r'["“”]', text)),
        bool(_NARRATIVE_SEQUENCE.search(text)),
        bool(_NARRATIVE_THIRD_PERSON.search(text)),
    ])
    return narrative_signals >= 2

def detect_format(text: str) -> str:
    """Deterministic format detection for session text.

    Returns one of:
        CONVERSATION, DOCUMENT, AGENT_TRACE, JSON_CONV,
        WEB_DOM, GAME_BOARD, CODE_TRACE, FACT_LIST, NARRATIVE
    """
    return _detect_format_impl(text)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_json_conv(text: str):
    """Convert raw JSON conversation string to readable text.

    Handles both Python literal (ast.literal_eval) and standard JSON formats.
    Supports flat dicts, nested lists of dicts, and mixed string/list items.
    """
    return _preprocess_json_conv_impl(text)


def _chunk_document(text: str, chunk_size: int = MAX_DOC_CHUNK_CHARS) -> list:
    """Split long document into overlapping chunks.

    Uses a 500-character overlap to avoid losing context at chunk boundaries.
    Returns a list of non-empty chunk strings.
    """
    return _chunk_document_impl(text, chunk_size=chunk_size)

# ═══════════════════════════════════════════════════════════════════════════
# 3-tier decision helper
# ═══════════════════════════════════════════════════════════════════════════

def _needs_3tier(sessions: list) -> bool:
    """Determine if 3-tier indexing helps or hurts.

    3-tier helps: haystack data (many sessions, multi-conversation)
    3-tier hurts: single conversation data (few sessions, same speakers)

    Threshold from EV-14/EV-16: 3-tier helps when N >= 10 sessions.
    Below 10 sessions, granular-only retrieval performs better.
    """
    return len(sessions) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# L1 Classification (lightweight metadata enrichment for pre-extracted facts)
# ═══════════════════════════════════════════════════════════════════════════

async def classify_fact(text: str, model: str, call_extract_fn) -> dict:
    """Classify a single fact text into kind/entities/tags via LLM.

    Uses the classify.md prompt. Returns dict with kind, entities, tags,
    event_date, supersedes_topic.  Returns {} on any error.
    """
    prompt = _load_extraction_prompt("classify")
    system = prompt
    try:
        result = await call_extract_fn(
            model,
            system,
            _render_source_text_payload(text),
            max_tokens=256,
        )
        valid_kinds = {"fact", "rule", "constraint", "decision", "lesson_learned",
                       "preference", "count_item", "action_item", "observation"}
        if result.get("kind") not in valid_kinds:
            result["kind"] = "fact"
        if not isinstance(result.get("entities"), list):
            result["entities"] = []
        if not isinstance(result.get("tags"), list):
            result["tags"] = []
        return result
    except Exception:
        return {}


def merge_l1_metadata(fact: dict, metadata: dict) -> None:
    """Merge L1 classification metadata into an existing fact dict.

    Only overwrites fields that are missing or at default values in the fact.
    """
    if not metadata:
        return
    if metadata.get("kind", "fact") != "fact" and fact.get("kind", "fact") == "fact":
        fact["kind"] = metadata["kind"]
    if metadata.get("entities") and not fact.get("entities"):
        fact["entities"] = metadata["entities"]
    if metadata.get("tags") and not fact.get("tags"):
        fact["tags"] = metadata["tags"]
    if metadata.get("event_date") and not fact.get("event_date"):
        fact["event_date"] = metadata["event_date"]
    if metadata.get("supersedes_topic") and not fact.get("supersedes_topic"):
        fact["supersedes_topic"] = metadata["supersedes_topic"]
