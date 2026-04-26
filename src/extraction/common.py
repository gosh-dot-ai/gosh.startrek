# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import re
from datetime import datetime

_CONVERSATION_MARKERS = (
    re.compile(r"^\s*(user|assistant|system|tool)\s*:", re.I | re.M),
    re.compile(r"^\[D\d+:[^\]]*\]", re.I | re.M),
)
_DOCUMENT_MARKERS = (
    re.compile(r"^#{1,3}\s", re.M),
    re.compile(r"^---$", re.M),
    re.compile(r"^\*\*[A-Z][^*]+\*\*:", re.M),
)
_FACT_LIST_LINE = re.compile(r"^\s*(\d+[\.\)]\s+|[-*•]\s+)")
_NARRATIVE_SEQUENCE = re.compile(r"\b(then|after|before|later|when|eventually|meanwhile)\b", re.I)
_NARRATIVE_THIRD_PERSON = re.compile(r"\b(he|she|they|his|her|their)\b", re.I)

MAX_DOC_CHUNK_CHARS = 8000


def format_session(session_turns, session_num):
    lines = []
    for turn in session_turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", f"D{session_num}:?")
        lines.append(f"[{dia_id}] {speaker}: {text}")
    return "\n".join(lines)


def _has_conversation_markers(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CONVERSATION_MARKERS)


def _has_document_markers(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DOCUMENT_MARKERS)


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
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
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
    if "RootWebArea" in text and "focused:" in text:
        return "WEB_DOM"
    if re.search(r"\[Step \d+\]\nAction:", text):
        return "AGENT_TRACE"
    if re.search(r"^[A-Z]\|(\s[A-Z]){3,}", text, re.M):
        return "GAME_BOARD"
    if re.search(r"execute_bash|EXECUTION RESULT|^\$\s", text, re.M):
        return "CODE_TRACE"
    if text.strip().startswith("[") and ('"role"' in text[:500] or "'Chat Time:" in text[:500]):
        return "JSON_CONV"
    if _is_fact_list(text):
        return "FACT_LIST"
    if _has_document_markers(text):
        return "DOCUMENT"
    if _is_narrative(text):
        return "NARRATIVE"
    return "CONVERSATION"


def chunk_document(text: str, chunk_size: int = MAX_DOC_CHUNK_CHARS) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - 500
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def preprocess_json_conv(text: str):
    import ast
    import json as _json

    try:
        data = ast.literal_eval(text)
    except Exception:
        try:
            data = _json.loads(text)
        except Exception:
            return text

    lines = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, list):
                for msg in item:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        if content:
                            lines.append(f"{role}: {content}")
            elif isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                if content:
                    lines.append(f"{role}: {content}")
    readable = "\n".join(lines) if lines else text
    if len(readable) > MAX_DOC_CHUNK_CHARS:
        return chunk_document(readable, chunk_size=MAX_DOC_CHUNK_CHARS)
    return readable


def resolve_session_dates(session_date: str) -> tuple[str, str]:
    try:
        dt = datetime.fromisoformat(session_date.replace("Z", "+00:00"))
        return dt.strftime("%d %B %Y"), str(dt.year - 1)
    except Exception:
        return "2023", "2022"
