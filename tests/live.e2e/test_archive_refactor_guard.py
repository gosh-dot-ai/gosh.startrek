# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from multibench.judges.loaders import find_karnali_source, load_karnali, load_locomo, prep_kar, prep_loco

from tests._archive_repo import archive_repo_root
from tests._live_provider_stack import LiveProviderStack


pytestmark = pytest.mark.e2e

LOCOMO_INFERENCE_MODEL = "qwen/qwen3-32b"
KARNALI_INFERENCE_MODEL = "qwen/qwen3-32b"


def _sprint_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"sprint-(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def _result_candidates(root: Path, pattern: str) -> list[Path]:
    return sorted(
        root.glob(pattern),
        key=lambda path: (_sprint_number(path), path.as_posix()),
        reverse=True,
    )


def _karnali_result_recency_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"results_t(\d+)v(\d+)", path.stem)
    if match:
        return (int(match.group(1)), int(match.group(2)), path.as_posix())
    return (-1, -1, path.as_posix())


def _collapse(text: str) -> str:
    cleaned = re.sub(r"[*_`#]+", " ", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _lexical_variants(token: str) -> set[str]:
    word = str(token or "").strip().lower()
    if not word:
        return set()
    variants = {word}
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        variants.add(stem)
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
        if not stem.endswith("e"):
            variants.add(stem + "e")
    if word.endswith("es") and len(word) > 3:
        variants.add(word[:-2])
        variants.add(word[:-1])
    if word.endswith("s") and len(word) > 3:
        variants.add(word[:-1])
    return {variant for variant in variants if variant}


def _answer_matches_gold(answer: str, gold: str) -> bool:
    answer_norm = _collapse(answer)
    gold_norm = _collapse(gold)
    if not answer_norm or not gold_norm:
        return False
    if gold_norm.startswith("by "):
        gold_tail = gold_norm[3:].strip()
        if gold_tail in answer_norm:
            return True
        gold_tokens = re.findall(r"[a-z0-9']+", gold_tail)
        answer_tokens = set(re.findall(r"[a-z0-9']+", answer_norm))
        if len(gold_tokens) == 1 and _lexical_variants(gold_tokens[0]) & answer_tokens:
            return True
    if gold_norm in {"yes", "no"}:
        return bool(re.search(rf"\b{re.escape(gold_norm)}\b", answer_norm))
    gold_numbers = re.findall(r"\d+(?:\.\d+)?", gold_norm)
    if gold_numbers and all(number in answer_norm for number in gold_numbers):
        return True
    gold_without_parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", gold_norm).strip()
    if gold_without_parenthetical and gold_without_parenthetical != gold_norm:
        if gold_without_parenthetical in answer_norm:
            return True
    if gold_norm in answer_norm:
        return True
    if gold_norm.replace("litres per second", "l/s") in answer_norm:
        return True
    if gold_norm.replace("liters per second", "l/s") in answer_norm:
        return True
    return False


def _gold_complexity(gold: str) -> tuple[int, int]:
    normalized = _collapse(gold)
    return (len(normalized.split()), len(normalized))


def _select_latest_usable_locomo_case() -> dict:
    root = archive_repo_root()
    dataset = load_locomo()
    for path in _result_candidates(root, "validation/sprint-*/results/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        usable_cases: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not row.get("judge_correct"):
                continue
            if not {"conv_id", "qa_idx", "category", "question", "gold"} <= set(row):
                continue
            gold = str(row.get("gold") or "").strip()
            if len(_collapse(gold)) > 80:
                continue
            conv_id = str(row["conv_id"])
            conv = dataset.get(conv_id)
            if not conv:
                continue
            prepared = prep_loco(conv, int(row["category"]))
            if prepared is None:
                continue
            sessions, dates, speakers, question, prepared_gold, category = prepared
            if _collapse(question) != _collapse(str(row["question"])):
                continue
            if not _answer_matches_gold(prepared_gold, gold) and not _answer_matches_gold(gold, prepared_gold):
                continue
            usable_cases.append({
                "results_path": path,
                "conv_id": conv_id,
                "qa_idx": int(row["qa_idx"]),
                "category": int(category),
                "question": question,
                "gold": prepared_gold,
                "sessions": sessions,
                "dates": dates,
                "speakers": speakers,
            })
        if usable_cases:
            return min(
                usable_cases,
                key=lambda case: (
                    len(case["sessions"]),
                    sum(len(session) for session in case["sessions"]),
                    _gold_complexity(case["gold"]),
                    case["qa_idx"],
                    case["conv_id"],
                ),
            )
    raise AssertionError("No usable LoCoMo archive result with recoverable live case was found")


def _artifact_text(artifact: dict) -> str:
    parts = []
    for key in ("title", "summary", "body", "content", "text"):
        value = artifact.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts).strip()


def _chunk_document_text(text: str, size: int = 4000) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    return [text[idx: idx + size] for idx in range(0, len(text), size) if text[idx: idx + size].strip()]


def _build_karnali_document_text(query: dict, docs: dict, arts: dict) -> str:
    sections: list[str] = []
    for document_id in query.get("source_documents", []):
        document = find_karnali_source(docs, document_id)
        text = str(document.get("content") or document.get("text") or "").strip()
        if not text:
            continue
        sections.extend([f"[Document {document_id}]", text, ""])
    for artifact_id in query.get("source_artifacts", []):
        artifact = arts.get(artifact_id, {})
        text = _artifact_text(artifact)
        if not text:
            continue
        sections.extend([f"[Artifact {artifact_id}]", text, ""])
    return "\n".join(sections).strip()


def _select_latest_usable_karnali_case() -> dict:
    root = archive_repo_root()
    queries, convs, docs, arts = load_karnali()
    candidates = sorted(
        root.glob("karnali/results/results_*.jsonl"),
        key=_karnali_result_recency_key,
        reverse=True,
    )
    for path in candidates:
        usable_cases: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                query_id = str(row.get("query_id") or "").strip()
                if not query_id or not row.get("is_correct"):
                    continue
                if not {"query", "ground_truth"} <= set(row):
                    continue
                gold = str(row.get("ground_truth") or "").strip()
                if not re.search(r"\d", gold):
                    continue
                query = queries.get(query_id)
                if not query:
                    continue
                prepared = prep_kar(query, convs, docs, arts, _chunk_document_text)
                if prepared is None:
                    continue
                _sessions, _dates, _speakers, prepared_question, prepared_gold, _category, _canary = prepared
                if _collapse(prepared_question) != _collapse(str(row["query"])):
                    continue
                if not _answer_matches_gold(prepared_gold, gold) and not _answer_matches_gold(gold, prepared_gold):
                    continue
                if not query.get("source_documents") and not query.get("source_artifacts"):
                    continue
                document_text = _build_karnali_document_text(query, docs, arts)
                if not document_text:
                    continue
                usable_cases.append({
                    "results_path": path,
                    "query_id": query_id,
                    "question": prepared_question,
                    "gold": prepared_gold,
                    "document_text": document_text,
                })
        if usable_cases:
            return min(
                usable_cases,
                key=lambda case: (
                    len(case["document_text"]),
                    _gold_complexity(case["gold"]),
                    case["query_id"],
                ),
            )
    raise AssertionError("No usable Karnali archive result with recoverable document source was found")


def _build_index(stack: LiveProviderStack, *, key: str, agent_id: str) -> dict:
    return stack.agent_mcp(agent_id, "memory_build_index", {"key": key, "agent_id": agent_id})


def test_locomo_latest_archive_conversation_e2e_qwen(live_provider_stack: LiveProviderStack) -> None:
    case = _select_latest_usable_locomo_case()
    key = live_provider_stack.unique_key("locomo_live_guard")
    agent_id = "alice"
    live_provider_stack.set_config(key, agent_id=agent_id)

    for idx, session_text in enumerate(case["sessions"], start=1):
        stored = live_provider_stack.agent_mcp(
            agent_id,
            "memory_store",
            {
                "key": key,
                "agent_id": agent_id,
                "scope": "agent-private",
                "session_num": idx,
                "session_date": str(case["dates"][idx - 1] or ""),
                "speakers": case["speakers"],
                "content": session_text,
            },
        )
        assert stored["status"] == "ok", stored

    build = _build_index(live_provider_stack, key=key, agent_id=agent_id)
    assert build.get("granular", 0) > 0, build

    result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ask",
        {
            "key": key,
            "agent_id": agent_id,
            "query": case["question"],
            "inference_model": LOCOMO_INFERENCE_MODEL,
        },
    )

    assert _answer_matches_gold(result["answer"], case["gold"]), result
    assert result["retrieval_families"] == ["conversation"], result


def test_karnali_latest_archive_document_e2e_qwen(live_provider_stack: LiveProviderStack) -> None:
    case = _select_latest_usable_karnali_case()
    key = live_provider_stack.unique_key("karnali_live_guard")
    agent_id = "alice"
    live_provider_stack.set_config(key, agent_id=agent_id)

    ingested = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ingest_document",
        {
            "key": key,
            "agent_id": agent_id,
            "scope": "agent-private",
            "source_id": case["query_id"],
            "content": case["document_text"],
        },
    )
    assert ingested["status"] == "ok", ingested
    assert ingested["facts_extracted"] > 0, ingested

    build = _build_index(live_provider_stack, key=key, agent_id=agent_id)
    assert build.get("granular", 0) > 0, build

    result = live_provider_stack.agent_mcp(
        agent_id,
        "memory_ask",
        {
            "key": key,
            "agent_id": agent_id,
            "query": case["question"],
            "inference_model": KARNALI_INFERENCE_MODEL,
        },
    )

    assert _answer_matches_gold(result["answer"], case["gold"]), result
    assert result["retrieval_families"] == ["document"], result
