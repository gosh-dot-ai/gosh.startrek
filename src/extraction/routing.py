# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from typing import Any

from ..object_reports import build_report, normalize_legacy_report_fields
from ..prompt_safety import render_data_block, render_kv_block
from .common import detect_format, preprocess_json_conv, resolve_session_dates
from .conversation import extract_conversation_session
from .document import extract_document_session


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


async def extract_session_via_routing(
    *,
    session_text,
    session_num,
    session_date,
    conv_id,
    speakers,
    model,
    call_extract_fn,
    prompt_bundle: dict[str, str],
    fmt=None,
    block_prompt_overrides=None,
    return_report: bool = False,
    return_diagnostics: bool | None = None,
):
    if return_diagnostics is not None:
        return_report = return_report or bool(return_diagnostics)

    if fmt is None:
        fmt = detect_format(session_text)

    if fmt == "JSON_CONV":
        parsed = preprocess_json_conv(session_text)
        if isinstance(parsed, list):
            all_facts, all_tlinks, all_report_entries = [], [], []
            fact_offset = 0
            for chunk in parsed:
                chunk_result = await extract_session_via_routing(
                    session_text=chunk,
                    session_num=session_num,
                    session_date=session_date,
                    conv_id=conv_id,
                    speakers=speakers,
                    model=model,
                    call_extract_fn=call_extract_fn,
                    prompt_bundle=prompt_bundle,
                    fmt="CONVERSATION",
                    block_prompt_overrides=block_prompt_overrides,
                    return_report=return_report,
                )
                if return_report:
                    _, _, _, chunk_facts, chunk_tlinks, chunk_report = chunk_result
                    if isinstance(chunk_report, dict):
                        all_report_entries.extend(list(chunk_report.get("entries") or []))
                else:
                    _, _, _, chunk_facts, chunk_tlinks = chunk_result
                chunk_map = {}
                for idx, fact in enumerate(chunk_facts):
                    new_id = f"f_{fact_offset + idx + 1:02d}"
                    old_id = fact.get("id", "")
                    if old_id:
                        chunk_map[old_id] = new_id
                    fact["id"] = new_id
                for link in chunk_tlinks:
                    link["before"] = chunk_map.get(link.get("before", ""), link.get("before", ""))
                    link["after"] = chunk_map.get(link.get("after", ""), link.get("after", ""))
                fact_offset += len(chunk_facts)
                all_facts.extend(chunk_facts)
                all_tlinks.extend(chunk_tlinks)
            if return_report:
                merged_report = build_report(
                    report_kind="extraction",
                    producer="extraction.routing",
                    status="partial" if all_report_entries else "ok",
                    entries=all_report_entries,
                    summary={"entry_count": len(all_report_entries)},
                )
                return conv_id, session_num, session_date, all_facts, all_tlinks, merged_report
            return conv_id, session_num, session_date, all_facts, all_tlinks
        session_text = parsed
        fmt = "CONVERSATION"

    date_str, year_minus_1 = resolve_session_dates(session_date)
    if fmt in {"AGENT_TRACE", "WEB_DOM", "GAME_BOARD", "CODE_TRACE"}:
        system_prompt = prompt_bundle["agent_trace"].format(
            episode_id=conv_id,
            domain=fmt,
            chunk_num=session_num,
            total_chunks="?",
        )
        user_msg = _render_session_payload(
            fmt=fmt,
            session_num=session_num,
            session_date=date_str,
            speakers=speakers,
            conv_id=conv_id,
            source_text=session_text,
        )
    elif fmt == "DOCUMENT":
        return await extract_document_session(
            session_text=session_text,
            session_num=session_num,
            session_date=date_str,
            conv_id=conv_id,
            speakers=speakers,
            model=model,
            call_extract_fn=call_extract_fn,
            block_prompt_overrides=block_prompt_overrides,
            return_report=return_report,
        )
    elif fmt == "FACT_LIST":
        system_prompt = prompt_bundle["fact_list"].format(session_num=session_num)
        user_msg = _render_session_payload(
            fmt=fmt,
            session_num=session_num,
            session_date=date_str,
            speakers=speakers,
            conv_id=conv_id,
            source_text=session_text,
        )
    elif fmt == "NARRATIVE":
        system_prompt = prompt_bundle["narrative"].format(
            session_date=date_str,
            session_num=session_num,
            speakers=speakers,
        )
        user_msg = _render_session_payload(
            fmt=fmt,
            session_num=session_num,
            session_date=date_str,
            speakers=speakers,
            conv_id=conv_id,
            source_text=session_text,
        )
    elif fmt == "CONVERSATION":
        return await extract_conversation_session(
            session_text=session_text,
            session_num=session_num,
            session_date=date_str,
            conv_id=conv_id,
            speakers=speakers,
            model=model,
            call_extract_fn=call_extract_fn,
            block_prompt_overrides=block_prompt_overrides,
            return_report=return_report,
        )
    else:
        system_prompt = prompt_bundle["fallback"].format(
            session_date=date_str,
            year_minus_1=year_minus_1,
            session_num=session_num,
        )
        user_msg = _render_session_payload(
            fmt=fmt,
            session_num=session_num,
            session_date=date_str,
            speakers=speakers,
            conv_id=conv_id,
            source_text=session_text,
        )

    result = await call_extract_fn(model, system_prompt, user_msg, max_tokens=8192)
    extraction_report = build_report(report_kind="extraction", producer="extraction.routing", status="ok", entries=[], summary={"entry_count": 0})
    if isinstance(result, list):
        facts = [fact for fact in result if isinstance(fact, dict)]
        temporal_links = []
    elif isinstance(result, dict):
        facts = [fact for fact in result.get("facts", []) if isinstance(fact, dict)]
        temporal_links = result.get("temporal_links", [])
        normalized_result = normalize_legacy_report_fields(result, producer="extraction.routing", report_kind="extraction")
        if isinstance(normalized_result, dict):
            normalized_extraction_report = normalized_result.get("extraction_report")
            if isinstance(normalized_extraction_report, dict):
                extraction_report = normalized_extraction_report
    else:
        facts, temporal_links = [], []
    for fact in facts:
        fact["session"] = session_num
        fact.setdefault("speaker", None)
        fact.setdefault("speaker_role", None)
        fact.setdefault("kind", "fact")
    print(f"  [{conv_id}] S{session_num}: {len(facts)} facts, {len(temporal_links)} temporal links ({fmt})")
    if return_report:
        return conv_id, session_num, session_date, facts, temporal_links, extraction_report
    return conv_id, session_num, session_date, facts, temporal_links
