# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from ..block_extractor import extract_block
from ..block_merger import merge_block_results
from ..block_segmenter import segment_document_blocks


async def extract_document_session(
    *,
    session_text: str,
    session_num: int,
    session_date: str,
    conv_id: str,
    speakers: str,
    model,
    call_extract_fn,
    block_prompt_overrides=None,
    return_report: bool = False,
    return_diagnostics: bool | None = None,
):
    if return_diagnostics is not None:
        return_report = return_report or bool(return_diagnostics)

    blocks = segment_document_blocks(session_text)
    session_metadata = {
        "container_kind": "document",
        "session_date": session_date,
        "session_num": session_num,
        "speakers": speakers,
    }
    block_results = []
    for block in blocks:
        result = await extract_block(
            block,
            session_metadata,
            model=model,
            call_extract_fn=call_extract_fn,
            prompt_overrides=block_prompt_overrides,
        )
        block_results.append((block, result))
    merged = merge_block_results(block_results, session_num)
    facts = merged["facts"]
    temporal_links = merged["temporal_links"]
    extraction_report = merged.get("extraction_report") or {}
    for fact in facts:
        fact.setdefault("speaker", None)
        fact.setdefault("speaker_role", None)
        fact.setdefault("kind", "fact")
    print(
        f"  [{conv_id}] S{session_num}: {len(facts)} facts, "
        f"{len(temporal_links)} temporal links (DOCUMENT, {len(blocks)} blocks)"
    )
    if return_report:
        return conv_id, session_num, session_date, facts, temporal_links, extraction_report
    return conv_id, session_num, session_date, facts, temporal_links
