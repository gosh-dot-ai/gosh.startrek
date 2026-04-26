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

import copy
from typing import Any

from .block_segmenter import Block
from .object_reports import build_report, normalize_legacy_report_fields


def merge_block_results(
    block_results: list[tuple[Block, dict]],
    session_num: int,
) -> dict:
    """Merge block extraction results into session-level output.

    Args:
        block_results: list of (Block, extraction_result) tuples,
            ordered by block.order.
        session_num: session number to set on all facts.

    Returns:
        dict with "facts" and "temporal_links" matching downstream schema.
    """
    block_results = sorted(block_results, key=lambda br: br[0].order)

    all_facts = []
    all_tlinks = []
    all_report_entries = []
    fact_counter = 0

    for block, result in block_results:
        normalized_result: dict[str, Any] = {}
        if isinstance(result, dict):
            maybe_normalized = normalize_legacy_report_fields(result, producer="block_extractor", report_kind="extraction")
            if isinstance(maybe_normalized, dict):
                normalized_result = maybe_normalized
            else:
                normalized_result = result
        facts = normalized_result.get("facts", [])
        tlinks = normalized_result.get("temporal_links", [])
        extraction_report_obj = normalized_result.get("extraction_report")
        extraction_report: dict[str, Any] = extraction_report_obj if isinstance(extraction_report_obj, dict) else {}
        report_entries: list[dict[str, Any]] = extraction_report.get("entries", []) if isinstance(extraction_report.get("entries"), list) else []

        local_to_final: dict[str, str] = {}

        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact_counter += 1
            final_id = f"f_{fact_counter:02d}"
            local_id = fact.get("local_id") or fact.get("id", "")
            if local_id:
                local_to_final[local_id] = final_id

            merged: dict[str, Any] = {
                "id": final_id,
                "session": session_num,
                "fact": fact.get("fact", ""),
                "kind": fact.get("kind", "fact"),
                "entities": copy.deepcopy(fact.get("entities", [])),
                "tags": copy.deepcopy(fact.get("tags", [])),
                "depends_on": copy.deepcopy(fact.get("depends_on", [])),
                "supersedes_topic": fact.get("supersedes_topic"),
                "confidence": fact.get("confidence"),
                "event_date": fact.get("event_date"),
            }
            if "flags" in fact:
                merged["flags"] = copy.deepcopy(fact.get("flags") or [])
            if isinstance(fact.get("metadata"), dict):
                merged["metadata"] = copy.deepcopy(fact["metadata"])

            if not fact.get("speaker") and block.speaker:
                merged["speaker"] = block.speaker
            elif fact.get("speaker"):
                merged["speaker"] = fact["speaker"]
            else:
                merged["speaker"] = None

            if not fact.get("speaker_role") and block.speaker_role:
                merged["speaker_role"] = block.speaker_role
            elif fact.get("speaker_role"):
                merged["speaker_role"] = fact["speaker_role"]
            else:
                merged["speaker_role"] = None

            if block.section_path:
                meta = merged.setdefault("metadata", {})
                assert isinstance(meta, dict)
                meta["section_path"] = block.section_path

            if merged.get("supersedes_topic"):
                meta = merged.setdefault("metadata", {})
                assert isinstance(meta, dict)
                meta["version_status"] = "current"
                meta["version_supersedes"] = merged["supersedes_topic"]

            all_facts.append(merged)

        for tlink in tlinks:
            if not isinstance(tlink, dict):
                continue
            before_local = tlink.get("before", "")
            after_local = tlink.get("after", "")
            before_final = local_to_final.get(before_local, before_local)
            after_final = local_to_final.get(after_local, after_local)
            merged_link = {
                "before": before_final,
                "after": after_final,
                "signal": tlink.get("signal", ""),
                "relation": tlink.get("relation", "before"),
            }
            if "flags" in tlink:
                merged_link["flags"] = copy.deepcopy(tlink.get("flags") or [])
            all_tlinks.append(merged_link)
        for entry in report_entries:
            if not isinstance(entry, dict):
                continue
            tagged: dict[str, Any] = copy.deepcopy(entry)
            details_obj = tagged.get("details")
            details: dict[str, Any] = details_obj if isinstance(details_obj, dict) else {}
            details.setdefault("block_order", block.order)
            details.setdefault("block_family", block.family)
            tagged["details"] = details
            all_report_entries.append(tagged)

    report_status = "partial" if all_report_entries else "ok"
    return {
        "facts": all_facts,
        "temporal_links": all_tlinks,
        "extraction_report": build_report(
            report_kind="extraction",
            producer="block_merger",
            status=report_status,
            entries=all_report_entries,
            summary={"entry_count": len(all_report_entries)},
        ),
    }
