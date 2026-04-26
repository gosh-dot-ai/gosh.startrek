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

import re

ARTIFACT_MARKER_RE = re.compile(r"\[Artifact\s+([^\]\n]+)\]", re.IGNORECASE)
BARE_ARTIFACT_MARKER_RE = re.compile(
    r"(?im)^(?:artifact)\s*[:#-]?\s*([0-9A-Za-z._-]+)\s*[:#-]?\s*$",
)
PROMPT_RESPONSE_START_RE = re.compile(
    r"(?is)^\s*(?:instruction|prompt|request|question)\s*:\s*.+?(?:\n+\s*(?:response|answer|output)\s*:)",
)


def _normalize_artifact_marker(raw_marker: str) -> str:
    marker = re.sub(r"[^0-9A-Za-z._-]+", "_", raw_marker.strip()).strip("._-")
    return marker


def _episode_candidate_texts(episode: dict) -> list[str]:
    values: list[str] = []
    for value in (
        episode.get("raw_original"),
        episode.get("raw_text"),
        (episode.get("provenance") or {}).get("source_section_path"),
        (episode.get("metadata") or {}).get("source_section_path"),
        episode.get("section_path"),
    ):
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)
    return values


def extract_artifact_markers(text: str) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    raw_text = str(text or "")
    for pattern in (ARTIFACT_MARKER_RE, BARE_ARTIFACT_MARKER_RE):
        for raw_marker in pattern.findall(raw_text):
            marker = _normalize_artifact_marker(raw_marker)
            if not marker or marker in seen:
                continue
            markers.append(marker)
            seen.add(marker)
    return markers


def _episode_artifact_markers(episode: dict) -> list[str]:
    for text in _episode_candidate_texts(episode):
        markers = extract_artifact_markers(text)
        if markers:
            return markers
    return []


def _looks_like_artifact_restart(episode: dict) -> bool:
    for text in _episode_candidate_texts(episode):
        if PROMPT_RESPONSE_START_RE.search(text[:1600]):
            return True
    return False


def assign_document_artifact_span_ids(
    episodes: list[dict],
    *,
    source_id: str,
) -> list[dict]:
    current_span_id: str | None = None
    for idx, episode in enumerate(episodes, start=1):
        explicit_span_id = str(episode.get("artifact_span_id") or "").strip()
        if explicit_span_id:
            current_span_id = explicit_span_id
        else:
            markers = _episode_artifact_markers(episode)
            if markers:
                current_span_id = f"{source_id}::artifact::{markers[-1]}"
            elif _looks_like_artifact_restart(episode) or current_span_id is None:
                current_span_id = f"{source_id}::artifact::episode_{idx:04d}"
            episode["artifact_span_id"] = current_span_id
            provenance = dict(episode.get("provenance") or {})
            provenance["artifact_span_id"] = current_span_id
            if markers:
                provenance["artifact_markers"] = markers
            if provenance:
                episode["provenance"] = provenance
            continue

        provenance = dict(episode.get("provenance") or {})
        provenance["artifact_span_id"] = current_span_id
        if provenance:
            episode["provenance"] = provenance
    return episodes
