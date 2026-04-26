# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import src.memory as memory_mod
import src.query_executors.document as document_executor
from src.episode_extraction import build_singleton_episodes
from src.memory import MemoryServer


pytestmark = pytest.mark.e2e

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "mrcr_cases"


def _fixture_manifest() -> dict:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _load_case(case_id: str) -> dict:
    manifest = _fixture_manifest()
    for case in manifest["cases"]:
        if case["case_id"] != case_id:
            continue
        example = json.loads((FIXTURE_ROOT / case["example_path"]).read_text(encoding="utf-8"))
        question_text = (FIXTURE_ROOT / case["question_path"]).read_text(encoding="utf-8").strip()
        assert example["query"]["query"] == question_text
        return {
            "manifest": case,
            "example": example,
            "question": question_text,
        }
    raise AssertionError(f"fixture case not found: {case_id}")


def _install_deterministic_document_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_group_document(model, source_id, title, date, block_dicts, grouping_config, sem):
        return build_singleton_episodes(source_id, date, block_dicts), "fixture", "fixture"

    async def _fake_extract_session(**kwargs):
        session_text = str(kwargs.get("session_text") or "")
        session_num = int(kwargs.get("session_num") or 1)
        conv_id = str(kwargs.get("conv_id") or "fixture")
        session_date = str(kwargs.get("session_date") or "2026-01-01")
        return (
            conv_id,
            session_num,
            session_date,
            [
                {
                    "id": f"fact_{session_num}",
                    "fact": session_text[:220],
                    "kind": "detail",
                    "entities": [],
                    "tags": ["fixture"],
                    "session": session_num,
                }
            ],
            [],
        )

    async def _fake_extract_source_aggregation_facts(self, **kwargs):
        return []

    monkeypatch.setattr(memory_mod, "group_document", _fake_group_document)
    monkeypatch.setattr(memory_mod, "extract_session", _fake_extract_session)
    monkeypatch.setattr(MemoryServer, "_extract_source_aggregation_facts", _fake_extract_source_aggregation_facts)


async def _seed_case_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
) -> tuple[MemoryServer, dict, dict[str, dict]]:
    _install_deterministic_document_extract(monkeypatch)
    loaded = _load_case(case_id)
    example = loaded["example"]
    ingest_item = deepcopy(example["ingest_items"][0]["arguments"])
    ingest_item.pop("key", None)
    ingest_item.setdefault("swarm_id", "mrcr-v2-bench")

    server = MemoryServer(
        str(tmp_path),
        example["benchmark_id"],
        extract_model="fixture-extract",
        swarm_id="mrcr-v2-bench",
    )
    result = await server.ingest_document(**ingest_item)
    assert result["status"] == "ok"

    episodes = server._get_episode_documents(ingest_item["source_id"], "document")
    episode_lookup = {episode["episode_id"]: episode for episode in episodes}
    return server, loaded, episode_lookup


def _episode_suffix(ep_num: int) -> str:
    return f"_e{ep_num:02d}"


def _pick_episode_facts(server: MemoryServer, episode_numbers: list[int]) -> list[dict]:
    selected: list[dict] = []
    for episode_number in episode_numbers:
        suffix = _episode_suffix(episode_number)
        fact = next(
            (
                item
                for item in server._all_granular
                if str((item.get("metadata") or {}).get("episode_id") or "").endswith(suffix)
            ),
            None,
        )
        if fact is None:
            raise AssertionError(f"fact for episode suffix {suffix} not found")
        selected.append(fact)
    return selected


async def _augment_packet_for_selected_facts(
    server: MemoryServer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    episode_lookup: dict[str, dict],
    selected_facts: list[dict],
    budget: int,
) -> dict:
    monkeypatch.setattr(
        document_executor,
        "build_bounded_chain_candidate_bundle",
        lambda *args, **kwargs: {
            "facts": selected_facts,
            "trace": {
                "mode": "fixture",
                "selected_fact_ids": [fact["id"] for fact in selected_facts],
            },
        },
    )
    packet = {
        "query_operator_plan": {"bounded_chain": {"enabled": True}},
        "retrieved_episode_ids": [
            (fact.get("metadata") or {}).get("episode_id", "")
            for fact in selected_facts
        ],
        "retrieved_fact_ids": [fact["id"] for fact in selected_facts],
        "selector_config": {"budget": budget},
        "tuning_snapshot": {
            "packet": {
                "snippet_chars": 1200,
                "query_specificity_bonus": 0.0,
            }
        },
    }
    augmented, _retrieved = await server._augment_document_structural_packet(
        query=question,
        packet=packet,
        episode_lookup=episode_lookup,
        fact_filter=lambda fact: True,
    )
    return augmented


def _raw_section(context: str) -> str:
    marker = "--- SOURCE EPISODE RAW TEXT ---"
    assert marker in context
    return context.split(marker, 1)[1]


@pytest.mark.asyncio
async def test_mrcr_0421_closes_missing_middle_target_span_from_real_fixture(tmp_path, monkeypatch):
    server, loaded, episode_lookup = await _seed_case_server(tmp_path, monkeypatch, case_id="mrcr-0421")
    selected_facts = _pick_episode_facts(server, [4, 1, 6])

    augmented = await _augment_packet_for_selected_facts(
        server,
        monkeypatch,
        question=loaded["question"],
        episode_lookup=episode_lookup,
        selected_facts=selected_facts,
        budget=8000,
    )

    expected_episode_ids = [
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e04",
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e05",
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e06",
    ]
    assert augmented["actual_injected_episode_ids"] == expected_episode_ids
    assert augmented["document_target_span_episode_ids"] == expected_episode_ids
    assert augmented["document_target_span_snippet_mode"] is False
    assert len(augmented["document_target_span_ids"]) == 1

    raw_section = _raw_section(augmented["context"])
    assert 'Title: "Echoes in the Kitchen"' in raw_section
    assert "It's the beauty of being a mother, Lila. We both grow." in raw_section
    assert "I hope I'll be as good a mother to her as you were to me." in raw_section
    assert "[Artifact 0001]" not in raw_section


@pytest.mark.asyncio
async def test_mrcr_0458_renders_target_span_in_source_order_from_real_fixture(tmp_path, monkeypatch):
    server, loaded, episode_lookup = await _seed_case_server(tmp_path, monkeypatch, case_id="mrcr-0458")
    selected_facts = _pick_episode_facts(server, [2, 1, 3])

    augmented = await _augment_packet_for_selected_facts(
        server,
        monkeypatch,
        question=loaded["question"],
        episode_lookup=episode_lookup,
        selected_facts=selected_facts,
        budget=8000,
    )

    source_id = loaded["example"]["ingest_items"][0]["arguments"]["source_id"]
    expected_episode_ids = [f"{source_id}_e01", f"{source_id}_e02", f"{source_id}_e03"]
    assert augmented["actual_injected_episode_ids"] == expected_episode_ids
    assert augmented["document_target_span_episode_ids"] == expected_episode_ids
    assert augmented["document_target_span_snippet_mode"] is False

    expected_raw = "\n\n".join(
        (episode_lookup[episode_id].get("raw_original") or episode_lookup[episode_id]["raw_text"])
        for episode_id in expected_episode_ids
    )
    raw_section = _raw_section(augmented["context"])
    assert expected_raw in raw_section
    assert raw_section.index("[Artifact 0001]") < raw_section.index("Response:\n**Verse 1:**")
    assert raw_section.index("Response:\n**Verse 1:**") < raw_section.index("**Verse 3:**")


@pytest.mark.asyncio
async def test_mrcr_0421_target_span_respects_packet_budget(tmp_path, monkeypatch):
    server, loaded, episode_lookup = await _seed_case_server(tmp_path, monkeypatch, case_id="mrcr-0421")
    selected_facts = _pick_episode_facts(server, [4, 1, 6])

    augmented = await _augment_packet_for_selected_facts(
        server,
        monkeypatch,
        question=loaded["question"],
        episode_lookup=episode_lookup,
        selected_facts=selected_facts,
        budget=600,
    )

    assert len(augmented["context"]) < 2200
    assert augmented["document_target_span_mode"] == "budget_fallback"
    assert augmented["document_target_span_snippet_mode"] is True
    assert augmented["actual_injected_episode_ids"][0] == (
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e04"
    )
    assert augmented["actual_injected_episode_ids"] != [
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e04",
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e05",
        f"{loaded['example']['ingest_items'][0]['arguments']['source_id']}_e06",
    ]
