#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from multibench.judges.loaders import prep_kar, prep_loco


def test_prep_kar_includes_artifacts_even_when_sessions_exist():
    query = {
        "query": "What is the current pipe cover depth at km 2.3?",
        "ground_truth": "980mm",
        "category": "supersession",
        "expected_canary": "KAR_FCAE1B",
        "source_conversations": ["conv_021"],
        "source_documents": ["DOC-018"],
        "source_artifacts": ["ART-091"],
    }
    convs = {
        "conv_021": {
            "id": "conv_021",
            "date": "2026-07-22",
            "turns": [{"speaker": "engineer_alpha", "text": "Conversation text"}],
        }
    }
    docs = {
        "DOC-018": {
            "id": "DOC-018",
            "date": "2026-07-22",
            "content": "Document text",
        }
    }
    arts = {
        "ART-091": {
            "id": "ART-091",
            "title": "Pipe Laying km 2.1-5.5 Complete",
            "summary": "KAR_FCAE1B summary",
            "body": "Pipe installation update with 980mm depth at km 2.3.",
            "date": "2026-07-22",
        }
    }

    sessions, dates, speakers, question, gold, category, canary = prep_kar(
        query,
        convs,
        docs,
        arts,
        lambda text: [text],
    )

    assert speakers == "agents"
    assert question == "What is the current pipe cover depth at km 2.3?"
    assert gold == "980mm"
    assert category == "supersession"
    assert canary == "KAR_FCAE1B"
    assert any("Conversation text" in session for session in sessions)
    assert any("Document text" in session for session in sessions)
    assert any("980mm depth at km 2.3" in session for session in sessions)
    assert len(sessions) == 3


def test_prep_kar_uses_artifacts_as_fallback_source_text():
    query = {
        "query": "What changed?",
        "ground_truth": "980mm",
        "category": "supersession",
        "expected_canary": "KAR_FCAE1B",
        "source_artifacts": ["ART-091"],
    }
    arts = {
        "ART-091": {
            "id": "ART-091",
            "title": "Pipe Laying km 2.1-5.5 Complete",
            "summary": "Summary text",
            "body": "Pipe installation update with 980mm depth at km 2.3.",
            "date": "2026-07-22",
        }
    }

    sessions, dates, _speakers, _question, _gold, _category, _canary = prep_kar(
        query,
        {},
        {},
        arts,
        lambda text: [text],
    )

    assert len(sessions) == 1
    assert "Pipe installation update with 980mm depth at km 2.3." in sessions[0]
    assert dates == ["2026-07-22"]


def test_prep_loco_preserves_turn_query_as_semantic_source_text():
    conv = {
        "qa": [
            {"category": 2, "question": "When did Joanna first watch it?", "answer": "2019"},
        ],
        "conversation": {
            "speaker_a": "Joanna",
            "speaker_b": "Nate",
            "session_1": [
                {
                    "speaker": "Joanna",
                    "text": "Yep, that movie is awesome. I first watched it around 3 years ago.",
                    "query": "eternal sunshine of the spotless mind dvd cover",
                }
            ],
            "session_1_date_time": "2022-06-01",
        },
    }

    sessions, dates, speakers, question, gold, category = prep_loco(conv, 2)

    assert speakers == "Joanna and Nate"
    assert question == "When did Joanna first watch it?"
    assert gold == "2019"
    assert category == 2
    assert dates == ["2022-06-01"]
    assert "[Turn query] eternal sunshine of the spotless mind dvd cover" in sessions[0]


def test_prep_loco_does_not_duplicate_query_when_already_present_in_text():
    conv = {
        "qa": [
            {"category": 4, "question": "What movie is one of Joanna's favorites?", "answer": "Eternal Sunshine"},
        ],
        "conversation": {
            "speaker_a": "Joanna",
            "speaker_b": "Nate",
            "session_1": [
                {
                    "speaker": "Joanna",
                    "text": "Eternal Sunshine of the Spotless Mind is one of my favorites.",
                    "query": "Eternal Sunshine of the Spotless Mind",
                }
            ],
            "session_1_date_time": "2022-06-01",
        },
    }

    sessions, *_rest = prep_loco(conv, 4)

    assert sessions[0].count("[Turn query]") == 0
