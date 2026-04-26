# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import argparse
import json

import pytest

from multibench.sprint37 import run_conversation_production_batch as conv_batch


def test_sprint37_conversation_harness_requires_explicit_or_env_models(monkeypatch):
    monkeypatch.delenv("GOSH_EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("GOSH_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_JUDGE_MODEL", raising=False)

    args = argparse.Namespace(
        extraction_model=None,
        inference_model=None,
        judge_model=None,
    )

    with pytest.raises(RuntimeError, match="Extraction model is empty"):
        conv_batch._resolve_models(args)


def test_sprint37_conversation_harness_accepts_explicit_models(monkeypatch):
    monkeypatch.delenv("GOSH_EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("GOSH_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_JUDGE_MODEL", raising=False)

    args = argparse.Namespace(
        extraction_model="inception/mercury-2",
        inference_model="qwen/qwen3-32b",
        judge_model="gpt-4.1",
    )

    models = conv_batch._resolve_models(args)

    assert models["extraction_model"] == "inception/mercury-2"
    assert models["inference_model"] == "qwen/qwen3-32b"
    assert models["judge_model"] == "gpt-4.1"


def test_sprint37_conversation_harness_picks_three_plus_three(monkeypatch):
    monkeypatch.setattr(
        conv_batch,
        "pick_locomo",
        lambda n: [{"benchmark": "locomo", "qid": f"l{i}", "source_key": f"l{i}", "category": "c"} for i in range(n)],
    )
    monkeypatch.setattr(
        conv_batch,
        "pick_longmemeval",
        lambda n: [{"benchmark": "longmemeval", "qid": f"m{i}", "source_key": f"m{i}", "category": "c"} for i in range(n)],
    )

    rows = conv_batch.pick_conversation_examples(
        locomo_count=3,
        longmemeval_count=3,
        locomo_offset=0,
        longmemeval_offset=0,
    )

    assert len(rows) == 6
    assert sum(1 for row in rows if row["benchmark"] == "locomo") == 3
    assert sum(1 for row in rows if row["benchmark"] == "longmemeval") == 3


def test_sprint37_resume_manifest_rejects_model_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    expected = {
        "experiment": "conversation_production_batch",
        "sprint": 37,
        "seed": 37,
        "question_count": 6,
        "models": {
            "extraction_model": "inception/mercury-2",
            "inference_model": "qwen/qwen3-32b",
            "judge_model": "gpt-4.1",
            "embed_model": "",
        },
        "benchmarks": ["locomo", "longmemeval"],
        "examples": [{"benchmark": "locomo", "query_id": "a", "source_key": "a", "category": "c"}],
    }
    actual = dict(expected)
    actual["models"] = dict(expected["models"])
    actual["models"]["inference_model"] = "other/model"
    (run_dir / "manifest.json").write_text(json.dumps(actual))

    with pytest.raises(RuntimeError, match="manifest mismatch"):
        conv_batch._validate_resume_manifest(run_dir, expected)


def test_sprint37_resume_manifest_accepts_exact_match(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "experiment": "conversation_production_batch",
        "sprint": 37,
        "seed": 37,
        "question_count": 6,
        "models": {
            "extraction_model": "inception/mercury-2",
            "inference_model": "qwen/qwen3-32b",
            "judge_model": "gpt-4.1",
            "embed_model": "",
        },
        "benchmarks": ["locomo", "longmemeval"],
        "examples": [{"benchmark": "locomo", "query_id": "a", "source_key": "a", "category": "c"}],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    assert conv_batch._validate_resume_manifest(run_dir, manifest) == manifest


def test_sprint37_manifest_includes_payload_fingerprint():
    row = {
        "benchmark": "locomo",
        "qid": "conv-1_cat1",
        "source_key": "conv-1",
        "category": "single-session-user",
        "question": "What did Alice study?",
        "gold": "Biology",
        "speakers": "User and Assistant",
        "dates": ["2024-01-01"],
        "sessions": ["User: Alice studied Biology."],
    }

    manifest_row = conv_batch._example_manifest_row(row)

    assert manifest_row["benchmark"] == "locomo"
    assert manifest_row["query_id"] == "conv-1_cat1"
    assert manifest_row["source_key"] == "conv-1"
    assert manifest_row["category"] == "single-session-user"
    assert len(manifest_row["payload_sha256"]) == 64


def test_sprint37_resume_manifest_rejects_payload_drift(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    expected = {
        "experiment": "conversation_production_batch",
        "sprint": 37,
        "seed": 37,
        "question_count": 1,
        "models": {
            "extraction_model": "inception/mercury-2",
            "inference_model": "qwen/qwen3-32b",
            "judge_model": "gpt-4.1",
            "embed_model": "",
        },
        "benchmarks": ["locomo"],
        "examples": [
            conv_batch._example_manifest_row(
                {
                    "benchmark": "locomo",
                    "qid": "conv-1_cat1",
                    "source_key": "conv-1",
                    "category": "single-session-user",
                    "question": "What did Alice study?",
                    "gold": "Biology",
                    "speakers": "User and Assistant",
                    "dates": ["2024-01-01"],
                    "sessions": ["User: Alice studied Biology."],
                }
            )
        ],
    }
    actual = dict(expected)
    actual["examples"] = [
        conv_batch._example_manifest_row(
            {
                "benchmark": "locomo",
                "qid": "conv-1_cat1",
                "source_key": "conv-1",
                "category": "single-session-user",
                "question": "What did Alice study?",
                "gold": "Chemistry",
                "speakers": "User and Assistant",
                "dates": ["2024-01-01"],
                "sessions": ["User: Alice studied Chemistry."],
            }
        )
    ]
    (run_dir / "manifest.json").write_text(json.dumps(actual))

    with pytest.raises(RuntimeError, match="manifest mismatch"):
        conv_batch._validate_resume_manifest(run_dir, expected)
