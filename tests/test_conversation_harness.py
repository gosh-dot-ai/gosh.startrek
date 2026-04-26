# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import argparse

import pytest

from multibench.sprint36 import run_conversation_production_batch as conv_batch


def test_conversation_harness_requires_explicit_or_env_models(monkeypatch):
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


def test_conversation_harness_accepts_explicit_models(monkeypatch):
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
