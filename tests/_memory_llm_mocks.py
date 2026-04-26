# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from copy import deepcopy


def patch_memory_llm_runtime(monkeypatch, *, answer: str = "test answer", extract_payload=None) -> None:
    payload = {} if extract_payload is None else deepcopy(extract_payload)

    async def mock_call_extract(*args, **kwargs):
        return deepcopy(payload)

    async def mock_call_oai(*args, **kwargs):
        return answer

    async def mock_call_model(*args, **kwargs):
        return answer

    monkeypatch.setattr("src.memory.call_extract", mock_call_extract)
    monkeypatch.setattr("src.common.call_extract", mock_call_extract)
    monkeypatch.setattr("src.memory.call_oai", mock_call_oai)
    monkeypatch.setattr("src.common.call_oai", mock_call_oai)
    monkeypatch.setattr("src.memory._call_model", mock_call_model)
    monkeypatch.setattr("src.common._call_model", mock_call_model)
