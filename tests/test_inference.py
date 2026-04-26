# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

def test_synthesis_prompt_loaded():
    """Verify synthesize prompt loads and is usable."""
    from src.inference import INF_PROMPTS
    prompt = INF_PROMPTS["synthesize"]
    assert len(prompt) > 10


def test_all_inference_prompts_load():
    """Verify all inference prompt .md files load without error."""
    from src.inference import INF_PROMPTS, _INF_PROMPT_TYPES
    for name in _INF_PROMPT_TYPES:
        assert name in INF_PROMPTS, f"Missing inference prompt: {name}"
        assert len(INF_PROMPTS[name]) > 10, f"Prompt {name} too short"


def test_backward_compat_aliases():
    """Verify backward-compat prompt aliases load without error."""
    from src.inference import (
        INF_PROMPT, INF_PROMPT_ADV, INF_PROMPT_COUNTING,
        INF_PROMPT_SYNTHESIS, INF_PROMPT_TEMP, INF_PROMPT_TEMPORAL,
        INF_PROMPT_TEMPORAL_NOTOOL,
    )
    for name, prompt in [
        ("INF_PROMPT", INF_PROMPT),
        ("INF_PROMPT_TEMP", INF_PROMPT_TEMP),
        ("INF_PROMPT_ADV", INF_PROMPT_ADV),
        ("INF_PROMPT_TEMPORAL", INF_PROMPT_TEMPORAL),
        ("INF_PROMPT_TEMPORAL_NOTOOL", INF_PROMPT_TEMPORAL_NOTOOL),
        ("INF_PROMPT_COUNTING", INF_PROMPT_COUNTING),
        ("INF_PROMPT_SYNTHESIS", INF_PROMPT_SYNTHESIS),
    ]:
        assert isinstance(prompt, str), f"{name} is not a string"
        assert len(prompt) > 10, f"{name} is too short"


def test_codebase_mixed_prompt_loaded():
    from src.inference import INF_PROMPTS
    prompt = INF_PROMPTS["codebase_mixed"]
    assert "BOTH prose memory evidence and codebase evidence" in prompt
    assert "SOURCE FILES" in prompt


def test_code_slot_prompt_loaded():
    from src.inference import INF_PROMPTS
    prompt = INF_PROMPTS["code_slot"]
    assert "exact code object lookup question" in prompt
    assert "SOURCE FILES" in prompt


def test_code_chain_prompt_loaded():
    from src.inference import INF_PROMPTS
    prompt = INF_PROMPTS["code_chain"]
    assert "dependency, call-chain, or impact-trace question" in prompt
    assert "SOURCE FILES" in prompt


def test_risk_review_prompt_loaded():
    from src.inference import INF_PROMPTS
    prompt = INF_PROMPTS["risk_review"]
    assert "code change risk or review question" in prompt
    assert "SOURCE FILES" in prompt
