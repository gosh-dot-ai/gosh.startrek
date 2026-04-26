#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
import json
from typing import Any, Optional

import numpy as np

# ── Lazy imports ──

_litellm = None
_st_model_cache = {}


def _get_litellm():
    global _litellm
    if _litellm is not None:
        return _litellm
    raise ImportError(
        "litellm has been disabled due to CVE-2026-33634 (supply chain attack, "
        "CVSS 9.4). Use direct provider SDKs via src/common.py instead. "
        "See: https://docs.litellm.ai/blog/security-update-march-2026"
    )


def _get_st_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    if model_name not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_name] = SentenceTransformer(model_name)
    return _st_model_cache[model_name]


# ── Provider resolution ──


def provider_from_model(model: str) -> str:
    """Extract provider from model name prefix.

    openai/gpt-oss-120b → groq (OpenAI-compat models on Groq)
    qwen/...            → groq
    anthropic/...       → anthropic
    google/...          → google
    gpt-4.1-mini        → openai (bare names)
    """
    if "/" in model:
        prefix = model.split("/")[0]
        if prefix in ("openai", "qwen"):
            return "groq"
        if prefix == "inception":
            return "inception"
        return prefix
    return "openai"


def _resolve_runtime_secret(secret_ref: dict[str, Any]) -> str:
    from .common import resolve_runtime_secret_value
    return resolve_runtime_secret_value(secret_ref)


def ensure_api_key(secret_ref: dict[str, Any]) -> str:
    """Resolve one persisted runtime secret from an explicit secret_ref."""
    return _resolve_runtime_secret(secret_ref)


# ── LLM completion ──

async def acomplete(model: str, messages: list[dict[str, str]],
                    max_tokens: int = 300, temperature: float = 0.0,
                    json_mode: bool = False, timeout: int = 60,
                    secret_ref: dict[str, Any] | None = None) -> dict:
    """Unified async chat completion via litellm.

    Returns: {"content": str, "usage": {"prompt_tokens": int, "completion_tokens": int}}
    """
    if secret_ref is None:
        raise RuntimeError("acompletion requires an explicit persisted secret_ref")
    api_key = ensure_api_key(secret_ref)
    lm = _get_litellm()

    kw = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if temperature is not None:
        kw["temperature"] = temperature
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    kw["api_key"] = api_key

    response = await lm.acompletion(**kw)

    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }

    return {
        "content": response.choices[0].message.content.strip(),
        "usage": usage,
    }


async def aextract(model: str, system: str, user_msg: str,
                   max_tokens: int = 8192,
                   secret_ref: dict[str, Any] | None = None) -> dict:
    """Extraction call — returns parsed JSON dict via litellm."""
    result = await acomplete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        secret_ref=secret_ref,
    )
    from .common import parse_json_response
    return parse_json_response(result["content"])


# ── Embeddings ──

def embed(texts: list[str], model: str = "text-embedding-3-large",
          provider: str = "openai",
          secret_ref: dict[str, Any] | None = None) -> np.ndarray:
    """Batch embed texts.

    provider="openai" → OpenAI API (via litellm or direct)
    provider="local"  → sentence-transformers (no API key needed)
    """
    if not texts:
        dim = 384 if provider == "local" else 3072
        return np.zeros((0, dim))

    texts = [t if t.strip() else "[empty]" for t in texts]

    if provider == "local":
        st = _get_st_model(model)
        return np.array(st.encode(texts, convert_to_numpy=True))

    # OpenAI embedding via litellm
    if secret_ref is None:
        raise RuntimeError("remote embeddings require an explicit persisted secret_ref")
    api_key = ensure_api_key(secret_ref)
    lm = _get_litellm()
    response = lm.embedding(model=model, input=texts, api_key=api_key)
    return np.array([e["embedding"] for e in response.data])


def embed_one(text: str, model: str = "text-embedding-3-large",
              provider: str = "openai",
              secret_ref: dict[str, Any] | None = None) -> np.ndarray:
    """Embed a single text string."""
    return embed([text], model=model, provider=provider, secret_ref=secret_ref)[0]
