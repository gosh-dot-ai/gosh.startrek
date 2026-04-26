# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src import providers, setup_store
from src.common import _supports_json_response_format, runtime_secret_context
from src.memory import MemoryServer


@pytest.fixture(autouse=True)
def _allow_plaintext_secrets(monkeypatch):
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")


def _secret_ref(name: str, *, scope: str = "system-wide", agent_id: str | None = None, swarm_id: str | None = None, owner_id: str | None = None) -> dict:
    ref = {"name": name, "scope": scope}
    if agent_id is not None:
        ref["agent_id"] = agent_id
    if swarm_id is not None:
        ref["swarm_id"] = swarm_id
    if owner_id is not None:
        ref["owner_id"] = owner_id
    return ref


def _server_with_secret(tmp_path, name: str, value: str, *, scope: str = "system-wide", swarm_id: str | None = None, caller_id: str = "agent:operator") -> MemoryServer:
    server = MemoryServer(str(tmp_path), f"prov_{name}")
    stored = server.store_secret(
        name,
        value,
        scope=scope,
        swarm_id=swarm_id,
        caller_id=caller_id,
    )
    assert stored["stored"] is True
    return server


# ── provider_from_model ──


def test_provider_from_model_bare_names():
    assert providers.provider_from_model("gpt-4.1-mini") == "openai"
    assert providers.provider_from_model("text-embedding-3-large") == "openai"


def test_provider_from_model_groq_prefixes():
    assert providers.provider_from_model("openai/gpt-oss-120b") == "groq"
    assert providers.provider_from_model("qwen/qwen3-32b") == "groq"


def test_qwen_groq_extraction_disables_openai_json_mode():
    assert _supports_json_response_format("qwen/qwen3-32b") is False
    assert _supports_json_response_format("openai/gpt-oss-120b") is True
    assert _supports_json_response_format("gpt-5.4") is True


def test_provider_from_model_other_providers():
    assert providers.provider_from_model("anthropic/claude-sonnet-4-6") == "anthropic"
    assert providers.provider_from_model("google/gemini-2.5-pro") == "google"
    assert providers.provider_from_model("inception/mercury-2") == "inception"


# ── ensure_api_key ──


def test_ensure_api_key_resolves_from_persisted_secret_store(tmp_path):
    secret_ref = _secret_ref("team-alpha-openai")
    server = _server_with_secret(tmp_path, "team-alpha-openai", "dummy-store-123")
    with runtime_secret_context(server, secret_ref):
        key = providers.ensure_api_key(secret_ref)
    assert key == "dummy-store-123"


def test_ensure_api_key_missing_fails_closed_without_env_or_setup_fallback(monkeypatch, tmp_path):
    server = MemoryServer(str(tmp_path), "prov_missing")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-env-should-not-be-used")
    monkeypatch.setattr(
        setup_store,
        "get_api_key",
        lambda provider: (_ for _ in ()).throw(AssertionError("setup_store.get_api_key must not be used at runtime")),
    )
    secret_ref = _secret_ref("missing-groq-runtime")
    with (
        runtime_secret_context(server, secret_ref),
        pytest.raises(RuntimeError, match="Missing runtime secret 'missing-groq-runtime' in persisted secret store"),
    ):
        providers.ensure_api_key(secret_ref)


def test_ensure_api_key_supports_arbitrary_secret_names_and_multiple_refs(tmp_path):
    server = MemoryServer(str(tmp_path), "prov_map")
    refs = {
        "openai_primary": _secret_ref("team-alpha-openai-primary"),
        "openai_secondary": _secret_ref("team-alpha-openai-secondary"),
        "anthropic_shared": _secret_ref("shared-anthropic-prod", scope="swarm-shared", swarm_id="alpha"),
    }
    for name, ref in refs.items():
        stored = server.store_secret(
            ref["name"],
            f"dummy-{name}",
            scope=ref["scope"],
            swarm_id=ref.get("swarm_id"),
            caller_id="agent:operator",
        )
        assert stored["stored"] is True
    with runtime_secret_context(server, refs["openai_primary"]):
        assert providers.ensure_api_key(refs["openai_primary"]) == "dummy-openai_primary"
    with runtime_secret_context(server, refs["openai_secondary"]):
        assert providers.ensure_api_key(refs["openai_secondary"]) == "dummy-openai_secondary"
    with runtime_secret_context(server, refs["anthropic_shared"]):
        assert providers.ensure_api_key(refs["anthropic_shared"]) == "dummy-anthropic_shared"


# ── acomplete / aextract ──


def test_acomplete_passes_store_secret_to_litellm(tmp_path):
    secret_ref = _secret_ref("llm-secret-A")
    server = _server_with_secret(tmp_path, "llm-secret-A", "dummy-litellm-token")

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5

    mock_choice = MagicMock()
    mock_choice.message.content = "  Hello world  "

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_lm = MagicMock()
    mock_lm.acompletion = AsyncMock(return_value=mock_response)
    providers._litellm = mock_lm

    try:
        with runtime_secret_context(server, secret_ref):
            result = asyncio.run(
                providers.acomplete(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": "Hi"}],
                    secret_ref=secret_ref,
                )
            )
        assert result["content"] == "Hello world"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert mock_lm.acompletion.call_args.kwargs["api_key"] == "dummy-litellm-token"
    finally:
        providers._litellm = None


def test_aextract_returns_parsed_json_from_store_backed_call(tmp_path):
    secret_ref = _secret_ref("extract-secret-main")
    server = _server_with_secret(tmp_path, "extract-secret-main", "dummy-extract-token")

    mock_choice = MagicMock()
    mock_choice.message.content = '```json\n{"facts": [{"fact": "test"}]}\n```'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_lm = MagicMock()
    mock_lm.acompletion = AsyncMock(return_value=mock_response)
    providers._litellm = mock_lm

    try:
        with runtime_secret_context(server, secret_ref):
            result = asyncio.run(
                providers.aextract(
                    model="gpt-4.1-mini",
                    system="Extract facts.",
                    user_msg="Alice met Bob.",
                    secret_ref=secret_ref,
                )
            )
        assert "facts" in result
        assert result["facts"][0]["fact"] == "test"
        assert mock_lm.acompletion.call_args.kwargs["api_key"] == "dummy-extract-token"
    finally:
        providers._litellm = None


# ── embed ──


def test_embed_empty_texts():
    result = providers.embed([], provider="openai", secret_ref=_secret_ref("unused"))
    assert result.shape == (0, 3072)

    result_local = providers.embed([], provider="local")
    assert result_local.shape == (0, 384)


def test_embed_local_sentence_transformers(monkeypatch):
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        result = providers.embed(
            ["hello", "world"],
            model="all-MiniLM-L6-v2",
            provider="local",
        )
        assert result.shape == (2, 3)
        mock_st.encode.assert_called_once()
    finally:
        providers._st_model_cache.clear()


def test_embed_openai_via_litellm_uses_persisted_secret_store(tmp_path):
    secret_ref = _secret_ref("embed-secret-main")
    server = _server_with_secret(tmp_path, "embed-secret-main", "dummy-embed-token")

    mock_data = [
        {"embedding": [0.1] * 3072},
        {"embedding": [0.2] * 3072},
    ]
    mock_response = MagicMock()
    mock_response.data = mock_data

    mock_lm = MagicMock()
    mock_lm.embedding.return_value = mock_response
    providers._litellm = mock_lm

    try:
        with runtime_secret_context(server, secret_ref):
            result = providers.embed(["hello", "world"], provider="openai", secret_ref=secret_ref)
        assert result.shape == (2, 3072)
        assert mock_lm.embedding.call_args.kwargs["api_key"] == "dummy-embed-token"
    finally:
        providers._litellm = None


def test_embed_openai_missing_secret_fails_closed_without_env_fallback(monkeypatch, tmp_path):
    server = MemoryServer(str(tmp_path), "prov_embed_missing")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-env-should-not-be-used")
    monkeypatch.setattr(
        setup_store,
        "get_api_key",
        lambda provider: (_ for _ in ()).throw(AssertionError("setup_store.get_api_key must not be used at runtime")),
    )
    secret_ref = _secret_ref("missing-openai-embed")

    with (
        runtime_secret_context(server, secret_ref),
        pytest.raises(RuntimeError, match="Missing runtime secret 'missing-openai-embed' in persisted secret store"),
    ):
        providers.embed(["hello"], provider="openai", secret_ref=secret_ref)


def test_embed_one_returns_1d(monkeypatch):
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        result = providers.embed_one("hello", model="all-MiniLM-L6-v2", provider="local")
        assert result.ndim == 1
        assert len(result) == 3
    finally:
        providers._st_model_cache.clear()


def test_embed_replaces_empty_strings(monkeypatch):
    mock_st = MagicMock()
    mock_st.encode.return_value = np.array([[0.1, 0.2]])

    providers._st_model_cache["all-MiniLM-L6-v2"] = mock_st

    try:
        providers.embed(["", "  "], model="all-MiniLM-L6-v2", provider="local")
        call_args = mock_st.encode.call_args[0][0]
        assert call_args == ["[empty]", "[empty]"]
    finally:
        providers._st_model_cache.clear()
