# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import argparse
import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.common import runtime_secret_context
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


def _server_with_runtime_secret(tmp_path, name: str, value: str, *, scope: str = "system-wide", swarm_id: str | None = None, caller_id: str = "agent:operator") -> MemoryServer:
    server = MemoryServer(str(tmp_path), f"common_{name}")
    stored = server.store_secret(
        name,
        value,
        scope=scope,
        swarm_id=swarm_id,
        caller_id=caller_id,
    )
    assert stored["stored"] is True
    return server


@pytest.mark.asyncio
async def test_call_extract_routes_via_get_client(monkeypatch):
    """call_extract must route to correct client via _get_client."""
    called = {}

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = '{"facts": [], "temporal_links": []}'

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            called["model"] = kw.get("model")
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract
    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert called.get("model") == "gpt-4.1-mini"
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_call_extract_salvages_json_from_markdown_fence(monkeypatch):
    """call_extract must salvage valid JSON wrapped in markdown fences."""

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = """```json
{"facts": [], "temporal_links": []}
```"""

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract

    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert result == {"facts": [], "temporal_links": []}


@pytest.mark.asyncio
async def test_call_extract_salvages_json_from_surrounding_text(monkeypatch):
    """call_extract must salvage valid JSON when the model adds prose around it."""

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30

    class FakeChoice:
        class message:
            content = (
                "Here is the extracted JSON:\\n"
                '{"facts": [{"local_id": "f1"}], "temporal_links": []}\\n'
                "Done."
            )

    class FakeResp:
        usage = FakeUsage()
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kw):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import call_extract

    result = await call_extract("gpt-4.1-mini", "sys", "user")
    assert result == {"facts": [{"local_id": "f1"}], "temporal_links": []}


def test_embed_texts_sync_calls_openai(monkeypatch):
    """embed_texts_sync must call oai_sync().embeddings.create."""
    calls = {}

    class FakeUsage:
        total_tokens = 10
    class FakeEmbData:
        def __init__(self): self.embedding = [0.0] * 1024
    class FakeResp:
        usage = FakeUsage()
        data = [FakeEmbData(), FakeEmbData()]

    class FakeEmbeddings:
        def create(self, **kw):
            calls["called"] = True
            calls["model"] = kw.get("model")
            return FakeResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common._oai_sync", FakeClient())

    from src.common import embed_texts_sync
    result = embed_texts_sync(["hello", "world"])
    assert calls.get("called")
    assert result.shape == (2, 1024)


@pytest.mark.asyncio
async def test_embed_query_async(monkeypatch):
    """embed_query (async) must call oai_async().embeddings.create."""
    calls = {}

    class FakeUsage:
        total_tokens = 5
    class FakeEmbData:
        def __init__(self): self.embedding = [0.0] * 1024
    class FakeResp:
        usage = FakeUsage()
        data = [FakeEmbData()]

    class FakeEmbeddings:
        async def create(self, **kw):
            calls["called"] = True
            return FakeResp()

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.common._oai_async", FakeClient())

    from src.common import embed_query
    result = await embed_query("test query")
    assert calls.get("called")
    assert result.shape == (1024,)


@pytest.mark.asyncio
async def test_embed_texts_empty_no_api_call():
    """embed_texts([]) must return zeros without calling any API."""
    from src.common import embed_texts
    result = await embed_texts([])
    assert result.shape[0] == 0


def test_bge_is_default_local_model():
    """BAAI/bge-large-en-v1.5 must be the default local embedding model."""
    import inspect

    from src import providers
    sig = inspect.signature(providers._get_st_model)
    default = sig.parameters["model_name"].default
    assert default == "BAAI/bge-large-en-v1.5", \
        f"Expected BAAI/bge-large-en-v1.5, got {default}"


def test_oai_async_uses_persisted_secret_store_without_env(monkeypatch, tmp_path):
    from src import common

    created = {}

    class FakeClient:
        def __init__(self, **kw):
            created.update(kw)

    secret_ref = _secret_ref("runtime-openai-main")
    server = _server_with_runtime_secret(tmp_path, "runtime-openai-main", "dummy-openai-store-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(common, "_oai_async", None)
    monkeypatch.setattr(common, "AsyncOpenAI", FakeClient)

    with runtime_secret_context(server, secret_ref):
        client = common.oai_async()

    assert isinstance(client, FakeClient)
    assert created["api_key"] == "dummy-openai-store-token"
    assert created["timeout"] == common._OPENAI_TIMEOUT


def test_groq_async_fails_closed_without_secret_and_ignores_env(monkeypatch, tmp_path):
    from src import common, setup_store

    server = MemoryServer(str(tmp_path), "common_groq_missing")
    secret_ref = _secret_ref("team-groq-prod")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-env-should-not-be-used")
    monkeypatch.setattr(
        setup_store,
        "get_api_key",
        lambda provider: (_ for _ in ()).throw(AssertionError("setup_store.get_api_key must not be used at runtime")),
    )
    monkeypatch.setattr(common, "_groq_async", None)

    with (
        runtime_secret_context(server, secret_ref),
        pytest.raises(RuntimeError, match="Missing runtime secret 'team-groq-prod' in persisted secret store"),
    ):
        common.groq_async()


def test_google_client_uses_persisted_secret_store_without_env(monkeypatch, tmp_path):
    from src import common

    configured = {}

    google_pkg = ModuleType("google")
    fake_genai = ModuleType("google.generativeai")

    def _configure(**kw):
        configured.update(kw)

    fake_genai.configure = _configure  # type: ignore[attr-defined]
    google_pkg.generativeai = fake_genai  # type: ignore[attr-defined]

    secret_ref = _secret_ref("google-service-account-prod")
    server = _server_with_runtime_secret(tmp_path, "google-service-account-prod", "google-store-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(common, "_google_client", None)
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    with runtime_secret_context(server, secret_ref):
        client = common.google_client()

    assert client is fake_genai
    assert configured["api_key"] == "google-store-secret"


def test_runtime_secret_context_supports_exact_non_system_secret_refs(monkeypatch, tmp_path):
    from src import common

    created = {}

    class FakeClient:
        def __init__(self, **kw):
            created.update(kw)

    secret_ref = _secret_ref("team-shared-openai", scope="swarm-shared", swarm_id="alpha")
    server = _server_with_runtime_secret(
        tmp_path,
        "team-shared-openai",
        "dummy-shared-openai-token",
        scope="swarm-shared",
        swarm_id="alpha",
    )
    monkeypatch.setattr(common, "_oai_async", None)
    monkeypatch.setattr(common, "AsyncOpenAI", FakeClient)

    with runtime_secret_context(server, secret_ref):
        client = common.oai_async()

    assert isinstance(client, FakeClient)
    assert created["api_key"] == "dummy-shared-openai-token"


# ── Fix 3: key source transparency ──

def test_setup_show_prints_key_source(capsys, monkeypatch, tmp_path):
    """--show must print which storage backend holds the API key."""
    from src import setup_store

    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path / ".gosh-memory")
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / ".gosh-memory" / "config.json")
    # Clear env var so config file path is visible
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    (tmp_path / ".gosh-memory").mkdir()
    setup_store.save_config({
        "provider": "openai",
        "models": {"extraction": "gpt-4.1-mini"},
        "api_keys": {"openai": "dummy-config-openai-token"},
    })

    from src.cli import cmd_setup
    args = argparse.Namespace(show=True, provider=None, api_key=None, embed_provider=None)
    cmd_setup(args)

    out = capsys.readouterr().out
    assert "config.json" in out
    assert "***" in out


def test_setup_show_prints_env_source(capsys, monkeypatch, tmp_path):
    """--show must detect API key from env var and report it."""
    from src import setup_store

    monkeypatch.setattr(setup_store, "CONFIG_DIR", tmp_path / ".gosh-memory")
    monkeypatch.setattr(setup_store, "CONFIG_FILE", tmp_path / ".gosh-memory" / "config.json")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-test-env-key")

    (tmp_path / ".gosh-memory").mkdir()
    setup_store.save_config({
        "provider": "groq",
        "models": {"extraction": "qwen/qwen3-32b"},
    })

    from src.cli import cmd_setup
    args = argparse.Namespace(show=True, provider=None, api_key=None, embed_provider=None)
    cmd_setup(args)

    out = capsys.readouterr().out
    assert "env var $GROQ_API_KEY" in out
    assert "***" in out


def test_inception_client_uses_persisted_secret_store_without_nameerror(monkeypatch, tmp_path):
    """Inception client must boot cleanly from the persisted secret store."""
    from src import common

    created = {}

    class FakeClient:
        def __init__(self, **kw):
            created.update(kw)

    secret_ref = _secret_ref("mercury-runtime-prod")
    server = _server_with_runtime_secret(tmp_path, "mercury-runtime-prod", "inception-store-key")
    monkeypatch.setattr(common, "_inception_async", None)
    monkeypatch.setattr(common, "AsyncOpenAI", FakeClient)

    with runtime_secret_context(server, secret_ref):
        client = common.inception_async()

    assert isinstance(client, FakeClient)
    assert created["base_url"] == "https://api.inceptionlabs.ai/v1"
    assert created["api_key"] == "inception-store-key"
    assert created["timeout"] == common._OPENAI_TIMEOUT
