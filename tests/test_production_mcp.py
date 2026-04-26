# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import pytest

from multibench import production_mcp as pm
from src.config import MemoryConfig


class _FakeSocket:
    def __init__(self, fd=55, port=43123):
        self._fd = fd
        self._port = port
        self.closed = False

    def fileno(self):
        return self._fd

    def getsockname(self):
        return ("127.0.0.1", self._port)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _FakeHttpClient:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakeClientCm:
    def __init__(self):
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return ("read", "write", lambda: "sid")

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True


class _FailingEnterClientCm:
    def __init__(self):
        self.exited = False

    async def __aenter__(self):
        raise RuntimeError("client enter boom")

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True


class _FakeSession:
    def __init__(self, *args, **kwargs):
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

    async def initialize(self):
        return None


@pytest.mark.asyncio
async def test_production_case_runtime_cleans_up_on_initialize_failure(monkeypatch, tmp_path):
    fake_sock = _FakeSocket()
    fake_proc = _FakeProc()
    fake_http = _FakeHttpClient()
    fake_cm = _FakeClientCm()

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    async def _fake_wait_until_ready(self):
        return None

    class _FailingSession:
        def __init__(self, *args, **kwargs):
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self.exited = True

        async def initialize(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(pm, "_bind_listen_socket", lambda: fake_sock)
    monkeypatch.setattr(pm.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(pm.ProductionCaseRuntime, "_wait_until_ready", _fake_wait_until_ready)
    monkeypatch.setattr(pm, "create_mcp_http_client", lambda **kwargs: fake_http)
    monkeypatch.setattr(pm, "streamable_http_client", lambda *args, **kwargs: fake_cm)
    monkeypatch.setattr(pm, "ClientSession", _FailingSession)

    runtime = pm.ProductionCaseRuntime(
        data_dir=tmp_path / "data",
        cfg=MemoryConfig(extraction_model="e", inference_model="i", judge_model="j"),
        log_path=tmp_path / "server.log",
    )

    with pytest.raises(RuntimeError, match="boom"):
        await runtime.__aenter__()

    assert fake_proc.terminated is True
    assert fake_proc.waited is True
    assert fake_http.closed is True
    assert fake_cm.exited is True
    assert fake_sock.closed is True


@pytest.mark.asyncio
async def test_production_case_runtime_cleans_up_on_client_cm_enter_failure(monkeypatch, tmp_path):
    fake_sock = _FakeSocket()
    fake_proc = _FakeProc()
    fake_http = _FakeHttpClient()
    fake_cm = _FailingEnterClientCm()

    async def _fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc

    async def _fake_wait_until_ready(self):
        return None

    monkeypatch.setattr(pm, "_bind_listen_socket", lambda: fake_sock)
    monkeypatch.setattr(pm.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(pm.ProductionCaseRuntime, "_wait_until_ready", _fake_wait_until_ready)
    monkeypatch.setattr(pm, "create_mcp_http_client", lambda **kwargs: fake_http)
    monkeypatch.setattr(pm, "streamable_http_client", lambda *args, **kwargs: fake_cm)

    runtime = pm.ProductionCaseRuntime(
        data_dir=tmp_path / "data",
        cfg=MemoryConfig(extraction_model="e", inference_model="i", judge_model="j"),
        log_path=tmp_path / "server.log",
    )

    with pytest.raises(RuntimeError, match="client enter boom"):
        await runtime.__aenter__()

    assert fake_proc.terminated is True
    assert fake_proc.waited is True
    assert fake_http.closed is True
    assert fake_cm.exited is True
    assert fake_sock.closed is True


@pytest.mark.asyncio
async def test_production_case_runtime_passes_bound_fd_to_child(monkeypatch, tmp_path):
    fake_sock = _FakeSocket(fd=77, port=43177)
    fake_proc = _FakeProc()
    fake_http = _FakeHttpClient()
    fake_cm = _FakeClientCm()
    captured = {}

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured["kwargs"] = kwargs
        return fake_proc

    async def _fake_wait_until_ready(self):
        return None

    fake_session = _FakeSession()

    monkeypatch.setattr(pm, "_bind_listen_socket", lambda: fake_sock)
    monkeypatch.setattr(pm.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(pm.ProductionCaseRuntime, "_wait_until_ready", _fake_wait_until_ready)
    monkeypatch.setattr(pm, "create_mcp_http_client", lambda **kwargs: fake_http)
    monkeypatch.setattr(pm, "streamable_http_client", lambda *args, **kwargs: fake_cm)
    monkeypatch.setattr(pm, "ClientSession", lambda *args, **kwargs: fake_session)

    async with pm.production_case_runtime(
        data_dir=tmp_path / "data",
        cfg=MemoryConfig(extraction_model="e", inference_model="i", judge_model="j"),
        log_path=tmp_path / "server.log",
    ):
        pass

    assert captured["kwargs"]["pass_fds"] == (77,)
    assert captured["kwargs"]["env"]["GOSH_BENCH_FD"] == "77"
    assert "GOSH_BENCH_PORT" not in captured["kwargs"]["env"]
    assert fake_session.entered is True
    assert fake_session.exited is True
