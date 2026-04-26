# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import contextlib
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from tests.e2e.conftest import (
    LiveStack,
    _base_runtime_config,
    e2e_runtime_secret_values,
)


def _live_secret(name: str) -> str:
    try:
        import keyring  # type: ignore
    except Exception:
        keyring = None
    if keyring is not None:
        try:
            value = str(keyring.get_password("gosh-memory", name) or "").strip()
        except Exception:
            value = ""
        if value:
            return value
    return str(os.environ.get(name) or "").strip()


class LiveProviderStack(LiveStack):
    extraction_model: str | None
    default_inference_model: str | None
    embed_model: str | None

    def __init__(
        self,
        *,
        extraction_model: str | None,
        default_inference_model: str | None,
        embed_model: str | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.extraction_model = extraction_model
        self.default_inference_model = default_inference_model
        self.embed_model = embed_model
        self._endpoint = f"http://127.0.0.1:{self.memory_port}"
        self._server_token = f"srv_{secrets.token_urlsafe(32)}"
        self._memory_proc: subprocess.Popen[str] | None = None
        self._memory_log = self.state_dir / "memory.log"
        self._memory_log_handle = None

    def write_services_config(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def start_memory(self) -> None:
        self.write_services_config()
        env = os.environ.copy()
        env["GOSH_MEMORY_ADMIN_TOKEN"] = self.bootstrap_admin_token
        env["GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS"] = "1"
        env["GOSH_MEMORY_TEST_DETERMINISTIC_E2E"] = "0"
        self._memory_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self._memory_log.open("w", encoding="utf-8")
        self._memory_log_handle = log_handle
        cmd = [
            sys.executable,
            "-m",
            "src.mcp_server",
            "--data-dir",
            str(self.data_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.memory_port),
            "--server-token",
            self._server_token,
        ]
        if self.extraction_model:
            cmd.extend(["--extraction-model", self.extraction_model])
        if self.default_inference_model:
            cmd.extend(["--inference-model", self.default_inference_model])
        if self.embed_model:
            cmd.extend(["--embed-model", self.embed_model])
        self._memory_proc = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._wait_for_health()
        if self.admin_token is None:
            self.bootstrap_admin()

    def _runtime_secret_values(self) -> dict[str, str]:
        inference_model = str(self.default_inference_model or self.extraction_model or "").strip() or None
        return e2e_runtime_secret_values(
            extraction_model=self.extraction_model,
            default_inference_model=inference_model,
            embed_model=self.embed_model,
            fast_profile_model=inference_model,
            balanced_profile_model=inference_model,
            strong_profile_model=inference_model,
            judge_model=inference_model,
        )

    def _base_runtime_config(self) -> dict[str, object]:
        inference_model = str(self.default_inference_model or self.extraction_model or "").strip() or None
        return _base_runtime_config(
            extraction_model=self.extraction_model,
            default_inference_model=inference_model,
            embed_model=self.embed_model,
            fast_profile_model=inference_model,
            balanced_profile_model=inference_model,
            strong_profile_model=inference_model,
            judge_model=inference_model,
        )

    def stop_memory(self) -> None:
        proc = self._memory_proc
        self._memory_proc = None
        try:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=20)
                except Exception:
                    with contextlib.suppress(Exception):
                        os.killpg(proc.pid, signal.SIGKILL)
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=5)
        finally:
            if self._memory_log_handle is not None:
                self._memory_log_handle.close()
                self._memory_log_handle = None

    def memory_endpoint(self) -> str:
        return self._endpoint

    def memory_token(self) -> str:
        return self._server_token

    def _wait_for_health(self) -> None:
        deadline = time.time() + 60.0
        last_error: Exception | None = None
        while time.time() < deadline:
            proc = self._memory_proc
            if proc is not None and proc.poll() is not None:
                log_text = self._memory_log.read_text(encoding="utf-8") if self._memory_log.exists() else ""
                raise AssertionError(
                    f"memory server exited early with code {proc.returncode}\nlog:\n{log_text}"
                )
            try:
                response = requests.get(f"{self._endpoint}/health", timeout=2)
                response.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1.0)
        log_text = self._memory_log.read_text(encoding="utf-8") if self._memory_log.exists() else ""
        raise AssertionError(f"memory server failed health check: {last_error}\nlog:\n{log_text}")
