# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
import subprocess
from typing import Any


class LocalCliTimeoutError(RuntimeError):
    """Raised when a local_cli subprocess exceeds the configured timeout."""


def _parse_timeout_secs(raw: Any) -> float | None:
    if raw is None:
        return None
    raw_s = str(raw).strip()
    if not raw_s:
        return None
    try:
        value = float(raw_s)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _local_cli_timeout_secs() -> float | None:
    return _parse_timeout_secs(os.getenv("GOSH_LOCAL_CLI_TIMEOUT_SECS", "120")) or 120.0


def _local_cli_empty_failure_retries() -> int:
    raw = os.getenv("GOSH_LOCAL_CLI_EMPTY_FAILURE_RETRIES", "1")
    try:
        value = int(str(raw).strip())
    except ValueError:
        return 1
    return min(max(value, 0), 3)


def _is_empty_local_cli_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    if completed.returncode == 0:
        return False
    return not (completed.stdout or "").strip() and not (completed.stderr or "").strip()


def render_local_cli_prompt(system: str, messages: list[dict]) -> str:
    blocks = [f"SYSTEM:\n{system}"]
    for message in messages:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", ""))
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)


def run_local_cli(
    prompt: str,
    cli_bin: str,
    cli_args_prefix: list[str],
    timeout_secs: float | None = None,
) -> str:
    resolved_timeout_secs = (
        _parse_timeout_secs(timeout_secs)
        if timeout_secs is not None
        else _local_cli_timeout_secs()
    )
    cmd = [cli_bin, *cli_args_prefix]
    attempts = 1 + _local_cli_empty_failure_retries()
    for attempt_index in range(attempts):
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=resolved_timeout_secs,
            )
        except subprocess.TimeoutExpired as exc:
            raise LocalCliTimeoutError(
                "local_cli subprocess timed out "
                f"(timeout_secs={resolved_timeout_secs}, cmd={cmd!r})"
            ) from exc
        if completed.returncode == 0:
            return completed.stdout or ""
        if _is_empty_local_cli_failure(completed) and attempt_index + 1 < attempts:
            continue
        stdout_summary = (completed.stdout or "").strip()
        stderr_summary = (completed.stderr or "").strip()
        raise RuntimeError(
            "local_cli subprocess failed "
            f"(exit_code={completed.returncode}, stdout={stdout_summary!r}, stderr={stderr_summary!r})"
        )
    return ""
