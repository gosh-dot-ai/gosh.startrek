# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import subprocess

import pytest

from src.local_cli_backend import LocalCliTimeoutError, render_local_cli_prompt, run_local_cli


def test_render_local_cli_prompt_renders_exact_expected_string():
    prompt = render_local_cli_prompt(
        "You are a test system.",
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ],
    )

    assert prompt == (
        "SYSTEM:\n"
        "You are a test system.\n\n"
        "USER:\n"
        "First question\n\n"
        "ASSISTANT:\n"
        "First answer\n\n"
        "USER:\n"
        "Second question"
    )


def test_run_local_cli_passes_cli_bin_args_and_prompt_via_stdin(monkeypatch):
    captured = {}
    large_prompt = "PROMPT BODY " * 10000

    def _fake_run(cmd, capture_output, text, shell, check, input, timeout):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["shell"] = shell
        captured["check"] = check
        captured["input"] = input
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = run_local_cli(large_prompt, "/abs/path/to/my-cli", ["run"])

    assert result == "ok"
    assert captured["cmd"] == ["/abs/path/to/my-cli", "run"]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["shell"] is False
    assert captured["check"] is False
    assert captured["input"] == large_prompt
    assert captured["timeout"] == 120.0


def test_run_local_cli_returns_stdout(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="cli answer\n", stderr=""),
    )

    assert run_local_cli("PROMPT", "/abs/path/to/my-cli", ["run"]) == "cli answer\n"


def test_run_local_cli_raises_on_non_zero_exit(monkeypatch):
    calls = []

    def _fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args[0],
            7,
            stdout="partial output",
            stderr="fatal error",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run,
    )

    with pytest.raises(RuntimeError, match="local_cli subprocess failed"):
        run_local_cli("PROMPT", "/abs/path/to/my-cli", ["run"])
    assert len(calls) == 1


def test_run_local_cli_retries_empty_non_zero_exit_once(monkeypatch):
    calls = []

    def _fake_run(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args[0], 0, stdout="ok after retry", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert run_local_cli("PROMPT", "/abs/path/to/my-cli", ["run"]) == "ok after retry"
    assert len(calls) == 2


def test_run_local_cli_empty_non_zero_exit_retry_can_be_disabled(monkeypatch):
    calls = []

    def _fake_run(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

    monkeypatch.setenv("GOSH_LOCAL_CLI_EMPTY_FAILURE_RETRIES", "0")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="local_cli subprocess failed"):
        run_local_cli("PROMPT", "/abs/path/to/my-cli", ["run"])
    assert len(calls) == 1


def test_run_local_cli_raises_timeout_error_on_subprocess_timeout(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(LocalCliTimeoutError, match="local_cli subprocess timed out"):
        run_local_cli("PROMPT", "/abs/path/to/my-cli", ["run"])
