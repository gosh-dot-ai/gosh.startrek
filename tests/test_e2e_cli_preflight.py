# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import (
    _assert_agent_binary_matches_dev_contract,
    _assert_cli_binary_matches_dev_contract,
    _resolve_binary,
)


def _write_script(
    path: Path,
    *,
    top_help: str,
    serve_help: str | None = None,
    memory_start_help: str | None = None,
    top_exit: int = 0,
    serve_exit: int = 0,
    memory_start_exit: int = 0,
) -> Path:
    body = [
        "#!/usr/bin/env bash",
        'if [ "$#" -eq 1 ] && [ "$1" = "--help" ]; then',
        "cat <<'EOH'",
        top_help,
        "EOH",
        f"exit {top_exit}",
        "fi",
    ]
    if serve_help is not None:
        body.extend(
            [
                'if [ "$#" -eq 2 ] && [ "$1" = "serve" ] && [ "$2" = "--help" ]; then',
                "cat <<'EOH'",
                serve_help,
                "EOH",
                f"exit {serve_exit}",
                "fi",
            ]
        )
    if memory_start_help is not None:
        body.extend(
            [
                'if [ "$#" -eq 3 ] && [ "$1" = "memory" ] && [ "$2" = "start" ] && [ "$3" = "--help" ]; then',
                "cat <<'EOH'",
                memory_start_help,
                "EOH",
                f"exit {memory_start_exit}",
                "fi",
            ]
        )
    body.extend([
        'echo "unexpected args: $*" >&2',
        'exit 1',
    ])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_cli_preflight_accepts_current_dev_markers(tmp_path: Path) -> None:
    cli_bin = _write_script(
        tmp_path / "gosh",
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  agent   Manage gosh-agent instances
  status  Show status of all running services
  help     Print this message or the help of the given subcommand(s)

Options:
      --test-mode  Test mode: use file-based keychain instead of OS keychain
  -h, --help       Print help
""".strip(),
        memory_start_help="""
Start a local memory instance

Usage: gosh memory start [OPTIONS]

Options:
      --instance <INSTANCE>  Instance name (defaults to current)
      --test-mode            Test mode: use file-based keychain instead of OS keychain
  -h, --help                 Print help
""".strip(),
    )

    _assert_cli_binary_matches_dev_contract(cli_bin)


def test_cli_preflight_rejects_stale_binary_missing_state_dir(tmp_path: Path) -> None:
    cli_bin = _write_script(
        tmp_path / "gosh",
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  agent   Manage gosh-agent instances
  status  Show status of all running services
  help    Print this message or the help of the given subcommand(s)
""".strip(),
    )

    with pytest.raises(AssertionError, match="memory start --help"):
        _assert_cli_binary_matches_dev_contract(cli_bin)


def test_agent_preflight_accepts_current_dev_markers(tmp_path: Path) -> None:
    agent_bin = _write_script(
        tmp_path / "gosh-agent",
        top_help="""
GOSH AI Agent — MCP server, capture plugin, MCP proxy

Usage: gosh-agent <COMMAND>

Commands:
  serve  Run as MCP server (default agent mode)
  help   Print this message or the help of the given subcommand(s)
""".strip(),
        serve_help="""
Run as MCP server (default agent mode)

Usage: gosh-agent serve [OPTIONS]

Options:
      --bootstrap-file <BOOTSTRAP_FILE>
  -h, --help
""".strip(),
    )

    _assert_agent_binary_matches_dev_contract(agent_bin)


def test_agent_preflight_rejects_stale_binary_missing_bootstrap_file(tmp_path: Path) -> None:
    agent_bin = _write_script(
        tmp_path / "gosh-agent",
        top_help="""
GOSH AI Agent — MCP server, capture plugin, MCP proxy

Usage: gosh-agent <COMMAND>

Commands:
  serve  Run as MCP server (default agent mode)
  help   Print this message or the help of the given subcommand(s)
""".strip(),
        serve_help="""
Run as MCP server (default agent mode)

Usage: gosh-agent serve [OPTIONS]

Options:
  -h, --help
""".strip(),
    )

    with pytest.raises(AssertionError, match="missing --bootstrap-file"):
        _assert_agent_binary_matches_dev_contract(agent_bin)


def test_resolve_binary_falls_back_to_release_when_debug_remains_incompatible_after_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "gosh.cli"
    debug_bin = repo_dir / "target" / "debug" / "gosh"
    release_bin = repo_dir / "target" / "release" / "gosh"
    debug_bin.parent.mkdir(parents=True)
    release_bin.parent.mkdir(parents=True)

    _write_script(
        debug_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  agent   Manage gosh-agent instances
  status  Show status of all running services
  help    Print this message or the help of the given subcommand(s)
""".strip(),
    )
    _write_script(
        release_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  agent   Manage gosh-agent instances
  status  Show status of all running services
  help     Print this message or the help of the given subcommand(s)

Options:
      --test-mode  Test mode: use file-based keychain instead of OS keychain
  -h, --help       Print help
""".strip(),
        memory_start_help="""
Start a local memory instance

Usage: gosh memory start [OPTIONS]

Options:
      --instance <INSTANCE>  Instance name (defaults to current)
      --test-mode            Test mode: use file-based keychain instead of OS keychain
  -h, --help                 Print help
""".strip(),
    )

    cargo_bin = tmp_path / "cargo"
    cargo_bin.write_text(
        "#!/usr/bin/env bash\n"
        "exit 0\n",
        encoding="utf-8",
    )
    cargo_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    resolved = _resolve_binary(
        "GOSH_CLI_BIN",
        repo_dir=repo_dir,
        binary_name="gosh",
        validator=_assert_cli_binary_matches_dev_contract,
    )

    assert resolved == release_bin


def test_resolve_binary_uses_release_when_stale_debug_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "gosh.cli"
    debug_bin = repo_dir / "target" / "debug" / "gosh"
    release_bin = repo_dir / "target" / "release" / "gosh"
    repo_dir.mkdir()
    debug_bin.parent.mkdir(parents=True)
    release_bin.parent.mkdir(parents=True)

    _write_script(
        debug_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  help    Print this message or the help of the given subcommand(s)
""".strip(),
    )
    _write_script(
        release_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  agent   Manage gosh-agent instances
  status  Show status of all running services
  help     Print this message or the help of the given subcommand(s)

Options:
      --test-mode  Test mode: use file-based keychain instead of OS keychain
  -h, --help       Print help
""".strip(),
        memory_start_help="""
Start a local memory instance

Usage: gosh memory start [OPTIONS]

Options:
      --instance <INSTANCE>  Instance name (defaults to current)
      --test-mode            Test mode: use file-based keychain instead of OS keychain
  -h, --help                 Print help
""".strip(),
    )

    cargo_bin = tmp_path / "cargo"
    cargo_bin.write_text(
        "#!/usr/bin/env bash\n"
        "echo broken build >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    cargo_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    resolved = _resolve_binary(
        "GOSH_CLI_BIN",
        repo_dir=repo_dir,
        binary_name="gosh",
        validator=_assert_cli_binary_matches_dev_contract,
    )

    assert resolved == release_bin


def test_resolve_binary_rebuilds_stale_existing_debug_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "gosh.cli"
    debug_bin = repo_dir / "target" / "debug" / "gosh"
    release_bin = repo_dir / "target" / "release" / "gosh"
    repo_dir.mkdir()
    debug_bin.parent.mkdir(parents=True)
    release_bin.parent.mkdir(parents=True)

    _write_script(
        debug_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  help    Print this message or the help of the given subcommand(s)
""".strip(),
    )
    _write_script(
        release_bin,
        top_help="""
CLI for gosh.memory and gosh-agent

Usage: gosh [OPTIONS] <COMMAND>

Commands:
  memory  Manage gosh.memory instances
  help    Print this message or the help of the given subcommand(s)
""".strip(),
    )

    cargo_bin = tmp_path / "cargo"
    cargo_bin.write_text(
        "#!/usr/bin/env bash\n"
        f"cat > '{debug_bin}' <<'EOH'\n"
        "#!/usr/bin/env bash\n"
        "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"--help\" ]; then\n"
        "cat <<'EOX'\n"
        "CLI for gosh.memory and gosh-agent\n\n"
        "Usage: gosh [OPTIONS] <COMMAND>\n\n"
        "Commands:\n"
        "  memory  Manage gosh.memory instances\n"
        "  agent   Manage gosh-agent instances\n"
        "  status  Show status of all running services\n"
        "  help     Print this message or the help of the given subcommand(s)\n\n"
        "Options:\n"
        "      --test-mode  Test mode: use file-based keychain instead of OS keychain\n"
        "  -h, --help       Print help\n"
        "EOX\n"
        "exit 0\n"
        "fi\n"
        "if [ \"$#\" -eq 3 ] && [ \"$1\" = \"memory\" ] && [ \"$2\" = \"start\" ] && [ \"$3\" = \"--help\" ]; then\n"
        "cat <<'EOX'\n"
        "Start a local memory instance\n\n"
        "Usage: gosh memory start [OPTIONS]\n\n"
        "Options:\n"
        "      --instance <INSTANCE>  Instance name (defaults to current)\n"
        "      --test-mode            Test mode: use file-based keychain instead of OS keychain\n"
        "  -h, --help                 Print help\n"
        "EOX\n"
        "exit 0\n"
        "fi\n"
        "exit 1\n"
        "EOH\n"
        f"chmod +x '{debug_bin}'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    cargo_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    resolved = _resolve_binary(
        "GOSH_CLI_BIN",
        repo_dir=repo_dir,
        binary_name="gosh",
        validator=_assert_cli_binary_matches_dev_contract,
    )

    assert resolved == debug_bin
    assert debug_bin.is_file()
    _assert_cli_binary_matches_dev_contract(debug_bin)


def test_resolve_binary_builds_missing_debug_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "gosh.cli"
    repo_dir.mkdir()
    cargo_bin = tmp_path / "cargo"
    debug_bin = repo_dir / "target" / "debug" / "gosh"
    cargo_bin.write_text(
        "#!/usr/bin/env bash\n"
        f"mkdir -p '{debug_bin.parent}'\n"
        f"cat > '{debug_bin}' <<'EOH'\n"
        "#!/usr/bin/env bash\n"
        "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"--help\" ]; then\n"
        "cat <<'EOX'\n"
        "CLI for gosh.memory and gosh-agent\n\n"
        "Usage: gosh [OPTIONS] <COMMAND>\n\n"
        "Commands:\n"
        "  memory  Manage gosh.memory instances\n"
        "  agent   Manage gosh-agent instances\n"
        "  status  Show status of all running services\n"
        "  help     Print this message or the help of the given subcommand(s)\n\n"
        "Options:\n"
        "      --test-mode  Test mode: use file-based keychain instead of OS keychain\n"
        "  -h, --help       Print help\n"
        "EOX\n"
        "exit 0\n"
        "fi\n"
        "if [ \"$#\" -eq 3 ] && [ \"$1\" = \"memory\" ] && [ \"$2\" = \"start\" ] && [ \"$3\" = \"--help\" ]; then\n"
        "cat <<'EOX'\n"
        "Start a local memory instance\n\n"
        "Usage: gosh memory start [OPTIONS]\n\n"
        "Options:\n"
        "      --instance <INSTANCE>  Instance name (defaults to current)\n"
        "      --test-mode            Test mode: use file-based keychain instead of OS keychain\n"
        "  -h, --help                 Print help\n"
        "EOX\n"
        "exit 0\n"
        "fi\n"
        "exit 1\n"
        "EOH\n"
        f"chmod +x '{debug_bin}'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    cargo_bin.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    resolved = _resolve_binary(
        "GOSH_CLI_BIN",
        repo_dir=repo_dir,
        binary_name="gosh",
        validator=_assert_cli_binary_matches_dev_contract,
    )

    assert resolved == debug_bin
    assert debug_bin.is_file()
