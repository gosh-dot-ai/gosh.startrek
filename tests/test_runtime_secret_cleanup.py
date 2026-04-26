# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path


RUNTIME_FILES = {
    "src/common.py",
    "src/providers.py",
    "src/memory.py",
    "src/mcp_server.py",
    "src/gosh_secrets.py",
}

FORBIDDEN_RUNTIME_PATTERNS = {
    "setup_store.get_api_key",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "INCEPTION_API_KEY",
    "MERCURY_API_KEY",
    "resolve_runtime_provider_secret(",
    "_resolve_runtime_provider_secret(",
}


def test_runtime_files_no_longer_reference_legacy_provider_env_or_setup_store_paths():
    for rel_path in RUNTIME_FILES:
        content = Path(rel_path).read_text(encoding="utf-8")
        for pattern in FORBIDDEN_RUNTIME_PATTERNS:
            assert pattern not in content, f"{pattern} leaked back into {rel_path}"


def test_runtime_files_do_not_merge_plaintext_api_keys_into_runtime_config():
    for rel_path in ("src/common.py", "src/providers.py", "src/memory.py"):
        content = Path(rel_path).read_text(encoding="utf-8")
        assert "api_keys" not in content, f"legacy plaintext api_keys path leaked back into {rel_path}"
