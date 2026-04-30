# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import asyncio

import pytest

from src.mcp_server import mcp


def test_all_tools_use_underscore_naming():
    """All registered MCP tools must use underscore naming (no dots)."""
    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        assert "." not in t.name, (
            f"Tool '{t.name}' uses dot notation — must use underscore"
        )


def test_required_tools_registered():
    """All required active MCP tools are registered."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    required = {
        "memory_store", "memory_recall", "get_more_context", "memory_plan_inference", "memory_ingest_document", "memory_ingest",
        "memory_build_index", "memory_flush", "memory_migrate_jsonnpz", "memory_stats",
        "memory_reextract", "memory_list", "memory_get",
        "memory_import",
        "membership_grant", "membership_revoke", "membership_register", "membership_unregister", "membership_list",
        "courier_subscribe", "courier_unsubscribe",
        "memory_store_secret", "memory_list_secrets", "memory_delete_secret",
    }
    missing = required - tool_names
    assert not missing, f"Missing tools: {missing}"


def test_removed_secret_readback_tools_not_registered():
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert "memory_get_secret" not in tool_names
    assert "memory_rotate_secret" not in tool_names
