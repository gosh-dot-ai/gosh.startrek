# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from pathlib import Path

import pytest

from src.prompt_registry import BUILTIN_PROMPTS, PromptRegistry

# ── key-based mode (existing behavior) ──


def test_key_mode_reads_from_librarian_prompts(tmp_path):
    prompts_dir = tmp_path / "librarian_prompts" / "mykey"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "technical.md").write_text("custom technical prompt")
    reg = PromptRegistry(data_dir=str(tmp_path), key="mykey")
    assert reg.get("technical") == "custom technical prompt"


def test_key_mode_falls_back_to_builtin(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key="mykey")
    prompt = reg.get("technical")
    assert prompt == BUILTIN_PROMPTS["technical"]


def test_key_mode_falls_back_to_default_builtin(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key="mykey")
    prompt = reg.get("nonexistent_type")
    assert prompt == BUILTIN_PROMPTS["default"]


# ── direct-dir mode (key=None, for MAL generations) ──


def test_direct_dir_mode_reads_from_conversation_subdir(tmp_path):
    conv_dir = tmp_path / "prompts" / "conversation"
    conv_dir.mkdir(parents=True)
    (conv_dir / "technical.md").write_text("generation-local technical prompt")
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    assert reg.get("technical") == "generation-local technical prompt"


def test_direct_dir_mode_falls_back_to_builtin(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    prompt = reg.get("technical")
    assert prompt == BUILTIN_PROMPTS["technical"]


def test_direct_dir_mode_falls_back_to_default(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    prompt = reg.get("nonexistent_type")
    assert prompt == BUILTIN_PROMPTS["default"]


def test_direct_dir_mode_exists_with_custom_file(tmp_path):
    conv_dir = tmp_path / "prompts" / "conversation"
    conv_dir.mkdir(parents=True)
    (conv_dir / "technical.md").write_text("custom")
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    assert reg.exists("technical") is True


def test_direct_dir_mode_exists_for_builtin(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    assert reg.exists("technical") is True
    assert reg.exists("nonexistent_xyz") is False


def test_direct_dir_mode_list_includes_custom_and_builtin(tmp_path):
    conv_dir = tmp_path / "prompts" / "conversation"
    conv_dir.mkdir(parents=True)
    (conv_dir / "custom_type.md").write_text("custom")
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    entries = reg.list()
    content_types = [e["content_type"] for e in entries]
    assert "custom_type" in content_types
    assert "default" in content_types
    assert "technical" in content_types


def test_direct_dir_mode_does_not_read_librarian_prompts(tmp_path):
    librarian_dir = tmp_path / "librarian_prompts"
    librarian_dir.mkdir(parents=True)
    (librarian_dir / "technical.md").write_text("this should NOT be found")
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    prompt = reg.get("technical")
    assert prompt != "this should NOT be found"
    assert prompt == BUILTIN_PROMPTS["technical"]


# ── set in direct-dir mode ──


def test_direct_dir_mode_set_writes_to_conversation_subdir(tmp_path):
    reg = PromptRegistry(data_dir=str(tmp_path), key=None)
    reg.set("technical", "new generation prompt")
    path = tmp_path / "prompts" / "conversation" / "technical.md"
    assert path.exists()
    assert path.read_text() == "new generation prompt"
    assert reg.get("technical") == "new generation prompt"
