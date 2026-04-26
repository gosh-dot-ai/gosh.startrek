#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".gosh-memory"
CONFIG_FILE = CONFIG_DIR / "config.json"

_ENV_MAP = {
    "openai":    "OPENAI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "inception": "INCEPTION_API_KEY",
}
_ENV_ALIASES = {
    "inception": ["INCEPTION_API_KEY", "MERCURY_API_KEY"],
}


# ── Config file ──

def load_config() -> dict:
    """Load config from ~/.gosh-memory/config.json. Returns {} if missing."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config: dict):
    """Save config to ~/.gosh-memory/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")


# ── API keys (env var → config file → None) ──

def store_api_key(provider: str, key: str):
    """Store API key in config file (chmod 600).

    Keys are stored in plaintext in ~/.gosh-memory/config.json.
    File permissions are set to 0o600 (owner read/write only).
    For production, prefer environment variables instead.
    """
    cfg = load_config()
    if "api_keys" not in cfg:
        cfg["api_keys"] = {}
    cfg["api_keys"][provider] = key
    save_config(cfg)
    # Tighten permissions on config file since it now contains keys
    try:
        os.chmod(str(CONFIG_FILE), 0o600)
    except OSError:
        pass


def get_api_key(provider: str) -> str | None:
    """Read one legacy bootstrap/import key from env or config.

    This helper is intentionally not a runtime source of truth.
    """
    # 1. Env var
    env_names = list(_ENV_ALIASES.get(provider, []))
    primary_env = _ENV_MAP.get(provider)
    if primary_env and primary_env not in env_names:
        env_names.insert(0, primary_env)
    for env_name in env_names:
        if not env_name:
            continue
        val = os.environ.get(env_name)
        if val:
            return val
    # 2. Config file
    cfg = load_config()
    api_keys = cfg.get("api_keys", {})
    if provider in api_keys:
        return api_keys[provider]
    return None


def delete_api_key(provider: str):
    """Delete API key from config file."""
    cfg = load_config()
    api_keys = cfg.get("api_keys", {})
    if provider in api_keys:
        del api_keys[provider]
        cfg["api_keys"] = api_keys
        save_config(cfg)


def is_configured() -> bool:
    """Check if gosh-memory has been configured."""
    config = load_config()
    return bool(config.get("provider"))


def list_configured_providers() -> list:
    """List providers that have API keys available."""
    return [p for p in _ENV_MAP if get_api_key(p)]


def get_config() -> dict:
    """Return non-secret config only.

    Runtime secret material is intentionally excluded. Provider secrets must be
    resolved from the persisted secret store, not from env/config.
    """
    cfg = load_config()
    # Resolve models from nested "models" dict written by setup wizard.
    # Defaults come from src/config.py (single source of truth).
    from .config import MemoryConfig
    models = cfg.get("models", {})
    defaults = MemoryConfig()
    cfg.setdefault("extraction_model", models.get("extraction", defaults.extraction_model))
    cfg.setdefault("embed_provider",   cfg.get("embedding", {}).get("provider", "openai"))
    cfg.setdefault("embed_model",      cfg.get("embedding", {}).get("model", defaults.embed_model))
    return cfg
