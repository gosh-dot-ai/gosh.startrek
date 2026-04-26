#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

_PLUGIN_PACKAGE_DIR = [str(Path(__file__).resolve().parent)]
_SKIP_MODULES = {"__init__", "base", "registry", "runner"}

def get_codebase_semantic_plugins():
    plugins = []
    package_name = __package__ or "src.codebase_semantic_plugins"
    for module_info in sorted(iter_modules(_PLUGIN_PACKAGE_DIR), key=lambda item: item.name):
        module_name = module_info.name
        if module_name in _SKIP_MODULES or module_name.startswith("_"):
            continue
        module = import_module(f"{package_name}.{module_name}")
        plugin = getattr(module, "SEMANTIC_PLUGIN", None)
        if plugin is not None:
            plugins.append(plugin)
    return sorted(plugins, key=lambda plugin: plugin.plugin_name)
