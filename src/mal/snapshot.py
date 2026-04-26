# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from copy import deepcopy


class Snapshot:
    def __init__(self, server):
        """Capture frozen state from a live MemoryServer."""
        self.raw_sessions = list(server._raw_sessions)
        self.raw_docs = dict(server._raw_docs)
        self.all_granular = list(server._all_granular)
        self.all_cons = list(server._all_cons)
        self.all_cross = list(server._all_cross)
        self.episode_corpus = deepcopy(server._episode_corpus) if hasattr(server, '_episode_corpus') else {}
        self.config = {}  # active config at snapshot time
        self.prompts = {}  # active prompts at snapshot time

    def eval_top_k(self) -> int:
        """Fixed eval_top_k from snapshot config."""
        from ..episode_retrieval import resolve_selection_config
        selector = resolve_selection_config(self.config.get("selector_config_overrides"))
        return selector.get("max_episodes_default", 3)
