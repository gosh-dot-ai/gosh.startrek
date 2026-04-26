#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import os
from dataclasses import dataclass, field


def _env_default(name: str) -> str:
    return os.getenv(name, "")


@dataclass
class MemoryConfig:
    """Runtime configuration for gosh.memory.

    Controls which model is used for each pipeline stage.
    Stages are independent — mix providers freely.

    Example (production):
        cfg = MemoryConfig()  # reads from env

    Example (experiment):
        cfg = MemoryConfig(inference_model="anthropic/claude-sonnet-4-6")

    Example (CLI):
        cfg = MemoryConfig.from_args(args)
    """
    extraction_model: str = field(default_factory=lambda: _env_default("GOSH_EXTRACTION_MODEL"))
    inference_model:  str = field(default_factory=lambda: _env_default("GOSH_INFERENCE_MODEL"))
    judge_model:      str = field(default_factory=lambda: _env_default("GOSH_JUDGE_MODEL"))
    embed_model:      str = field(default_factory=lambda: _env_default("GOSH_EMBED_MODEL"))

    def summary(self) -> str:
        return (
            f"extraction={self.extraction_model} | "
            f"inference={self.inference_model} | "
            f"judge={self.judge_model} | "
            f"embed={self.embed_model}"
        )

    @classmethod
    def from_args(cls, args) -> "MemoryConfig":
        """Build from argparse Namespace. Individual flags override --model shortcut."""
        base = getattr(args, "model", None)
        return cls(
            extraction_model=getattr(args, "extraction_model", None) or base or _env_default("GOSH_EXTRACTION_MODEL"),
            inference_model =getattr(args, "inference_model",  None) or base or _env_default("GOSH_INFERENCE_MODEL"),
            judge_model     =getattr(args, "judge_model",      None) or base or _env_default("GOSH_JUDGE_MODEL"),
            embed_model     =getattr(args, "embed_model",      None) or _env_default("GOSH_EMBED_MODEL"),
        )
