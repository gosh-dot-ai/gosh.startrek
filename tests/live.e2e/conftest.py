# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests._live_provider_stack import LiveProviderStack
from tests.e2e.conftest import (
    E2E_ADMIN_PRINCIPAL,
    E2E_BOOTSTRAP_ADMIN_TOKEN,
    PROJECT_ROOT,
    _missing_runtime_secret_requirements,
    _reserve_port,
)


DEFAULT_LIVE_EXTRACTION_MODEL = os.environ.get("GOSH_LIVE_E2E_EXTRACTION_MODEL", "qwen/qwen3-32b")
DEFAULT_LIVE_INFERENCE_MODEL = os.environ.get("GOSH_LIVE_E2E_INFERENCE_MODEL", "qwen/qwen3-32b")
DEFAULT_LIVE_EMBED_MODEL = os.environ.get("GOSH_LIVE_E2E_EMBED_MODEL") or None


@pytest.fixture(scope="module")
def live_provider_stack(tmp_path_factory: pytest.TempPathFactory) -> LiveProviderStack:
    inference_model = DEFAULT_LIVE_INFERENCE_MODEL or DEFAULT_LIVE_EXTRACTION_MODEL
    missing = _missing_runtime_secret_requirements(
        extraction_model=DEFAULT_LIVE_EXTRACTION_MODEL,
        default_inference_model=inference_model,
        embed_model=DEFAULT_LIVE_EMBED_MODEL,
        fast_profile_model=inference_model,
        balanced_profile_model=inference_model,
        strong_profile_model=inference_model,
        judge_model=inference_model,
    )
    if missing:
        pytest.skip("Requires real live runtime secrets: " + "; ".join(missing))
    temp_root = tmp_path_factory.mktemp("live_refactor_guard")
    stack = LiveProviderStack(
        project_root=PROJECT_ROOT,
        state_dir=Path(temp_root) / "state",
        data_dir=Path(temp_root) / "data",
        home_dir=Path(temp_root) / "home",
        tmp_dir=Path(temp_root) / "tmp",
        cli_bin=Path(sys.executable),
        agent_bin=Path(sys.executable),
        memory_port=_reserve_port(),
        bootstrap_admin_token=E2E_BOOTSTRAP_ADMIN_TOKEN,
        admin_principal_id=E2E_ADMIN_PRINCIPAL,
        extraction_model=DEFAULT_LIVE_EXTRACTION_MODEL,
        default_inference_model=DEFAULT_LIVE_INFERENCE_MODEL,
        embed_model=DEFAULT_LIVE_EMBED_MODEL,
    )
    stack.start_memory()
    yield stack
    stack.cleanup()
