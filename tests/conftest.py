# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from copy import deepcopy

import pytest

from tests._archive_repo import ensure_archive_repo_on_sys_path
from src.memory import MemoryServer

ensure_archive_repo_on_sys_path(auto_clone=False)


_TEST_RUNTIME_SECRET_REFS = {
    "embedding_secret_ref": {"name": "test-runtime-embedding", "scope": "system-wide"},
    "librarian_secret_ref": {"name": "test-runtime-librarian", "scope": "system-wide"},
    "inference_secret_ref": {"name": "test-runtime-inference", "scope": "system-wide"},
    "judge_secret_ref": {"name": "test-runtime-judge", "scope": "system-wide"},
}

_TEST_RUNTIME_SECRET_VALUES = {
    "test-runtime-embedding": "dummy-test-runtime-embedding-token",
    "test-runtime-librarian": "dummy-test-runtime-librarian-token",
    "test-runtime-inference": "dummy-test-runtime-inference-token",
    "test-runtime-judge": "dummy-test-runtime-judge-token",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests that run in the separate e2e suite")


@pytest.fixture(autouse=True)
def _enable_plaintext_secret_store_for_tests(monkeypatch):
    """All tests run in the explicit dev/test plaintext-secret mode unless they opt out."""
    monkeypatch.setenv("GOSH_MEMORY_ALLOW_PLAINTEXT_SECRETS", "1")


@pytest.fixture(autouse=True)
def _seed_default_runtime_secret_refs(monkeypatch, request):
    """Hermetic tests get explicit persisted secret refs instead of implicit env/config lookup."""

    node_path = str(getattr(request.node, "fspath", ""))
    skip_seed_files = {"test_secrets.py", "test_storage.py"}
    skip_seeding = any(node_path.endswith(name) for name in skip_seed_files)

    original_default_memory_config = MemoryServer._default_memory_config
    original_init = MemoryServer.__init__

    def _patched_default_memory_config(self):
        cfg = original_default_memory_config(self)
        for field_name, secret_ref in _TEST_RUNTIME_SECRET_REFS.items():
            if cfg.get(field_name) is None:
                cfg[field_name] = deepcopy(secret_ref)
        return cfg

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if skip_seeding:
            return
        secret_storage = self._secret_storage()
        if secret_storage is None:
            return
        for secret_name, secret_value in _TEST_RUNTIME_SECRET_VALUES.items():
            self.store_secret(
                secret_name,
                secret_value,
                scope="system-wide",
                caller_id="agent:test-runtime",
            )

    monkeypatch.setattr(MemoryServer, "_default_memory_config", _patched_default_memory_config)
    monkeypatch.setattr(MemoryServer, "__init__", _patched_init)
