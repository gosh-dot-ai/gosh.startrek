# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import os
from collections.abc import Mapping

DETERMINISTIC_E2E_ENV = "GOSH_MEMORY_TEST_DETERMINISTIC_E2E"


def deterministic_e2e_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(DETERMINISTIC_E2E_ENV) or "").strip() == "1"
