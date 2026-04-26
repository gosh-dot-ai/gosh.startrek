# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / "audit.jsonl"

    def log(self, event: str, caller_id: str, details: dict = None) -> None:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "event": event, "caller_id": caller_id}
        if details:
            entry.update(details)
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
