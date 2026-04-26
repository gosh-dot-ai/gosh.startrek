# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

import json
from pathlib import Path

DEFAULT_MIN_SIGNALS = 10


class ControlStore:

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)

    def _control_path(self, key: str, agent_id: str) -> Path:
        return self._data_dir / "mal" / key / agent_id / "control.json"

    def get(self, key: str, agent_id: str) -> dict:
        path = self._control_path(key, agent_id)
        if not path.exists():
            return {
                "enabled": False,
                "auto_collect_feedback": False,
                "auto_trigger": False,
                "min_signals": DEFAULT_MIN_SIGNALS,
            }
        data = json.loads(path.read_text())
        data.setdefault("min_signals", DEFAULT_MIN_SIGNALS)
        return data

    def set(self, key: str, agent_id: str, **fields) -> None:
        current = self.get(key, agent_id)
        current.update(fields)
        path = self._control_path(key, agent_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current))

    def is_enabled(self, key: str, agent_id: str) -> bool:
        return self.get(key, agent_id)["enabled"]
