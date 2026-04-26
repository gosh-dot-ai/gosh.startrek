# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import re
from typing import Any

_TAG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _validate_tag(tag: str) -> str:
    normalized = str(tag or "").strip()
    if not _TAG_RE.fullmatch(normalized):
        raise ValueError(f"prompt data block tag must be ASCII uppercase, got {tag!r}")
    return normalized


def escape_prompt_data(text: str) -> str:
    value = str(text or "")
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_data_block(tag: str, content: str) -> str:
    block_tag = _validate_tag(tag)
    return f"<{block_tag}>\n{escape_prompt_data(content)}\n</{block_tag}>"


def render_kv_block(tag: str, data: dict[str, Any]) -> str:
    block_tag = _validate_tag(tag)
    lines: list[str] = []
    for key, value in data.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        value_text = str(value if value is not None else "")
        value_text = value_text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        lines.append(f"{key_text}={escape_prompt_data(value_text)}")
    return f"<{block_tag}>\n" + "\n".join(lines) + f"\n</{block_tag}>"
