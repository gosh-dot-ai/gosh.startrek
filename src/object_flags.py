# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import copy
import hashlib
from typing import Any

ALLOWED_FLAG_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
ALLOWED_FLAG_STATUSES = frozenset({"open", "resolved", "suppressed", "dropped"})
ALLOWED_OPTIONAL_FLAG_KEYS = frozenset({
    "code",
    "path",
    "object_id",
    "resolution",
    "repair_attempted",
    "details",
})
_REQUIRED_FLAG_KEYS = frozenset({
    "flag_id",
    "producer",
    "category",
    "severity",
    "message",
    "status",
})


def build_object_flag(
    *,
    producer: str,
    category: str,
    severity: str,
    message: str,
    status: str,
    code: str | None = None,
    path: str | None = None,
    object_id: str | None = None,
    resolution: str | None = None,
    repair_attempted: bool | None = None,
    details: dict | None = None,
    flag_id: str | None = None,
) -> dict[str, Any]:
    producer = str(producer or '').strip()
    category = str(category or '').strip()
    message = str(message or '').strip()
    if not producer:
        raise ValueError('producer must be a non-empty string')
    if not category:
        raise ValueError('category must be a non-empty string')
    if severity not in ALLOWED_FLAG_SEVERITIES:
        raise ValueError(f'unsupported flag severity: {severity}')
    if not message:
        raise ValueError('message must be a non-empty string')
    if not isinstance(status, str) or not status.strip():
        raise ValueError('status must be a non-empty string')
    if code is not None and not isinstance(code, str):
        raise ValueError('code must be a string or None')
    if path is not None and not isinstance(path, str):
        raise ValueError('path must be a string or None')
    if object_id is not None and not isinstance(object_id, str):
        raise ValueError('object_id must be a string or None')
    if resolution is not None and not isinstance(resolution, str):
        raise ValueError('resolution must be a string or None')
    if repair_attempted is not None and not isinstance(repair_attempted, bool):
        raise ValueError('repair_attempted must be a bool or None')
    if details is not None and not isinstance(details, dict):
        raise ValueError('details must be a dict or None')
    if flag_id is None:
        material = '|'.join([
            producer,
            category,
            severity,
            status.strip(),
            code or '',
            path or '',
            object_id or '',
            message,
        ])
        digest = hashlib.sha1(material.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
        flag_id = f'flag_{digest}'
    if not isinstance(flag_id, str) or not flag_id.strip():
        raise ValueError('flag_id must be a non-empty string')
    flag: dict[str, Any] = {
        'flag_id': flag_id,
        'producer': producer,
        'category': category,
        'severity': severity,
        'message': message,
        'status': status.strip(),
    }
    optional: dict[str, Any] = {
        'code': code,
        'path': path,
        'object_id': object_id,
        'resolution': resolution,
        'repair_attempted': repair_attempted,
        'details': copy.deepcopy(details) if details is not None else None,
    }
    for key, value in optional.items():
        if value is not None:
            flag[key] = value
    return flag


def normalize_legacy_object_flag(flag: Any) -> dict[str, Any] | None:
    if not isinstance(flag, dict):
        return None
    if 'category' in flag and 'producer' in flag and 'severity' in flag and 'message' in flag and 'status' in flag:
        return copy.deepcopy(flag)
    if flag.get('origin') != 'extraction':
        return copy.deepcopy(flag)
    details = {
        'raw_code': flag.get('raw_code'),
    }
    for key in ('layer', 'family', 'expected', 'actual', 'repair_hint'):
        if key in flag:
            details[key] = copy.deepcopy(flag.get(key))
    details = {k: v for k, v in details.items() if v is not None}
    return build_object_flag(
        flag_id=str(flag.get('flag_id') or '').strip() or None,
        producer=str(flag.get('producer') or '').strip() or 'unknown_extraction_producer',
        category='extraction',
        severity=str(flag.get('severity') or 'medium'),
        message=str(flag.get('message') or '').strip() or 'Legacy extraction flag',
        status=str(flag.get('status') or 'open').strip() or 'open',
        code=(str(flag.get('normalized_code')).strip() if flag.get('normalized_code') is not None else None),
        path=(str(flag.get('path')).strip() if flag.get('path') is not None else None),
        object_id=(str(flag.get('object_id')).strip() if flag.get('object_id') is not None else None),
        resolution=(str(flag.get('resolution')).strip() if flag.get('resolution') is not None else None),
        repair_attempted=flag.get('repair_attempted') if isinstance(flag.get('repair_attempted'), bool) else None,
        details=details or None,
    )


def normalize_legacy_object_flags(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    for item in value:
        normalized.append(normalize_legacy_object_flag(item))
    return normalized


def validate_object_flags(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return f'flags must be a list, got {type(value).__name__}'
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            return f'flags[{idx}] must be a dict, got {type(item).__name__}'
        missing = _REQUIRED_FLAG_KEYS - set(item)
        if missing:
            return f'flags[{idx}] missing required keys: {sorted(missing)}'
        if not isinstance(item.get('flag_id'), str) or not item['flag_id'].strip():
            return f'flags[{idx}].flag_id must be a non-empty string'
        if not isinstance(item.get('producer'), str) or not item['producer'].strip():
            return f'flags[{idx}].producer must be a non-empty string'
        if not isinstance(item.get('category'), str) or not item['category'].strip():
            return f'flags[{idx}].category must be a non-empty string'
        if item.get('severity') not in ALLOWED_FLAG_SEVERITIES:
            return f'flags[{idx}].severity is invalid'
        if not isinstance(item.get('message'), str) or not item['message'].strip():
            return f'flags[{idx}].message must be a non-empty string'
        if not isinstance(item.get('status'), str) or not item['status'].strip():
            return f'flags[{idx}].status must be a non-empty string'
        if 'code' in item and item.get('code') is not None and not isinstance(item.get('code'), str):
            return f'flags[{idx}].code must be a string or null'
        if 'path' in item and item.get('path') is not None and not isinstance(item.get('path'), str):
            return f'flags[{idx}].path must be a string or null'
        if 'object_id' in item and item.get('object_id') is not None and not isinstance(item.get('object_id'), str):
            return f'flags[{idx}].object_id must be a string or null'
        if 'resolution' in item and item.get('resolution') is not None and not isinstance(item.get('resolution'), str):
            return f'flags[{idx}].resolution must be a string or null'
        if 'repair_attempted' in item and item.get('repair_attempted') is not None and not isinstance(item.get('repair_attempted'), bool):
            return f'flags[{idx}].repair_attempted must be a boolean or null'
        if 'details' in item and item.get('details') is not None and not isinstance(item.get('details'), dict):
            return f'flags[{idx}].details must be a dict or null'
    return None


def normalize_object_flags_field(obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return obj
    if 'flags' not in obj:
        return obj
    normalized_flags = normalize_legacy_object_flags(obj.get('flags'))
    if normalized_flags is None:
        obj.pop('flags', None)
        return obj
    obj['flags'] = normalized_flags
    return obj
