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

from .object_flags import normalize_legacy_object_flags, normalize_object_flags_field, validate_object_flags

_ALLOWED_REPORT_STATUSES = frozenset({'ok', 'partial', 'failed', 'dropped', 'warning'})
_ALLOWED_REPORT_ENTRY_STATUSES = frozenset({'ok', 'partial', 'failed', 'dropped', 'warning', 'resolved'})


def _looks_like_report_object(obj: dict[str, Any]) -> bool:
    return all(key in obj for key in ('report_kind', 'producer', 'status', 'entries'))


def validate_report_object(report: Any) -> str | None:
    if not isinstance(report, dict):
        return f'report must be a dict, got {type(report).__name__}'
    if 'report_id' in report and (not isinstance(report.get('report_id'), str) or not str(report.get('report_id') or '').strip()):
        return 'report.report_id must be a non-empty string'
    if not isinstance(report.get('report_kind'), str) or not report['report_kind'].strip():
        return 'report.report_kind must be a non-empty string'
    if not isinstance(report.get('producer'), str) or not report['producer'].strip():
        return 'report.producer must be a non-empty string'
    if report.get('status') not in _ALLOWED_REPORT_STATUSES:
        return f"unsupported report status: {report.get('status')}"
    if not isinstance(report.get('entries'), list):
        return 'report.entries must be a list'
    if 'summary' in report and report.get('summary') is not None and not isinstance(report.get('summary'), dict):
        return 'report.summary must be a dict or None'
    flag_error = validate_object_flags(report.get('flags'))
    if flag_error:
        return flag_error
    for idx, entry in enumerate(report.get('entries') or []):
        if not isinstance(entry, dict):
            return f'report.entries[{idx}] must be a dict'
        if not isinstance(entry.get('entry_id'), str) or not entry['entry_id'].strip():
            return f'report.entries[{idx}].entry_id must be a non-empty string'
        if entry.get('target_path') is not None and not isinstance(entry.get('target_path'), str):
            return f'report.entries[{idx}].target_path must be a string or None'
        if entry.get('status') not in _ALLOWED_REPORT_ENTRY_STATUSES:
            return f"unsupported report entry status: {entry.get('status')}"
        if entry.get('repair_attempted') is not None and not isinstance(entry.get('repair_attempted'), bool):
            return f'report.entries[{idx}].repair_attempted must be a bool or None'
        if entry.get('issue') is not None and not isinstance(entry.get('issue'), dict):
            return f'report.entries[{idx}].issue must be a dict or None'
        if entry.get('details') is not None and not isinstance(entry.get('details'), dict):
            return f'report.entries[{idx}].details must be a dict or None'
        flag_error = validate_object_flags(entry.get('flags'))
        if flag_error:
            return flag_error.replace('flags[', f'report.entries[{idx}].flags[')
    return None


def build_report_entry(
    *,
    target_path: str | None,
    status: str,
    repair_attempted: bool | None = None,
    issue: dict | None = None,
    details: dict | None = None,
    flags: list[dict] | None = None,
    entry_id: str | None = None,
) -> dict[str, Any]:
    if status not in _ALLOWED_REPORT_ENTRY_STATUSES:
        raise ValueError(f'unsupported report entry status: {status}')
    if issue is not None and not isinstance(issue, dict):
        raise ValueError('report entry issue must be a dict or None')
    if details is not None and not isinstance(details, dict):
        raise ValueError('report entry details must be a dict or None')
    normalized_flags = normalize_legacy_object_flags(copy.deepcopy(flags)) if flags is not None else None
    flag_error = validate_object_flags(normalized_flags)
    if flag_error:
        raise ValueError(flag_error)
    if entry_id is None:
        material = "|".join([
            status.strip(),
            str(target_path or ""),
            str((issue or {}).get('normalized_code') or ""),
            str((issue or {}).get('message') or ""),
        ])
        digest = hashlib.sha1(material.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
        entry_id = f'report_entry_{digest}'
    entry = {
        'entry_id': entry_id,
        'target_path': target_path,
        'status': status.strip(),
        'repair_attempted': repair_attempted,
        'issue': copy.deepcopy(issue) if issue is not None else None,
        'details': copy.deepcopy(details) if details is not None else None,
    }
    if normalized_flags:
        entry['flags'] = normalized_flags
    normalize_object_flags_field(entry)
    validation_error = validate_report_object({
        'report_id': 'report_validate_stub',
        'report_kind': 'validation',
        'producer': 'object_reports',
        'status': 'ok',
        'entries': [entry],
        'summary': None,
    })
    if validation_error:
        raise ValueError(validation_error)
    return entry


def build_report(
    *,
    report_kind: str,
    producer: str,
    status: str,
    entries: list[dict] | None = None,
    summary: dict | None = None,
    flags: list[dict] | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    report_kind = str(report_kind or "").strip()
    producer = str(producer or "").strip()
    if not report_kind:
        raise ValueError('report_kind must be a non-empty string')
    if not producer:
        raise ValueError('producer must be a non-empty string')
    if status not in _ALLOWED_REPORT_STATUSES:
        raise ValueError(f'unsupported report status: {status}')
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise ValueError('entries must be a list')
    if summary is not None and not isinstance(summary, dict):
        raise ValueError('summary must be a dict or None')
    normalized_flags = normalize_legacy_object_flags(copy.deepcopy(flags)) if flags is not None else None
    flag_error = validate_object_flags(normalized_flags)
    if flag_error:
        raise ValueError(flag_error)
    if report_id is None:
        material = "|".join([producer, report_kind, status, str(len(entries))])
        digest = hashlib.sha1(material.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
        report_id = f'report_{digest}'
    report: dict[str, Any] = {
        'report_id': report_id,
        'report_kind': report_kind,
        'producer': producer,
        'status': status,
        'entries': copy.deepcopy(entries),
        'summary': copy.deepcopy(summary) if summary is not None else None,
    }
    if normalized_flags:
        report['flags'] = normalized_flags
    normalize_object_flags_field(report)
    entries_list = report.get('entries')
    if isinstance(entries_list, list):
        for entry in entries_list:
            if isinstance(entry, dict):
                normalize_object_flags_field(entry)
    validation_error = validate_report_object(report)
    if validation_error:
        raise ValueError(validation_error)
    return report


def _report_status_from_entries(entries: list[dict]) -> str:
    if not entries:
        return 'ok'
    statuses = {str((entry or {}).get('status') or "").strip() for entry in entries if isinstance(entry, dict)}
    if 'failed' in statuses:
        return 'failed'
    if 'dropped' in statuses:
        return 'partial'
    if 'warning' in statuses:
        return 'warning'
    if 'partial' in statuses:
        return 'partial'
    return 'ok'


def normalize_legacy_report_fields(obj: dict[str, Any] | None, *, producer: str | None = None, report_kind: str | None = None) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return obj
    normalize_object_flags_field(obj)
    if 'diagnostics' in obj and 'report_id' not in obj:
        entries = []
        for item in obj.get('diagnostics') or []:
            if not isinstance(item, dict):
                continue
            entries.append(build_report_entry(
                entry_id=item.get('entry_id'),
                target_path=item.get('target_path'),
                status=str(item.get('status') or 'warning'),
                repair_attempted=item.get('repair_attempted') if isinstance(item.get('repair_attempted'), bool) else None,
                issue=item.get('issue') if isinstance(item.get('issue'), dict) else None,
                details={k: copy.deepcopy(v) for k, v in item.items() if k not in {'entry_id', 'target_path', 'status', 'repair_attempted', 'issue', 'flags'}},
                flags=item.get('flags') if isinstance(item.get('flags'), list) else None,
            ))
        normalized_report = build_report(
            report_id=obj.get('report_id'),
            report_kind=str(obj.get('report_kind') or report_kind or 'validation'),
            producer=str(obj.get('producer') or producer or 'runtime'),
            status=str(obj.get('status') or _report_status_from_entries(entries)),
            entries=entries,
            summary=obj.get('summary') if isinstance(obj.get('summary'), dict) else ({'entry_count': len(entries)} if entries else None),
            flags=obj.get('flags') if isinstance(obj.get('flags'), list) else None,
        )
        obj.clear()
        obj.update(normalized_report)
        return obj
    for old_key, kind in (
        ('extraction_diagnostics', 'extraction'),
        ('source_aggregation_diagnostics', 'source_aggregation'),
    ):
        new_key = old_key.replace('_diagnostics', '_report')
        if old_key in obj and new_key not in obj:
            normalized = normalize_legacy_report_fields(obj.get(old_key), producer=producer, report_kind=kind)
            if isinstance(normalized, dict):
                obj[new_key] = normalized
            obj.pop(old_key, None)
    for key in ('extraction_report', 'source_aggregation_report'):
        if isinstance(obj.get(key), dict):
            nested = normalize_legacy_report_fields(obj[key], producer=producer, report_kind=report_kind)
            if isinstance(nested, dict):
                obj[key] = nested
    if isinstance(obj.get('entries'), list):
        for entry in obj['entries']:
            if isinstance(entry, dict):
                normalize_object_flags_field(entry)
    if _looks_like_report_object(obj):
        validation_error = validate_report_object(obj)
        if validation_error:
            raise ValueError(validation_error)
    if 'diagnostics' in obj:
        obj.pop('diagnostics', None)
    return obj
