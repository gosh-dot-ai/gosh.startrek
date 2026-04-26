#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..codebase_semantic_bundle import stable_semantic_id
from .base import relative_file_path
from .base import span as _make_span


class EcosystemSemanticPlugin:
    plugin_name = "ecosystem_semantics"
    supported_extensions = frozenset({
        ".cfg",
        ".conf",
        ".gradle",
        ".hbs",
        ".html",
        ".ini",
        ".j2",
        ".jinja",
        ".json",
        ".mustache",
        ".po",
        ".properties",
        ".sql",
        ".tmpl",
        ".toml",
        ".xml",
        ".yaml",
        ".yml",
    })

    def supports_file(self, path: Path) -> bool:
        rel_path = str(path).replace("\\", "/")
        name = path.name
        return bool(
            path.suffix.lower() in self.supported_extensions
            or _manifest_kind(name)
            or _is_workflow_path(rel_path)
            or _generic_config_kind(name)
        )

    def build_bundle(
        self,
        *,
        repo_root: Path,
        repo_id: str,
        revision: str,
        files: list[Path],
    ) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        sidecars: list[dict[str, Any]] = []
        notes: list[str] = []
        object_ids_by_key: dict[tuple[str, str], str] = {}

        def add_object(
            *,
            object_type: str,
            rel_path: str,
            name: str,
            region: dict[str, int | None],
            payload: dict[str, Any],
        ) -> str:
            object_id = stable_semantic_id("obj", repo_id, revision, rel_path, object_type, name, region["start_line"], region["end_line"])
            objects.append(
                {
                    "id": object_id,
                    "object_type": object_type,
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_path,
                    "span": region,
                    "language": "config",
                    "analyzer_id": self.plugin_name,
                    "analyzer_version": "1",
                    "derivation_type": "observed",
                    "payload": dict(payload),
                }
            )
            object_ids_by_key[(object_type, rel_path + ":" + name)] = object_id
            return object_id

        def add_relation(relation_type: str, from_id: str, to_id: str, rel_path: str, region: dict[str, int | None], payload: dict[str, Any]) -> None:
            relations.append(
                {
                    "id": stable_semantic_id("rel", repo_id, revision, rel_path, relation_type, from_id, to_id),
                    "relation_type": relation_type,
                    "from_id": from_id,
                    "to_id": to_id,
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_path,
                    "span": region,
                    "language": "config",
                    "analyzer_id": self.plugin_name,
                    "analyzer_version": "1",
                    "derivation_type": "resolved",
                    "payload": dict(payload),
                }
            )

        def add_sidecar(rel_path: str, node_id: str, node_kind: str, region: dict[str, int | None], payload: dict[str, Any]) -> None:
            sidecar_id = stable_semantic_id("sidecar", repo_id, revision, rel_path, node_id, node_kind)
            sidecars.append(
                {
                    "sidecar_id": sidecar_id,
                    "sidecar_kind": "semantic_snapshot",
                    "format_family": "ecosystem_config",
                    "format_name": f"{node_kind}_v1",
                    "format_version": "1",
                    "encoding": "utf-8",
                    "compression": "none",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_path,
                    "span": region,
                    "node_id": node_id,
                    "storage_ref": f"inline:{sidecar_id}",
                    "content_hash": stable_semantic_id("payload", sidecar_id, json.dumps(payload, sort_keys=True, ensure_ascii=False)),
                    "byte_size": len(json.dumps(payload, ensure_ascii=False)),
                    "producer": self.plugin_name,
                    "metadata": {"language": "config", "node_kind": node_kind},
                    "payload": dict(payload),
                }
            )

        for file_path in sorted(files):
            rel_path = relative_file_path(repo_root, file_path)
            name = file_path.name
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                notes.append(f"ecosystem file skipped due to non-utf8 content: {rel_path}")
                continue
            lines = text.splitlines() or [""]
            file_region = _make_span(1, max(1, len(lines)))
            manifest_kind = _manifest_kind(rel_path)
            is_workflow = _is_workflow_path(rel_path)
            if manifest_kind is None and not is_workflow:
                config_kind = _generic_config_kind(rel_path)
                if config_kind is None:
                    continue
                root_type = config_kind
            else:
                root_type = "workflow" if is_workflow else "dependency_manifest"
            root_id = add_object(
                object_type=root_type,
                rel_path=rel_path,
                name=name,
                region=file_region,
                payload={
                    "name": name,
                    "qualified_name": rel_path,
                    "manifest_kind": manifest_kind,
                    "workflow": is_workflow,
                    "config_kind": root_type if root_type in {"config", "lockfile"} else None,
                },
            )
            add_sidecar(
                rel_path,
                root_id,
                root_type,
                file_region,
                {"file_path": rel_path, "span": file_region, "code": text, "manifest_kind": manifest_kind},
            )

            for command_name, command_text, command_line in _commands_for_file(rel_path, text, manifest_kind):
                command_region = _make_span(command_line, command_line)
                command_id = add_object(
                    object_type="command",
                    rel_path=rel_path,
                    name=command_name,
                    region=command_region,
                    payload={
                        "name": command_name,
                        "qualified_name": f"{rel_path}:{command_name}",
                        "command": command_text,
                        "cwd": ".",
                        "manifest_kind": manifest_kind,
                    },
                )
                add_relation(
                    "command_targets",
                    command_id,
                    root_id,
                    rel_path,
                    command_region,
                    {"command": command_text, "manifest_kind": manifest_kind},
                )

            for entry in _config_option_entries(text):
                option_region = _make_span(int(entry["start_line"]), int(entry["end_line"]))
                option_id = add_object(
                    object_type="config",
                    rel_path=rel_path,
                    name=str(entry["option"]),
                    region=option_region,
                    payload={
                        "name": str(entry["option"]),
                        "qualified_name": f"{rel_path}:{entry['option']}",
                        "config_object_kind": "option",
                        "option": str(entry["option"]),
                        "option_prefix": str(entry.get("option_prefix") or ""),
                        "type": str(entry.get("type") or ""),
                        "default": str(entry.get("default") or ""),
                        "references": list(entry.get("references") or []),
                        "default_tokens": list(entry.get("default_tokens") or []),
                        "leading_literal_token": str(entry.get("leading_literal_token") or ""),
                        "proof_source_fields": [
                            "semantic_bundle.object.payload.option",
                            "semantic_bundle.object.payload.default",
                            "semantic_bundle.object.payload.type",
                            "semantic_bundle.object.payload.references",
                        ],
                    },
                )
                add_relation(
                    "configures",
                    root_id,
                    option_id,
                    rel_path,
                    option_region,
                    {"config_object_kind": "option", "option": str(entry["option"])},
                )

            if is_workflow:
                for job_name, job_line in _workflow_jobs(text):
                    job_region = _make_span(job_line, job_line)
                    job_id = add_object(
                        object_type="workflow_job",
                        rel_path=rel_path,
                        name=job_name,
                        region=job_region,
                        payload={
                            "name": job_name,
                            "qualified_name": f"{rel_path}:{job_name}",
                            "workflow_id": rel_path,
                        },
                    )
                    add_relation("job_targets", job_id, root_id, rel_path, job_region, {"workflow_id": rel_path})

        return {
            "objects": objects,
            "relations": relations,
            "provenance": {
                "repo_id": repo_id,
                "revision": revision,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(repo_root),
            },
            "capability_report": {
                "supported_languages": ["config"],
                "supported_capabilities": [
                    "command_semantics",
                    "generic_config_semantics",
                    "dependency_manifest_semantics",
                    "workflow_semantics",
                ],
                "analyzer_protocol_version": "codebase-v1",
                "schema_version": "1",
            },
            "gap_report": {"skipped_files": [], "notes": notes},
            "sidecars": sidecars,
        }


def _manifest_kind(rel_path: str) -> str | None:
    lower = rel_path.lower()
    name = Path(rel_path).name.lower()
    if name == "package.json":
        return "npm_package_manifest"
    if name == "pyproject.toml":
        return "python_project_manifest"
    if name in {"tox.ini", "noxfile.py"}:
        return "python_test_manifest"
    if name == "cargo.toml":
        return "rust_package_manifest"
    if name == "go.mod":
        return "go_module_manifest"
    if name in {"pom.xml", "build.gradle", "build.gradle.kts"}:
        return "jvm_build_manifest"
    if name in {"makefile", "gnumakefile"}:
        return "make_manifest"
    if lower.endswith("requirements.txt"):
        return "python_requirements_manifest"
    return None


def _is_workflow_path(rel_path: str) -> bool:
    lower = rel_path.lower()
    return (
        lower.startswith(".github/workflows/")
        or "/.github/workflows/" in lower
    ) and (lower.endswith(".yml") or lower.endswith(".yaml"))


def _generic_config_kind(rel_path: str) -> str | None:
    lower = rel_path.lower()
    name = Path(rel_path).name.lower()
    if name in {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "cargo.lock", "poetry.lock"}:
        return "lockfile"
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "config"
    if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
        return "config"
    if "/migrations/" in f"/{lower}" and lower.endswith(".sql"):
        return "config"
    if "/templates/" in f"/{lower}" or lower.endswith((".j2", ".jinja", ".tmpl", ".mustache", ".hbs", ".html")):
        return "config"
    if "/i18n/" in f"/{lower}" or "/locales/" in f"/{lower}" or lower.endswith((".po", ".properties")):
        return "config"
    if lower.endswith((".json", ".toml", ".yaml", ".yml", ".xml", ".gradle", ".ini", ".cfg", ".conf", ".sql")):
        return "config"
    return None


def _commands_for_file(rel_path: str, text: str, manifest_kind: str | None) -> list[tuple[str, str, int]]:
    if manifest_kind == "npm_package_manifest":
        return _npm_script_commands(text)
    if manifest_kind == "python_project_manifest":
        commands = [("pytest", "pytest", 1)]
        if "[tool.tox]" in text:
            commands.append(("tox", "tox", _line_number(text, "[tool.tox]")))
        return commands
    if manifest_kind == "python_test_manifest":
        return [("pytest", "pytest", 1)]
    if manifest_kind == "rust_package_manifest":
        return [("cargo test", "cargo test", 1)]
    if manifest_kind == "go_module_manifest":
        return [("go test", "go test ./...", 1)]
    if manifest_kind == "jvm_build_manifest":
        command = "./gradlew test" if "gradle" in rel_path.lower() else "mvn test"
        return [(command, command, 1)]
    if manifest_kind == "make_manifest":
        return [("make test", "make test", _line_number(text, "test:"))]
    return []


def _npm_script_commands(text: str) -> list[tuple[str, str, int]]:
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    scripts = parsed.get("scripts")
    if not isinstance(scripts, dict):
        return []
    commands = []
    for name, command in sorted(scripts.items()):
        if not isinstance(name, str) or not isinstance(command, str):
            continue
        commands.append((f"npm:{name}", f"npm run {name}", _line_number(text, f'"{name}"')))
    return commands


def _workflow_jobs(text: str) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    in_jobs = False
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "jobs:":
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line and not line.startswith((" ", "\t")):
            break
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            job_name = stripped[:-1].strip()
            if job_name:
                jobs.append((job_name, idx))
    return jobs


def _line_number(text: str, needle: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return 1


def _config_default_tokens(default_value: str) -> list[dict[str, str]]:
    tokens: list[dict[str, str]] = []
    for raw in re.findall(r"[^\s]+", str(default_value or "")):
        token = raw.strip("\"'")
        lowered = token.lower()
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:pt|px)", lowered):
            classification = "size_literal"
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            classification = "symbol"
        elif re.fullmatch(r"[0-9]+", token):
            classification = "number_literal"
        else:
            classification = "literal"
        tokens.append({"token": token, "kind": classification})
    return tokens


def _config_option_entries(text: str) -> list[dict[str, Any]]:
    """Extract portable config option declarations at ingest time.

    This intentionally emits semantic objects only. Query/runtime code consumes
    these typed objects and does not parse config syntax during retrieval.
    """

    lines = text.splitlines()
    if not lines:
        return []
    key_rows: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^([A-Za-z0-9_.-]+):(?:\s*(.*))?$", line)
        if match and "." in match.group(1):
            key_rows.append((idx, match.group(1)))
    entries: list[dict[str, Any]] = []
    for row_idx, (start, option) in enumerate(key_rows):
        end = key_rows[row_idx + 1][0] if row_idx + 1 < len(key_rows) else len(lines)
        block_lines = lines[start:end]
        type_name = ""
        default_value = ""
        for offset, line in enumerate(block_lines):
            type_match = re.match(r"\s*type:\s*([A-Za-z0-9_.-]+)?\s*$", line)
            if type_match:
                type_name = str(type_match.group(1) or "").strip()
                if not type_name:
                    for nested in block_lines[offset + 1 : offset + 8]:
                        name_match = re.match(r"\s*name:\s*([A-Za-z0-9_.-]+)\s*$", nested)
                        if name_match:
                            type_name = name_match.group(1).strip()
                            break
            default_match = re.match(r"\s*default:\s*(.*)\s*$", line)
            if default_match:
                default_value = default_match.group(1).strip()
        default_tokens = _config_default_tokens(default_value)
        leading_literal_token = ""
        if default_tokens and default_tokens[0]["kind"] in {"size_literal", "number_literal", "literal"}:
            leading_literal_token = default_tokens[0]["token"]
        references = sorted(
            {
                row["token"]
                for row in default_tokens
                if row.get("kind") == "symbol" and str(row.get("token") or "").strip()
            }
        )
        entries.append(
            {
                "option": option,
                "option_prefix": option.lower().split(".", 1)[0],
                "type": type_name,
                "default": default_value,
                "start_line": start + 1,
                "end_line": max(start + 1, end),
                "references": references,
                "default_tokens": default_tokens,
                "leading_literal_token": leading_literal_token,
            }
        )
    return entries


SEMANTIC_PLUGIN = EcosystemSemanticPlugin()
