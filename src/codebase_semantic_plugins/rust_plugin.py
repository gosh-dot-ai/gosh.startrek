#!/usr/bin/env python3
# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

import html
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..codebase_semantic_bundle import merge_semantic_bundles, stable_semantic_id
from .base import relative_file_path
from .base import snippet_for_span as _snippet_for_span
from .base import span as _make_span

_RUST_SIDEBAR_RE = re.compile(r"window\.SIDEBAR_ITEMS\s*=\s*(\{.*\});", re.S)
_RUST_SOURCE_LINK_RE = re.compile(r'href="\.\./src/([^"#]+)#(\d+)(?:-(\d+))?"')
_RUST_SIGNATURE_RE = re.compile(r'<pre class="rust item-decl"><code>(.*?)</code></pre>', re.S)
_RUST_FIELD_RE = re.compile(r'<span id="structfield\.([^"]+)"[^>]*><[^>]*>§</a><code>(.*?)</code></span>', re.S)
_RUST_PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:")

def _run_cargo_metadata(manifest_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def _normalize_crate_name(name: str) -> str:
    return name.replace("-", "_")


def _parse_rust_source_href(crate_root: Path, crate_name: str, href: str) -> tuple[Path, dict[str, int | None]]:
    match = _RUST_SOURCE_LINK_RE.search(f'href="{href}"')
    if not match:
        raise ValueError(f"unable to parse rustdoc source href: {href}")
    relative_html = match.group(1)
    start_line = int(match.group(2))
    end_line = int(match.group(3) or start_line)
    html_path = Path(relative_html)
    parts = html_path.parts
    if len(parts) < 2 or parts[0] != crate_name:
        raise ValueError(f"unexpected rustdoc source path: {relative_html}")
    source_rel = Path("src").joinpath(*parts[1:]).with_suffix("")
    return crate_root / source_rel, _make_span(start_line, end_line)


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return html.unescape(text).strip()


def _rust_signature(html_text: str) -> str:
    match = _RUST_SIGNATURE_RE.search(html_text)
    if not match:
        return ""
    return _strip_html(match.group(1))


def _rustdoc_bundle_for_manifest(manifest_path: Path, repo_id: str, revision: str) -> dict[str, Any]:
    metadata = _run_cargo_metadata(manifest_path)
    packages = metadata.get("packages") or []
    package = next((row for row in packages if row.get("manifest_path") == str(manifest_path)), packages[0] if packages else None)
    if not isinstance(package, dict):
        raise ValueError(f"unable to resolve cargo package for {manifest_path}")
    crate_name = _normalize_crate_name(str(package.get("name") or manifest_path.parent.name))
    crate_root = manifest_path.parent
    target_dir = Path(str(metadata.get("target_directory") or (crate_root / "target")))
    subprocess.run(
        [
            "cargo",
            "doc",
            "--no-deps",
            "--document-private-items",
            "--manifest-path",
            str(manifest_path),
        ],
        cwd=crate_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    docs_root = target_dir / "doc" / crate_name
    sidebar_path = docs_root / "sidebar-items.js"
    if not sidebar_path.exists():
        raise ValueError(f"rustdoc sidebar not found for crate {crate_name}")
    sidebar_text = sidebar_path.read_text(encoding="utf-8")
    match = _RUST_SIDEBAR_RE.search(sidebar_text)
    if not match:
        raise ValueError(f"unable to parse rustdoc sidebar for crate {crate_name}")
    sidebar_items = json.loads(match.group(1))

    plugin_id = "rustdoc"
    plugin_version = subprocess.run(
        ["rustdoc", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    objects: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []

    module_objects: dict[str, str] = {}

    def ensure_module(file_path: Path) -> tuple[str, str]:
        rel_path = relative_file_path(crate_root, file_path)
        existing = module_objects.get(rel_path)
        if existing:
            return existing, rel_path
        source = file_path.read_text(encoding="utf-8")
        span = _make_span(1, max(1, len(source.splitlines())))
        module_name = f"{crate_name}::{rel_path.replace('/', '::').removesuffix('.rs')}"
        module_id = stable_semantic_id("obj", repo_id, revision, rel_path, "module", module_name)
        sidecar_id = stable_semantic_id("sidecar", repo_id, revision, rel_path, module_id, "module")
        sidecar_payload: dict[str, Any] = {
            "fragment_id": stable_semantic_id("fragment", module_id),
            "root_node_id": module_id,
            "root_kind": "rust_source_file",
            "file_path": rel_path,
            "span": span,
            "parent_node_id": None,
            "child_node_ids": [],
            "payload_ref": None,
            "code": _snippet_for_span(source, span),
        }
        sidecars.append(
            {
                "sidecar_id": sidecar_id,
                "sidecar_kind": "semantic_snapshot",
                "format_family": "rustdoc",
                "format_name": "rustdoc_fragment_v1",
                "format_version": "1",
                "encoding": "utf-8",
                "compression": "none",
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "node_id": module_id,
                "storage_ref": f"inline:{sidecar_id}",
                "content_hash": stable_semantic_id("payload", sidecar_id, json.dumps(sidecar_payload, sort_keys=True)),
                "byte_size": len(json.dumps(sidecar_payload, ensure_ascii=False)),
                "producer": plugin_id,
                "metadata": {"language": "rust", "node_kind": "module"},
                "payload": sidecar_payload,
            }
        )
        objects.append(
            {
                "id": module_id,
                "object_type": "module",
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "language": "rust",
                "analyzer_id": plugin_id,
                "analyzer_version": plugin_version,
                "derivation_type": "observed",
                "payload": {
                    "name": module_name,
                    "qualified_name": module_name,
                    "file_name": file_path.name,
                    "sidecar_id": sidecar_id,
                },
            }
        )
        module_objects[rel_path] = module_id
        return module_id, rel_path

    object_lookup: dict[str, dict[str, Any]] = {}

    for struct_name in sidebar_items.get("struct", []):
        html_path = docs_root / f"struct.{struct_name}.html"
        html_text = html_path.read_text(encoding="utf-8")
        href_match = re.search(r'<a class="src" href="([^"]+)">Source</a>', html_text)
        if not href_match:
            continue
        source_file, span = _parse_rust_source_href(crate_root, crate_name, href_match.group(1))
        module_id, rel_path = ensure_module(source_file)
        signature = _rust_signature(html_text)
        class_id = stable_semantic_id("obj", repo_id, revision, rel_path, "class", struct_name)
        sidecar_id = stable_semantic_id("sidecar", repo_id, revision, rel_path, class_id, "struct")
        source_text = source_file.read_text(encoding="utf-8")
        sidecar_payload: dict[str, Any] = {
            "fragment_id": stable_semantic_id("fragment", class_id),
            "root_node_id": class_id,
            "root_kind": "rust_struct",
            "file_path": rel_path,
            "span": span,
            "parent_node_id": module_id,
            "child_node_ids": [],
            "payload_ref": None,
            "code": _snippet_for_span(source_text, span),
            "signature": signature,
        }
        sidecars.append(
            {
                "sidecar_id": sidecar_id,
                "sidecar_kind": "semantic_snapshot",
                "format_family": "rustdoc",
                "format_name": "rustdoc_fragment_v1",
                "format_version": "1",
                "encoding": "utf-8",
                "compression": "none",
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "node_id": class_id,
                "storage_ref": f"inline:{sidecar_id}",
                "content_hash": stable_semantic_id("payload", sidecar_id, json.dumps(sidecar_payload, sort_keys=True)),
                "byte_size": len(json.dumps(sidecar_payload, ensure_ascii=False)),
                "producer": plugin_id,
                "metadata": {"language": "rust", "node_kind": "struct"},
                "payload": sidecar_payload,
            }
        )
        class_row = {
            "id": class_id,
            "object_type": "class",
            "repo_id": repo_id,
            "revision": revision,
            "file_path": rel_path,
            "span": span,
            "language": "rust",
            "analyzer_id": plugin_id,
            "analyzer_version": plugin_version,
            "derivation_type": "observed",
            "payload": {
                "name": struct_name,
                "qualified_name": f"{crate_name}::{struct_name}",
                "signature": signature,
                "language_object_kind": "struct",
                "sidecar_id": sidecar_id,
            },
        }
        objects.append(class_row)
        object_lookup[class_id] = class_row
        relations.append(
            {
                "id": stable_semantic_id("rel", repo_id, revision, rel_path, "declares", module_id, class_id),
                "relation_type": "declares",
                "from_id": module_id,
                "to_id": class_id,
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "language": "rust",
                "analyzer_id": plugin_id,
                "analyzer_version": plugin_version,
                "derivation_type": "observed",
                "payload": {"from_type": "module", "to_type": "class"},
            }
        )
        for field_name, field_signature in _RUST_FIELD_RE.findall(html_text):
            field_id = stable_semantic_id("obj", repo_id, revision, rel_path, "field", f"{struct_name}.{field_name}")
            field_row = {
                "id": field_id,
                "object_type": "field",
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "language": "rust",
                "analyzer_id": plugin_id,
                "analyzer_version": plugin_version,
                "derivation_type": "observed",
                "payload": {
                    "name": field_name,
                    "qualified_name": f"{crate_name}::{struct_name}.{field_name}",
                    "signature": _strip_html(field_signature),
                    "owner_id": class_id,
                },
            }
            objects.append(field_row)
            object_lookup[field_id] = field_row
            relations.append(
                {
                    "id": stable_semantic_id("rel", repo_id, revision, rel_path, "declares", class_id, field_id),
                    "relation_type": "declares",
                    "from_id": class_id,
                    "to_id": field_id,
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_path": rel_path,
                    "span": span,
                    "language": "rust",
                    "analyzer_id": plugin_id,
                    "analyzer_version": plugin_version,
                    "derivation_type": "observed",
                    "payload": {"from_type": "class", "to_type": "field"},
                }
            )

    for fn_name in sidebar_items.get("fn", []):
        html_path = docs_root / f"fn.{fn_name}.html"
        html_text = html_path.read_text(encoding="utf-8")
        href_match = re.search(r'<a class="src" href="([^"]+)">Source</a>', html_text)
        if not href_match:
            continue
        source_file, span = _parse_rust_source_href(crate_root, crate_name, href_match.group(1))
        module_id, rel_path = ensure_module(source_file)
        signature = _rust_signature(html_text)
        callable_id = stable_semantic_id("obj", repo_id, revision, rel_path, "callable", fn_name)
        sidecar_id = stable_semantic_id("sidecar", repo_id, revision, rel_path, callable_id, "fn")
        source_text = source_file.read_text(encoding="utf-8")
        sidecar_payload = {
            "fragment_id": stable_semantic_id("fragment", callable_id),
            "root_node_id": callable_id,
            "root_kind": "rust_function",
            "file_path": rel_path,
            "span": span,
            "parent_node_id": module_id,
            "child_node_ids": [],
            "payload_ref": None,
            "code": _snippet_for_span(source_text, span),
            "signature": signature,
        }
        sidecars.append(
            {
                "sidecar_id": sidecar_id,
                "sidecar_kind": "semantic_snapshot",
                "format_family": "rustdoc",
                "format_name": "rustdoc_fragment_v1",
                "format_version": "1",
                "encoding": "utf-8",
                "compression": "none",
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "node_id": callable_id,
                "storage_ref": f"inline:{sidecar_id}",
                "content_hash": stable_semantic_id("payload", sidecar_id, json.dumps(sidecar_payload, sort_keys=True)),
                "byte_size": len(json.dumps(sidecar_payload, ensure_ascii=False)),
                "producer": plugin_id,
                "metadata": {"language": "rust", "node_kind": "function"},
                "payload": sidecar_payload,
            }
        )
        callable_row = {
            "id": callable_id,
            "object_type": "callable",
            "repo_id": repo_id,
            "revision": revision,
            "file_path": rel_path,
            "span": span,
            "language": "rust",
            "analyzer_id": plugin_id,
            "analyzer_version": plugin_version,
            "derivation_type": "observed",
            "payload": {
                "name": fn_name,
                "qualified_name": f"{crate_name}::{fn_name}",
                "signature": signature,
                "sidecar_id": sidecar_id,
            },
        }
        objects.append(callable_row)
        object_lookup[callable_id] = callable_row
        relations.append(
            {
                "id": stable_semantic_id("rel", repo_id, revision, rel_path, "declares", module_id, callable_id),
                "relation_type": "declares",
                "from_id": module_id,
                "to_id": callable_id,
                "repo_id": repo_id,
                "revision": revision,
                "file_path": rel_path,
                "span": span,
                "language": "rust",
                "analyzer_id": plugin_id,
                "analyzer_version": plugin_version,
                "derivation_type": "observed",
                "payload": {"from_type": "module", "to_type": "callable"},
            }
        )
        params_block_match = re.search(r"fn\\s+[^\\(]+\\((.*?)\\)", signature)
        if params_block_match:
            for index, param_name in enumerate(_RUST_PARAM_RE.findall(params_block_match.group(1))):
                param_id = stable_semantic_id("obj", repo_id, revision, rel_path, "parameter", f"{fn_name}.{param_name}")
                objects.append(
                    {
                        "id": param_id,
                        "object_type": "parameter",
                        "repo_id": repo_id,
                        "revision": revision,
                        "file_path": rel_path,
                        "span": span,
                        "language": "rust",
                        "analyzer_id": plugin_id,
                        "analyzer_version": plugin_version,
                        "derivation_type": "observed",
                        "payload": {
                            "name": param_name,
                            "qualified_name": f"{crate_name}::{fn_name}.{param_name}",
                            "position": index,
                            "callable_id": callable_id,
                        },
                    }
                )
                relations.append(
                    {
                        "id": stable_semantic_id("rel", repo_id, revision, rel_path, "declares", callable_id, param_id),
                        "relation_type": "declares",
                        "from_id": callable_id,
                        "to_id": param_id,
                        "repo_id": repo_id,
                        "revision": revision,
                        "file_path": rel_path,
                        "span": span,
                        "language": "rust",
                        "analyzer_id": plugin_id,
                        "analyzer_version": plugin_version,
                        "derivation_type": "observed",
                        "payload": {"from_type": "callable", "to_type": "parameter"},
                    }
                )

    return {
        "objects": objects,
        "relations": relations,
        "provenance": {
            "repo_id": repo_id,
            "revision": revision,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(crate_root),
        },
        "capability_report": {
            "supported_languages": ["rust"],
            "supported_capabilities": ["module_graph", "structural_semantics"],
            "analyzer_protocol_version": "codebase-v1",
            "schema_version": "1",
        },
        "gap_report": {
            "skipped_files": [],
            "notes": [],
        },
        "sidecars": sidecars,
    }

def _normalize_skipped_files(existing: Any, extra: list[str]) -> list[str]:
    values = []
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
    values.extend(extra)
    return values


@dataclass(frozen=True)
class RustSemanticPlugin:
    plugin_name: str = "rustdoc"
    supported_extensions: frozenset[str] = frozenset({".rs"})

    def build_bundle(
        self,
        *,
        repo_root: Path,
        repo_id: str,
        revision: str,
        files: list[Path],
    ) -> dict[str, Any]:
        rust_groups: dict[Path, list[Path]] = {}
        skipped_rust: list[str] = []
        for rust_file in files:
            manifest_dir = next(
                (candidate for candidate in [rust_file.parent, *rust_file.parents] if (candidate / "Cargo.toml").exists()),
                None,
            )
            if manifest_dir is None:
                skipped_rust.append(relative_file_path(repo_root, rust_file))
                continue
            manifest_path = manifest_dir / "Cargo.toml"
            rust_groups.setdefault(manifest_path, []).append(rust_file)

        bundles: list[dict[str, Any]] = []
        failure_notes: list[str] = []
        for manifest_path in sorted(rust_groups):
            try:
                bundles.append(_rustdoc_bundle_for_manifest(manifest_path, repo_id, revision))
            except Exception as exc:
                skipped_rust.extend(relative_file_path(repo_root, rust_file) for rust_file in rust_groups[manifest_path])
                failure_notes.append(
                    f"rust manifest {relative_file_path(repo_root, manifest_path)} failed: {exc}"
                )
        if not bundles:
            if failure_notes:
                raise ValueError("; ".join(failure_notes))
            raise ValueError("codebase Repository context supports only Rust sources with Cargo metadata")

        merged = merge_semantic_bundles(bundles)
        gap_report = dict(merged.get("gap_report") or {})
        notes = list(gap_report.get("notes") or [])
        notes.extend(failure_notes)
        gap_report["notes"] = notes
        gap_report["skipped_files"] = list(
            dict.fromkeys(_normalize_skipped_files(gap_report.get("skipped_files"), skipped_rust))
        )
        merged["gap_report"] = gap_report
        return merged


RUST_SEMANTIC_PLUGIN = RustSemanticPlugin()
SEMANTIC_PLUGIN = RUST_SEMANTIC_PLUGIN
