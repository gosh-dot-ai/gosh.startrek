# Copyright (c) 2026 GOSH Operating Co., GOSH Technology Ltd., and Mitja Goroshevsky
# SPDX-License-Identifier: LicenseRef-GOSH-Noncommercial-1.0
#
# Licensed under the GOSH.AI Noncommercial License, Version 1.0.
# See LICENSE.md in the root of this repository for the full license
# text. Use of this file for any Commercial Purpose requires a separate
# written license. Contact: legal@gosh.sh

from __future__ import annotations

from typing import Final, Literal

FactLikelihood = Literal["high", "medium", "uncertain"]

FACT_LIKELIHOOD_HIGH: Final[FactLikelihood] = "high"
FACT_LIKELIHOOD_MEDIUM: Final[FactLikelihood] = "medium"
FACT_LIKELIHOOD_UNCERTAIN: Final[FactLikelihood] = "uncertain"

RAW_CONVERSATION_WINDOW_RADIUS: Final[int] = 1
RAW_SOURCE_WINDOW_BUDGET_CHARS: Final[int] = 4000
RAW_EPISODE_RETRIEVAL_LIMIT: Final[int] = 4

# Mirror of the extraction prompts used only to decide whether fact evidence is
# likely complete enough for recall or whether bounded raw evidence is needed.
# Keep source_prompt/rule_ids in sync with src/prompts/extraction/*.md.
RECALL_EXTRACTION_POLICY_MIRROR: Final[dict[str, dict[str, dict[str, object]]]] = {
    "conversation": {
        "exact_values": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 1", "RULE 3", "RULE 4", "RULE 7c"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["exact_names", "numbers", "dates", "quotes"],
            "query_feature_terms": {
                "asks_for_exact_value": ["exact", "value", "number"],
                "asks_for_identifier_like_value": ["id", "identifier", "code", "token"],
                "asks_for_date_or_time": ["date", "time", "when", "year", "month", "day"],
                "asks_for_named_entity": ["where", "place", "city", "location"],
            },
        },
        "named_targets": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 1b"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["venues", "events", "titles", "products", "destinations"],
            "query_feature_terms": {
                "asks_for_named_entity": [
                    "name",
                    "named",
                    "target",
                    "venue",
                    "event",
                    "title",
                    "product",
                    "destination",
                    "place",
                    "city",
                    "location",
                    "where",
                    "go",
                ],
            },
        },
        "one_fact_per_item": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 2"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["separate_items"],
        },
        "identity": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 5"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["direct_identity"],
        },
        "relationships": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 6"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["stated_relationships"],
        },
        "physical_objects": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 7"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["objects_with_attributes"],
        },
        "tables_schedules": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 7b"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["table_rows", "schedule_rows"],
        },
        "verbatim_quotes": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 7c"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["quoted_text"],
        },
        "temporal_ordering": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 7d"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["explicit_ordering"],
            "prompt_declared_signals": ["then", "after that", "later", "before", "the next day", "two weeks later"],
            "query_signal_lemmas": ["before", "after", "later", "then", "next"],
        },
        "acquisition_events": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["DELTA D"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["acquisition_or_change_events"],
            "prompt_declared_signals": ["bought", "purchased", "ordered", "booked", "got", "acquired"],
            "query_signal_lemmas": ["buy", "purchase", "order", "book", "get", "acquire"],
        },
        "knowledge_updates": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 8"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["updates_or_contradictions"],
            "prompt_declared_signals": [
                "actually",
                "changed",
                "now",
                "no longer",
                "moved to",
                "switched to",
                "updated",
            ],
            "query_signal_lemmas": ["change", "changed", "move", "moved", "switch", "switched", "update", "updated"],
        },
        "assistant_material_facts": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 9", "RULE 10"],
            "source_requirements": ["assistant_material"],
            "evidence_contract": ["recommendations", "provided_data", "commitments"],
            "query_feature_terms": {
                "asks_for_decision_rule_requirement_constraint": [
                    "recommendation",
                    "recommend",
                    "provided",
                    "commitment",
                    "committed",
                ],
            },
        },
        "quality_priority": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["RULE 10"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["memory_relevance"],
        },
        "financial_components": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["DELTA A"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["amounts", "formulas", "components"],
            "query_feature_terms": {
                "asks_for_quantity": ["amount", "cost", "price", "tax", "fee", "budget"],
            },
        },
        "preferences_with_reason": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["DELTA B"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["preferences", "reasons"],
            "query_feature_terms": {
                "asks_for_preference_or_reason": ["preference", "prefer", "favorite", "reason", "because"],
            },
        },
        "organizational_facts": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["DELTA C"],
            "source_requirements": ["user", "source"],
            "evidence_contract": ["roles", "channels", "responsibilities"],
        },
        "kind_classification": {
            "source_prompt": "src/prompts/extraction/conversation.md",
            "rule_ids": ["KIND CLASSIFICATION"],
            "source_requirements": ["user", "source", "assistant_material"],
            "evidence_contract": ["rule", "constraint", "decision", "preference", "action_item"],
            "query_feature_terms": {
                "asks_for_decision_rule_requirement_constraint": [
                    "rule",
                    "constraint",
                    "decision",
                    "requirement",
                    "policy",
                    "procedure",
                ],
            },
        },
    },
    "document": {
        "exact_technical_values": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 1"],
            "source_requirements": ["document"],
            "evidence_contract": ["technical_values", "numbers"],
            "query_feature_terms": {
                "asks_for_exact_value": ["exact", "value", "number"],
                "asks_for_date_or_time": ["date", "time", "when", "year", "month", "day"],
                "asks_for_quantity": ["quantity", "amount", "cost", "price"],
            },
        },
        "one_fact_per_item": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 2"],
            "source_requirements": ["document"],
            "evidence_contract": ["separate_items"],
        },
        "tables_rows": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 3"],
            "source_requirements": ["document"],
            "evidence_contract": ["table_rows"],
            "query_feature_terms": {
                "asks_for_list_or_table_item": ["table", "row"],
            },
        },
        "decisions_with_alternatives": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 4"],
            "source_requirements": ["document"],
            "evidence_contract": ["chosen", "rejected", "rationale"],
            "query_feature_terms": {
                "asks_for_decision_rule_requirement_constraint": ["decision", "alternative", "selected", "rejected"],
            },
        },
        "unique_identifiers": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 5"],
            "source_requirements": ["document"],
            "evidence_contract": ["ticket_ids", "codes", "references"],
            "query_feature_terms": {
                "asks_for_identifier_like_value": ["identifier", "id", "ticket", "code", "reference"],
            },
        },
        "requirements_constraints": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 6"],
            "source_requirements": ["document"],
            "evidence_contract": ["requirements", "constraints"],
            "query_feature_terms": {
                "asks_for_decision_rule_requirement_constraint": ["requirement", "constraint"],
            },
        },
        "version_tracking": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 7"],
            "source_requirements": ["document"],
            "evidence_contract": ["supersedes_topic"],
            "query_feature_terms": {
                "asks_for_temporal_order_or_update": ["version", "supersedes", "updated"],
            },
        },
        "boilerplate_zero_correct": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["RULE 8"],
            "source_requirements": ["document"],
            "evidence_contract": ["boilerplate_can_be_zero"],
        },
        "policy_conditions_exceptions": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["DELTA A"],
            "source_requirements": ["document"],
            "evidence_contract": ["conditions", "exceptions"],
        },
        "financial_components": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["DELTA B"],
            "source_requirements": ["document"],
            "evidence_contract": ["amounts", "formulas", "components"],
        },
        "requirement_rejection_kinds": {
            "source_prompt": "src/prompts/extraction/document.md",
            "rule_ids": ["KIND"],
            "source_requirements": ["document"],
            "evidence_contract": ["requirement", "rejection"],
        },
    },
}
