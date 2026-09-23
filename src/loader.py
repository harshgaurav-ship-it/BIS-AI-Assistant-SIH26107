"""Load the reviewed seed corpus without copying raw BIS documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FACT_FIELDS = {
    "fact_id",
    "source_id",
    "title",
    "source_url",
    "topic",
    "language",
    "content",
    "time_sensitivity",
    "reviewed_at",
}


@dataclass(frozen=True)
class SourceDocument:
    """A short, reviewed piece of evidence with the source data needed for citation."""

    document_id: str
    text: str
    metadata: dict[str, str]


def load_curated_facts(path: Path) -> list[SourceDocument]:
    """Load validated team-authored facts from a local JSON file.

    This loader intentionally performs no web scraping or PDF downloading. Any new
    material must first pass the review process in ``data/metadata/README.md``.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Curated facts file was not found: {path}")

    with path.open(encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)

    facts = payload.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("The curated facts file must contain a non-empty 'facts' list.")

    documents: list[SourceDocument] = []
    seen_fact_ids: set[str] = set()

    for position, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact {position} must be a JSON object.")

        missing_fields = REQUIRED_FACT_FIELDS - fact.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Fact {position} is missing required fields: {missing}.")

        fact_id = _required_text(fact, "fact_id", position)
        if fact_id in seen_fact_ids:
            raise ValueError(f"Duplicate fact_id found: {fact_id}.")
        seen_fact_ids.add(fact_id)

        content = _required_text(fact, "content", position)
        metadata = {
            key: _required_text(fact, key, position)
            for key in REQUIRED_FACT_FIELDS - {"fact_id", "content"}
        }
        metadata["content_type"] = "team_authored_paraphrase"

        documents.append(
            SourceDocument(document_id=fact_id, text=content, metadata=metadata)
        )

    return documents


def _required_text(fact: dict[str, Any], field_name: str, position: int) -> str:
    """Return a required non-empty text field with a beginner-friendly error."""

    value = fact.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Fact {position} field '{field_name}' must be non-empty text.")
    return value.strip()
