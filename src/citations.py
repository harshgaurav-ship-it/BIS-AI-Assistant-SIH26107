"""Create stable, de-duplicated citations from retrieved BIS evidence."""

from __future__ import annotations

from dataclasses import dataclass

from src.retriever import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """A source users can open to verify an answer."""

    number: int
    source_id: str
    title: str
    url: str


def citations_from_results(results: list[RetrievalResult]) -> list[Citation]:
    """Build one citation per unique official source, in retrieval order."""

    citations: list[Citation] = []
    seen_sources: set[str] = set()

    for result in results:
        metadata = result.chunk.metadata
        source_id = metadata["source_id"]
        if source_id in seen_sources:
            continue

        seen_sources.add(source_id)
        citations.append(
            Citation(
                number=len(citations) + 1,
                source_id=source_id,
                title=metadata["title"],
                url=metadata["source_url"],
            )
        )

    return citations
