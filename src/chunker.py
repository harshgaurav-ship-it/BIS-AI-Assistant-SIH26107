"""Split reviewed source documents into citation-ready retrieval chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.loader import SourceDocument


@dataclass(frozen=True)
class Chunk:
    """One searchable text unit and the metadata required to cite it."""

    chunk_id: str
    text: str
    metadata: dict[str, str]


def chunk_documents(
    documents: list[SourceDocument], max_chars: int = 600, overlap_chars: int = 80
) -> list[Chunk]:
    """Create small chunks while preserving source metadata on every chunk."""

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be zero or more and smaller than max_chars.")

    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(_split_text(document.text, max_chars, overlap_chars), start=1):
            metadata = dict(document.metadata)
            metadata["document_id"] = document.document_id
            metadata["chunk_number"] = str(index)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-chunk-{index:03d}",
                    text=text,
                    metadata=metadata,
                )
            )
    return chunks


def chunks_as_dicts(chunks: list[Chunk]) -> list[dict[str, object]]:
    """Convert chunks to JSON-safe dictionaries for the later vector-index step."""

    return [asdict(chunk) for chunk in chunks]


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        raise ValueError("A source document cannot be empty.")

    pieces: list[str] = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        if end < text_length:
            natural_break = normalized_text.rfind(" ", start, end)
            if natural_break > start:
                end = natural_break

        piece = normalized_text[start:end].strip()
        if not piece:
            raise ValueError("Chunking could not create a non-empty text piece.")
        pieces.append(piece)

        if end >= text_length:
            break
        start = end - overlap_chars

    return pieces
