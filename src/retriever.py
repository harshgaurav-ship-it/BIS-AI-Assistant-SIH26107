"""FAISS-backed semantic search with preserved citation metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.chunker import Chunk, chunk_documents, chunks_as_dicts
from src.embeddings import embed_documents
from src.loader import SourceDocument


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved evidence chunk and its cosine-similarity score."""

    chunk: Chunk
    score: float


class FaissRetriever:
    """A small local vector index appropriate for the SIH prototype corpus."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Cannot create a retriever without chunks.")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        if vectors.shape[1] == 0:
            raise ValueError("Embeddings must have at least one dimension.")

        self.chunks = chunks
        self.dimension = int(vectors.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(_as_faiss_vectors(vectors, self.dimension))

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievalResult]:
        """Return the most semantically similar evidence chunks for one query."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape != (1, self.dimension):
            raise ValueError(
                f"Query embedding must have shape (1, {self.dimension}) or ({self.dimension},)."
            )

        scores, positions = self.index.search(query, min(top_k, len(self.chunks)))
        results: list[RetrievalResult] = []
        for score, position in zip(scores[0], positions[0], strict=True):
            if position >= 0:
                results.append(
                    RetrievalResult(chunk=self.chunks[int(position)], score=float(score))
                )
        return results

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Persist the FAISS index and its human-readable citation records locally."""

        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        chunks_path.write_text(
            json.dumps(chunks_as_dicts(self.chunks), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "FaissRetriever":
        """Load a previously built local index and verify its chunk count."""

        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                "The local search index is missing. Run 'python scripts/build_index.py' first."
            )

        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=record["chunk_id"],
                text=record["text"],
                metadata=record["metadata"],
            )
            for record in raw_chunks
        ]
        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks):
            raise ValueError("The saved FAISS index and chunk records do not match.")

        retriever = cls.__new__(cls)
        retriever.chunks = chunks
        retriever.dimension = index.d
        retriever.index = index
        return retriever


def build_retriever(documents: list[SourceDocument], model: object) -> FaissRetriever:
    """Chunk approved documents, embed them, and build an in-memory FAISS index."""

    chunks = chunk_documents(documents)
    vectors = embed_documents(model, [chunk.text for chunk in chunks])
    return FaissRetriever(chunks, vectors)


def _as_faiss_vectors(vectors: np.ndarray, dimension: int) -> np.ndarray:
    checked_vectors = np.asarray(vectors, dtype=np.float32)
    if checked_vectors.shape[1] != dimension:
        raise ValueError("Embedding dimension does not match the FAISS index dimension.")
    return np.ascontiguousarray(checked_vectors)
