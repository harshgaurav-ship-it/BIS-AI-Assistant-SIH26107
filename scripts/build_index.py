"""Build the local FAISS search index from the reviewed BIS seed corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import DEFAULT_EMBEDDING_MODEL, load_embedding_model  # noqa: E402
from src.loader import load_curated_facts  # noqa: E402
from src.retriever import build_retriever  # noqa: E402


CURATED_FACTS_PATH = PROJECT_ROOT / "data" / "metadata" / "curated_facts.json"
INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIRECTORY / "bis_seed.faiss"
CHUNKS_PATH = INDEX_DIRECTORY / "bis_seed_chunks.json"
MANIFEST_PATH = INDEX_DIRECTORY / "index_manifest.json"


def main() -> None:
    documents = load_curated_facts(CURATED_FACTS_PATH)
    print(f"Loaded {len(documents)} reviewed BIS guidance facts.")

    print("Loading the local multilingual embedding model. The first run may download model files.")
    model = load_embedding_model()
    retriever = build_retriever(documents, model)
    retriever.save(INDEX_PATH, CHUNKS_PATH)

    manifest = {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "document_count": len(documents),
        "chunk_count": len(retriever.chunks),
        "embedding_dimension": retriever.dimension,
        "index_type": "faiss.IndexFlatIP",
        "normalization": "L2 normalized vectors; inner product equals cosine similarity",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built {len(retriever.chunks)} searchable chunks with {retriever.dimension}-dimension embeddings.")
    print(f"Saved local index to: {INDEX_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
