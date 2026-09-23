"""Ask one question against the local FAISS evidence index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import load_embedding_model, embed_query  # noqa: E402
from src.retriever import FaissRetriever  # noqa: E402


INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIRECTORY / "bis_seed.faiss"
CHUNKS_PATH = INDEX_DIRECTORY / "bis_seed_chunks.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the reviewed BIS seed corpus.")
    parser.add_argument("question", help="Natural-language question to search for.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence chunks to show.")
    arguments = parser.parse_args()

    model = load_embedding_model()
    retriever = FaissRetriever.load(INDEX_PATH, CHUNKS_PATH)
    results = retriever.search(embed_query(model, arguments.question), top_k=arguments.top_k)

    print(f"Question: {arguments.question}\n")
    for rank, result in enumerate(results, start=1):
        metadata = result.chunk.metadata
        print(f"{rank}. Similarity: {result.score:.3f}")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Evidence: {result.chunk.text}")
        print(f"   Source: {metadata['title']}")
        print(f"   Link: {metadata['source_url']}\n")


if __name__ == "__main__":
    main()
