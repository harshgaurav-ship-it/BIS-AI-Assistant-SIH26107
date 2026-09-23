"""Ask the evidence-first BIS assistant from the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import load_embedding_model  # noqa: E402
from src.rag_pipeline import answer_question  # noqa: E402
from src.retriever import FaissRetriever  # noqa: E402


INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
INDEX_PATH = INDEX_DIRECTORY / "bis_seed.faiss"
CHUNKS_PATH = INDEX_DIRECTORY / "bis_seed_chunks.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the evidence-first BIS assistant.")
    parser.add_argument("question", help="Natural-language BIS question.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum evidence chunks to consider.")
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=0.55,
        help="Minimum similarity score required before the assistant can answer.",
    )
    arguments = parser.parse_args()

    model = load_embedding_model()
    retriever = FaissRetriever.load(INDEX_PATH, CHUNKS_PATH)
    grounded_answer = answer_question(
        arguments.question,
        retriever,
        model,
        top_k=arguments.top_k,
        minimum_score=arguments.minimum_score,
    )

    print(f"Question: {grounded_answer.question}\n")
    print(grounded_answer.answer)
    print(f"\nNote: {grounded_answer.advisory_notice}")
    if grounded_answer.citations:
        print("\nSources:")
        for citation in grounded_answer.citations:
            print(f"[{citation.number}] {citation.title} — {citation.url}")


if __name__ == "__main__":
    main()
