"""Tests for FAISS result ordering and citation preservation."""

import unittest

import numpy as np

from src.chunker import Chunk
from src.retriever import FaissRetriever


class FaissRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="FACT-HM-001-chunk-001",
                text="Hallmarking evidence.",
                metadata={"source_id": "BIS-HM-001", "source_url": "https://bis.gov.in/hallmark"},
            ),
            Chunk(
                chunk_id="FACT-LAB-001-chunk-001",
                text="Laboratory evidence.",
                metadata={"source_id": "BIS-LAB-001", "source_url": "https://bis.gov.in/labs"},
            ),
        ]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.retriever = FaissRetriever(self.chunks, vectors)

    def test_search_returns_best_match_with_citation_metadata(self) -> None:
        results = self.retriever.search(np.array([0.9, 0.1], dtype=np.float32), top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "FACT-HM-001-chunk-001")
        self.assertEqual(results[0].chunk.metadata["source_id"], "BIS-HM-001")
        self.assertGreater(results[0].score, 0.8)

    def test_search_rejects_wrong_query_dimension(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search(np.array([1.0, 0.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
