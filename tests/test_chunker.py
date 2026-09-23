"""Tests for source loading and citation-ready chunking."""

import unittest
from pathlib import Path

from src.chunker import chunk_documents
from src.loader import SourceDocument, load_curated_facts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_FACTS_PATH = PROJECT_ROOT / "data" / "metadata" / "curated_facts.json"


class CuratedCorpusTests(unittest.TestCase):
    def test_seed_corpus_has_citation_metadata(self) -> None:
        documents = load_curated_facts(CURATED_FACTS_PATH)

        self.assertEqual(len(documents), 16)
        for document in documents:
            self.assertTrue(document.document_id.startswith("FACT-"))
            self.assertTrue(document.metadata["source_id"].startswith("BIS-"))
            self.assertTrue(document.metadata["source_url"].startswith("https://"))
            self.assertEqual(document.metadata["content_type"], "team_authored_paraphrase")

    def test_long_document_is_split_with_source_metadata(self) -> None:
        document = SourceDocument(
            document_id="FACT-TEST-001",
            text="Verified BIS guidance. " * 80,
            metadata={
                "source_id": "BIS-TEST-001",
                "source_url": "https://example.bis.gov.in/source",
            },
        )

        chunks = chunk_documents([document], max_chars=180, overlap_chars=20)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "FACT-TEST-001-chunk-001")
        self.assertEqual(chunks[0].metadata["source_id"], "BIS-TEST-001")
        self.assertEqual(chunks[0].metadata["document_id"], "FACT-TEST-001")
        self.assertTrue(all(len(chunk.text) <= 180 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
