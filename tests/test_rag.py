"""Tests for grounded-answer and safe-refusal behavior."""

import unittest

from src.chunker import Chunk
from src.rag_pipeline import INSUFFICIENT_EVIDENCE_MESSAGE, answer_from_results
from src.retriever import RetrievalResult


def make_result(score: float, source_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=f"{source_id}-chunk-001",
            text=text,
            metadata={
                "source_id": source_id,
                "title": f"Title for {source_id}",
                "source_url": f"https://example.bis.gov.in/{source_id.lower()}",
            },
        ),
        score=score,
    )


class GroundedAnswerTests(unittest.TestCase):
    def test_supported_answer_uses_only_high_scoring_evidence(self) -> None:
        strong = make_result(0.88, "BIS-LAB-001", "Use the LIMS portal for current lab details.")
        weak = make_result(0.22, "BIS-PC-001", "Unrelated certification material.")

        answer = answer_from_results("Where can I find a lab?", [strong, weak])

        self.assertTrue(answer.supported)
        self.assertIn(strong.chunk.text, answer.answer)
        self.assertNotIn(weak.chunk.text, answer.answer)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].source_id, "BIS-LAB-001")

    def test_unsupported_question_refuses_without_citations(self) -> None:
        weak = make_result(0.31, "BIS-PC-001", "Unrelated certification material.")

        answer = answer_from_results("What is the capital of Japan?", [weak])

        self.assertFalse(answer.supported)
        self.assertEqual(answer.answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.evidence, [])

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            answer_from_results("Test", [], minimum_score=1.1)


if __name__ == "__main__":
    unittest.main()
