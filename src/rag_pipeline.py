"""Evidence-first answer generation for the BIS assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.citations import Citation, citations_from_results
from src.embeddings import embed_query
from src.retriever import FaissRetriever, RetrievalResult


DEFAULT_MIN_RETRIEVAL_SCORE = 0.55
DEFAULT_TOP_K = 3
INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have sufficient verified BIS information in this prototype to answer "
    "that safely. Please check the current BIS website or refine the question with "
    "a product name, Indian Standard number, or certification context."
)
ADVISORY_NOTICE = (
    "This is general guidance from the cited BIS sources, not a certification or legal decision."
)


@dataclass(frozen=True)
class GroundedAnswer:
    """A safe response together with its evidence and official citations."""

    question: str
    answer: str
    supported: bool
    evidence: list[RetrievalResult]
    citations: list[Citation]
    advisory_notice: str


def answer_question(
    question: str,
    retriever: FaissRetriever,
    embedding_model: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    minimum_score: float = DEFAULT_MIN_RETRIEVAL_SCORE,
) -> GroundedAnswer:
    """Retrieve evidence for a question and produce a source-bound response."""

    query_vector = embed_query(embedding_model, question)
    results = retriever.search(query_vector, top_k=top_k)
    return answer_from_results(question, results, minimum_score=minimum_score)


def answer_from_results(
    question: str,
    results: list[RetrievalResult],
    *,
    minimum_score: float = DEFAULT_MIN_RETRIEVAL_SCORE,
) -> GroundedAnswer:
    """Produce an answer only from results that clear the evidence threshold.

    Keeping this deterministic for the first version makes every answer traceable:
    the displayed text is drawn from reviewed evidence rather than an LLM's memory.
    """

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0.0 and 1.0.")

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("The user question cannot be empty.")

    eligible_evidence = [result for result in results if result.score >= minimum_score]
    if not eligible_evidence:
        return GroundedAnswer(
            question=cleaned_question,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            supported=False,
            evidence=[],
            citations=[],
            advisory_notice=ADVISORY_NOTICE,
        )

    evidence_lines = [f"- {result.chunk.text}" for result in eligible_evidence]
    answer = "Based on the reviewed BIS evidence:\n\n" + "\n".join(evidence_lines)
    return GroundedAnswer(
        question=cleaned_question,
        answer=answer,
        supported=True,
        evidence=eligible_evidence,
        citations=citations_from_results(eligible_evidence),
        advisory_notice=ADVISORY_NOTICE,
    )
