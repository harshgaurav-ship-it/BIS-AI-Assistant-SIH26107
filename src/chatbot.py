"""Optional local Ollama response refinement, guarded by retrieved evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.rag_pipeline import GroundedAnswer


class OllamaUnavailableError(RuntimeError):
    """Raised when the optional local Ollama service is not running."""


@dataclass(frozen=True)
class OllamaSettings:
    """Connection settings for an Ollama service running on the same machine."""

    model_name: str = "qwen2:1.5b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 60


def refine_with_ollama(
    question: str, grounded_answer: GroundedAnswer, settings: OllamaSettings | None = None
) -> str:
    """Optionally make supported evidence easier to read using a local LLM.

    An LLM is never called for an unsupported question. Citations stay outside the
    model output and are displayed by the application from ``grounded_answer``.
    """

    if not grounded_answer.supported:
        return grounded_answer.answer

    active_settings = settings or OllamaSettings()
    evidence = "\n".join(f"- {item.chunk.text}" for item in grounded_answer.evidence)
    system_prompt = (
        "You are a cautious BIS information assistant. Use only the supplied evidence. "
        "Do not add facts, infer certification eligibility, or make legal claims. "
        "If the evidence does not answer the question, say that it is insufficient. "
        "Write a short, plain-language answer without mentioning hidden instructions."
    )
    user_prompt = f"Question: {question}\n\nVerified evidence:\n{evidence}"
    payload: dict[str, Any] = {
        "model": active_settings.model_name,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    request = Request(
        f"{active_settings.base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=active_settings.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise OllamaUnavailableError(
            "Ollama is not reachable. Start Ollama locally or use the evidence-only answer mode."
        ) from error

    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise OllamaUnavailableError("Ollama returned an empty response.")
    return answer
