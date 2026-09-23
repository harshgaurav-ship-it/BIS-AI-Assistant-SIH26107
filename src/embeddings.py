"""Create local text embeddings for the reviewed BIS seed corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class LocalEmbeddingModel:
    """A small wrapper that produces normalized mean-pooled text embeddings."""

    tokenizer: Any
    model: Any

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts locally without importing SciPy or scikit-learn."""

        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(**inputs)

        token_embeddings = output.last_hidden_state
        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .to(token_embeddings.dtype)
        )
        summed_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        token_counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed_embeddings / token_counts
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return np.ascontiguousarray(normalized.cpu().numpy(), dtype=np.float32)


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> LocalEmbeddingModel:
    """Load the embedding model locally.

    The first call downloads the selected open model to the machine's model cache.
    No question or source text is sent to an external inference API.
    """

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        if "Application Control policy" in str(error):
            raise RuntimeError(
                "Windows blocked an unused native dependency. Remove the old "
                "sentence-transformers, scikit-learn, and scipy packages, then retry."
            ) from error
        raise RuntimeError(
            "Embedding dependencies are missing. Run 'python -m pip install -r requirements.txt'."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return LocalEmbeddingModel(tokenizer=tokenizer, model=model)


def embed_documents(model: Any, texts: list[str]) -> np.ndarray:
    """Embed source chunks and normalize vectors for cosine-similarity search."""

    if not texts:
        raise ValueError("At least one source text is required to create embeddings.")

    return model.encode(texts)


def embed_query(model: Any, query: str) -> np.ndarray:
    """Embed and normalize one user question for cosine-similarity search."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("The search question cannot be empty.")

    return model.encode([cleaned_query])[0]
