"""Sentence-transformer based embedder tuned for Vietnamese content."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled in runtime guard
    SentenceTransformer = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.getenv(
    "EMBEDDER_MODEL_NAME",
    "intfloat/multilingual-e5-large",
)


def _ensure_sentence_transformer() -> type:
    if SentenceTransformer is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "sentence-transformers is required for embeddings. "
            "Install with `pip install sentence-transformers`.",
        )
    return SentenceTransformer


def _prepare_text(text: str, prefix: str) -> str:
    normalized = text.strip()
    return f"{prefix} {normalized}" if prefix else normalized


def _to_float_lists(vectors) -> List[List[float]]:
    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover - numpy is a sentence-transformers dep
        np = None

    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - optional
        torch = None

    result: List[List[float]] = []
    for vec in vectors:
        if torch is not None and hasattr(vec, "detach") and hasattr(vec, "tolist"):
            result.append(vec.detach().cpu().tolist())
        elif np is not None and isinstance(vec, np.ndarray):
            result.append(vec.tolist())
        elif isinstance(vec, (list, tuple)):
            result.append(list(vec))
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported embedding vector type: {type(vec)!r}")
    return result


@dataclass
class VietnameseE5Embedder:
    """Embedder that prefixes inputs as recommended for multilingual E5 models."""

    model: Optional[object] = None
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    batch_size: int = 32
    passage_prefix: str = "passage:"
    query_prefix: str = "query:"
    normalize: bool = True
    _encoder: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model is not None:
            self._encoder = self.model
            return

        transformer_cls = _ensure_sentence_transformer()
        logger.info("Loading sentence transformer model: %s", self.model_name)
        self._encoder = transformer_cls(self.model_name, device=self.device)

    def encode_passages(self, passages: Sequence[str]) -> List[List[float]]:
        prepared = [_prepare_text(text, self.passage_prefix) for text in passages]
        vectors = self._call_encoder(prepared)
        return _to_float_lists(vectors)

    def encode_queries(self, queries: Sequence[str]) -> List[List[float]]:
        prepared = [_prepare_text(text, self.query_prefix) for text in queries]
        vectors = self._call_encoder(prepared)
        return _to_float_lists(vectors)

    def _call_encoder(self, texts: Sequence[str]):
        encode = getattr(self._encoder, "encode", None)
        if encode is None:  # pragma: no cover - defensive
            raise AttributeError("Encoder object must provide an `encode` method.")
        return encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )


__all__ = ["VietnameseE5Embedder", "DEFAULT_MODEL_NAME"]
