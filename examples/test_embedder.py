"""Unit tests for the Vietnamese embedding helper."""

from __future__ import annotations

from typing import List, Sequence

import pytest

from src.data_pipeline.embedder import VietnameseE5Embedder, _prepare_text


class _FakeModel:
    def __init__(self) -> None:
        self.calls: List[Sequence[str]] = []

    def encode(self, texts, batch_size=None, normalize_embeddings=None):
        self.calls.append(tuple(texts))
        # Return a simple embedding (vector length = len(text) % 5 + 1)
        return [
            [float((len(t) + idx) % 7)] for idx, t in enumerate(texts)
        ]


def test_prepare_text_trims_and_prefixes() -> None:
    assert _prepare_text("  xin chào  ", "passage:") == "passage: xin chào"


def test_embedder_applies_e5_prefixes() -> None:
    model = _FakeModel()
    embedder = VietnameseE5Embedder(model=model, normalize=False)

    embeddings = embedder.encode_passages(["dịch vụ SEO", "chăm sóc khách hàng"])

    assert len(embeddings) == 2
    assert model.calls[0][0].startswith("passage:")
    assert isinstance(embeddings[0][0], float)


def test_embedder_handles_queries() -> None:
    model = _FakeModel()
    embedder = VietnameseE5Embedder(model=model)

    embedder.encode_queries(["dịch vụ nào nổi bật?"])

    assert model.calls[0][0].startswith("query:")


def test_missing_sentence_transformers_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import data_pipeline

    monkeypatch.setattr(data_pipeline.embedder, "SentenceTransformer", None)

    with pytest.raises(RuntimeError):
        VietnameseE5Embedder(model=None, model_name="dummy-model")._call_encoder([])
