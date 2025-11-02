"""Central configuration objects for the GSoft chatbot services."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


logger = logging.getLogger(__name__)


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:  # pragma: no cover - defensive guard
        logger.warning("Invalid integer for %s (%r); using default %s", name, value, default)
        return default


@dataclass(frozen=True)
class MilvusSettings:
    uri: str = _get_env("MILVUS_URI", "http://localhost:19530")
    alias: str = _get_env("MILVUS_ALIAS", "default")
    db_name: str = _get_env("MILVUS_DB_NAME", "default")
    doc_dense_dim: int = _get_env_int("DOC_DENSE_DIM", 1024)
    faq_dense_dim: int = _get_env_int("FAQ_DENSE_DIM", 1024)

@dataclass(frozen=True)
class Neo4jSetting:
    uri: str = _get_env("NEO4J_URI", "bolt://localhost:7687")
    user: str = _get_env("NEO4J_USER", "neo4j")
    password: str = _get_env("NEO4J_PASSWORD", "password123")

@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = _get_env("EMBEDDER_MODEL_NAME", "intfloat/multilingual-e5-large")
    device: Optional[str] = os.getenv("EMBEDDER_DEVICE")
    batch_size: int = _get_env_int("EMBEDDER_BATCH_SIZE", 32)
    normalize_embeddings: bool = os.getenv("EMBEDDER_NORMALIZE", "true").lower() != "false"


@dataclass(frozen=True)
class AppConfig:
    milvus: MilvusSettings = MilvusSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    neo4j: Neo4jSetting = Neo4jSetting()


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load the application configuration once per process."""
    return AppConfig()


__all__ = [
    "AppConfig",
    "MilvusSettings",
    "Neo4jSetting",
    "EmbeddingSettings",
    "load_config",
]
