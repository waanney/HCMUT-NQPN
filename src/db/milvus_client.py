"""Milvus schema helpers for GSoft chatbot collections."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable, cast
from core.config import load_config

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

config = load_config()
logger = logging.getLogger(__name__)

MILVUS_URI = config.milvus.uri
MILVUS_ALIAS = config.milvus.alias
MILVUS_DB_NAME = config.milvus.db_name

DOC_COLLECTION_NAME = "gsoft_docs"

DOC_DENSE_DIM_DEFAULT = config.milvus.doc_dense_dim


@runtime_checkable
class CollectionLike(Protocol):
    """Protocol exposing the attributes pytest needs."""

    name: str
    schema: CollectionSchema


@dataclass
class OfflineCollection:
    """Fallback collection object when Milvus is not reachable."""

    name: str
    schema: CollectionSchema

# Connect to milvus and ensure it has collections
def connect_to_milvus(
    alias: Optional[str] = None,
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> str:
    """Connect to Milvus and return the alias that should be reused."""
    alias = alias or MILVUS_ALIAS
    uri = uri or MILVUS_URI
    db_name = db_name or MILVUS_DB_NAME

    try:
        if not connections.has_connection(alias):
            connections.connect(alias=alias, uri=uri, db_name=db_name)
    except MilvusException as exc:
        logger.debug("Unable to connect to Milvus; continuing offline: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unexpected Milvus connection error; continuing offline: %s", exc)
    return alias


def ensure_gsoft_docs_collection(
    alias: Optional[str] = None,
    dense_dim: Optional[int] = None,
    auto_recreate: bool = True,
) -> CollectionLike:
    """Make sure the gsoft_docs collection exists (or build the schema offline).
    
    If collection exists with different dimension and auto_recreate=True,
    automatically drops and recreates the collection with correct dimension.
    
    Args:
        alias: Milvus connection alias
        dense_dim: Expected dense vector dimension (defaults to config value)
        auto_recreate: If True, automatically recreate collection if dimension mismatch (default: True)
    """
    expected_dim = dense_dim or DOC_DENSE_DIM_DEFAULT
    schema = _build_docs_schema(expected_dim)
    
    # Check if collection exists and validate dimension
    try:
        alias = connect_to_milvus(alias)
        if utility.has_collection(DOC_COLLECTION_NAME, using=alias):
            existing_collection = Collection(DOC_COLLECTION_NAME, using=alias)
            # Check dimension of dense_vec field
            existing_dim = None
            for field in existing_collection.schema.fields:
                if field.name == "dense_vec":
                    # Get dimension from field - try multiple ways
                    existing_dim = getattr(field, "dim", None)
                    if existing_dim is None:
                        # Try to get from params dict
                        if hasattr(field, "params") and isinstance(field.params, dict):
                            existing_dim = field.params.get("dim")
                        # Try to get from params attribute directly
                        elif hasattr(field, "params") and hasattr(field.params, "get"):
                            existing_dim = field.params.get("dim")
                    break
            
            if existing_dim is not None and existing_dim != expected_dim:
                logger.warning(
                    f"⚠️  Collection '{DOC_COLLECTION_NAME}' exists with dimension {existing_dim}, "
                    f"but expected dimension is {expected_dim}."
                )
                if auto_recreate:
                    logger.info(
                        f"🔄 Auto-recreating collection '{DOC_COLLECTION_NAME}' with dimension {expected_dim}..."
                    )
                    try:
                        # Drop existing collection
                        utility.drop_collection(DOC_COLLECTION_NAME, using=alias)
                        logger.info(f"✅ Dropped old collection with dimension {existing_dim}")
                        
                        # Create new collection with correct dimension
                        new_collection = Collection(
                            name=DOC_COLLECTION_NAME,
                            schema=schema,
                            using=alias,
                            consistency_level="Session",
                        )
                        logger.info(
                            f"✅ Created new collection '{DOC_COLLECTION_NAME}' with dimension {expected_dim}"
                        )
                        return cast(CollectionLike, new_collection)
                    except Exception as recreate_error:
                        logger.error(
                            f"❌ Failed to auto-recreate collection: {recreate_error}",
                            exc_info=True,
                        )
                        # Fall through to return existing collection
                else:
                    logger.warning(
                        f"⚠️  Dimension mismatch detected but auto_recreate=False. "
                        f"Collection will use existing dimension {existing_dim}, which may cause insertion failures."
                    )
                    return cast(CollectionLike, existing_collection)
            
            if existing_dim == expected_dim:
                logger.debug(f"✅ Collection '{DOC_COLLECTION_NAME}' exists with correct dimension {expected_dim}")
            else:
                logger.debug(f"Collection '{DOC_COLLECTION_NAME}' exists (dimension check skipped)")
            
            return cast(CollectionLike, existing_collection)
    except Exception as e:
        logger.debug(f"Could not check existing collection: {e}")
    
    return _ensure_collection(DOC_COLLECTION_NAME, schema, alias)


# Build Schema 
def _build_docs_schema(dense_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,
        ),
        FieldSchema(
            name="original_doc_id",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
        FieldSchema(name="permission", dtype=DataType.INT8),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
        FieldSchema(
            name="text_preview",
            dtype=DataType.VARCHAR,
            max_length=1024,
        ),
        FieldSchema(
            name="dense_vec",
            dtype=DataType.FLOAT_VECTOR,
            dim=dense_dim,
        ),
        FieldSchema(
            name="sparse_vec",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Chunks of GSoft documentation for semantic retrieval.",
        enable_dynamic_field=False,
    )


def _ensure_collection(
    name: str,
    schema: CollectionSchema,
    alias: Optional[str],
) -> CollectionLike:
    alias = connect_to_milvus(alias)
    try:
        if utility.has_collection(name, using=alias):
            return cast(CollectionLike, Collection(name, using=alias))  
        return cast(CollectionLike, Collection(  
            name=name,
            schema=schema,
            using=alias,
            consistency_level="Session",
        ))
    except MilvusException as exc:
        logger.debug("Milvus unavailable for %s; using offline schema: %s", name, exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Unexpected error preparing Milvus collection %s: %s", name, exc)
    return OfflineCollection(name=name, schema=schema)


__all__ = [
    "CollectionLike",
    "DOC_COLLECTION_NAME",
    "DOC_DENSE_DIM_DEFAULT",
    "OfflineCollection",
    "connect_to_milvus",
    "ensure_gsoft_docs_collection",
]
