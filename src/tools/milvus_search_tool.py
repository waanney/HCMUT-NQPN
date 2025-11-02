"""Milvus search tool for function calling.

This tool provides a function-callable interface for searching Milvus Knowledge Base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pymilvus import Collection
from pymilvus.exceptions import MilvusException

from core.config import load_config
from db.milvus_client import (
    DOC_COLLECTION_NAME,
    connect_to_milvus,
    ensure_gsoft_docs_collection,
)
from data_pipeline.embedder import VietnameseE5Embedder

logger = logging.getLogger(__name__)


class MilvusSearchInput(BaseModel):
    """Input model for Milvus search tool.
    
    Format:
    {
        "query": "user question or search text",
        "top_k": 5  # Optional, default is 5
    }
    """
    
    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=15, description="Number of top results to return", ge=1, le=50)


class MilvusSearchOutput(BaseModel):
    """Output model for Milvus search tool.
    
    Format:
    {
        "doc_ids": ["doc_001", "doc_002", ...],  # Document IDs
        "results": [
            {
                "id": "doc_001",
                "text_preview": "...",
                "source": "...",
                "url": "...",
                "score": 0.95
            },
            ...
        ]
    }
    """
    
    doc_ids: List[str] = Field(default_factory=list, description="Array of document IDs")
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed search results with scores and metadata"
    )


class MilvusSearchTool:
    """Tool for searching Milvus Knowledge Base."""
    
    def __init__(
        self,
        collection: Optional[Collection] = None,
        embedder: Optional[VietnameseE5Embedder] = None,
    ):
        """Initialize Milvus search tool.
        
        Args:
            collection: Milvus collection instance (optional, will create if not provided)
            embedder: Embedder instance (optional, will create if not provided)
        """
        self.config = load_config()
        
        # Initialize Milvus
        self.milvus_alias = connect_to_milvus()
        self.collection = collection or ensure_gsoft_docs_collection(
            alias=self.milvus_alias,
            dense_dim=self.config.milvus.doc_dense_dim
        )
        
        # Initialize Embedder
        self.embedder = embedder or VietnameseE5Embedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
            normalize=self.config.embedding.normalize_embeddings,
        )
    
    def _check_collection_has_data(self) -> bool:
        """Check if collection has any data."""
        try:
            if hasattr(self.collection, "num_entities"):
                return self.collection.num_entities > 0
            return False
        except:
            return False
    
    def search(self, input_data: MilvusSearchInput) -> MilvusSearchOutput:
        """Search Milvus collection for similar document chunks.
        
        This is the main function for function calling.
        
        Args:
            input_data: MilvusSearchInput with query and top_k
            
        Returns:
            MilvusSearchOutput with doc_ids and detailed results
        """
        try:
            # Load collection (only if it's a real Collection object, not OfflineCollection)
            if not hasattr(self.collection, "load"):
                logger.warning("Milvus collection is offline or unavailable. Skipping search.")
                return MilvusSearchOutput()
            
            # Check if collection has data
            if not self._check_collection_has_data():
                logger.info("Collection is empty. No data to search.")
                return MilvusSearchOutput()
            
            # Load collection
            try:
                self.collection.load()
            except MilvusException as e:
                if "index not found" in str(e).lower() or e.code == 700:
                    logger.warning("Collection exists but has no index. This usually means the collection is empty or index was not created after inserting data.")
                    return MilvusSearchOutput()
                elif "sparse_vec" in str(e).lower():
                    logger.warning("Sparse vector index missing. This is expected if sparse vectors are not used.")
                    return MilvusSearchOutput()
                raise

            # Encode query for dense vector
            query_vectors = self.embedder.encode_queries([input_data.query])
            if not query_vectors:
                return MilvusSearchOutput()
            
            # Create sparse vector for query
            from agents.ingest_agent import IngestAgent
            ingest_agent = IngestAgent(embedder=self.embedder)
            query_sparse_vec = ingest_agent.create_sparse_vector(input_data.query)

            # Hybrid search: search both dense and sparse vectors
            # First, search with dense vector
            dense_search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            dense_results = self.collection.search(
                data=query_vectors,
                anns_field="dense_vec",
                param=dense_search_params,
                limit=input_data.top_k * 2,  # Get more results to merge
                output_fields=["id", "original_doc_id", "text_preview", "source", "url"],
            )
            
            # Search with sparse vector if available
            sparse_results = []
            if query_sparse_vec:
                try:
                    sparse_search_params = {
                        "metric_type": "IP",  # Inner Product for sparse vectors
                    }
                    
                    sparse_results = self.collection.search(
                        data=[query_sparse_vec],
                        anns_field="sparse_vec",
                        param=sparse_search_params,
                        limit=input_data.top_k * 2,  # Get more results to merge
                        output_fields=["id", "original_doc_id", "text_preview", "source", "url"],
                    )
                except Exception as e:
                    logger.warning(f"Sparse vector search failed (will use dense only): {e}")
                    sparse_results = []
            
            # Merge results from dense and sparse search
            # Combine scores using weighted combination
            combined_results = {}
            
            # Process dense results
            if dense_results and len(dense_results) > 0:
                for hits in dense_results:
                    for hit in hits:
                        doc_id = hit.entity.get("id", "")
                        if doc_id:
                            # Convert distance to similarity score (lower distance = higher similarity)
                            dense_score = float(hit.score) if hasattr(hit, "score") else 0.0
                            # Normalize: for L2 distance, lower is better, convert to similarity (0-1)
                            # Use exponential decay: similarity = 1 / (1 + distance)
                            similarity_score = 1.0 / (1.0 + dense_score)
                            
                            if doc_id not in combined_results:
                                combined_results[doc_id] = {
                                    "id": doc_id,
                                    "original_doc_id": hit.entity.get("original_doc_id"),
                                    "text_preview": hit.entity.get("text_preview", ""),
                                    "source": hit.entity.get("source", ""),
                                    "url": hit.entity.get("url", ""),
                                    "dense_score": dense_score,
                                    "sparse_score": 0.0,
                                    "combined_score": 0.0,
                                }
                            combined_results[doc_id]["dense_score"] = dense_score
                            combined_results[doc_id]["dense_similarity"] = similarity_score
            
            # Process sparse results
            if sparse_results and len(sparse_results) > 0:
                for hits in sparse_results:
                    for hit in hits:
                        doc_id = hit.entity.get("id", "")
                        if doc_id:
                            sparse_score = float(hit.score) if hasattr(hit, "score") else 0.0
                            # For IP (Inner Product), higher is better, normalize to 0-1
                            # Assuming scores are already normalized (cosine similarity range)
                            sparse_similarity = max(0.0, min(1.0, (sparse_score + 1) / 2))
                            
                            if doc_id not in combined_results:
                                combined_results[doc_id] = {
                                    "id": doc_id,
                                    "original_doc_id": hit.entity.get("original_doc_id"),
                                    "text_preview": hit.entity.get("text_preview", ""),
                                    "source": hit.entity.get("source", ""),
                                    "url": hit.entity.get("url", ""),
                                    "dense_score": 0.0,
                                    "sparse_score": sparse_score,
                                    "combined_score": 0.0,
                                }
                            combined_results[doc_id]["sparse_score"] = sparse_score
                            combined_results[doc_id]["sparse_similarity"] = sparse_similarity
            
            # Calculate combined scores (weighted combination)
            # Default weights: 0.7 for dense, 0.3 for sparse (can be adjusted)
            dense_weight = 0.7
            sparse_weight = 0.3
            
            for doc_id, result in combined_results.items():
                dense_sim = result.get("dense_similarity", 0.0)
                sparse_sim = result.get("sparse_similarity", 0.0)
                
                # Combined score: weighted average
                if dense_sim > 0 and sparse_sim > 0:
                    combined_score = dense_weight * dense_sim + sparse_weight * sparse_sim
                elif dense_sim > 0:
                    combined_score = dense_sim
                elif sparse_sim > 0:
                    combined_score = sparse_sim
                else:
                    combined_score = 0.0
                
                result["combined_score"] = combined_score
            
            # Sort by combined score (descending)
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )[:input_data.top_k]
            
            # Use combined results instead of direct search results
            results_to_process = sorted_results

            # Format results
            doc_ids = []
            detailed_results = []
            
            logger.info(f"Searching Milvus with hybrid search (dense + sparse) for query: '{input_data.query}', top_k: {input_data.top_k}")
            
            if results_to_process and len(results_to_process) > 0:
                for i, result in enumerate(results_to_process, 1):
                    doc_id = result.get("id", "")
                    if doc_id:
                        doc_ids.append(doc_id)
                        text_preview = result.get("text_preview", "")
                        source = result.get("source", "")
                        url = result.get("url", "")
                        
                        # Use combined score for final ranking
                        # But also store dense_score for backward compatibility
                        combined_score = result.get("combined_score", 0.0)
                        dense_score = result.get("dense_score", 0.0)
                        sparse_score = result.get("sparse_score", 0.0)
                        
                        detailed_results.append({
                            "id": doc_id,
                            "original_doc_id": result.get("original_doc_id"),
                            "text_preview": text_preview,
                            "source": source,
                            "url": url,
                            "score": dense_score,  # Keep dense_score for backward compatibility
                            "combined_score": combined_score,
                            "dense_score": dense_score,
                            "sparse_score": sparse_score,
                        })
                        
                        # Log search results with hybrid scores
                        logger.info(f"Milvus Hybrid Result {i}:")
                        logger.info(f"  - Doc ID: {doc_id}")
                        logger.info(f"  - Combined Score: {combined_score:.4f} (higher = more relevant)")
                        logger.info(f"  - Dense Score: {dense_score:.4f} (distance, lower = more relevant)")
                        logger.info(f"  - Sparse Score: {sparse_score:.4f}")
                        logger.info(f"  - Source: {source}")
                        if text_preview:
                            preview = text_preview[:200] + "..." if len(text_preview) > 200 else text_preview
                            logger.info(f"  - Preview: {preview}")
                        if url:
                            logger.info(f"  - URL: {url}")
                        
                        # Print score to console for visibility
                        print(f"[Milvus Hybrid Search] Doc {i}: {doc_id} | Combined: {combined_score:.4f} | Dense: {dense_score:.4f} | Sparse: {sparse_score:.4f}")
            else:
                logger.warning("No results found in Milvus hybrid search")
            
            # Log summary with hybrid scores
            if detailed_results:
                scores_summary = ", ".join([f"#{i+1}: combined={r['combined_score']:.4f}, dense={r['dense_score']:.4f}, sparse={r['sparse_score']:.4f}" for i, r in enumerate(detailed_results)])
                logger.info(f"Milvus hybrid search completed: Found {len(doc_ids)} documents")
                logger.info(f"Milvus hybrid scores: {scores_summary}")
                print(f"[Milvus Hybrid Search Summary] Found {len(doc_ids)} documents | Scores: {scores_summary}")
            else:
                logger.info(f"Milvus hybrid search completed: Found {len(doc_ids)} documents")
                logger.info(f"Milvus doc_ids: {doc_ids}")

            return MilvusSearchOutput(
                doc_ids=doc_ids,
                results=detailed_results,
            )

        except Exception as e:
            logger.error(f"Error searching Milvus: {e}", exc_info=True)
            return MilvusSearchOutput()


# Function for OpenAI function calling format
def get_milvus_search_function_schema() -> Dict[str, Any]:
    """Get OpenAI function calling schema for Milvus search tool.
    
    Returns:
        Dictionary with function schema for OpenAI function calling
    """
    return {
        "type": "function",
        "function": {
            "name": "search_milvus",
            "description": "Search the Knowledge Base (Milvus) for relevant document chunks based on semantic similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (default: 5, max: 50)",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["query"]
            }
        }
    }


__all__ = ["MilvusSearchTool", "MilvusSearchInput", "MilvusSearchOutput", "get_milvus_search_function_schema"]

