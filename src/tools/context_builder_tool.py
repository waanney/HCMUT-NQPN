"""Tool for building context strings from KB and KG results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BuildKBContextInput(BaseModel):
    """Input model for KB context builder tool.
    
    Format:
    {
        "kb_results": [
            {
                "doc_name": "...",
                "text_preview": "...",
                "similarity_score": 0.95
            },
            ...
        ],
        "max_results": 3  # Optional, default is 3
    }
    """
    
    kb_results: List[Dict[str, Any]] = Field(..., description="Knowledge Base search results")
    max_results: int = Field(default=3, description="Maximum number of results to include", ge=1, le=10)


class BuildKGContextInput(BaseModel):
    """Input model for KG context builder tool.
    
    Format:
    {
        "kg_results": [
            {
                "doc_name": "...",
                "source_type": "...",
                "relationship": "...",
                "target_type": "...",
                "concept": "..."
            },
            ...
        ],
        "max_results": 3  # Optional, default is 3
    }
    """
    
    kg_results: List[Dict[str, Any]] = Field(..., description="Knowledge Graph search results")
    max_results: int = Field(default=3, description="Maximum number of results to include", ge=1, le=10)


class ContextBuilderTool:
    """Tool for building context strings from search results."""
    
    def build_kb_context(self, input_data: BuildKBContextInput) -> str:
        """Build context string from Knowledge Base results.
        
        Args:
            input_data: BuildKBContextInput with kb_results and max_results
            
        Returns:
            Formatted context string
        """
        if not input_data.kb_results:
            return "No similar requirements found in knowledge base."
        
        context_parts = []
        for i, result in enumerate(input_data.kb_results[:input_data.max_results], 1):
            doc_name = result.get("doc_name", "N/A")
            text_preview = result.get("text_preview", "N/A")
            # Truncate text preview to 200 characters
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            
            similarity_score = result.get("similarity_score", 0)
            if isinstance(similarity_score, float):
                similarity_str = f"{similarity_score:.2f}"
            else:
                similarity_str = str(similarity_score)
            
            context_parts.append(
                f"{i}. Document: {doc_name}\n"
                f"   Content: {text_preview}\n"
                f"   Similarity: {similarity_str}"
            )
        
        return "\n\n".join(context_parts)
    
    def build_kg_context(self, input_data: BuildKGContextInput) -> str:
        """Build context string from Knowledge Graph results.
        
        Args:
            input_data: BuildKGContextInput with kg_results and max_results
            
        Returns:
            Formatted context string
        """
        if not input_data.kg_results:
            return "No related patterns found in knowledge graph."
        
        context_parts = []
        for i, result in enumerate(input_data.kg_results[:input_data.max_results], 1):
            source_type = result.get("source_type", "N/A")
            relationship = result.get("relationship", "N/A")
            target_type = result.get("target_type", "N/A")
            concept = result.get("concept", "N/A")
            
            context_parts.append(
                f"{i}. Relationship: {source_type} "
                f"-[{relationship}]-> "
                f"{target_type}\n"
                f"   Concept: {concept}"
            )
        
        return "\n\n".join(context_parts)


# Convenience functions
def build_kb_context(kb_results: List[Dict[str, Any]], max_results: int = 3) -> str:
    """Quick function to build KB context.
    
    Args:
        kb_results: Knowledge Base search results
        max_results: Maximum number of results to include
        
    Returns:
        Formatted context string
    """
    tool = ContextBuilderTool()
    input_data = BuildKBContextInput(kb_results=kb_results, max_results=max_results)
    return tool.build_kb_context(input_data)


def build_kg_context(kg_results: List[Dict[str, Any]], max_results: int = 3) -> str:
    """Quick function to build KG context.
    
    Args:
        kg_results: Knowledge Graph search results
        max_results: Maximum number of results to include
        
    Returns:
        Formatted context string
    """
    tool = ContextBuilderTool()
    input_data = BuildKGContextInput(kg_results=kg_results, max_results=max_results)
    return tool.build_kg_context(input_data)


__all__ = [
    "ContextBuilderTool",
    "BuildKBContextInput",
    "BuildKGContextInput",
    "build_kb_context",
    "build_kg_context",
]



