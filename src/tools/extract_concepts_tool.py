"""Tool for extracting key concepts from requirements text."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractConceptsInput(BaseModel):
    """Input model for concept extraction tool.
    
    Format:
    {
        "text": "requirements text to extract concepts from",
        "max_concepts": 5  # Optional, default is 5
    }
    """
    
    text: str = Field(..., description="The text to extract concepts from")
    max_concepts: int = Field(default=5, description="Maximum number of concepts to extract", ge=1, le=10)


class ExtractConceptsOutput(BaseModel):
    """Output model for concept extraction tool.
    
    Format:
    {
        "concepts": ["concept1", "concept2", ...],
        "count": 3
    }
    """
    
    concepts: List[str] = Field(default_factory=list, description="List of extracted concepts")
    count: int = Field(default=0, description="Number of concepts extracted")


class ExtractConceptsTool:
    """Tool for extracting key business concepts from requirements text."""
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        """Initialize concept extraction tool.
        
        Args:
            llm_client: OpenAI client instance (optional, will create if not provided)
        """
        if llm_client is None:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
            else:
                self.llm_client = None
                logger.warning("OpenAI API key not set. Concept extraction will use fallback method.")
        else:
            self.llm_client = llm_client
    
    def extract_concepts(self, input_data: ExtractConceptsInput) -> ExtractConceptsOutput:
        """Extract key concepts from requirements text.
        
        Args:
            input_data: ExtractConceptsInput with text and max_concepts
            
        Returns:
            ExtractConceptsOutput with concepts list and count
        """
        if not input_data.text or not input_data.text.strip():
            return ExtractConceptsOutput(concepts=[], count=0)
        
        # Try to use LLM for better extraction
        if self.llm_client:
            try:
                concepts = self._extract_with_llm(input_data.text, input_data.max_concepts)
                if concepts:
                    return ExtractConceptsOutput(concepts=concepts, count=len(concepts))
            except Exception as e:
                logger.warning(f"LLM concept extraction failed, using fallback: {e}")
        
        # Fallback to simple keyword extraction
        concepts = self._extract_simple(input_data.text, input_data.max_concepts)
        return ExtractConceptsOutput(concepts=concepts, count=len(concepts))
    
    def _extract_with_llm(self, text: str, max_concepts: int) -> List[str]:
        """Extract concepts using OpenAI LLM."""
        prompt = f"""
Extract {max_concepts} key business concepts or entities from this requirements text.
Focus on: user roles, features, business processes, data entities, system components.

Text: {text}

Return only the concepts as a comma-separated list. Do not include any explanation.
"""
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        concepts_text = response.choices[0].message.content.strip()
        # Remove any markdown formatting
        concepts_text = concepts_text.replace("```", "").replace("json", "").strip()
        
        # Split by comma and clean
        concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]
        
        # Remove empty strings and limit
        concepts = [c for c in concepts if c][:max_concepts]
        
        return concepts
    
    def _extract_simple(self, text: str, max_concepts: int) -> List[str]:
        """Fallback: simple keyword extraction using regex."""
        # Extract words with at least 3 characters
        words = re.findall(r'\b[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]{3,}\b', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'this',
            'with', 'from', 'have', 'will', 'when', 'what', 'where', 'which',
            'would', 'could', 'should', 'might', 'must', 'shall', 'than', 'then',
            'them', 'these', 'those', 'there', 'their', 'they', 'been', 'being',
            'about', 'above', 'after', 'again', 'against', 'before', 'below',
            'between', 'during', 'except', 'inside', 'outside', 'through', 'under',
            'until', 'while', 'within', 'without'
        }
        
        # Filter stop words and get unique words
        filtered_words = [w.lower() for w in words if w.lower() not in stop_words]
        unique_words = list(set(filtered_words))
        
        # Sort by length (longer words often more meaningful) and limit
        unique_words.sort(key=len, reverse=True)
        
        return unique_words[:max_concepts]


# Convenience function
def extract_key_concepts(text: str, max_concepts: int = 5) -> List[str]:
    """Quick function to extract concepts from text.
    
    Args:
        text: Text to extract concepts from
        max_concepts: Maximum number of concepts to extract
        
    Returns:
        List of extracted concepts
    """
    tool = ExtractConceptsTool()
    input_data = ExtractConceptsInput(text=text, max_concepts=max_concepts)
    output = tool.extract_concepts(input_data)
    return output.concepts


__all__ = [
    "ExtractConceptsTool",
    "ExtractConceptsInput",
    "ExtractConceptsOutput",
    "extract_key_concepts",
]



