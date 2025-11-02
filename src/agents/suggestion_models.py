"""
Pydantic models for Suggestion Agent I/O
Defines the exact input/output contract for the Suggestion Agent.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ========================
# Input Models (from BA Agent)
# ========================

class KBDocument(BaseModel):
    """Knowledge Base document reference."""
    doc_name: str = Field(..., description="Document name in Milvus")
    
    @field_validator('doc_name')
    @classmethod
    def validate_doc_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_name cannot be empty")
        return v.strip()


class KGNode(BaseModel):
    """Knowledge Graph node reference."""
    doc_name: str = Field(..., description="Node name/ID in Neo4j")
    
    @field_validator('doc_name')
    @classmethod
    def validate_doc_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_name cannot be empty")
        return v.strip()


class BAAgentInput(BaseModel):
    """
    Input from BA Agent containing conflict analysis.
    
    Format:
    {
        "BA_answer_text": "conflict description...",
        "KB": ["doc_name1", "doc_name2", ...],
        "KG": ["doc_name1", "doc_name2", ...]
    }
    """
    BA_answer_text: str = Field(..., description="Text describing the conflict from BA agent")
    KB: List[str] = Field(default_factory=list, description="List of Knowledge Base document names in Milvus")
    KG: List[str] = Field(default_factory=list, description="List of Knowledge Graph node names in Neo4j")
    
    @field_validator('BA_answer_text')
    @classmethod
    def validate_answer_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("BA_answer_text cannot be empty")
        return v.strip()
    
    @field_validator('KB')
    @classmethod
    def validate_kb(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        return [doc.strip() for doc in v if doc and doc.strip()]
    
    @field_validator('KG')
    @classmethod
    def validate_kg(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        return [node.strip() for node in v if node and node.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "BA_answer_text": "Conflict detected: REQ-001 requires email authentication but REQ-002 specifies phone-only authentication",
                "KB": ["requirements_spec_v2.pdf", "authentication_policy.docx", "security_standards.pdf"],
                "KG": ["REQ-001", "REQ-002", "AUTH-POLICY-001"]
            }
        }


# ========================
# Output Models (from Suggestion Agent)
# ========================

class SuggestionAgentOutput(BaseModel):
    """
    Output from Suggestion Agent with suggested solution.
    
    Format:
    {
        "answer_text": "suggested solution...",
        "source": ["url1", "url2", ...],  # Array of URLs from web search
        "KB": ["doc_name1", "doc_name2", ...],  # Pass through from BA input
        "KG": ["identifier1", "identifier2", ...]  # Pass through from BA input
    }
    """
    answer_text: str = Field(..., description="Suggested solution to resolve the conflict")
    source: List[str] = Field(default_factory=list, description="List of URLs or references supporting the suggestion")
    KB: List[str] = Field(default_factory=list, description="List of Knowledge Base document names (pass through from BA input)")
    KG: List[str] = Field(default_factory=list, description="List of Knowledge Graph node identifiers (pass through from BA input)")
    
    @field_validator('answer_text')
    @classmethod
    def validate_answer_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer_text cannot be empty")
        return v.strip()
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        return [url.strip() for url in v if url and url.strip()]
    
    @field_validator('KB')
    @classmethod
    def validate_kb(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        return [doc.strip() for doc in v if doc and doc.strip()]
    
    @field_validator('KG')
    @classmethod
    def validate_kg(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        return [node.strip() for node in v if node and node.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer_text": "Implement multi-factor authentication (MFA) supporting both email and phone number methods. Allow users to configure their preferred primary and secondary authentication methods in account settings.",
                "source": [
                    "https://owasp.org/www-community/controls/Multifactor_Authentication",
                    "https://en.wikipedia.org/wiki/Multi-factor_authentication"
                ],
                "KB": ["requirements_spec_v2.pdf", "authentication_policy.docx"],
                "KG": ["REQ-001", "REQ-002"]
            }
        }


# ========================
# Helper Functions
# ========================

def parse_ba_input(input_dict: Dict[str, Any]) -> BAAgentInput:
    """
    Parse and validate BA agent input dictionary.
    
    Args:
        input_dict: Dictionary with BA_answer_text, KB, and KG keys
    
    Returns:
        Validated BAAgentInput object
    
    Raises:
        ValueError: If input is invalid
    """
    return BAAgentInput(**input_dict)


def create_suggestion_output(
    answer_text: str, 
    source: List[str], 
    kb: Optional[List[str]] = None,
    kg: Optional[List[str]] = None
) -> SuggestionAgentOutput:
    """
    Create and validate suggestion agent output.
    
    Args:
        answer_text: The suggested solution text
        source: List of URLs or references
        kb: Optional list of KB document names (pass through from BA input)
        kg: Optional list of KG node identifiers (pass through from BA input)
    
    Returns:
        Validated SuggestionAgentOutput object
    
    Raises:
        ValueError: If output is invalid
    """
    return SuggestionAgentOutput(
        answer_text=answer_text, 
        source=source or [],
        KB=kb or [],
        KG=kg or []
    )


def validate_ba_input_json(json_str: str) -> BAAgentInput:
    """
    Parse and validate BA agent input from JSON string.
    
    Args:
        json_str: JSON string with BA agent output
    
    Returns:
        Validated BAAgentInput object
    
    Raises:
        ValueError: If JSON is invalid or validation fails
    """
    import json
    try:
        data = json.loads(json_str)
        return parse_ba_input(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def suggestion_output_to_json(output: SuggestionAgentOutput) -> str:
    """
    Convert suggestion output to JSON string.
    
    Args:
        output: SuggestionAgentOutput object
    
    Returns:
        JSON string
    """
    return output.model_dump_json(indent=2)


# ========================
# Exports
# ========================

__all__ = [
    # Input models
    "BAAgentInput",
    "KBDocument",
    "KGNode",
    
    # Output models
    "SuggestionAgentOutput",
    
    # Helper functions
    "parse_ba_input",
    "create_suggestion_output",
    "validate_ba_input_json",
    "suggestion_output_to_json",
]
