"""
Pydantic models for Guardrail Agent I/O
Defines the exact input/output contract for the Guardrail Agent.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class GuardrailAgentInput(BaseModel):
    """
    Input to Guardrail Agent for evaluation.
    
    Format:
    {
        "user_input": "original user query or text",
        "answer_text": "answer from suggestion_agent or rag_agent",
        "source": ["url1", "url2", ...],  # Optional, array of URLs
        "KB": ["doc_name1", "doc_name2", ...],  # Optional
        "KG": ["identifier1", "identifier2", ...]  # Optional
    }
    """
    user_input: Optional[str] = Field(default="", description="Original user query or input text (optional, recommended for better evaluation)")
    answer_text: str = Field(..., description="Answer text from suggestion_agent or rag_agent")
    source: Optional[List[str]] = Field(default=None, description="List of source URLs (from suggestion_agent) or None (from rag_agent)")
    KB: List[str] = Field(default_factory=list, description="List of Knowledge Base document names")
    KG: List[str] = Field(default_factory=list, description="List of Knowledge Graph node identifiers")
    
    @field_validator('user_input')
    @classmethod
    def validate_user_input(cls, v: Optional[str]) -> str:
        if v is None:
            return ""
        return v.strip() if v else ""
    
    @field_validator('answer_text')
    @classmethod
    def validate_answer_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer_text cannot be empty")
        return v.strip()
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
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
                "user_input": "What features does Project Alpha include?",
                "answer_text": "Project Alpha includes user authentication, data storage, and reporting features.",
                "source": ["https://example.com/project-alpha"],
                "KB": ["doc_014", "doc_015"],
                "KG": ["PROJ-001", "REQ-002"]
            }
        }


class GuardrailAgentOutput(BaseModel):
    """
    Output from Guardrail Agent with evaluation results.
    
    Format:
    {
        "guardrail_status": "APPROVED" | "REJECTED" | "REVIEW_REQUIRED",
        "issues_found": [
            {"type": "clarity", "description": "..."},
            ...
        ],
        "suggested_action": "approve" | "rewrite" | "verify_source",
        "summary": "evaluation summary text"
    }
    """
    guardrail_status: str = Field(..., description="Evaluation status: APPROVED, REJECTED, or REVIEW_REQUIRED")
    issues_found: List[Dict[str, str]] = Field(default_factory=list, description="List of issues found during evaluation")
    suggested_action: str = Field(..., description="Suggested action: approve, rewrite, or verify_source")
    summary: str = Field(..., description="Summary of the evaluation")
    
    @field_validator('guardrail_status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = ["APPROVED", "REJECTED", "REVIEW_REQUIRED"]
        if v not in valid_statuses:
            raise ValueError(f"guardrail_status must be one of: {valid_statuses}")
        return v
    
    @field_validator('suggested_action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = ["approve", "rewrite", "verify_source"]
        if v not in valid_actions:
            raise ValueError(f"suggested_action must be one of: {valid_actions}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "guardrail_status": "APPROVED",
                "issues_found": [],
                "suggested_action": "approve",
                "summary": "Response meets quality standards"
            }
        }


# ========================
# Helper Functions
# ========================

def parse_guardrail_input(input_dict: Dict[str, Any]) -> GuardrailAgentInput:
    """
    Parse and validate guardrail agent input dictionary.
    
    Args:
        input_dict: Dictionary with user_input, answer_text, and optional source, KB, KG
    
    Returns:
        Validated GuardrailAgentInput object
    
    Raises:
        ValueError: If input is invalid
    """
    return GuardrailAgentInput(**input_dict)


def validate_guardrail_input_json(json_str: str) -> GuardrailAgentInput:
    """
    Parse and validate guardrail agent input from JSON string.
    
    Args:
        json_str: JSON string with guardrail agent input
    
    Returns:
        Validated GuardrailAgentInput object
    
    Raises:
        ValueError: If JSON is invalid or validation fails
    """
    import json
    try:
        data = json.loads(json_str)
        return parse_guardrail_input(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def guardrail_output_to_json(output: Dict[str, Any]) -> str:
    """
    Convert guardrail output to JSON string.
    
    Args:
        output: GuardrailAgentOutput dictionary
    
    Returns:
        JSON string
    """
    import json
    return json.dumps(output, ensure_ascii=False, indent=2)


# ========================
# Exports
# ========================

__all__ = [
    # Input models
    "GuardrailAgentInput",
    
    # Output models
    "GuardrailAgentOutput",
    
    # Helper functions
    "parse_guardrail_input",
    "validate_guardrail_input_json",
    "guardrail_output_to_json",
]

