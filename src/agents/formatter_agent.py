"""Formatter Agent for formatting RAG output with color tags.

This agent takes RAG agent output and formats it with color tags:
- <g></g> (green) - content to be added
- <y></y> (yellow) - content to be modified  
- <r></r> (red) - content to be deleted

Input format (from RAG Agent):
{
    "answer_text": "Generated answer text",
    "KB": ["doc_001", "doc_002", ...],
    "KG": ["PROJ-001", "REQ-002", ...]
}

Output format:
{
    "formatted_text": "Formatted text with <g>, <y>, <r> tags",
    "original_text": "Original answer text",
    "KB": ["doc_001", "doc_002", ...],
    "KG": ["PROJ-001", "REQ-002", ...]
}
"""

from __future__ import annotations

import logging
import os
import sys
import importlib
import importlib.metadata
from typing import Any, Dict, List, Optional

# Import from openai-agents package
try:
    # Get the distribution location
    dist = importlib.metadata.distribution('openai-agents')
    agents_path = None
    
    # Find the agents module in the package
    for file in dist.files:
        if 'agents' in str(file) and '__init__.py' in str(file):
            # Extract the path to the agents module
            file_path = str(file.locate())
            if 'site-packages' in file_path or 'dist-packages' in file_path:
                # This is the installed package
                agents_path = file_path.replace('__init__.py', '').replace('\\', '/')
                break
    
    if agents_path:
        # Add the package parent directory to path if needed
        parent_path = os.path.dirname(agents_path.rstrip('/'))
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        
        # Temporarily remove our local agents module
        _temp_agents = sys.modules.pop('agents', None)
        try:
            # Import from the installed openai-agents package
            _openai_agents = importlib.import_module('agents')
            Agent = getattr(_openai_agents, 'Agent')
            ModelSettings = getattr(_openai_agents, 'ModelSettings')
            function_tool = getattr(_openai_agents, 'function_tool')
            Runner = getattr(_openai_agents, 'Runner', None)
        finally:
            # Restore our local agents module
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
    else:
        # Fallback: try direct import (this works if run from src directory)
        _temp_agents = sys.modules.pop('agents', None)
        try:
            from agents import Agent, ModelSettings, function_tool, Runner
        except ImportError as e:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
            raise ImportError(f"Could not import Agent, ModelSettings, function_tool, Runner from openai-agents package. Make sure openai-agents>=0.4.2 is installed: {e}")
        finally:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
except Exception as e:
    # Last resort: try importing directly (might work if package structure allows)
    _temp_agents = sys.modules.pop('agents', None)
    try:
        from agents import Agent, ModelSettings, function_tool, Runner
    except ImportError:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents
        raise ImportError(f"Could not import from openai-agents package: {e}")
    finally:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field

from agents.rag_agent import RAGOutput

logger = logging.getLogger(__name__)


class FormatterInput(BaseModel):
    """Input model for Formatter Agent.
    
    Format:
    {
        "answer_text": "Generated answer text from RAG agent",
        "KB": ["doc_001", "doc_002", ...],
        "KG": ["PROJ-001", "REQ-002", ...]
    }
    """
    
    answer_text: str = Field(..., description="Answer text from RAG agent")
    KB: List[str] = Field(default_factory=list, description="Document IDs from Knowledge Base")
    KG: List[str] = Field(default_factory=list, description="Node identifiers from Knowledge Graph")


class FormatterOutput(BaseModel):
    """Output model for Formatter Agent.
    
    Format:
    {
        "formatted_text": "Formatted text with <g>, <y>, <r> tags",
        "original_text": "Original answer text",
        "KB": ["doc_001", "doc_002", ...],
        "KG": ["PROJ-001", "REQ-002", ...]
    }
    """
    
    formatted_text: str = Field(..., description="Formatted text with color tags (<g>, <y>, <r>)")
    original_text: str = Field(..., description="Original answer text from RAG agent")
    KB: List[str] = Field(default_factory=list, description="Document IDs from Knowledge Base")
    KG: List[str] = Field(default_factory=list, description="Node identifiers from Knowledge Graph")


class FormatterAgent:
    """Agent for formatting text with color tags."""
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        """Initialize Formatter Agent.
        
        Args:
            llm_client: OpenAI client instance (optional)
        """
        # Initialize LLM using OpenAI SDK
        api_key = os.getenv("OPENAI_API_KEY")
        if llm_client is None:
            if api_key:
                try:
                    self.llm_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully for Formatter Agent")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.llm_client = None
            else:
                self.llm_client = None
                logger.warning("OpenAI API key not set. LLM functionality will be limited.")
        else:
            self.llm_client = llm_client
            if self.llm_client:
                logger.info("OpenAI client provided via parameter")
    
    def format_with_tags(self, text: str) -> str:
        """Format text with color tags using LLM.
        
        Args:
            text: Input text to format
            
        Returns:
            Formatted text with <g>, <y>, <r> tags
        """
        if not self.llm_client:
            # Fallback: return text as-is if no LLM client
            logger.warning("No LLM client available. Returning text as-is.")
            return text
        
        prompt = f"""You are a Text Formatter.

**Goal**:
Analyze the given text and apply color-coded tags to indicate additions, modifications, and deletions.

**Guide**:

Use <g></g> around content that should be added (green color).
Use <y></y> around content that should be modified (yellow color).
Use <r></r> around content that should be deleted (red color).

Mark new information or additions with <g></g>.
Mark changes or modifications with <y></y>.
Mark content to be removed with <r></r>.
Leave unmodified text as-is.
IMPORTANT: Preserve the original structure and meaning.
Return only the formatted text — no explanations or comments.
Text to format: {text}
"""

        try:
            # Use OpenAI SDK to create chat completion
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a text formatter that adds color tags to text for highlighting additions, modifications, and deletions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,  # Lower temperature for more consistent formatting
                max_tokens=2000,
            )
            
            # Extract content from response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                else:
                    logger.warning("OpenAI response has no content")
                    return text
            else:
                logger.warning("OpenAI response has no choices")
                return text
                
        except RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}")
            return text
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            return text
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout error: {e}")
            return text
        except APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return text
        except Exception as e:
            logger.error(f"Unexpected error formatting text: {e}", exc_info=True)
            return text
    
    def format_rag_output(self, rag_output: RAGOutput) -> FormatterOutput:
        """Format RAG output with color tags.
        
        Args:
            rag_output: RAGOutput from RAG agent
            
        Returns:
            FormatterOutput with formatted text and original references
        """
        formatted_text = self.format_with_tags(rag_output.answer_text)
        
        return FormatterOutput(
            formatted_text=formatted_text,
            original_text=rag_output.answer_text,
            KB=rag_output.KB,
            KG=rag_output.KG,
        )


# Import tool from tools directory (lazy import to avoid circular dependency)
def _get_format_text_with_tags_tool():
    """Get format_text_with_tags tool (lazy import)."""
    from tools.formatter_tool import format_text_with_tags
    return format_text_with_tags


def create_formatter_agent(
    name: str = "Formatter Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.05,
    max_tokens: int = 2000,
    top_p: Optional[float] = None,
) -> Agent:
    """Create a Formatter Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for randomness (default: 0.3 for consistent formatting)
        max_tokens: Maximum tokens in response (default: 2000)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for text formatting with color tags
        
    Example:
        from agents import Runner
        from agents.formatter_agent import create_formatter_agent
        
        agent = create_formatter_agent()
        result = await Runner.run(agent, "Format this text: ...")
        print(result.final_output)
    """
    if instructions is None:
        instructions = (f"""You are a Text Formatter.

**Goal**:
Analyze the given text and apply color-coded tags to indicate additions, modifications, and deletions.

**Guide**:

Use <g></g> around content that should be added (green color).
Use <y></y> around content that should be modified (yellow color).
Use <r></r> around content that should be deleted (red color).

Mark new information or additions with <g></g>.
Mark changes or modifications with <y></y>.
Mark content to be removed with <r></r>.
Leave unmodified text as-is.
IMPORTANT: Preserve the original structure and meaning.
Return only the formatted text — no explanations or comments.
Text to format: {text}
"""
        )
    
    # Create model settings
    model_settings_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        model_settings_kwargs["top_p"] = top_p
    
    model_settings = ModelSettings(**model_settings_kwargs)
    
    # Create agent with tools
    # Get tool (lazy import to avoid circular dependency)
    format_tool = _get_format_text_with_tags_tool()
    
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[format_tool],
    )
    
    logger.info(f"Created Formatter Agent '{name}' with model {model}")
    return agent


__all__ = [
    "FormatterAgent",
    "FormatterInput",
    "FormatterOutput",
    "create_formatter_agent",
]
