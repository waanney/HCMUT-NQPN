"""Formatter tools for formatting text with color tags (green, yellow, red)."""

from __future__ import annotations

import logging
import os
import sys
import importlib
import importlib.metadata
from pathlib import Path

# Import from openai-agents package (same pattern as other tools)
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
            function_tool = getattr(_openai_agents, 'function_tool')
        finally:
            # Restore our local agents module
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
    else:
        # Fallback: try direct import (this works if run from src directory)
        _temp_agents = sys.modules.pop('agents', None)
        try:
            from agents import function_tool
        except ImportError as e:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
            raise ImportError(f"Could not import function_tool from openai-agents package. Make sure openai-agents>=0.4.2 is installed: {e}")
        finally:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
except Exception as e:
    # Last resort: try importing directly (might work if package structure allows)
    _temp_agents = sys.modules.pop('agents', None)
    try:
        from agents import function_tool
    except ImportError:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents
        raise ImportError(f"Could not import from openai-agents package: {e}")
    finally:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents

logger = logging.getLogger(__name__)


@function_tool
def format_text_with_tags(text: str) -> str:
    """Format text with color tags for additions, modifications, and deletions.
    
    This tool analyzes the text and identifies parts that should be:
    - <g></g> (green) - content to be added
    - <y></y> (yellow) - content to be modified
    - <r></r> (red) - content to be deleted
    
    Args:
        text: Input text to format (can be plain text or already contain some formatting hints)
        
    Returns:
        Formatted text with appropriate color tags (<g>, <y>, <r>)
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.formatter_agent import FormatterAgent
        agent = FormatterAgent()
        formatted_text = agent.format_with_tags(text)
        return formatted_text
    except Exception as e:
        logger.error(f"Error formatting text with tags: {e}", exc_info=True)
        # Fallback: return text as-is if formatting fails
        return text


__all__ = [
    "format_text_with_tags",
]
