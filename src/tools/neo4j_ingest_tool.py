"""Neo4j ingestion tools for creating and feeding JSON files into Neo4j."""

from __future__ import annotations

import json
import logging
import os
import sys
import importlib
import importlib.metadata
from pathlib import Path

# Import from openai-agents package (same pattern as ingest_agent.py)
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
def create_neo4j_json_file(
    projects: str = "[]",
    requirements: str = "[]",
    user_stories: str = "[]",
    output_file: str = "neo4j_data.json",
) -> str:
    """Create JSON file with proper Neo4j schema and automatically feed it into Neo4j database.
    
    This tool will:
    1. Create the JSON file with proper Neo4j schema
    2. Automatically feed the JSON file into Neo4j database (Knowledge Graph)
    
    Args:
        projects: JSON string of projects array
        requirements: JSON string of requirements array
        user_stories: JSON string of user stories array
        output_file: Path to output JSON file (default: neo4j_data.json)
        
    Returns:
        Message indicating success, file path, and number of nodes created in Neo4j
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        proj_list = json.loads(projects) if isinstance(projects, str) else projects
        req_list = json.loads(requirements) if isinstance(requirements, str) else requirements
        story_list = json.loads(user_stories) if isinstance(user_stories, str) else user_stories
        
        # Step 1: Create JSON file
        file_path = agent.create_neo4j_json(proj_list, req_list, story_list, output_file)
        logger.info(f"Created Neo4j JSON file: {file_path}")
        
        # Step 2: Automatically feed into Neo4j
        feed_result = agent.feed_neo4j_from_json(file_path)
        logger.info(f"Fed Neo4j JSON into database: {feed_result}")
        
        return f"Successfully created Neo4j JSON file: {file_path}\n{feed_result}"
    except Exception as e:
        logger.error(f"Error creating/feeding Neo4j JSON file: {e}", exc_info=True)
        return f"Error creating/feeding Neo4j JSON file: {str(e)}"


@function_tool
def feed_neo4j_from_json_file(json_file: str) -> str:
    """Feed Neo4j JSON file into Neo4j database (Knowledge Graph).
    
    Args:
        json_file: Path to Neo4j JSON file (e.g., "neo4j_data.json")
        
    Returns:
        Message indicating success and number of nodes created
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        result = agent.feed_neo4j_from_json(json_file)
        return result
    except Exception as e:
        logger.error(f"Error feeding Neo4j from JSON file: {e}", exc_info=True)
        return f"Error feeding Neo4j from JSON file: {str(e)}"


__all__ = [
    "create_neo4j_json_file",
    "feed_neo4j_from_json_file",
]
