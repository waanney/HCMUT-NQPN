"""Milvus ingestion tools for creating and feeding JSON files into Milvus."""

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
def create_milvus_json_file(
    documents: str,
    output_file: str = "milvus_data.json",
) -> str:
    """Create JSON file with proper Milvus schema and automatically feed it into Milvus collection.
    
    This tool will:
    1. Create the JSON file with proper Milvus schema (including embeddings)
    2. Automatically feed the JSON file into Milvus collection (Knowledge Base)
    
    Args:
        documents: JSON string of documents array. Each document should have: id, text, original_doc_id (optional), source (optional), url (optional), permission (optional), updated_at (optional)
        output_file: Path to output JSON file (default: milvus_data.json)
        
    Returns:
        Message indicating success, file path, and number of documents inserted into Milvus
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        docs_list = json.loads(documents) if isinstance(documents, str) else documents
        
        # Step 1: Create JSON file
        file_path = agent.create_milvus_json(docs_list, output_file)
        logger.info(f"Created Milvus JSON file: {file_path}")
        
        # Step 2: Automatically feed into Milvus
        feed_result = agent.feed_milvus_from_json(file_path)
        logger.info(f"Fed Milvus JSON into collection: {feed_result}")
        
        return f"Successfully created Milvus JSON file: {file_path}\n{feed_result}"
    except Exception as e:
        logger.error(f"Error creating/feeding Milvus JSON file: {e}", exc_info=True)
        return f"Error creating/feeding Milvus JSON file: {str(e)}"


@function_tool
def feed_milvus_from_json_file(json_file: str) -> str:
    """Feed Milvus JSON file into Milvus collection (Knowledge Base).
    
    Args:
        json_file: Path to Milvus JSON file (e.g., "milvus_data.json")
        
    Returns:
        Message indicating success and number of documents inserted
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        result = agent.feed_milvus_from_json(json_file)
        return result
    except Exception as e:
        logger.error(f"Error feeding Milvus from JSON file: {e}", exc_info=True)
        return f"Error feeding Milvus from JSON file: {str(e)}"


@function_tool
def process_file_for_ingestion(
    file_path: str,
    chunk_size: int = 500,
    extract_requirements: bool = True,
) -> str:
    """Automatically process a file: parse, chunk, create embeddings, and ingest into Milvus and Neo4j.
    
    This tool automatically:
    1. Parses the file (supports .txt, .docx, .pdf)
    2. Chunks the text into smaller pieces
    3. Creates embeddings for Milvus
    4. Extracts requirements/user stories for Neo4j (if enabled)
    5. Creates JSON files with proper schemas
    6. Automatically feeds into Milvus (Knowledge Base) and Neo4j (Knowledge Graph)
    
    Args:
        file_path: Path to file to process (supports .txt, .docx, .pdf)
        chunk_size: Size of chunks for Milvus in characters (default: 500)
        extract_requirements: Whether to extract requirements/user stories for Neo4j (default: True)
        
    Returns:
        JSON string with processing results including file paths, counts, and status
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        
        result = agent.process_file_automatically(
            file_path=file_path,
            chunk_size=chunk_size,
            extract_requirements=extract_requirements,
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error processing file for ingestion: {e}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@function_tool
def process_text_for_ingestion(
    text: str,
    source: str = "user_input",
    chunk_size: int = 500,
    extract_requirements: bool = True,
) -> str:
    """Automatically process text: chunk, create embeddings, and ingest into Milvus and Neo4j.
    
    This tool automatically:
    1. Chunks the text into smaller pieces
    2. Creates embeddings for Milvus
    3. Extracts requirements/user stories for Neo4j (if enabled)
    4. Creates JSON files with proper schemas
    5. Automatically feeds into Milvus (Knowledge Base) and Neo4j (Knowledge Graph)
    
    Args:
        text: Text content to process
        source: Source identifier for the text (default: "user_input")
        chunk_size: Size of chunks for Milvus in characters (default: 500)
        extract_requirements: Whether to extract requirements/user stories for Neo4j (default: True)
        
    Returns:
        JSON string with processing results including file paths, counts, and status
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.ingest_agent import IngestAgent
        agent = IngestAgent()
        
        result = agent.process_text_automatically(
            text=text,
            source=source,
            chunk_size=chunk_size,
            extract_requirements=extract_requirements,
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error processing text for ingestion: {e}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


__all__ = [
    "create_milvus_json_file",
    "feed_milvus_from_json_file",
    "process_file_for_ingestion",
    "process_text_for_ingestion",
]
