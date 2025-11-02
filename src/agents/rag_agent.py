"""
RAG Agent that retrieves information from Milvus (KB) and Neo4j (KG).

This module provides both a traditional RAGAgent class and a create_rag_agent()
function that returns an OpenAI Agents Agent instance following the standard pattern.

Basic configuration using OpenAI Agents:
from agents.rag_agent import create_rag_agent, Runner

agent = create_rag_agent(
    name="RAG Agent",
    model="gpt-4o",
    temperature=0.5,
    max_tokens=1000,
)

result = Runner.run_sync(starting_agent=agent, input="What features does Project Alpha include?")
print(result.final_output)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import from openai-agents package
# Note: openai-agents package exposes 'Agent', 'ModelSettings', 'function_tool', 'Runner' at top-level 'agents' module
# Since we have a local 'agents' module, we need to import from the installed package before it loads
import sys
import importlib.util
import importlib.metadata

# Find the openai-agents package location and import from it directly
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
        import os
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
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field
from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException

from core.config import load_config
from db.milvus_client import (
    DOC_COLLECTION_NAME,
    connect_to_milvus,
    ensure_gsoft_docs_collection,
)
# Import local agents modules using relative import (since we're in the agents package)
from .ingest_agent import IngestAgent
from tools.milvus_search_tool import MilvusSearchTool, MilvusSearchInput
from tools.neo4j_search_tool import Neo4jSearchTool, Neo4jSearchInput
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
config = load_config()

# Neo4j queries for Project/Requirement/UserStory schema
SEARCH_PROJECTS_QUERY = """
MATCH (p:Project)
WHERE $term = '' OR 
      toLower(coalesce(p.name, '')) CONTAINS $term OR
      toLower(coalesce(p.description, '')) CONTAINS $term OR
      toLower(coalesce(p.project_id, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.description, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.project_id, '')) CONTAINS keyword)
RETURN id(p) AS node_id, labels(p) AS labels, p.project_id AS identifier,
       p.name AS name, p.description AS description,
       p.status AS status, p.version AS version,
       p.created_date AS created_date, p.updated_date AS updated_date,
       p.stakeholders AS stakeholders
ORDER BY 
  CASE WHEN toLower(coalesce(p.name, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) THEN 2
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.project_id, '')) CONTAINS keyword) THEN 3
       ELSE 4 END,
  p.created_date DESC
LIMIT $limit
"""

SEARCH_REQUIREMENTS_QUERY = """
MATCH (r:Requirement)
WHERE $term = '' OR 
      toLower(coalesce(r.title, '')) CONTAINS $term OR
      toLower(coalesce(r.description, '')) CONTAINS $term OR
      toLower(coalesce(r.req_id, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.title, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.description, '')) CONTAINS keyword)
RETURN id(r) AS node_id, labels(r) AS labels, r.req_id AS identifier,
       r.title AS title, r.description AS description,
       r.type AS type, r.priority AS priority, r.status AS status,
       r.version AS version, r.source AS source,
       r.acceptance_criteria AS acceptance_criteria,
       r.constraints AS constraints, r.assumptions AS assumptions,
       r.created_date AS created_date, r.updated_date AS updated_date
ORDER BY 
  CASE 
    WHEN r.priority = 'critical' THEN 1
    WHEN r.priority = 'high' THEN 2
    WHEN r.priority = 'medium' THEN 3
    ELSE 4
  END,
  CASE WHEN toLower(coalesce(r.title, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.title, '')) CONTAINS keyword) THEN 2
       ELSE 3 END
LIMIT $limit
"""

SEARCH_USER_STORIES_QUERY = """
MATCH (us:UserStory)
WHERE $term = '' OR 
      toLower(coalesce(us.title, '')) CONTAINS $term OR
      toLower(coalesce(us.description, '')) CONTAINS $term OR
      toLower(coalesce(us.story_id, '')) CONTAINS $term OR
      toLower(coalesce(us.i_want, '')) CONTAINS $term OR
      toLower(coalesce(us.as_a, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.title, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.description, '')) CONTAINS keyword)
RETURN id(us) AS node_id, labels(us) AS labels, us.story_id AS identifier,
       us.title AS title, 
       us.as_a AS as_a,
       us.i_want AS i_want,
       us.so_that AS so_that,
       us.description AS description,
       us.status AS status, us.priority AS priority,
       us.story_points AS story_points, us.sprint AS sprint,
       us.acceptance_criteria AS acceptance_criteria,
       us.created_date AS created_date, us.updated_date AS updated_date
ORDER BY 
  CASE 
    WHEN us.priority = 'critical' THEN 1
    WHEN us.priority = 'high' THEN 2
    WHEN us.priority = 'medium' THEN 3
    ELSE 4
  END,
  CASE WHEN toLower(coalesce(us.title, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.title, '')) CONTAINS keyword) THEN 2
       ELSE 3 END
LIMIT $limit
"""

GET_NODE_CONTEXT_QUERY = """
MATCH (n) WHERE id(n) = $node_id
OPTIONAL MATCH (n)-[r]->(m)
RETURN 'outgoing' AS direction, type(r) AS rel_type, id(m) AS target_id, null AS source_id,
       labels(m) AS target_labels, null AS source_labels,
       CASE 
         WHEN 'Project' IN labels(m) THEN m.project_id
         WHEN 'Requirement' IN labels(m) THEN m.req_id
         WHEN 'UserStory' IN labels(m) THEN m.story_id
         ELSE null
       END AS target_identifier, null AS source_identifier,
       properties(m) AS target_props, null AS source_props,
       properties(r) AS rel_props
UNION ALL
MATCH (m)-[r]->(n) WHERE id(n) = $node_id
RETURN 'incoming' AS direction, type(r) AS rel_type, null AS target_id, id(m) AS source_id,
       null AS target_labels, labels(m) AS source_labels,
       null AS target_identifier,
       CASE 
         WHEN 'Project' IN labels(m) THEN m.project_id
         WHEN 'Requirement' IN labels(m) THEN m.req_id
         WHEN 'UserStory' IN labels(m) THEN m.story_id
         ELSE null
       END AS source_identifier,
       null AS target_props, properties(m) AS source_props,
       properties(r) AS rel_props
LIMIT $limit
"""


@dataclass
class KBReference:
    """Knowledge Base reference from Milvus."""

    id: str
    original_doc_id: Optional[str]
    text_preview: Optional[str]
    source: Optional[str]
    url: Optional[str]
    score: float


@dataclass
class KGReference:
    """Knowledge Graph reference from Neo4j."""

    node_id: int
    labels: List[str]
    identifier: Optional[str]
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]


@dataclass
class RAGResponse:
    """RAG response with text and references (legacy format)."""

    text: str
    references: Dict[str, List[Any]]  # {"KB": List[KBReference], "KG": List[KGReference]}


# Pydantic Models for Input/Output
class RAGInput(BaseModel):
    """Input model for RAG Agent query.
    
    Format:
    {
        "user_text": "Your question here"
    }
    """
    
    user_text: str = Field(..., description="User's question or query text")


class RAGOutput(BaseModel):
    """Output model for RAG Agent response.
    
    Format:
    {
        "answer_text": "Generated answer text",
        "KB": [
            {"id": "doc_001", "score": 0.95},
            {"id": "doc_002", "score": 0.87},
            ...
        ],  # Document IDs/names from Milvus with relevance scores
        "KG": [
            {"id": "PROJ-001", "score": 0.92},
            {"id": "REQ-002", "score": 0.85},
            ...
        ]  # Node identifiers from Neo4j with relevance scores
    }
    """
    
    answer_text: str = Field(..., description="Generated answer text from LLM")
    KB: List[Dict[str, Any]] = Field(default_factory=list, description="Array of document references with id and score from Knowledge Base (Milvus)")
    KG: List[Dict[str, Any]] = Field(default_factory=list, description="Array of node references with id and score from Knowledge Graph (Neo4j)")


# Global tool instances (initialized on first use)
_milvus_tool: Optional[MilvusSearchTool] = None
_neo4j_tool: Optional[Neo4jSearchTool] = None


def _get_milvus_tool() -> MilvusSearchTool:
    """Get or create Milvus search tool instance."""
    global _milvus_tool
    if _milvus_tool is None:
        _milvus_tool = MilvusSearchTool()
    return _milvus_tool


def _get_neo4j_tool() -> Neo4jSearchTool:
    """Get or create Neo4j search tool instance."""
    global _neo4j_tool
    if _neo4j_tool is None:
        _neo4j_tool = Neo4jSearchTool()
    return _neo4j_tool


@function_tool
def search_milvus_kb(query: str, top_k: int = 15) -> str:
    """Search the Knowledge Base (Milvus) for relevant document chunks based on semantic similarity.
    
    Args:
        query: The search query or question text
        top_k: Number of top results to return (default: 5, max: 50)
        
    Returns:
        A formatted string containing document IDs and summaries of the search results
    """
    try:
        milvus_tool = _get_milvus_tool()
        input_data = MilvusSearchInput(query=query, top_k=top_k)
        output = milvus_tool.search(input_data)
        
        if not output.doc_ids:
            return "No relevant documents found in Knowledge Base."
        
        result_parts = [f"Found {len(output.doc_ids)} relevant documents:"]
        for i, doc_id in enumerate(output.doc_ids, 1):
            result_parts.append(f"{i}. {doc_id}")
        
        # Add summaries if available
        if output.results:
            result_parts.append("\nDocument summaries:")
            for i, result in enumerate(output.results[:5], 1):  # Limit to top 5 summaries
                preview = result.get("text_preview", "")[:200]  # Truncate to 200 chars
                if preview:
                    result_parts.append(f"{i}. {doc_id}: {preview}...")
        
        return "\n".join(result_parts)
    except Exception as e:
        logger.error(f"Error in search_milvus_kb tool: {e}", exc_info=True)
        return f"Error searching Knowledge Base: {str(e)}"


@function_tool
def search_neo4j_kg(query: str, top_k: int = 15) -> str:
    """Search the Knowledge Graph (Neo4j) for relevant Project, Requirement, and UserStory nodes.
    
    Args:
        query: The search query or question text
        top_k: Number of top results to return (default: 5, max: 50)
        
    Returns:
        A formatted string containing node identifiers and summaries of the search results
    """
    try:
        neo4j_tool = _get_neo4j_tool()
        input_data = Neo4jSearchInput(query=query, top_k=top_k)
        output = neo4j_tool.search(input_data)
        
        if not output.identifiers:
            return "No relevant nodes found in Knowledge Graph."
        
        result_parts = [f"Found {len(output.identifiers)} relevant nodes:"]
        for i, identifier in enumerate(output.identifiers, 1):
            result_parts.append(f"{i}. {identifier}")
        
        # Add node details if available with ALL properties
        if output.results:
            result_parts.append("\nNode details:")
            for i, result in enumerate(output.results[:5], 1):  # Limit to top 5 details
                identifier = result.get("identifier", "Unknown")
                labels = result.get("labels", [])
                props = result.get("properties", {})
                
                # Extract key info
                name = props.get("name") or props.get("title", "N/A")
                description = props.get("description", "")[:150]  # Truncate to 150 chars
                
                result_parts.append(f"{i}. {identifier} ({', '.join(labels)}): {name}")
                if description:
                    result_parts.append(f"   Description: {description}...")
                
                # Include ALL important properties for LLM context
                if "Project" in labels:
                    if props.get("version"):
                        result_parts.append(f"   Version: {props.get('version')}")
                    if props.get("status"):
                        result_parts.append(f"   Status: {props.get('status')}")
                    if props.get("created_date"):
                        result_parts.append(f"   Created Date: {props.get('created_date')}")
                    if props.get("updated_date"):
                        result_parts.append(f"   Updated Date: {props.get('updated_date')}")
                    if props.get("stakeholders"):
                        stakeholders = props.get("stakeholders", [])
                        if isinstance(stakeholders, list) and len(stakeholders) > 0:
                            result_parts.append(f"   Stakeholders: {', '.join(str(s) for s in stakeholders)}")
                
                elif "Requirement" in labels:
                    if props.get("type"):
                        result_parts.append(f"   Type: {props.get('type')}")
                    if props.get("priority"):
                        result_parts.append(f"   Priority: {props.get('priority')}")
                    if props.get("status"):
                        result_parts.append(f"   Status: {props.get('status')}")
                    if props.get("version"):
                        result_parts.append(f"   Version: {props.get('version')}")
                    if props.get("created_date"):
                        result_parts.append(f"   Created Date: {props.get('created_date')}")
                    if props.get("updated_date"):
                        result_parts.append(f"   Updated Date: {props.get('updated_date')}")
                    if props.get("acceptance_criteria"):
                        criteria = props.get("acceptance_criteria", [])
                        if isinstance(criteria, list) and len(criteria) > 0:
                            result_parts.append(f"   Acceptance Criteria: {', '.join(str(c) for c in criteria[:3])}")  # Limit to 3
                    if props.get("constraints"):
                        constraints = props.get("constraints", [])
                        if isinstance(constraints, list) and len(constraints) > 0:
                            result_parts.append(f"   Constraints: {', '.join(str(c) for c in constraints[:3])}")  # Limit to 3
                
                elif "UserStory" in labels:
                    if props.get("as_a"):
                        result_parts.append(f"   As a: {props.get('as_a')}")
                    if props.get("i_want"):
                        result_parts.append(f"   I want: {props.get('i_want')}")
                    if props.get("so_that"):
                        result_parts.append(f"   So that: {props.get('so_that')}")
                    if props.get("status"):
                        result_parts.append(f"   Status: {props.get('status')}")
                    if props.get("priority"):
                        result_parts.append(f"   Priority: {props.get('priority')}")
                    if props.get("story_points"):
                        result_parts.append(f"   Story Points: {props.get('story_points')}")
                    if props.get("sprint"):
                        result_parts.append(f"   Sprint: {props.get('sprint')}")
                    if props.get("created_date"):
                        result_parts.append(f"   Created Date: {props.get('created_date')}")
                    if props.get("updated_date"):
                        result_parts.append(f"   Updated Date: {props.get('updated_date')}")
                    if props.get("acceptance_criteria"):
                        criteria = props.get("acceptance_criteria", [])
                        if isinstance(criteria, list) and len(criteria) > 0:
                            result_parts.append(f"   Acceptance Criteria: {', '.join(str(c) for c in criteria[:3])}")  # Limit to 3
                
                # Include relationships if available
                relationships = result.get("relationships", [])
                if relationships:
                    result_parts.append(f"   Relationships:")
                    for rel in relationships[:5]:  # Limit to 5 relationships per node
                        direction = rel.get("direction", "N/A")
                        rel_type = rel.get("type", "N/A")
                        if direction == "outgoing":
                            target_id = rel.get("target_identifier") or rel.get("target_id")
                            target_labels = rel.get("target_labels", [])
                            result_parts.append(f"      - {identifier} --[{rel_type}]--> {target_id} ({', '.join(target_labels)})")
                        elif direction == "incoming":
                            source_id = rel.get("source_identifier") or rel.get("source_id")
                            source_labels = rel.get("source_labels", [])
                            result_parts.append(f"      - {source_id} ({', '.join(source_labels)}) --[{rel_type}]--> {identifier}")
                
                result_parts.append("")  # Empty line between nodes
        
        return "\n".join(result_parts)
    except Exception as e:
        logger.error(f"Error in search_neo4j_kg tool: {e}", exc_info=True)
        return f"Error searching Knowledge Graph: {str(e)}"


def create_rag_agent(
    name: str = "RAG Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.5,
    max_tokens: int = 10000,
    top_p: Optional[float] = None,
) -> Agent:
    """Create a RAG Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for randomness (default: 0.5)
        max_tokens: Maximum tokens in response (default: 10000)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for RAG with Milvus and Neo4j search tools
        
    Example:
        from agents import Runner
        
        agent = create_rag_agent()
        result = await Runner.run(agent, "What features does Project Alpha include?")
        print(result.final_output)
    """
    if instructions is None:
        instructions = (
        "You are an assistant that answers questions based on a Knowledge Base (KB) and a Knowledge Graph (KG).\n\n"
        "CRITICAL: BEFORE answering any question, you MUST:\n"
        "1. ALWAYS call search_milvus_kb(query) to search the Knowledge Base\n"
        "2. ALWAYS call search_neo4j_kg(query) to search the Knowledge Graph\n"
        "3. THEN use the retrieved information to answer the question\n\n"
        "Do NOT attempt to answer without first calling both search tools.\n\n"
        "PRIORITY RULES:\n"
        "- KB chunks are sorted by relevance (Rank #1 = most relevant)\n"
        "- Always prioritize information from chunks with higher ranking (smaller rank number = more relevant)\n\n"
        "WHEN ANSWERING:\n\n"
        "1. Introduction (1-2 sentences): State whether relevant information was found in the KB/KG.\n\n"
        "2. Main Content: Answer the question using information from the KB/KG.\n"
        " - Use bullet points and headings for readability\n"
        " - Clearly identify any missing information fields\n"
        " - ONLY use existing information, DO NOT fabricate details\n"
        " - If multiple chunks contain information about the same topic, ALWAYS use the information from the higher-ranked chunk (lower rank number = more relevant)\n\n"
        "3. Detailed Information (if available):\n"
        " - Projects: name, description, version, status, created_date, updated_date, stakeholders\n"
        " - Requirements: title, description, type, priority, status, version, created_date, updated_date, acceptance_criteria, constraints\n"
        " - User Stories: title, as_a, i_want, so_that, description, status, priority, story_points, sprint, acceptance_criteria\n\n"
        "4. Relationships (if found in the KG):\n"
        " - Format: NodeA --[relation]--> NodeB\n"
        " - Include key attributes\n\n"
        "5. Missing Data: List any missing information required for a complete answer\n\n"
        "6. Conclusion (1 sentence): Assess whether the answer is complete or if additional data is needed\n\n"
        "NOTES:\n"
        "- Do not fabricate any information\n"
        "- If data is missing, clearly state 'No information available'\n"
        "- You MUST call search_milvus_kb and search_neo4j_kg BEFORE answering\n"
        "- Always cite the documents and nodes used from the search results"
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
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[search_milvus_kb, search_neo4j_kg],
    )
    
    logger.info(f"Created RAG Agent '{name}' with model {model}")
    return agent


def run_rag_query(
    query: str,
    agent: Optional[Agent] = None,
    top_k: int = 15,
) -> RAGOutput:
    """Run RAG query and return structured Pydantic output.
    
    This function runs the agent with Runner, then extracts KB/KG references
    from tool calls to create a structured RAGOutput.
    
    Args:
        query: User query string
        agent: Agent instance (optional, creates new one if None)
        top_k: Number of top results for KB and KG search (default: 5)
        
    Returns:
        RAGOutput with answer_text, KB (list of doc IDs), and KG (list of node identifiers)
        
    Example:
        from agents.rag_agent import run_rag_query
        
        output = run_rag_query("What features does Project Alpha include?")
        print(output.model_dump_json(indent=2))
    """
    # Create agent if not provided
    if agent is None:
        agent = create_rag_agent()
    
    # Run agent with Runner
    runner_result = Runner.run_sync(starting_agent=agent, input=query)
    answer_text = runner_result.final_output
    
    # Extract KB and KG references by calling tools directly
    kb_refs = []
    kg_refs = []
    
    try:
        # Search Milvus KB
        milvus_tool = _get_milvus_tool()
        milvus_input = MilvusSearchInput(query=query, top_k=top_k)
        milvus_output = milvus_tool.search(milvus_input)
        
        # Extract KB references with scores
        for result in milvus_output.results:
            kb_refs.append({
                "id": result.get("id", ""),
                "score": result.get("score", 0.0)
            })
    except Exception as e:
        logger.warning(f"Error extracting KB references: {e}")
    
    try:
        # Search Neo4j KG
        neo4j_tool = _get_neo4j_tool()
        neo4j_input = Neo4jSearchInput(query=query, top_k=top_k)
        logger.info(f"Searching Neo4j for query: '{query}'")
        neo4j_output = neo4j_tool.search(neo4j_input)
        
        # Extract KG references with scores (if available)
        # Note: Neo4j may not have scores, so we use identifier as id
        for result in neo4j_output.results:
            kg_refs.append({
                "id": result.get("identifier") or result.get("node_id", ""),
                "score": result.get("score", 0.0)  # Default to 0.0 if no score
            })
        
        logger.info(f"Neo4j search returned {len(kg_refs)} identifiers")
        if not kg_refs:
            logger.warning(f"No Neo4j identifiers found for query: '{query}'")
    except Exception as e:
        logger.error(f"Error extracting KG references: {e}", exc_info=True)
    
    # Create and return RAGOutput
    return RAGOutput(
        answer_text=answer_text,
        KB=kb_refs,
        KG=kg_refs,
    )



__all__ = [
    "create_rag_agent",
    "run_rag_query",
    "search_milvus_kb",
    "search_neo4j_kg",
    "Agent",
    "ModelSettings",
    "function_tool",
    "Runner",
    "RAGAgent", 
    "RAGResponse", 
    "KBReference", 
    "KGReference",
    "RAGInput",
    "RAGOutput",
]
