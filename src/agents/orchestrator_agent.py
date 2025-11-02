"""
Orchestrator Agent for routing user requests to appropriate agents.

This orchestrator handles three main flows:
1. Ingestion Flow: When user requests ingestion or sends documents → calls ingest_agent
2. RAG Flow: When user makes normal queries → calls rag_agent
3. Business Analysis Flow: When user requests conflict detection and improvement suggestions → calls business_analysis_agent
"""

from __future__ import annotations

import logging
import json
from typing import Optional, Dict, Any

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
            Runner = getattr(_openai_agents, 'Runner', None)
        finally:
            # Restore our local agents module
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
    else:
        # Fallback: try direct import (this works if run from src directory)
        _temp_agents = sys.modules.pop('agents', None)
        try:
            from agents import Agent, ModelSettings, Runner
        except ImportError as e:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
            raise ImportError(f"Could not import Agent, ModelSettings, Runner from openai-agents package. Make sure openai-agents>=0.4.2 is installed: {e}")
        finally:
            if _temp_agents:
                sys.modules['agents'] = _temp_agents
except Exception as e:
    # Last resort: try importing directly (might work if package structure allows)
    _temp_agents = sys.modules.pop('agents', None)
    try:
        from agents import Agent, ModelSettings, Runner
    except ImportError:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents
        raise ImportError(f"Could not import from openai-agents package: {e}")
    finally:
        if _temp_agents:
            sys.modules['agents'] = _temp_agents

# Import agents
from .ingest_agent import create_ingest_agent
from .rag_agent import create_rag_agent, run_rag_query, RAGInput, RAGOutput
from .business_analysis_agent import create_business_analysis_agent
from openai import OpenAI

logger = logging.getLogger(__name__)


def create_orchestrator_agent(
    name: str = "Orchestrator Agent",
    model: str = "gpt-4o",
    temperature: float = 0.1,  # Low temperature for routing decisions
    max_tokens: int = 500,
) -> Agent:
    """
    Create an Orchestrator Agent that routes user requests to appropriate agents.
    
    The orchestrator handles three main flows:
    1. **Ingestion Flow**: Routes to Ingest Agent when user:
       - Requests to ingest documents
       - Uploads or sends documents for processing
       - Wants to create JSON files for Milvus or Neo4j
       - Keywords: "ingest", "upload", "process documents", "create json", "milvus", "neo4j"
    
    2. **RAG Flow**: Routes to RAG Agent when user:
       - Asks normal questions/queries
       - Wants information retrieval
       - Asks "what", "how", "which", "when", "where" questions
       - This is the DEFAULT flow for normal queries
    
    3. **Business Analysis Flow**: Routes to Business Analysis Agent when user:
       - Requests conflict/contradiction detection
       - Asks for requirement analysis
       - Wants improvement suggestions
       - Keywords: "analyze", "contradiction", "conflict", "improve", "business analysis", "requirements analysis"
    
    Args:
        name: Agent name identifier
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for routing decisions (default: 0.1 for consistency)
        max_tokens: Maximum tokens in response (default: 500)
        
    Returns:
        Agent instance configured for routing
        
    Example:
        from agents.orchestrator_agent import create_orchestrator_agent
        from agents import Runner
        
        orchestrator = create_orchestrator_agent()
        result = Runner.run_sync(starting_agent=orchestrator, input="What features does Project Alpha include?")
        print(result.final_output)
    """
    
    # Create specialized agents for handoff
    ingest_agent = create_ingest_agent(
        name="Ingest Agent",
        model=model,
        temperature=0.3,
        max_tokens=2000,
    )
    
    rag_agent = create_rag_agent(
        name="RAG Agent",
        model=model,
        temperature=0.7,
        max_tokens=2000,
    )
    
    # Business Analysis Agent doesn't use Agent pattern yet, so we'll handle it differently
    # For now, we'll route to it via instructions
    
    instructions = (
        "You are an Orchestrator Agent that routes user requests to the appropriate specialized agent.\n\n"
        "**ROUTING RULES:**\n\n"
        
        "1. **Route to Ingest Agent** when the user:\n"
        "   - Requests to ingest, upload, or process documents\n"
        "   - Asks to create JSON files for Milvus or Neo4j\n"
        "   - Wants to add documents to the knowledge base or knowledge graph\n"
        "   - Uses keywords like: 'ingest', 'upload', 'process documents', 'create json', 'add to milvus', 'add to neo4j'\n"
        "   - Action: Hand off to 'Ingest Agent'\n\n"
        
        "2. **Route to RAG Agent** (DEFAULT) when the user:\n"
        "   - Asks normal questions or queries\n"
        "   - Wants information retrieval from knowledge base/graph\n"
        "   - Asks questions starting with: 'what', 'how', 'which', 'when', 'where', 'who'\n"
        "   - Requests information about projects, requirements, or user stories\n"
        "   - This is the DEFAULT flow for normal queries\n"
        "   - Action: Hand off to 'RAG Agent'\n\n"
        
        "3. **Route to Business Analysis Agent** when the user:\n"
        "   - Requests conflict/contradiction detection\n"
        "   - Asks for requirement analysis or business analysis\n"
        "   - Wants improvement suggestions for requirements\n"
        "   - Provides multiple requirements or user stories for analysis\n"
        "   - Uses keywords like: 'analyze', 'contradiction', 'conflict', 'improve', 'business analysis', 'requirements analysis'\n"
        "   - Action: Respond with: 'BUSINESS_ANALYSIS:{user_input}' format\n\n"
        
        "**IMPORTANT:**\n"
        "- Always route to the appropriate agent. Do NOT answer the question yourself.\n"
        "- For normal questions, default to RAG Agent unless explicitly ingestion or analysis request.\n"
        "- Be decisive in routing - analyze the intent, not just keywords.\n"
        "- If uncertain, default to RAG Agent.\n\n"
        
        "**RESPONSE FORMAT:**\n"
        "- For ingestion: Hand off to Ingest Agent\n"
        "- For normal queries: Hand off to RAG Agent\n"
        "- For business analysis: Respond with 'BUSINESS_ANALYSIS:' followed by the user input\n"
    )
    
    # Create model settings
    model_settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Create orchestrator agent with handoffs to Ingest and RAG agents
    orchestrator = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        handoffs=[ingest_agent, rag_agent],  # Business Analysis will be handled via function
    )
    
    logger.info(f"Created Orchestrator Agent '{name}' with model {model}")
    return orchestrator


# Global OpenAI client for routing classification
_routing_client = None

def _get_routing_client() -> OpenAI:
    """Get or create OpenAI client for routing classification"""
    import os
    global _routing_client
    if _routing_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _routing_client = OpenAI(api_key=api_key)
        else:
            # Fallback: Initialize without explicit API key (will use default)
            _routing_client = OpenAI()
    return _routing_client


def classify_user_intent_with_llm(user_input: str) -> str:
    """
    Use LLM to analyze user input and classify it into one of four flows:
    - "ingestion": User wants to ingest/upload/process documents
    - "business_analysis": User wants to analyze requirements, detect conflicts, or get improvements
    - "web_generation": User wants to create/generate a website or web application
    - "rag": Normal query/question (default)
    
    Args:
        user_input: User's input text or query
        
    Returns:
        "ingestion", "business_analysis", "web_generation", or "rag"
    """
    client = _get_routing_client()
    
    classification_prompt = f"""Analyze the following user input and classify it into ONE of these four categories based on the user's INTENT and MEANING (not just keywords):

1. **ingestion**: User wants to:
   - Ingest, upload, or process documents
   - Create JSON files for Milvus or Neo4j
   - Add documents to knowledge base or knowledge graph
   - Store or index documents/data

2. **business_analysis**: User wants to:
   - Analyze requirements for conflicts or contradictions
   - Get business analysis or requirements analysis
   - Improve or suggest improvements for requirements
   - Compare multiple requirements/user stories
   - Detect issues in requirements

3. **web_generation**: User wants to:
   - Create, generate, or build a website or web application
   - Make a website, build a site, create a web app
   - Generate HTML/CSS/JS code
   - Design or develop a website

4. **rag**: User wants to:
   - Ask questions or queries
   - Retrieve information
   - Get information about projects, requirements, or user stories
   - Normal conversational queries

User Input: "{user_input}"

Respond with ONLY the category name (one of: "ingestion", "business_analysis", "web_generation", or "rag").
Be decisive and analyze the semantic meaning, not just keywords."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use smaller model for faster routing
            messages=[
                {"role": "system", "content": "You are a routing classifier. Analyze user intent and respond with ONLY the category name."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent routing
            max_tokens=50,  # Just need the category name
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate and normalize response
        valid_flows = ["ingestion", "business_analysis", "web_generation", "rag"]
        if classification not in valid_flows:
            logger.warning(f"LLM returned invalid classification '{classification}', defaulting to 'rag'")
            return "rag"
        
        logger.info(f"LLM classified user input as: {classification}")
        return classification
        
    except Exception as e:
        logger.error(f"Error in LLM classification: {e}, defaulting to 'rag'")
        return "rag"


def route_request(user_input: str, orchestrator: Optional[Agent] = None) -> Dict[str, Any]:
    """
    Route user request to appropriate agent and return result.
    Uses LLM to analyze semantic meaning of user input for routing.
    
    Args:
        user_input: User's input text or query
        orchestrator: Optional orchestrator agent instance (creates new one if None)
        
    Returns:
        Dictionary with:
        - "flow": "ingestion" | "rag" | "business_analysis"
        - "output": Agent output or result
        - "agent": Agent name used
    """
    # Use LLM to classify user intent based on semantic meaning
    flow = classify_user_intent_with_llm(user_input)
    logger.info(f"Routing user input to '{flow}' flow based on LLM classification")
    
    if flow == "business_analysis":
        # Route to Business Analysis Agent
        logger.info(f"Routing to Business Analysis Agent")
        
        # Try to parse as JSON if possible, otherwise use as single text
        try:
            user_input_dict = json.loads(user_input)
            if isinstance(user_input_dict, dict):
                # Already in expected format
                ba_input_json = user_input
            else:
                # Convert to expected format
                ba_input = {"user_text_1": user_input}
                ba_input_json = json.dumps(ba_input)
        except json.JSONDecodeError:
            # Not JSON, treat as single requirement text
            ba_input = {"user_text_1": user_input}
            ba_input_json = json.dumps(ba_input)
        
        # Use Agent pattern
        ba_agent = create_business_analysis_agent()
        
        # Build prompt for business analysis
        # The Agent will parse this and call analyze_requirements tool
        ba_prompt = f"Analyze these requirements: user_text_1='{user_input}'"
        
        try:
            # Use Runner to execute the agent
            result = Runner.run_sync(starting_agent=ba_agent, input=ba_prompt)
            ba_result = result.final_output
            
            return {
                "flow": "business_analysis",
                "output": ba_result,
                "agent": "BusinessAnalysisAgent"
            }
        except Exception as e:
            logger.error(f"Error in Business Analysis Agent: {e}", exc_info=True)
            return {
                "flow": "business_analysis",
                "output": f"Error in Business Analysis Agent: {str(e)}",
                "agent": "BusinessAnalysisAgent",
                "error": str(e)
            }
    
    elif flow == "ingestion":
        # Route to Ingest Agent via orchestrator handoff
        logger.info(f"Routing to Ingest Agent")
        if orchestrator is None:
            orchestrator = create_orchestrator_agent()
        
        # Use Runner to handoff to Ingest Agent
        try:
            result = Runner.run_sync(starting_agent=orchestrator, input=user_input)
            final_output = result.final_output
            
            return {
                "flow": "ingestion",
                "output": final_output,  # Ingest agent output
                "agent": "IngestAgent"
            }
        except Exception as e:
            logger.error(f"Error in Ingest Agent: {e}", exc_info=True)
            return {
                "flow": "ingestion",
                "output": f"Error in Ingest Agent: {str(e)}",
                "agent": "IngestAgent",
                "error": str(e)
            }
    
    elif flow == "web_generation":
        # Route to Web Generation Flow
        logger.info(f"Routing to Web Generation Flow")
        return {
            "flow": "web_generation",
            "output": user_input,  # Pass user input to app.py for processing
            "agent": "WebGeneratorAgent"
        }
    
    else:  # flow == "rag" (default)
        # Default to RAG Agent (for normal queries)
        logger.info(f"Routing to RAG Agent")
        try:
            rag_output = run_rag_query(user_input)
            return {
                "flow": "rag",
                "output": rag_output.model_dump_json(indent=2) if isinstance(rag_output, RAGOutput) else str(rag_output),
                "agent": "RAGAgent",
                "rag_output": rag_output  # Include Pydantic model if available
            }
        except Exception as e:
            logger.error(f"Error in RAG Agent: {e}", exc_info=True)
            return {
                "flow": "rag",
                "output": f"Error in RAG Agent: {str(e)}",
                "agent": "RAGAgent",
                "error": str(e)
            }


def run_orchestrator(user_input: str) -> str:
    """
    Main function to run orchestrator and get output as JSON string.
    
    Args:
        user_input: User's input text or query
        
    Returns:
        JSON string with flow, output, and agent information
    """
    try:
        result = route_request(user_input)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error in orchestrator: {e}", exc_info=True)
        return json.dumps({
            "flow": "error",
            "output": f"Error: {str(e)}",
            "agent": "OrchestratorAgent",
            "error": str(e)
        }, ensure_ascii=False, indent=2)


__all__ = [
    "create_orchestrator_agent",
    "route_request",
    "run_orchestrator",
    "Agent",
    "ModelSettings",
    "Runner",
]
