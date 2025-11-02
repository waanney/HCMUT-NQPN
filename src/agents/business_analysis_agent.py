"""
Business Analysis Agent for Requirements Engineering
Analyzes SRS/User Stories, detects contradictions, and suggests improvements
Uses Milvus (Knowledge Base) and Neo4j (Knowledge Graph) for enhanced analysis
Uses OpenAI SDK Agent and Runner for content analysis
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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

from openai import OpenAI

# Import tools
from tools.milvus_search_tool import MilvusSearchTool, MilvusSearchInput
from tools.neo4j_search_tool import Neo4jSearchTool, Neo4jSearchInput
from tools.extract_concepts_tool import ExtractConceptsTool, ExtractConceptsInput
from tools.context_builder_tool import ContextBuilderTool, BuildKBContextInput, BuildKGContextInput

logger = logging.getLogger(__name__)

# Global tool instances (initialized on first use)
_milvus_tool_ba: Optional[MilvusSearchTool] = None
_neo4j_tool_ba: Optional[Neo4jSearchTool] = None


def _get_milvus_tool_ba() -> MilvusSearchTool:
    """Get or create Milvus search tool instance for Business Analysis Agent."""
    global _milvus_tool_ba
    if _milvus_tool_ba is None:
        _milvus_tool_ba = MilvusSearchTool()
    return _milvus_tool_ba


def _get_neo4j_tool_ba() -> Neo4jSearchTool:
    """Get or create Neo4j search tool instance for Business Analysis Agent."""
    global _neo4j_tool_ba
    if _neo4j_tool_ba is None:
        _neo4j_tool_ba = Neo4jSearchTool()
    return _neo4j_tool_ba


@function_tool
def search_milvus_kb_ba(query: str, top_k: int = 15) -> str:
    """Search the Knowledge Base (Milvus) for relevant document chunks based on semantic similarity.
    
    This tool is used by Business Analysis Agent to find similar requirements and documents
    that can help in analyzing conflicts, contradictions, and suggesting improvements.
    
    Args:
        query: The search query or question text
        top_k: Number of top results to return (default: 15, max: 50)
        
    Returns:
        A formatted string containing document IDs and summaries of the search results
    """
    try:
        milvus_tool = _get_milvus_tool_ba()
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
                doc_id = result.get("id", "Unknown")
                preview = result.get("text_preview", "")[:200]  # Truncate to 200 chars
                if preview:
                    result_parts.append(f"{i}. {doc_id}: {preview}...")
        
        return "\n".join(result_parts)
    except Exception as e:
        logger.error(f"Error in search_milvus_kb_ba tool: {e}", exc_info=True)
        return f"Error searching Knowledge Base: {str(e)}"


@function_tool
def search_neo4j_kg_ba(query: str, top_k: int = 15) -> str:
    """Search the Knowledge Graph (Neo4j) for relevant Project, Requirement, UserStory, Meeting, Participant, and Event nodes.
    
    This tool is used by Business Analysis Agent to find related requirements, projects, and user stories
    that can help in analyzing conflicts, contradictions, and suggesting improvements.
    
    Args:
        query: The search query or question text
        top_k: Number of top results to return (default: 15, max: 50)
        
    Returns:
        A formatted string containing node identifiers and summaries of the search results
    """
    try:
        neo4j_tool = _get_neo4j_tool_ba()
        input_data = Neo4jSearchInput(query=query, top_k=top_k)
        output = neo4j_tool.search(input_data)
        
        if not output.identifiers:
            return "No relevant nodes found in Knowledge Graph."
        
        result_parts = [f"Found {len(output.identifiers)} relevant nodes:"]
        for i, identifier in enumerate(output.identifiers, 1):
            result_parts.append(f"{i}. {identifier}")
        
        # Add node details if available
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
        
        return "\n".join(result_parts)
    except Exception as e:
        logger.error(f"Error in search_neo4j_kg_ba tool: {e}", exc_info=True)
        return f"Error searching Knowledge Graph: {str(e)}"


@dataclass
class BAAnalysisResult:
    """Structure for Business Analysis results"""
    answer: str
    kb_sources: List[Dict[str, str]]  # Knowledge Base sources from Milvus
    kg_sources: List[Dict[str, str]]  # Knowledge Graph sources from Neo4j


class BusinessAnalysisAgent:
    """
    Business Analysis Agent for Requirements Engineering
    Integrates with Milvus KB and Neo4j KG for comprehensive analysis
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Initialize tools
        try:
            self.milvus_tool = MilvusSearchTool()
        except Exception as e:
            logger.warning(f"Could not initialize MilvusSearchTool: {str(e)}")
            self.milvus_tool = None
        
        try:
            self.neo4j_tool = Neo4jSearchTool()
        except Exception as e:
            logger.warning(f"Could not initialize Neo4jSearchTool: {str(e)}")
            self.neo4j_tool = None
        
        # Initialize concept extraction tool
        self.extract_concepts_tool = ExtractConceptsTool(llm_client=self.openai_client)
        
        # Initialize context builder tool
        self.context_builder_tool = ContextBuilderTool()
        
        logger.info("Business Analysis Agent initialized successfully")

    def analyze_requirements(self, user_input: Dict[str, str]) -> Dict[str, Any]:
        """
        Main analysis function that processes user requirements
        
        Args:
            user_input: Dictionary with user_text_1, user_text_2, user_text_3 format
            
        Returns:
            JSON format with answer, KB sources, and KG sources
        """
        try:
            # Step 1: Combine and preprocess user texts
            combined_text = self._combine_user_texts(user_input)
            
            # Step 2: Query Knowledge Base (Milvus) for similar requirements
            kb_results, kb_doc_names = self._query_knowledge_base(combined_text)
            
            # Step 3: Query Knowledge Graph (Neo4j) for related patterns
            kg_results, kg_identifiers = self._query_knowledge_graph(combined_text)
            
            # Step 4: Perform comprehensive BA analysis
            analysis_result = self._perform_ba_analysis(
                combined_text, kb_results, kg_results
            )
            
            # Step 5: Format output according to specification for suggestion_agent
            return self._format_output(analysis_result, kb_doc_names, kg_identifiers)
            
        except Exception as e:
            logger.error(f"Error in requirements analysis: {str(e)}")
            return self._create_error_response(str(e))

    def _combine_user_texts(self, user_input: Dict[str, str]) -> str:
        """Combine multiple user text inputs into coherent analysis text"""
        texts = []
        for key in sorted(user_input.keys()):
            if key.startswith('user_text_') and user_input[key].strip():
                texts.append(f"Requirement {key}: {user_input[key].strip()}")
        
        return "\n\n".join(texts)

    def _query_knowledge_base(self, text: str, top_k: int = 5) -> tuple[List[Dict[str, Any]], List[str]]:
        """Query Milvus Knowledge Base for similar requirements using MilvusSearchTool
        
        Returns:
            Tuple of (detailed_results, doc_names) where:
            - detailed_results: List of dicts with full result info (for context building)
            - doc_names: List of doc_name strings (for output format)
        """
        if not self.milvus_tool:
            logger.warning("MilvusSearchTool not available, skipping KB query")
            return [], []
            
        try:
            # Use MilvusSearchTool
            milvus_input = MilvusSearchInput(query=text, top_k=top_k)
            milvus_output = self.milvus_tool.search(milvus_input)
            
            # Convert tool output to format expected by agent
            kb_sources = []
            doc_names = []
            
            for result in milvus_output.results:
                doc_name = result.get("original_doc_id", "unknown")
                kb_sources.append({
                    "doc_name": doc_name,
                    "source": result.get("source", ""),
                    "text_preview": result.get("text_preview", ""),
                    "similarity_score": 1.0 - result.get("score", 0.0),  # Convert distance to similarity
                    "url": result.get("url", "")
                })
                doc_names.append(doc_name)
            
            # Also use doc_ids from tool output directly
            if milvus_output.doc_ids:
                doc_names.extend([doc_id for doc_id in milvus_output.doc_ids if doc_id not in doc_names])
            
            return kb_sources, doc_names
            
        except Exception as e:
            logger.warning(f"Error querying Knowledge Base: {str(e)}")
            return [], []

    def _query_knowledge_graph(self, text: str) -> tuple[List[Dict[str, Any]], List[str]]:
        """Query Neo4j Knowledge Graph for related requirements patterns using Neo4jSearchTool
        
        Returns:
            Tuple of (detailed_results, identifiers) where:
            - detailed_results: List of dicts with full result info (for context building)
            - identifiers: List of node identifier strings (for output format)
        """
        if not self.neo4j_tool:
            logger.warning("Neo4jSearchTool not available, skipping KG query")
            return [], []
            
        try:
            # Use Neo4jSearchTool
            neo4j_input = Neo4jSearchInput(query=text, top_k=5)
            neo4j_output = self.neo4j_tool.search(neo4j_input)
            
            # Convert tool output to format expected by agent
            kg_sources = []
            identifiers = []
            
            for result in neo4j_output.results:
                identifier = result.get("identifier", "unknown")
                labels = result.get("labels", [])
                properties = result.get("properties", {})
                relationships = result.get("relationships", [])
                
                # Add identifier to list
                if identifier and identifier not in identifiers:
                    identifiers.append(identifier)
                
                # Create entry for each relationship
                for rel in relationships[:3]:  # Limit to 3 relationships per node
                    rel_type = rel.get("type", "RELATED_TO")
                    direction = rel.get("direction", "outgoing")
                    
                    if direction == "outgoing":
                        target_id = rel.get("target_identifier") or rel.get("target_id")
                        target_labels = rel.get("target_labels", [])
                        kg_sources.append({
                            "doc_name": f"{identifier}-{target_id}",
                            "source_type": labels[0] if labels else "Unknown",
                            "relationship": rel_type,
                            "target_type": target_labels[0] if target_labels else "Unknown",
                            "concept": identifier
                        })
                        # Add target identifier if available
                        if target_id and target_id not in identifiers:
                            identifiers.append(str(target_id))
                    else:
                        source_id = rel.get("source_identifier") or rel.get("source_id")
                        source_labels = rel.get("source_labels", [])
                        kg_sources.append({
                            "doc_name": f"{source_id}-{identifier}",
                            "source_type": source_labels[0] if source_labels else "Unknown",
                            "relationship": rel_type,
                            "target_type": labels[0] if labels else "Unknown",
                            "concept": identifier
                        })
                        # Add source identifier if available
                        if source_id and str(source_id) not in identifiers:
                            identifiers.append(str(source_id))
                
                # If no relationships, still add the node itself
                if not relationships:
                    kg_sources.append({
                        "doc_name": identifier,
                        "source_type": labels[0] if labels else "Unknown",
                        "relationship": "N/A",
                        "target_type": "N/A",
                        "concept": identifier
                    })
            
            # Also use identifiers from tool output directly
            if neo4j_output.identifiers:
                for ident in neo4j_output.identifiers:
                    if ident and ident not in identifiers:
                        identifiers.append(ident)
            
            return kg_sources[:15], identifiers[:15]  # Limit total results
            
        except Exception as e:
            logger.warning(f"Error querying Knowledge Graph: {str(e)}")
            return [], []

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from requirements text using ExtractConceptsTool"""
        try:
            input_data = ExtractConceptsInput(text=text, max_concepts=5)
            output = self.extract_concepts_tool.extract_concepts(input_data)
            return output.concepts
        except Exception as e:
            logger.warning(f"Error extracting concepts: {str(e)}")
            return []

    def _perform_ba_analysis(
        self, 
        text: str, 
        kb_results: List[Dict[str, Any]], 
        kg_results: List[Dict[str, Any]]
    ) -> BAAnalysisResult:
        """Perform comprehensive Business Analysis using OpenAI with context"""
        
        # Build context from KB and KG results
        kb_context = self._build_kb_context(kb_results)
        kg_context = self._build_kg_context(kg_results)
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
        You are a senior Business Analyst specializing in Requirements Engineering.
        Analyze the following requirements for software projects.
        
        REQUIREMENTS TO ANALYZE:
        {text}
        
        KNOWLEDGE BASE CONTEXT (Similar Requirements):
        {kb_context}
        
        KNOWLEDGE GRAPH CONTEXT (Related Patterns):
        {kg_context}
        
        ANALYSIS TASKS:
        1. Parse and structure the requirements (identify user stories, business rules, functional requirements)
        2. Detect contradictions and conflicts between requirements
        3. Assess quality (clarity, completeness, consistency, feasibility, testability)
        4. Suggest specific improvements following software engineering best practices
        5. Identify missing requirements based on similar projects
        6. Provide recommendations for requirements engineering best practices
        
        OUTPUT FORMAT:
        Provide a comprehensive analysis in English, including:
        - Requirements Summary
        - Issues Identified
        - Contradictions and Conflicts
        - Quality Assessment (with scores)
        - Specific Improvement Suggestions
        - Missing Requirements
        - Next Steps Recommendations
        
        Be specific, actionable, and consider modern software development practices.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            analysis_answer = response.choices[0].message.content
            
            return BAAnalysisResult(
                answer=analysis_answer,
                kb_sources=[{"doc_name": item.get("doc_name", "unknown")} for item in kb_results],
                kg_sources=[{"doc_name": item.get("doc_name", "unknown")} for item in kg_results]
            )
            
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return BAAnalysisResult(
                answer=f"Error during analysis: {str(e)}",
                kb_sources=[],
                kg_sources=[]
            )

    def _build_kb_context(self, kb_results: List[Dict[str, Any]]) -> str:
        """Build context string from Knowledge Base results using ContextBuilderTool"""
        try:
            input_data = BuildKBContextInput(kb_results=kb_results, max_results=3)
            return self.context_builder_tool.build_kb_context(input_data)
        except Exception as e:
            logger.warning(f"Error building KB context: {str(e)}")
            return "No similar requirements found in knowledge base."

    def _build_kg_context(self, kg_results: List[Dict[str, Any]]) -> str:
        """Build context string from Knowledge Graph results using ContextBuilderTool"""
        try:
            input_data = BuildKGContextInput(kg_results=kg_results, max_results=3)
            return self.context_builder_tool.build_kg_context(input_data)
        except Exception as e:
            logger.warning(f"Error building KG context: {str(e)}")
            return "No related patterns found in knowledge graph."

    def _format_output(
        self, 
        analysis_result: BAAnalysisResult, 
        kb_doc_names: List[str], 
        kg_identifiers: List[str]
    ) -> Dict[str, Any]:
        """Format the final output according to specification for suggestion_agent
        
        Output format:
        {
            "BA_answer_text": "...",
            "KB": ["doc_name1", "doc_name2", ...],
            "KG": ["identifier1", "identifier2", ...]
        }
        """
        return {
            "BA_answer_text": analysis_result.answer,
            "KB": kb_doc_names,
            "KG": kg_identifiers
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response in format for suggestion_agent"""
        return {
            "BA_answer_text": f"Error during requirements analysis: {error_message}",
            "KB": [],
            "KG": []
        }


# Create a singleton instance for function tools
_ba_agent_instance = None

def _get_ba_agent_instance() -> BusinessAnalysisAgent:
    """Get or create the business analysis agent instance"""
    global _ba_agent_instance
    if _ba_agent_instance is None:
        _ba_agent_instance = BusinessAnalysisAgent()
    return _ba_agent_instance


@function_tool
def analyze_requirements(
    user_text_1: str = "",
    user_text_2: str = "",
    user_text_3: str = "",
) -> str:
    """
    Analyzes requirements (SRS/User Stories) for contradictions, quality, and improvements.
    Uses Knowledge Base (Milvus) and Knowledge Graph (Neo4j) for enhanced analysis.
    
    Args:
        user_text_1: First requirement or user story text (optional)
        user_text_2: Second requirement or user story text (optional)
        user_text_3: Third requirement or user story text (optional)
    
    Returns:
        JSON string with BA_answer_text, KB (list of doc names), and KG (list of identifiers)
    """
    ba_agent = _get_ba_agent_instance()
    
    try:
        # Build user input dictionary
        user_input = {}
        if user_text_1.strip():
            user_input["user_text_1"] = user_text_1.strip()
        if user_text_2.strip():
            user_input["user_text_2"] = user_text_2.strip()
        if user_text_3.strip():
            user_input["user_text_3"] = user_text_3.strip()
        
        # Validate that at least one text is provided
        if not user_input:
            return json.dumps({
                "BA_answer_text": "Error: At least one requirement text must be provided",
                "KB": [],
                "KG": []
            }, ensure_ascii=False)
        
        # Perform analysis
        result = ba_agent.analyze_requirements(user_input)
        
        # Return JSON string
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "BA_answer_text": f"Error: {str(e)}",
            "KB": [],
            "KG": []
        }
        return json.dumps(error_result, ensure_ascii=False)


def create_business_analysis_agent(
    name: str = "Business Analysis Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    top_p: Optional[float] = None,
) -> Agent:
    """Create a Business Analysis Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for randomness (default: 0.1)
        max_tokens: Maximum tokens in response (default: 2000)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for business analysis with analysis tools
        
    Example:
        from agents import Runner
        from agents.business_analysis_agent import create_business_analysis_agent
        
        agent = create_business_analysis_agent()
        result = await Runner.run(
            agent,
            "Analyze these requirements: user_text_1='As a customer, I want to login...', user_text_2='The system must allow guests...'"
        )
        print(result.final_output)
    """
    if instructions is None:
        instructions = (
            "You are a senior Business Analysis Agent specializing in Requirements Engineering "
            "for software projects.\n\n"
            "Your role is to:\n"
            "1. Analyze requirements (SRS/User Stories) for contradictions and conflicts\n"
            "2. Assess quality (clarity, completeness, consistency, feasibility, testability)\n"
            "3. Suggest specific improvements following software engineering best practices\n"
            "4. Identify missing requirements based on similar projects\n"
            "5. Provide recommendations for requirements engineering best practices\n\n"
            "When analyzing requirements:\n"
            "- ALWAYS call search_milvus_kb_ba(query) to search the Knowledge Base for similar requirements\n"
            "- ALWAYS call search_neo4j_kg_ba(query) to search the Knowledge Graph for related projects, requirements, and user stories\n"
            "- Use the analyze_requirements tool to perform comprehensive analysis\n"
            "- Provide at least one requirement text (user_text_1, user_text_2, or user_text_3)\n"
            "- The analyze_requirements tool will also automatically query Knowledge Base (Milvus) and Knowledge Graph (Neo4j) for context\n"
            "- You can use search_milvus_kb_ba and search_neo4j_kg_ba tools directly to get more detailed search results\n"
            "- Return structured analysis results in JSON format with BA_answer_text, KB, and KG\n"
            "- The analysis includes: requirements summary, issues identified, contradictions, "
            "quality assessment, improvement suggestions, missing requirements, and recommendations\n\n"
            "CRITICAL: You MUST call search_milvus_kb_ba and search_neo4j_kg_ba BEFORE using analyze_requirements tool "
            "to get the most relevant context from the Knowledge Base and Knowledge Graph."
        )
    
    # Create model settings
    model_settings_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        model_settings_kwargs["top_p"] = top_p
    
    model_settings = ModelSettings(**model_settings_kwargs)
    
    # Create agent with analysis tools (analyze_requirements, search_milvus_kb_ba, search_neo4j_kg_ba)
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[analyze_requirements, search_milvus_kb_ba, search_neo4j_kg_ba],
    )
    
    return agent


# Legacy function for backward compatibility
def create_business_analysis_agent_legacy():
    """Legacy factory function to create Business Analysis Agent (for backward compatibility)"""
    ba_agent_instance = BusinessAnalysisAgent()
    
    def analyze_user_requirements_legacy(user_input_json: str) -> str:
        """Main analysis function that takes JSON input and returns JSON output"""
        try:
            # Parse JSON input
            user_input = json.loads(user_input_json)
            
            # Validate input format
            if not isinstance(user_input, dict):
                raise ValueError("Input must be a JSON object")
            
            # Perform analysis
            result = ba_agent_instance.analyze_requirements(user_input)
            
            # Return JSON string
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError:
            return json.dumps({
                "BA_answer_text": "Error: Invalid JSON input format",
                "KB": [],
                "KG": []
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "BA_answer_text": f"Error: {str(e)}",
                "KB": [],
                "KG": []
            }, ensure_ascii=False)
    
    # Return the analysis function directly
    return analyze_user_requirements_legacy


if __name__ == "__main__":
    import asyncio
    
    async def test_business_analysis_agent():
        """Test the Business Analysis Agent using Runner"""
        # Create the Business Analysis Agent
        agent = create_business_analysis_agent()
        
        print("Testing Business Analysis Agent with Runner...")
        print("="*60)
        
        # Example input - English requirements
        test_input = (
            "Analyze these requirements: "
            "user_text_1='As a customer, I want to login to the system so that I can view my purchase history', "
            "user_text_2='The system must allow guest users to view products without requiring login', "
            "user_text_3='Admin can delete any account, including other admin accounts'"
        )
        
        print("\n📝 Test: Business Analysis with multiple requirements")
        print("Input:", test_input)
        print("\n" + "="*50 + "\n")
        
        # Run agent with Runner
        result = await Runner.run(agent, test_input)
        
        print("Output:", result.final_output)
        
        print("\n" + "="*60)
        print("✅ Business Analysis Agent testing completed!")
    
    # Run async test
    asyncio.run(test_business_analysis_agent())