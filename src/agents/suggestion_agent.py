"""
Suggestion Agent - Analyzes conflicts and generates solution suggestions for requirements.

This agent receives input from BA Agent (conflict detection), searches web for information,
and uses LLM to create detailed suggestions.

Architecture:
1. Parse BA Agent input
2. (Optional) Retrieve KB documents from Milvus - SKIPPED per user request
3. (Optional) Query KG nodes from Neo4j - SKIPPED per user request  
4. Search web for relevant solutions
5. Generate suggestion using LLM
6. Validate and return output

Note: User requested NOT to use retrieval adapter, so skip steps 2 and 3.
Uses OpenAI SDK Agent and Runner for content generation.
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

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
from pydantic import ValidationError

from src.agents.suggestion_models import (
    BAAgentInput,
    SuggestionAgentOutput,
    parse_ba_input,
    create_suggestion_output,
    validate_ba_input_json,
    suggestion_output_to_json,
)
from tools.web_search_tool import WebSearchTool, get_default_search_tool
from string import Template

logger = logging.getLogger(__name__)

# Prompt templates for Suggestion Agent
WEB_SEARCH_QUERY_GENERATION_PROMPT = Template("""
You are a search query generator for requirements engineering.

Given a conflict description and related requirements, generate 3-5 search queries that would help find solutions or best practices.

**Conflict Description:**
${conflict_description}

**Related Requirements:**
${requirements}

Generate a JSON object with the following format:
{
    "queries": ["query1", "query2", "query3", ...]
}

The queries should be:
- Specific to the conflict described
- Focused on finding solutions or best practices
- Clear and actionable
- In English

Respond with ONLY the JSON object, no additional text.
""")

SUGGESTION_AGENT_SYSTEM_PROMPT = """You are a Suggestion Agent specializing in Requirements Engineering.

Your role is to:
1. Analyze conflicts and issues detected by the Business Analysis Agent
2. Use web search results and knowledge base/graph references to find solutions
3. Generate detailed, actionable suggestions to resolve conflicts
4. Provide comprehensive solutions following software engineering best practices

**Guidelines:**
- Be specific and actionable in your suggestions
- Reference industry best practices when applicable
- Consider multiple perspectives and approaches
- Prioritize solutions that address the root cause of conflicts
- Provide clear, step-by-step recommendations when possible
- Reference relevant knowledge base documents and knowledge graph nodes when available
"""

SUGGESTION_AGENT_USER_PROMPT_TEMPLATE = Template("""
**Business Analysis Result:**
${ba_answer_text}

**Knowledge Base Context:**
${kb_context}

**Knowledge Graph Context:**
${kg_context}

**Web Search Results:**
${web_search_results}

**Your Task:**
Based on the above information, generate detailed suggestions to resolve the identified conflicts and improve the requirements.

Your suggestions should:
1. Address each conflict or issue identified
2. Provide specific, actionable recommendations
3. Reference industry best practices from web search results when applicable
4. Consider the context from knowledge base and knowledge graph
5. Be clear, concise, and implementable

Generate your response below:
""")


class SuggestionAgent:
    """
    Suggestion Agent cho Requirements Engineering.
    
    Nhận BA Agent output, tìm kiếm thông tin web, và tạo suggestions
    để giải quyết conflicts trong requirements.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        web_search_enabled: bool = True,
        max_web_results: int = 3,
    ):
        """
        Initialize Suggestion Agent.
        
        Args:
            openai_api_key: OpenAI API key (nếu None, lấy từ config)
            model: OpenAI model name
            temperature: LLM temperature (0.0 - 1.0)
            web_search_enabled: Có enable web search không
            max_web_results: Số lượng web search results tối đa
        """
        # Load config nếu không có API key
        if openai_api_key is None:
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.web_search_enabled = web_search_enabled
        self.max_web_results = max_web_results
        
        # Initialize web search tool nếu enabled
        if self.web_search_enabled:
            # get_default_search_tool doesn't take openai_api_key, it uses env var
            self.web_search_tool = get_default_search_tool()
        else:
            self.web_search_tool = None
        
        logger.info(
            f"SuggestionAgent initialized: model={model}, "
            f"temperature={temperature}, web_search={web_search_enabled}"
        )
    
    def generate_search_queries(self, ba_input: BAAgentInput) -> List[str]:
        """
        Generate web search queries từ BA agent input.
        
        Args:
            ba_input: Input từ BA agent
            
        Returns:
            List of search queries
        """
        try:
            # Extract requirements từ BA answer text
            requirements = []
            if ba_input.KG:
                requirements = ba_input.KG
            
            prompt = WEB_SEARCH_QUERY_GENERATION_PROMPT.substitute(
                conflict_description=ba_input.BA_answer_text[:500],  # Limit length
                requirements=", ".join(requirements) if requirements else "N/A"
            )
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature cho consistent queries
                max_tokens=500,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Remove markdown code blocks nếu có
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            data = json.loads(result_text)
            queries = data.get("queries", [])
            
            logger.info(f"Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            logger.warning(f"Failed to generate search queries: {e}")
            # Fallback: Tạo query đơn giản từ BA answer text
            return [ba_input.BA_answer_text[:200]]
    
    def search_web(self, queries: List[str]) -> tuple[str, List[str]]:
        """
        Search web với multiple queries và format kết quả.
        
        Args:
            queries: List of search queries
            
        Returns:
            Tuple of (formatted_results, urls) where:
            - formatted_results: Formatted web search results string (for LLM context)
            - urls: List of all URLs found (for output)
        """
        if not self.web_search_enabled or not self.web_search_tool:
            return "Web search is disabled.", []
        
        all_results = []
        all_urls = []
        
        for query in queries[:self.max_web_results]:
            try:
                response = self.web_search_tool.search(
                    query=query,
                    max_results=2  # 2 results per query
                )
                
                if response.error:
                    logger.warning(f"Web search error for '{query}': {response.error}")
                    continue
                
                all_results.extend(response.results)
                
                # Collect URLs
                for result in response.results:
                    if result.url and result.url not in all_urls:
                        all_urls.append(result.url)
                
            except Exception as e:
                logger.error(f"Error searching web for '{query}': {e}")
        
        # Format results for LLM context
        if not all_results:
            return "No web search results found.", []
        
        formatted = "### Web Search Results\n\n"
        for i, result in enumerate(all_results[:5], 1):  # Top 5 results
            formatted += f"**Result {i}:**\n"
            formatted += f"- Title: {result.title}\n"
            formatted += f"- URL: {result.url}\n"
            formatted += f"- Snippet: {result.snippet}\n"
            if hasattr(result, 'relevance_score') and result.relevance_score:
                formatted += f"- Relevance: {result.relevance_score}\n"
            formatted += "\n"
        
        return formatted, all_urls[:10]  # Limit to top 10 URLs
    
    def generate_suggestion(
        self,
        ba_input: BAAgentInput,
        web_search_results: str,
    ) -> str:
        """
        Generate suggestion using LLM.
        
        Args:
            ba_input: Input từ BA agent
            web_search_results: Formatted web search results
            
        Returns:
            Generated suggestion text
        """
        # Prepare KB context (from KB field in input)
        kb_context = "### Knowledge Base Documents\n"
        if ba_input.KB:
            kb_context += f"Referenced documents: {', '.join(ba_input.KB)}\n"
            kb_context += "(Document contents not retrieved - retrieval adapter disabled per user request)\n"
        else:
            kb_context += "No KB documents referenced.\n"
        
        # Prepare KG context (from KG field in input)
        kg_context = "### Knowledge Graph Nodes\n"
        if ba_input.KG:
            kg_context += f"Referenced requirements: {', '.join(ba_input.KG)}\n"
            kg_context += "(Node details not retrieved - retrieval adapter disabled per user request)\n"
        else:
            kg_context += "No KG nodes referenced.\n"
        
        # Build user prompt
        user_prompt = SUGGESTION_AGENT_USER_PROMPT_TEMPLATE.substitute(
            ba_answer_text=ba_input.BA_answer_text,
            kb_context=kb_context,
            kg_context=kg_context,
            web_search_results=web_search_results,
        )
        
        # Call LLM
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUGGESTION_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=4000,  # Allow longer responses
            )
            
            suggestion_text = response.choices[0].message.content.strip()
            
            logger.info(f"Generated suggestion ({len(suggestion_text)} chars)")
            return suggestion_text
            
        except Exception as e:
            logger.error(f"Error generating suggestion with LLM: {e}")
            raise
    
    def extract_source_urls(self, web_search_results: str) -> List[str]:
        """
        Extract all source URLs từ web search results formatted string.
        
        Args:
            web_search_results: Formatted web search results string
            
        Returns:
            List of URLs found in results
        """
        urls = []
        lines = web_search_results.split("\n")
        for line in lines:
            if line.startswith("- URL:"):
                url = line.replace("- URL:", "").strip()
                if url and url != "N/A" and url not in urls:
                    urls.append(url)
        
        # Fallback: Generic source if no URLs found
        if not urls:
            urls.append("https://example.com/requirements-engineering-best-practices")
        
        return urls
    
    def process(
        self,
        ba_input: BAAgentInput,
    ) -> SuggestionAgentOutput:
        """
        Process BA agent input và generate suggestion.
        
        Main pipeline:
        1. Parse input (already done via ba_input parameter)
        2. Generate search queries từ conflict description
        3. Search web for solutions
        4. Generate suggestion using LLM với all context
        5. Extract source URL
        6. Return validated output
        
        Args:
            ba_input: Parsed BA agent input
            
        Returns:
            Validated suggestion output
        """
        logger.info("=" * 80)
        logger.info("Starting Suggestion Agent pipeline")
        logger.info("=" * 80)
        
        # Step 1: Log input
        logger.info(f"BA answer text length: {len(ba_input.BA_answer_text)} chars")
        logger.info(f"KB documents referenced: {len(ba_input.KB)}")
        logger.info(f"KG nodes referenced: {len(ba_input.KG)}")
        
        # Step 2: Generate search queries
        logger.info("Generating web search queries...")
        search_queries = self.generate_search_queries(ba_input)
        logger.info(f"Generated queries: {search_queries}")
        
        # Step 3: Search web
        logger.info("Searching web for solutions...")
        web_search_results, source_urls = self.search_web(search_queries)
        logger.info(f"Found {len(source_urls)} URLs from web search")
        
        # Step 4: Generate suggestion
        logger.info("Generating suggestion with LLM...")
        suggestion_text = self.generate_suggestion(
            ba_input=ba_input,
            web_search_results=web_search_results,
        )
        
        # Step 5: Extract source URLs (already done in search_web, but also extract from formatted string as fallback)
        if not source_urls:
            source_urls = self.extract_source_urls(web_search_results)
        logger.info(f"Final source URLs: {source_urls}")
        
        # Step 6: Create validated output with KB and KG pass through
        output = create_suggestion_output(
            answer_text=suggestion_text,
            source=source_urls,
            kb=ba_input.KB,  # Pass through from BA input
            kg=ba_input.KG,  # Pass through from BA input
        )
        
        logger.info("=" * 80)
        logger.info("Suggestion Agent pipeline completed successfully")
        logger.info("=" * 80)
        
        return output
    
    def process_from_json(self, json_input: str) -> str:
        """
        Process JSON input string và return JSON output string.
        
        Convenience method cho API integration.
        
        Args:
            json_input: JSON string matching BAAgentInput format
            
        Returns:
            JSON string matching SuggestionAgentOutput format
        """
        # Parse input
        ba_input = validate_ba_input_json(json_input)
        
        # Process
        output = self.process(ba_input)
        
        # Return JSON
        return suggestion_output_to_json(output)
    
    def process_from_file(
        self,
        input_file: Path | str,
        output_file: Optional[Path | str] = None,
    ) -> SuggestionAgentOutput:
        """
        Process input từ file và optionally save output to file.
        
        Args:
            input_file: Path to JSON input file
            output_file: Optional path to save JSON output
            
        Returns:
            Suggestion output
        """
        input_file = Path(input_file)
        
        # Read input
        logger.info(f"Reading input from: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            json_input = f.read()
        
        # Process
        json_output = self.process_from_json(json_input)
        
        # Save output nếu specified
        if output_file:
            output_file = Path(output_file)
            logger.info(f"Saving output to: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json_output)
        
        # Return parsed output
        return create_suggestion_output(
            **json.loads(json_output)
        )


# Create a singleton instance for function tools
_suggestion_agent_instance = None

def _get_suggestion_agent_instance() -> SuggestionAgent:
    """Get or create the suggestion agent instance"""
    global _suggestion_agent_instance
    if _suggestion_agent_instance is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        _suggestion_agent_instance = SuggestionAgent(
            openai_api_key=api_key,
            model="gpt-4o",  # Default model
            temperature=0.7,  # Default temperature
            web_search_enabled=True,  # Default enabled
            max_web_results=3,  # Default max results
        )
    return _suggestion_agent_instance


@function_tool
def generate_suggestion(
    ba_answer_text: str = "",
    kb: str = "",
    kg: str = "",
) -> str:
    """
    Generates solution suggestions for conflicts in requirements.
    Analyzes BA agent input, searches web for solutions, and generates suggestions using LLM.
    
    Args:
        ba_answer_text: Conflict description from BA agent (required)
        kb: Comma-separated list of Knowledge Base document names (optional)
        kg: Comma-separated list of Knowledge Graph node identifiers (optional)
    
    Returns:
        JSON string with answer_text, source (URLs), KB, and KG
    """
    suggestion_agent = _get_suggestion_agent_instance()
    
    try:
        # Build BA input
        ba_input_dict = {
            "BA_answer_text": ba_answer_text.strip(),
        }
        
        # Parse KB if provided
        if kb:
            kb_list = [s.strip() for s in kb.split(",") if s.strip()]
            ba_input_dict["KB"] = kb_list
        
        # Parse KG if provided
        if kg:
            kg_list = [s.strip() for s in kg.split(",") if s.strip()]
            ba_input_dict["KG"] = kg_list
        
        # Validate and parse input
        ba_input = parse_ba_input(ba_input_dict)
        
        # Process and generate suggestion
        output = suggestion_agent.process(ba_input)
        
        # Return JSON string
        return suggestion_output_to_json(output)
        
    except Exception as e:
        error_result = {
            "answer_text": f"Error: {str(e)}",
            "source": [],
            "KB": kb.split(",") if kb else [],
            "KG": kg.split(",") if kg else []
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


def create_suggestion_agent(
    name: str = "Suggestion Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: Optional[float] = None,
) -> Agent:
    """Create a Suggestion Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for randomness (default: 0.7)
        max_tokens: Maximum tokens in response (default: 4000)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for suggestion generation with analysis tools
        
    Example:
        from agents import Runner
        from agents.suggestion_agent import create_suggestion_agent
        
        agent = create_suggestion_agent()
        result = await Runner.run(
            agent,
            "Generate suggestions for: ba_answer_text='Conflict detected: ...', kb='doc1,doc2', kg='REQ-001,REQ-002'"
        )
        print(result.final_output)
    """
    if instructions is None:
        instructions = (
            "You are a Suggestion Agent specializing in Requirements Engineering.\n\n"
            "Your role is to:\n"
            "1. Analyze conflicts and issues detected by the Business Analysis Agent\n"
            "2. Search the web for relevant solutions and best practices\n"
            "3. Generate detailed, actionable suggestions to resolve conflicts\n"
            "4. Provide comprehensive solutions following software engineering best practices\n\n"
            "When generating suggestions:\n"
            "- Use the generate_suggestion tool to create solution suggestions\n"
            "- Provide the BA_answer_text (conflict description) from the Business Analysis Agent\n"
            "- Include KB (Knowledge Base) and KG (Knowledge Graph) references when available\n"
            "- The tool will automatically search the web and generate suggestions\n"
            "- Return structured results in JSON format with answer_text, source URLs, KB, and KG\n"
            "- Suggestions should be specific, actionable, and based on industry best practices"
        )
    
    # Create model settings
    model_settings_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        model_settings_kwargs["top_p"] = top_p
    
    model_settings = ModelSettings(**model_settings_kwargs)
    
    # Create agent with suggestion tool
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[generate_suggestion],
    )
    
    return agent


# Legacy function for backward compatibility
def create_default_agent() -> SuggestionAgent:
    """
    Legacy factory function to create Suggestion Agent (for backward compatibility).
    
    Returns:
        Configured SuggestionAgent instance
    """
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return SuggestionAgent(
        openai_api_key=api_key,
        model="gpt-4o",  # Default model
        temperature=0.7,  # Default temperature
        web_search_enabled=True,  # Default enabled
        max_web_results=3,  # Default max results
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for Suggestion Agent."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Suggestion Agent - Analyze conflicts and generate suggestions"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to JSON input file (BA agent output)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save JSON output file (optional)",
    )
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Create agent
        agent = SuggestionAgent(
            model=args.model,
            temperature=args.temperature,
            web_search_enabled=not args.no_web_search,
        )
        
        # Process
        output = agent.process_from_file(
            input_file=args.input_file,
            output_file=args.output,
        )
        
        # Print result
        print("\n" + "=" * 80)
        print("SUGGESTION GENERATED")
        print("=" * 80)
        print(f"\nAnswer Text ({len(output.answer_text)} chars):\n")
        print(output.answer_text)
        print(f"\n\nSources ({len(output.source)} URLs):")
        for i, url in enumerate(output.source, 1):
            print(f"  {i}. {url}")
        print(f"\nKB Documents ({len(output.KB)}): {', '.join(output.KB) if output.KB else 'None'}")
        print(f"KG Nodes ({len(output.KG)}): {', '.join(output.KG) if output.KG else 'None'}")
        print("\n" + "=" * 80)
        
        if args.output:
            print(f"\n✅ Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
