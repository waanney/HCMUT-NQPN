"""
Guardrail Agent for Requirements Engineering
Evaluates and validates outputs from Suggestion_Agent and RAG_Agent
Uses OpenAI SDK Agent and Runner for content evaluation
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

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

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try multiple paths to find .env file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try project root (2 levels up from src/agents/)
    env_path = os.path.join(current_dir, '..', '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded .env from: {env_path}")
    else:
        # Try current working directory
        if os.path.exists('.env'):
            load_dotenv('.env')
            logger.info("Loaded .env from current directory")
        else:
            # Also try to load from environment
            load_dotenv()
            logger.info("Attempted to load .env from default locations")
except ImportError:
    # If python-dotenv is not installed, continue without it
    logger.warning("python-dotenv not installed, skipping .env file loading")
    pass


class GuardrailStatus(Enum):
    """Guardrail evaluation status"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SuggestedAction(Enum):
    """Suggested actions based on evaluation"""
    APPROVE = "approve"
    REWRITE = "rewrite"
    VERIFY_SOURCE = "verify_source"


class SourceType(Enum):
    """Classification of source types"""
    RESEARCH_PAPER = "research_paper"
    TECHNICAL_DOC = "technical_doc"
    OFFICIAL_WEBSITE = "official_website"
    BLOG = "blog"
    FORUM = "forum"
    UNKNOWN = "unknown"


@dataclass
class Issue:
    """Represents an issue found during evaluation"""
    type: str  # clarity, accuracy, consistency, risk, source_validity
    description: str


@dataclass
class GuardrailResult:
    """Structure for Guardrail evaluation results"""
    guardrail_status: str
    issues_found: List[Dict[str, str]]
    suggested_action: str
    summary: str


class GuardrailAgent:
    """
    Guardrail Agent for validating outputs from other agents
    Evaluates content based on risk, clarity, consistency, accuracy, and source validity
    """
    
    # Trusted source patterns
    TRUSTED_DOMAINS = [
        r'\.edu$', r'\.gov$', r'\.org$',  # Educational/Government/Organization domains
        r'arxiv\.org', r'ieee\.org', r'acm\.org',  # Research repositories
        r'github\.com', r'stackoverflow\.com',  # Technical platforms
        r'microsoft\.com', r'openai\.com', r'google\.com',  # Tech companies
        r'w3\.org', r'ietf\.org',  # Standards bodies
    ]
    
    # Weak/untrusted source patterns
    WEAK_DOMAINS = [
        r'medium\.com', r'blogger\.com', r'wordpress\.com',  # Blogs
        r'reddit\.com', r'quora\.com',  # Forums
        r'yahoo\.answers', r'ask\.com',  # Q&A sites
    ]
    
    # Harmful content keywords
    HARMFUL_KEYWORDS = [
        'malicious', 'exploit', 'hack', 'crack', 'pirate',
        'illegal', 'fraud', 'scam', 'phishing'
    ]
    
    def __init__(self):
        """Initialize the Guardrail Agent"""
        self.openai_client = OpenAI()
        logger.info("Guardrail Agent initialized successfully")
    
    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main evaluation function
        
        Args:
            input_data: Dictionary containing user_input, answer_text and optional source, KB, KG
                        Format: {
                            "user_input": "original user query",
                            "answer_text": "answer from agent",
                            "source": ["url1", ...],  # Optional
                            "KB": ["doc1", ...],  # Optional
                            "KG": ["node1", ...]  # Optional
                        }
            
        Returns:
            JSON format with guardrail_status, issues_found, suggested_action, summary
        """
        try:
            # Extract fields
            user_input = input_data.get("user_input", "")  # Original user query (optional for backward compatibility)
            answer_text = input_data.get("answer_text", "")
            source = input_data.get("source")
            kb_sources = input_data.get("KB", [])
            kg_sources = input_data.get("KG", [])
            
            if not answer_text:
                return self._create_rejection("Empty or missing answer_text")
            
            # Log user input if provided
            if user_input:
                logger.info(f"Evaluating response to user query: '{user_input[:100]}...'")
            
            issues = []
            
            # Step 1: Check for garbage text (highest priority)
            if self._is_garbage_text(answer_text):
                return self._create_rejection(
                    "Response contains nonsense or garbage text",
                    [Issue("clarity", "Text appears to be random characters or nonsense")]
                )
            
            # Step 2: Check for risk/safety issues (highest priority)
            risk_issues = self._check_risk_safety(answer_text)
            if risk_issues:
                return self._create_rejection(
                    "Response contains harmful or inappropriate content",
                    risk_issues
                )
            
            # Step 3: Evaluate source if provided
            source_classification = None
            source_issues = []
            
            if source:
                # Handle source as string or list
                if isinstance(source, list):
                    # If source is a list, classify and validate all sources
                    classifications = []
                    for src_url in source:
                        if src_url:
                            src_class = self._classify_source(src_url)
                            classifications.append(src_class)
                            src_issues = self._validate_source(src_url, src_class)
                            source_issues.extend(src_issues)
                    
                    # Use best classification (prioritize trusted sources)
                    if classifications:
                        # Priority: RESEARCH_PAPER > TECHNICAL_DOC > OFFICIAL_WEBSITE > others
                        priority_order = [SourceType.RESEARCH_PAPER, SourceType.TECHNICAL_DOC, 
                                         SourceType.OFFICIAL_WEBSITE, SourceType.BLOG, SourceType.FORUM, SourceType.UNKNOWN]
                        for priority_type in priority_order:
                            if priority_type in classifications:
                                source_classification = priority_type
                                break
                        if source_classification is None:
                            source_classification = classifications[0]
                else:
                    # Handle source as string (backward compatibility)
                    source_classification = self._classify_source(source)
                    source_issues = self._validate_source(source, source_classification)
                
                # If trustworthy source, approve immediately
                if source_classification in [SourceType.RESEARCH_PAPER, 
                                            SourceType.TECHNICAL_DOC, 
                                            SourceType.OFFICIAL_WEBSITE]:
                    # Still check for basic quality issues
                    clarity_issues = self._check_clarity(answer_text)
                    consistency_issues = self._check_consistency(answer_text)
                    
                    all_issues = clarity_issues + consistency_issues
                    
                    if not all_issues:
                        return self._create_approval(
                            "Response is clear, safe, and backed by trustworthy source",
                            []
                        )
                    else:
                        # Minor issues but trusted source
                        return self._create_approval(
                            "Response backed by trustworthy source with minor quality issues",
                            all_issues
                        )
                
                # If weak source, require review
                if source_classification in [SourceType.BLOG, SourceType.FORUM, SourceType.UNKNOWN]:
                    issues.extend(source_issues)
            
            # Step 4: Check clarity
            clarity_issues = self._check_clarity(answer_text)
            issues.extend(clarity_issues)
            
            # Step 5: Check consistency
            consistency_issues = self._check_consistency(answer_text)
            issues.extend(consistency_issues)
            
            # Step 6: Check accuracy (using OpenAI) with user input context
            # Convert source to string for accuracy check if it's a list
            source_str = None
            if source:
                if isinstance(source, list):
                    source_str = ", ".join(source[:3])  # Use top 3 URLs
                else:
                    source_str = source
            accuracy_issues = self._check_accuracy(
                answer_text, 
                user_input or "N/A",  # Include user input for relevance check (use "N/A" if not provided)
                source_str, 
                kb_sources, 
                kg_sources
            )
            issues.extend(accuracy_issues)
            
            # Decision logic
            if source:
                if source_classification in [SourceType.BLOG, SourceType.FORUM]:
                    return self._create_review_required(
                        "Weak or unverified source detected",
                        issues,
                        SuggestedAction.VERIFY_SOURCE
                    )
            else:
                # No source provided
                if len(issues) == 0:
                    return self._create_review_required(
                        "No source provided, but response appears plausible",
                        [],
                        SuggestedAction.VERIFY_SOURCE
                    )
                elif any(issue.type in ['accuracy', 'consistency'] for issue in issues):
                    return self._create_review_required(
                        "No source and potential accuracy issues detected",
                        issues,
                        SuggestedAction.VERIFY_SOURCE
                    )
            
            # If only minor clarity issues
            if all(issue.type == 'clarity' for issue in issues) and len(issues) <= 2:
                return self._create_approval(
                    "Response is acceptable with minor clarity improvements needed",
                    issues
                )
            
            # If significant issues
            if len(issues) >= 3:
                return self._create_review_required(
                    "Multiple quality issues detected",
                    issues,
                    SuggestedAction.VERIFY_SOURCE
                )
            
            # Default: approve if no major issues
            return self._create_approval(
                "Response meets quality standards",
                issues
            )
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return self._create_rejection(f"Evaluation error: {str(e)}")
    
    def _is_garbage_text(self, text: str) -> bool:
        """Detect nonsense or garbage text"""
        # Check for random character sequences
        if re.search(r'[a-z]{15,}', text.lower()) and not re.search(r'\s', text[:20]):
            return True
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\-\(\)]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            return True
        
        # Check for lack of spaces (gibberish)
        words = text.split()
        if len(words) < 3 and len(text) > 50:
            return True
        
        return False
    
    def _check_risk_safety(self, text: str) -> List[Issue]:
        """Check for harmful, false, or inappropriate content"""
        issues = []
        text_lower = text.lower()
        
        # Check for harmful keywords
        found_harmful = [kw for kw in self.HARMFUL_KEYWORDS if kw in text_lower]
        if found_harmful:
            issues.append(Issue(
                "risk",
                f"Potentially harmful content detected: {', '.join(found_harmful)}"
            ))
        
        # Use OpenAI to detect subtle safety issues
        try:
            safety_check = self._openai_safety_check(text)
            if safety_check:
                issues.extend(safety_check)
        except Exception as e:
            logger.warning(f"OpenAI safety check failed: {str(e)}")
        
        return issues
    
    def _check_clarity(self, text: str) -> List[Issue]:
        """Check if response is coherent, understandable, and complete"""
        issues = []
        
        # Check for very short responses
        if len(text.strip()) < 20:
            issues.append(Issue("clarity", "Response is too brief"))
        
        # Check for incomplete sentences
        if not text.strip().endswith(('.', '!', '?', '"', "'")):
            issues.append(Issue("clarity", "Response appears incomplete"))
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:
                issues.append(Issue("clarity", "Response contains excessive repetition"))
        
        return issues
    
    def _check_consistency(self, text: str) -> List[Issue]:
        """Detect logical or semantic contradictions"""
        issues = []
        
        # Look for contradiction indicators
        contradiction_patterns = [
            r'however.*but', r'although.*nevertheless',
            r'on the other hand.*on the other hand',
            r'not.*but.*is', r'cannot.*but.*can'
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, text.lower()):
                issues.append(Issue(
                    "consistency",
                    f"Potential logical contradiction detected"
                ))
                break
        
        return issues
    
    def _check_accuracy(self, text: str, user_input: str, source: Optional[str], 
                       kb_sources: List[str], kg_sources: List[str]) -> List[Issue]:
        """Verify if information is factually correct and relevant to user input"""
        issues = []
        
        # Use OpenAI for accuracy assessment
        try:
            accuracy_issues = self._openai_accuracy_check(
                text, 
                user_input,  # Include user input for relevance check
                source, 
                kb_sources, 
                kg_sources
            )
            issues.extend(accuracy_issues)
        except Exception as e:
            logger.warning(f"OpenAI accuracy check failed: {str(e)}")
        
        return issues
    
    def _classify_source(self, source: str) -> SourceType:
        """Classify the type and reliability of a source"""
        source_lower = source.lower()
        
        # Check for trusted domains
        for pattern in self.TRUSTED_DOMAINS:
            if re.search(pattern, source_lower):
                if 'arxiv' in source_lower or 'ieee' in source_lower or 'acm' in source_lower:
                    return SourceType.RESEARCH_PAPER
                elif '.edu' in source_lower or '.gov' in source_lower:
                    return SourceType.OFFICIAL_WEBSITE
                else:
                    return SourceType.TECHNICAL_DOC
        
        # Check for weak domains
        for pattern in self.WEAK_DOMAINS:
            if re.search(pattern, source_lower):
                if 'medium' in source_lower or 'blog' in source_lower:
                    return SourceType.BLOG
                else:
                    return SourceType.FORUM
        
        # Check for common technical documentation patterns
        if any(keyword in source_lower for keyword in ['docs', 'documentation', 'api', 'guide', 'manual']):
            return SourceType.TECHNICAL_DOC
        
        return SourceType.UNKNOWN
    
    def _validate_source(self, source: str, source_type: SourceType) -> List[Issue]:
        """Validate the credibility and relevance of a source"""
        issues = []
        
        if source_type == SourceType.BLOG:
            issues.append(Issue(
                "source_validity",
                "Source is a blog - credibility may vary"
            ))
        elif source_type == SourceType.FORUM:
            issues.append(Issue(
                "source_validity",
                "Source is a forum or Q&A site - information may be unverified"
            ))
        elif source_type == SourceType.UNKNOWN:
            issues.append(Issue(
                "source_validity",
                "Source type cannot be determined - verification needed"
            ))
        
        # Check if source is a valid URL
        if not re.match(r'https?://', source):
            issues.append(Issue(
                "source_validity",
                "Source does not appear to be a valid URL"
            ))
        
        return issues
    
    def _openai_safety_check(self, text: str) -> List[Issue]:
        """Use OpenAI to check for safety issues"""
        prompt = f"""Analyze this text for safety concerns:
        
Text: {text}

Check for:
- Harmful content
- Misleading information
- Inappropriate suggestions
- Security risks

Respond with ONLY a JSON array of issues found, or an empty array [] if none.
Format: [{{"type": "risk", "description": "issue description"}}]
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            issues_data = json.loads(result)
            
            return [Issue(item["type"], item["description"]) for item in issues_data]
        except Exception as e:
            logger.warning(f"OpenAI safety check parsing failed: {str(e)}")
            return []
    
    def _openai_accuracy_check(self, text: str, user_input: str, source: Optional[str],
                               kb_sources: List[str], kg_sources: List[str]) -> List[Issue]:
        """Use OpenAI to check for accuracy issues and relevance to user input"""
        context = f"Original User Query: {user_input}\n\n"
        context += f"Source: {source}\n" if source else ""
        context += f"KB References: {', '.join(kb_sources)}\n" if kb_sources else ""
        context += f"KG References: {', '.join(kg_sources)}\n" if kg_sources else ""
        
        prompt = f"""Evaluate the accuracy and relevance of this requirements engineering response:

{context}
Agent Response: {text}

Check for:
- Factual correctness
- Logical consistency
- Whether claims are supported by context
- Whether the response addresses the original user query
- Whether the response is relevant to what the user asked

Respond with ONLY a JSON array of accuracy issues, or [] if none.
Format: [{{"type": "accuracy", "description": "issue description"}}]
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            issues_data = json.loads(result)
            
            return [Issue(item["type"], item["description"]) for item in issues_data]
        except Exception as e:
            logger.warning(f"OpenAI accuracy check parsing failed: {str(e)}")
            return []
    
    def _create_approval(self, summary: str, issues: List[Issue]) -> Dict[str, Any]:
        """Create an APPROVED response"""
        return {
            "guardrail_status": GuardrailStatus.APPROVED.value,
            "issues_found": [{"type": issue.type, "description": issue.description} 
                           for issue in issues],
            "suggested_action": SuggestedAction.APPROVE.value,
            "summary": summary
        }
    
    def _create_rejection(self, summary: str, issues: Optional[List[Issue]] = None) -> Dict[str, Any]:
        """Create a REJECTED response"""
        if issues is None:
            issues = []
        
        return {
            "guardrail_status": GuardrailStatus.REJECTED.value,
            "issues_found": [{"type": issue.type, "description": issue.description} 
                           for issue in issues],
            "suggested_action": SuggestedAction.REWRITE.value,
            "summary": summary
        }
    
    def _create_review_required(self, summary: str, issues: List[Issue], 
                                action: SuggestedAction) -> Dict[str, Any]:
        """Create a REVIEW_REQUIRED response"""
        return {
            "guardrail_status": GuardrailStatus.REVIEW_REQUIRED.value,
            "issues_found": [{"type": issue.type, "description": issue.description} 
                           for issue in issues],
            "suggested_action": action.value,
            "summary": summary
        }


# Create a singleton instance for function tools
_guardrail_instance = None

def _get_guardrail_instance() -> GuardrailAgent:
    """Get or create the guardrail agent instance"""
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = GuardrailAgent()
    return _guardrail_instance


@function_tool
def evaluate_content(
    user_input: str = "",
    answer_text: str = "",
    source: Optional[str] = None,
    kb: Optional[str] = None,
    kg: Optional[str] = None,
) -> str:
    """
    Evaluates content from suggestion_agent or rag_agent for quality, safety, and accuracy.
    
    Args:
        user_input: Original user query (optional but recommended)
        answer_text: Answer text from suggestion_agent or rag_agent (required)
        source: Comma-separated list of source URLs (optional)
        kb: Comma-separated list of Knowledge Base document names (optional)
        kg: Comma-separated list of Knowledge Graph node identifiers (optional)
    
    Returns:
        JSON string with guardrail_status, issues_found, suggested_action, and summary
    """
    guardrail = _get_guardrail_instance()
    
    try:
        # Build input data
        input_data = {
            "user_input": user_input,
            "answer_text": answer_text,
        }
        
        # Parse source if provided
        if source:
            input_data["source"] = [s.strip() for s in source.split(",") if s.strip()]
        
        # Parse KB if provided
        if kb:
            input_data["KB"] = [s.strip() for s in kb.split(",") if s.strip()]
        
        # Parse KG if provided
        if kg:
            input_data["KG"] = [s.strip() for s in kg.split(",") if s.strip()]
        
        # Perform evaluation
        result = guardrail.evaluate(input_data)
        
        # Return JSON string (pure JSON, no extra text)
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_result = {
            "guardrail_status": "REJECTED",
            "issues_found": [{"type": "error", "description": str(e)}],
            "suggested_action": "rewrite",
            "summary": f"Evaluation failed: {str(e)}"
        }
        return json.dumps(error_result, ensure_ascii=False)


def create_guardrail_agent(
    name: str = "Guardrail Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 500,
    top_p: Optional[float] = None,
) -> Agent:
    """Create a Guardrail Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Model temperature for randomness (default: 0.1)
        max_tokens: Maximum tokens in response (default: 500)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for guardrail evaluation with evaluation tools
        
    Example:
        from agents import Runner
        from agents.guardrail_agent import create_guardrail_agent
        
        agent = create_guardrail_agent()
        result = await Runner.run(
            agent, 
            "Evaluate this content: user_input='What is authentication?', answer_text='Authentication is...', source='https://example.com'"
        )
        print(result.final_output)
    """
    if instructions is None:
        instructions = (
            "You are a Guardrail Agent that evaluates and validates outputs from other agents "
            "(Suggestion_Agent and RAG_Agent) for quality, safety, and accuracy.\n\n"
            "Your role is to:\n"
            "1. Evaluate content for safety, clarity, consistency, accuracy, and source validity\n"
            "2. Check for harmful or inappropriate content\n"
            "3. Verify that information is factually correct and relevant to the user's query\n"
            "4. Assess the credibility of sources\n"
            "5. Return structured evaluation results in JSON format\n\n"
            "When evaluating content:\n"
            "- Use the evaluate_content tool to perform the evaluation\n"
            "- Always provide user_input and answer_text at minimum\n"
            "- Include source URLs, KB references, and KG references when available\n"
            "- Return the evaluation result in JSON format with guardrail_status, issues_found, suggested_action, and summary\n"
            "- The guardrail_status can be: APPROVED, REJECTED, or REVIEW_REQUIRED\n"
            "- The suggested_action can be: approve, rewrite, or verify_source"
        )
    
    # Create model settings
    model_settings_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        model_settings_kwargs["top_p"] = top_p
    
    model_settings = ModelSettings(**model_settings_kwargs)
    
    # Create agent with evaluation tool
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[evaluate_content],
    )
    
    return agent


# Legacy function for backward compatibility
def create_guardrail_agent_legacy():
    """Legacy factory function to create Guardrail Agent (for backward compatibility)"""
    agent_instance = GuardrailAgent()
    
    def evaluate_content_legacy(input_json: str) -> str:
        """
        Main evaluation function that takes JSON input and returns JSON output
        
        Args:
            input_json: JSON string with user_input, answer_text and optional source, KB, KG
                        Format: {
                            "user_input": "original user query",  # Optional but recommended
                            "answer_text": "answer from agent",
                            "source": ["url1", ...],  # Optional
                            "KB": ["doc1", ...],  # Optional
                            "KG": ["node1", ...]  # Optional
                        }
            
        Returns:
            JSON string with guardrail_status, issues_found, suggested_action, summary
        """
        try:
            # Parse JSON input
            input_data = json.loads(input_json)
            
            # Perform evaluation
            result = agent_instance.evaluate(input_data)
            
            # Return JSON string (pure JSON, no extra text)
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError:
            error_result = {
                "guardrail_status": "REJECTED",
                "issues_found": [{"type": "input_error", "description": "Invalid JSON input format"}],
                "suggested_action": "rewrite",
                "summary": "Input is not valid JSON"
            }
            return json.dumps(error_result, ensure_ascii=False)
        except Exception as e:
            error_result = {
                "guardrail_status": "REJECTED",
                "issues_found": [{"type": "error", "description": str(e)}],
                "suggested_action": "rewrite",
                "summary": f"Evaluation failed: {str(e)}"
            }
            return json.dumps(error_result, ensure_ascii=False)
    
    return evaluate_content_legacy


if __name__ == "__main__":
    import asyncio
    
    async def test_guardrail_agent():
        """Test the Guardrail Agent using Runner"""
        # Create the Guardrail Agent
        agent = create_guardrail_agent()
        
        print("Testing Guardrail Agent with Runner...")
    print("="*60)
    
    # Test 1: From Suggestion_Agent with trusted source (with user_input)
        test1_input = (
            "Evaluate this content: "
            "user_input='How should I implement user authentication?', "
            "answer_text='The user authentication requirement should include multi-factor authentication for enhanced security.', "
            "source='https://www.nist.gov/publications/digital-identity-guidelines'"
        )
    
    print("\n📝 Test 1: Suggestion_Agent with trusted source (includes user_input)")
        print("Input:", test1_input)
        result1 = await Runner.run(agent, test1_input)
        print("Output:", result1.final_output)
    
    # Test 2: From RAG_Agent with no source (with user_input)
        test2_input = (
            "Evaluate this content: "
            "user_input='What conflicts exist between REQ-001 and REQ-002?', "
            "answer_text='Detect conflicts between REQ-001 and REQ-002: Both requirements specify different authentication methods which may cause implementation conflicts.', "
            "kb='doc_014,doc_015', "
            "kg='REQ-001,REQ-002'"
        )
    
    print("\n📝 Test 2: RAG_Agent with no source (includes user_input)")
        print("Input:", test2_input)
        result2 = await Runner.run(agent, test2_input)
        print("Output:", result2.final_output)
    
    # Test 3: Garbage text (with user_input)
        test3_input = (
            "Evaluate this content: "
            "user_input='What is the best authentication method?', "
            "answer_text='asdfkjaslkdfjalksjdflkajsdlfkjaslkdfj', "
            "source='https://example.com'"
        )
    
    print("\n📝 Test 3: Garbage text (includes user_input)")
        print("Input:", test3_input)
        result3 = await Runner.run(agent, test3_input)
        print("Output:", result3.final_output)
    
    # Test 4: Weak source (blog) with user_input
        test4_input = (
            "Evaluate this content: "
            "user_input='How should requirements be documented?', "
            "answer_text='Requirements should be written in clear, concise language.', "
            "source='https://medium.com/some-blog-post'"
        )
    
    print("\n📝 Test 4: Weak source (blog) with user_input")
        print("Input:", test4_input)
        result4 = await Runner.run(agent, test4_input)
        print("Output:", result4.final_output)
    
    # Test 5: Backward compatibility - without user_input
        test5_input = (
            "Evaluate this content: "
            "answer_text='Project Alpha includes authentication features.', "
            "kb='doc_014', "
            "kg='PROJ-001'"
        )
    
    print("\n📝 Test 5: Backward compatibility - without user_input")
        print("Input:", test5_input)
        result5 = await Runner.run(agent, test5_input)
        print("Output:", result5.final_output)
    
    print("\n" + "="*60)
    print("✅ Guardrail Agent testing completed!")
    
    # Run async test
    asyncio.run(test_guardrail_agent())
