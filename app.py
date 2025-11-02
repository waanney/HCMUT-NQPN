"""Chainlit UI using Orchestrator Agent to route requests to appropriate agents."""

import sys
from pathlib import Path
import json

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import chainlit as cl
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Import orchestrator
from agents.orchestrator_agent import route_request, create_orchestrator_agent
from agents.formatter_agent import FormatterAgent, FormatterOutput
from agents.rag_agent import RAGOutput
from agents.suggestion_agent import generate_suggestion
from agents.guardrail_agent import evaluate_content

load_dotenv()

# Agent instances
_orchestrator_agent = None
_formatter_agent: FormatterAgent = None


def get_orchestrator():
    """Get or create orchestrator instance."""
    global _orchestrator_agent
    if _orchestrator_agent is None:
        _orchestrator_agent = create_orchestrator_agent()
    return _orchestrator_agent


def get_formatter_agent() -> FormatterAgent:
    """Get or create FormatterAgent instance."""
    global _formatter_agent
    if _formatter_agent is None:
        _formatter_agent = FormatterAgent()
    return _formatter_agent


def parse_formatted_text(formatted_text: str) -> str:
    """Parse formatted text with color tags and convert to HTML for Chainlit.
    
    Args:
        formatted_text: Text with <g>, <y>, <r> tags
        
    Returns:
        HTML string with colored spans
    """
    import re
    
    # Replace tags with HTML spans
    # <g>content</g> -> <span style="background-color: #90EE90; padding: 2px;">content</span>
    formatted_text = re.sub(
        r'<g>(.*?)</g>',
        r'<span style="background-color: #90EE90; padding: 2px; border-radius: 3px;">\1</span>',
        formatted_text,
        flags=re.DOTALL
    )
    
    # <y>content</y> -> <span style="background-color: #FFE4B5; padding: 2px;">content</span>
    formatted_text = re.sub(
        r'<y>(.*?)</y>',
        r'<span style="background-color: #FFE4B5; padding: 2px; border-radius: 3px;">\1</span>',
        formatted_text,
        flags=re.DOTALL
    )
    
    # <r>content</r> -> <span style="background-color: #FFB6C1; padding: 2px;">content</span>
    formatted_text = re.sub(
        r'<r>(.*?)</r>',
        r'<span style="background-color: #FFB6C1; padding: 2px; border-radius: 3px; text-decoration: line-through;">\1</span>',
        formatted_text,
        flags=re.DOTALL
    )
    
    return formatted_text


def format_ba_output(ba_answer_text: str) -> str:
    """Format Business Analysis output for better display in Chainlit.
    
    Args:
        ba_answer_text: Raw markdown text from BA agent
        
    Returns:
        Formatted HTML string with improved styling and sections
    """
    import re
    
    # Validate input - should not be JSON
    if not ba_answer_text or not isinstance(ba_answer_text, str):
        return '<div style="padding: 16px; background: #ffe6e6; border-radius: 8px;"><p style="color: #cc0000;">Error: Invalid input for formatting</p></div>'
    
    # Check if input is JSON (should not be)
    if ba_answer_text.strip().startswith("{") or ba_answer_text.strip().startswith("["):
        # Try to extract BA_answer_text from JSON if present
        try:
            data = json.loads(ba_answer_text)
            if isinstance(data, dict) and "BA_answer_text" in data:
                ba_answer_text = data["BA_answer_text"]
            else:
                return '<div style="padding: 16px; background: #ffe6e6; border-radius: 8px;"><p style="color: #cc0000;">Error: Received JSON instead of markdown text</p></div>'
        except json.JSONDecodeError:
            # Not JSON, continue with original text
            pass
    
    # Parse sections from markdown
    sections = {}
    current_section = None
    current_content = []
    
    lines = ba_answer_text.split('\n')
    
    for line in lines:
        # Detect section headers (### or ##)
        if line.startswith('### '):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.replace('### ', '').strip()
            current_content = []
        elif line.startswith('## '):
            # Main section header
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.replace('## ', '').strip()
            current_content = []
        else:
            if current_section:
                current_content.append(line)
            elif line.strip():  # Only add non-empty lines
                # Content before first section
                if 'introduction' not in sections:
                    sections['introduction'] = []
                sections['introduction'].append(line)
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Section icons and colors mapping
    section_styles = {
        'Requirements Summary': {'icon': '📋', 'color': '#0066cc', 'bg': '#e6f2ff'},
        'Issues Identified': {'icon': '⚠️', 'color': '#ff6600', 'bg': '#fff4e6'},
        'Contradictions and Conflicts': {'icon': '⚡', 'color': '#cc0000', 'bg': '#ffe6e6'},
        'Quality Assessment': {'icon': '📊', 'color': '#0066cc', 'bg': '#e6f2ff'},
        'Specific Improvement Suggestions': {'icon': '💡', 'color': '#009900', 'bg': '#e6ffe6'},
        'Missing Requirements': {'icon': '❌', 'color': '#cc0000', 'bg': '#ffe6e6'},
        'Next Steps Recommendations': {'icon': '🚀', 'color': '#0066cc', 'bg': '#e6f2ff'},
    }
    
    # Build formatted HTML
    html_parts = []
    
    # Format each section
    for section_title, content in sections.items():
        if not content or (isinstance(content, list) and len(content) == 0):
            continue
        
        # Get section style
        style = section_styles.get(section_title, {'icon': '📌', 'color': '#666666', 'bg': '#f5f5f5'})
        
        # Format content
        if isinstance(content, list):
            content = '\n'.join(content).strip()
        
        if not content:
            continue
        
        # Convert markdown to HTML
        # Format bold text
        content_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        
        # Format numbered lists (1., 2., etc.)
        content_html = re.sub(r'^(\d+)\.\s+(.+)$', r'<li>\2</li>', content_html, flags=re.MULTILINE)
        
        # Format bullet points (-)
        content_html = re.sub(r'^-\s+(.+)$', r'<li>\1</li>', content_html, flags=re.MULTILINE)
        
        # Wrap consecutive <li> in <ul>
        # Find groups of <li> tags
        lines_list = content_html.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines_list:
            if line.strip().startswith('<li>'):
                if not in_list:
                    formatted_lines.append('<ul>')
                    in_list = True
                formatted_lines.append(line)
            else:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                if line.strip():
                    # Convert plain text lines to paragraphs
                    if not line.strip().startswith('<'):
                        formatted_lines.append(f'<p style="margin: 8px 0;">{line}</p>')
                    else:
                        formatted_lines.append(line)
        
        if in_list:
            formatted_lines.append('</ul>')
        
        content_html = '\n'.join(formatted_lines)
        
        # Format section
        section_html = f"""
<div style="margin-bottom: 24px; padding: 16px; background: {style['bg']}; border-radius: 8px; border-left: 4px solid {style['color']}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h3 style="margin-top: 0; margin-bottom: 12px; color: {style['color']}; font-size: 18px; font-weight: 600;">
        {style['icon']} {section_title}
    </h3>
    <div style="color: #333; line-height: 1.6; font-size: 14px;">
        {content_html}
    </div>
</div>
"""
        html_parts.append(section_html)
    
    # If no sections found, format as plain markdown
    if not html_parts:
        # Simple markdown to HTML conversion
        formatted = ba_answer_text
        formatted = re.sub(r'### (.+)', r'<h3 style="color: #0066cc; margin-top: 20px;">\1</h3>', formatted)
        formatted = re.sub(r'## (.+)', r'<h2 style="color: #0066cc; margin-top: 24px;">\1</h2>', formatted)
        formatted = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted)
        formatted = re.sub(r'^-\s+(.+)$', r'<li>\1</li>', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'^(\d+)\.\s+(.+)$', r'<li>\2</li>', formatted, flags=re.MULTILINE)
        return f'<div style="line-height: 1.6; padding: 16px;">{formatted}</div>'
    
    return '\n'.join(html_parts)


@cl.on_chat_start
async def start():
    """Called when a new chat session starts."""
    
    # Initialize agent status tracking
    agent_status = {
        "Orchestrator": {
            "name": "Orchestrator",
            "active": False,
            "duration": 0,
            "description": "Routes requests intelligently to appropriate agents"
        },
        "RAG Agent": {
            "name": "RAG Agent",
            "active": False,
            "duration": 0,
            "description": "Retrieves information from Knowledge Base and Knowledge Graph"
        },
        "Ingest Agent": {
            "name": "Ingest Agent",
            "active": False,
            "duration": 0,
            "description": "Processes documents, creates embeddings, and builds knowledge graphs"
        },
        "Business Analysis Agent": {
            "name": "Business Analysis Agent",
            "active": False,
            "duration": 0,
            "description": "Analyzes requirements, detects contradictions, and suggests improvements"
        },
        "Guardrail Agent": {
            "name": "Guardrail Agent",
            "active": False,
            "duration": 0,
            "description": "Ensures content safety and compliance"
        }
    }
    
    # Store agent status in session
    cl.user_session.set("agent_status", agent_status)
    
    # Welcome message with modern UI
    welcome_content = """
# 🚀 Welcome to Multi-Agent RAG System

A **powerful retrieval-augmented generation** system with intelligent agent collaboration.

---

## 📋 Available Flows

### 1️⃣ **Ingestion Flow** 📥
**Process & store documents automatically**

- Upload files (`.txt`, `.docx`, `.pdf`)
- Automatic chunking & embedding
- Build knowledge base (Milvus) & graph (Neo4j)

**Try:** `ingest document`, `upload file`, `process documents`

---

### 2️⃣ **RAG Flow** 🔍 *(Default)*
**Ask questions & get answers**

- Search Knowledge Base (semantic search)
- Query Knowledge Graph (relationships)
- Get comprehensive answers with sources

**Try:** `What is Project Aurora?`, `Tell me about the requirements`

---

### 3️⃣ **Business Analysis Flow** 📊
**Analyze requirements & find issues**

- Detect contradictions & conflicts
- Suggest improvements
- Generate insights

**Try:** `analyze requirements`, `find contradictions`, `improve requirements`

---

<div style="margin-top: 24px; padding: 16px; background: #f8f9fa; border-radius: 8px; border-left: 3px solid #0066cc;">
<strong>💡 Tip:</strong> Just start typing! The system will automatically route your request to the appropriate agent.
</div>
"""
    
    # Create custom element for agent status
    agent_list = list(agent_status.values())
    agent_dashboard = cl.CustomElement(
        name="MultiAgentDashboard",
        props={
            "agents": agent_list,
            "stats": {
                "totalMessages": 0,
                "successRate": 100
            }
        }
    )
    
    await cl.Message(
        content=welcome_content,
        author="System",
        elements=[agent_dashboard]
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Called every time the user sends a message."""
    user_query = message.content
    
    # Route request using orchestrator
    async with cl.Step(name="Orchestrator", type="tool") as orchestrator_step:
        orchestrator_step.input = user_query
        
        await cl.Message(
            content=f"🔀 **Routing request...**\n\n`{user_query}`",
            author="Orchestrator",
        ).send()
        
        try:
            # Route request
            result = route_request(user_query)
            flow = result.get("flow", "unknown")
            agent_name = result.get("agent", "Unknown")
            output = result.get("output", "")
            
            # Update agent status
            import time
            agent_status = cl.user_session.get("agent_status", {})
            if agent_name in agent_status:
                agent_status[agent_name]["active"] = True
                # Track start time for duration calculation
                if "start_time" not in agent_status[agent_name] or agent_status[agent_name]["start_time"] is None:
                    agent_status[agent_name]["start_time"] = time.time()
                agent_status[agent_name]["duration"] = agent_status[agent_name].get("duration", 0)
                cl.user_session.set("agent_status", agent_status)
            
            # Flow icons mapping
            flow_icons = {
                "ingestion": "📥",
                "rag": "🔍",
                "business_analysis": "📊"
            }
            
            flow_icon = flow_icons.get(flow, "🔀")
            
            orchestrator_step.output = f"Routed to: {agent_name} ({flow} flow)"
            
            # Update agent dashboard with current durations (including active time)
            agent_list = []
            for agent_name_key, agent_data in agent_status.items():
                agent_info = agent_data.copy()
                # Ensure all required fields are present
                agent_info["name"] = agent_data.get("name", agent_name_key)
                agent_info["active"] = agent_data.get("active", False)
                agent_info["duration"] = agent_data.get("duration", 0)
                agent_info["description"] = agent_data.get("description", "")
                
                # Calculate base duration (already accumulated) and start_time for active agents
                if agent_info.get("active") and agent_info.get("start_time"):
                    # Store base duration (already accumulated from previous runs)
                    agent_info["base_duration"] = agent_info.get("duration", 0)
                    # Keep start_time in seconds for JSX to calculate elapsed time
                    agent_info["start_time"] = agent_info.get("start_time")
                    # Duration will be calculated client-side, but show current value
                    # For now, calculate elapsed and add to base
                    import time
                    if agent_info["start_time"]:
                        elapsed_ms = int((time.time() - agent_info["start_time"]) * 1000)
                        agent_info["duration"] = agent_info.get("base_duration", agent_info.get("duration", 0)) + elapsed_ms
                else:
                    # For inactive agents, use stored duration
                    agent_info["duration"] = agent_info.get("duration", 0)
                    agent_info["base_duration"] = agent_info.get("duration", 0)
                    agent_info["start_time"] = None
                agent_list.append(agent_info)
            
            # Don't send dashboard with every routing message to reduce payload
            await cl.Message(
                content=f"{flow_icon} **Routed to:** `{agent_name}`\n"
                       f"**Flow:** `{flow}`\n"
                       f"\n⚡ Processing...",
                author="Orchestrator"
            ).send()
            
        except Exception as e:
            await cl.Message(
                content=f"Error routing request: {str(e)}",
                author="Orchestrator Agent",
            ).send()
            return
    
    # Handle different flows
    try:
        if flow == "ingestion":
            # Ingestion Flow
            async with cl.Step(name="Ingest Agent", type="tool") as ingest_step:
                ingest_step.input = user_query
                
                # Update agent status
                import time
                agent_status = cl.user_session.get("agent_status", {})
                if "Ingest Agent" in agent_status:
                    agent_status["Ingest Agent"]["active"] = True
                    # Track start time for duration calculation
                    if "start_time" not in agent_status["Ingest Agent"] or agent_status["Ingest Agent"]["start_time"] is None:
                        agent_status["Ingest Agent"]["start_time"] = time.time()
                cl.user_session.set("agent_status", agent_status)
                
                # Output from orchestrator is the ingest agent's response
                if output:
                    # Try to parse as JSON if possible
                    try:
                        output_data = json.loads(output)
                        formatted_output = json.dumps(output_data, indent=2, ensure_ascii=False)
                    except:
                        formatted_output = str(output)
                else:
                    formatted_output = "Ingestion completed successfully."
                
                ingest_step.output = formatted_output
                
                # Mark agent as inactive after processing
                if "Ingest Agent" in agent_status:
                    agent_status["Ingest Agent"]["active"] = False
                    # Calculate duration
                    if agent_status["Ingest Agent"].get("start_time"):
                        import time
                        elapsed = time.time() - agent_status["Ingest Agent"]["start_time"]
                        agent_status["Ingest Agent"]["duration"] = agent_status["Ingest Agent"].get("duration", 0) + int(elapsed * 1000)
                        agent_status["Ingest Agent"]["start_time"] = None
                cl.user_session.set("agent_status", agent_status)
                
                # Create updated dashboard with current durations
                agent_list = []
                for agent_name_key, agent_data in agent_status.items():
                    agent_info = agent_data.copy()
                    # Ensure all required fields are present
                    agent_info["name"] = agent_data.get("name", agent_name_key)
                    agent_info["active"] = agent_data.get("active", False)
                    agent_info["duration"] = agent_data.get("duration", 0)
                    agent_info["description"] = agent_data.get("description", "")
                    
                    # Calculate base duration (already accumulated) and start_time for active agents
                    if agent_info.get("active") and agent_info.get("start_time"):
                        # Store base duration (already accumulated from previous runs)
                        agent_info["base_duration"] = agent_info.get("duration", 0)
                        # Keep start_time in seconds for JSX to calculate elapsed time
                        agent_info["start_time"] = agent_info.get("start_time")
                        # Calculate current duration for display
                        import time
                        if agent_info["start_time"]:
                            elapsed_ms = int((time.time() - agent_info["start_time"]) * 1000)
                            agent_info["duration"] = agent_info.get("base_duration", agent_info.get("duration", 0)) + elapsed_ms
                    else:
                        # For inactive agents, use stored duration
                        agent_info["duration"] = agent_info.get("duration", 0)
                        agent_info["base_duration"] = agent_info.get("duration", 0)
                        agent_info["start_time"] = None
                    agent_list.append(agent_info)
                
                # Update stats
                stats = cl.user_session.get("stats", {})
                stats["totalMessages"] = stats.get("totalMessages", 0) + 1
                cl.user_session.set("stats", stats)
                
                    # Truncate very long output
                max_output_length = 30000
                display_output = formatted_output
                if len(display_output) > max_output_length:
                    display_output = display_output[:max_output_length] + "\n\n... (output truncated)"
                    
                await cl.Message(
                    content=f"## 📥 Ingestion Result\n\n```json\n{display_output}\n```",
                    author="Ingest Agent"
                ).send()
                    
                    # Send dashboard update separately
                agent_dashboard = cl.CustomElement(
                    name="MultiAgentDashboard",
                    props={
                        "agents": agent_list,
                        "stats": stats
                    }
                )
                await cl.Message(
                    content="",
                    elements=[agent_dashboard]
                ).send()
        
        elif flow == "rag":
            # RAG Flow
            rag_output = result.get("rag_output")
            
            # Update agent status
            import time
            agent_status = cl.user_session.get("agent_status", {})
            if "RAG Agent" in agent_status:
                agent_status["RAG Agent"]["active"] = True
                # Track start time for duration calculation
                if "start_time" not in agent_status["RAG Agent"] or agent_status["RAG Agent"]["start_time"] is None:
                    agent_status["RAG Agent"]["start_time"] = time.time()
            cl.user_session.set("agent_status", agent_status)
            
            # Step 1: Show RAG results
            async with cl.Step(name="RAG Agent", type="tool") as rag_step:
                rag_step.input = user_query
                
                if isinstance(rag_output, RAGOutput):
                    rag_step.output = f"Found {len(rag_output.KB)} documents in KB, {len(rag_output.KG)} nodes in KG"
                    
                    await cl.Message(
                        content=f"**📚 Retrieved:**\n"
                               f"- **KB:** `{len(rag_output.KB)}` docs\n"
                               f"- **KG:** `{len(rag_output.KG)}` nodes",
                        author="RAG Agent",
                    ).send()
                else:
                    rag_step.output = "RAG query completed"
                
                # Step 2: Call Guardrail with RAG output
                if isinstance(rag_output, RAGOutput) and rag_output.answer_text:
                    # Mark RAG Agent as inactive
                    if "RAG Agent" in agent_status:
                        agent_status["RAG Agent"]["active"] = False
                        # Calculate duration
                        if agent_status["RAG Agent"].get("start_time"):
                            import time
                            elapsed = time.time() - agent_status["RAG Agent"]["start_time"]
                            agent_status["RAG Agent"]["duration"] = agent_status["RAG Agent"].get("duration", 0) + int(elapsed * 1000)
                            agent_status["RAG Agent"]["start_time"] = None
                    cl.user_session.set("agent_status", agent_status)
                    
                    async with cl.Step(name="Guardrail Agent", type="tool") as guardrail_step:
                        guardrail_step.input = f"RAG answer: {rag_output.answer_text[:100]}..."
                        
                        # Update agent status
                        agent_status = cl.user_session.get("agent_status", {})
                        if "Guardrail Agent" in agent_status:
                            agent_status["Guardrail Agent"]["active"] = True
                            if "start_time" not in agent_status["Guardrail Agent"] or agent_status["Guardrail Agent"]["start_time"] is None:
                                import time
                                agent_status["Guardrail Agent"]["start_time"] = time.time()
                        cl.user_session.set("agent_status", agent_status)
                        
                        try:
                            # Prepare KB and KG as comma-separated strings
                            kb_str = ",".join([str(ref.get("id", "")) if isinstance(ref, dict) else str(ref) for ref in rag_output.KB])
                            kg_str = ",".join([str(ref.get("id", "")) if isinstance(ref, dict) else str(ref) for ref in rag_output.KG])
                            
                            # Call Guardrail
                            guardrail_result_json = evaluate_content(
                                user_input=user_query,
                                answer_text=rag_output.answer_text,
                                source="",  # RAG doesn't have web sources
                                kb=kb_str,
                                kg=kg_str
                            )
                            guardrail_data = json.loads(guardrail_result_json)
                            guardrail_status = guardrail_data.get("guardrail_status", "REVIEW_REQUIRED")
                            
                            guardrail_step.output = f"Status: {guardrail_status}"
                            
                            # Mark Guardrail Agent as inactive
                            if "Guardrail Agent" in agent_status:
                                agent_status["Guardrail Agent"]["active"] = False
                                if agent_status["Guardrail Agent"].get("start_time"):
                                    import time
                                    elapsed = time.time() - agent_status["Guardrail Agent"]["start_time"]
                                    agent_status["Guardrail Agent"]["duration"] = agent_status["Guardrail Agent"].get("duration", 0) + int(elapsed * 1000)
                                    agent_status["Guardrail Agent"]["start_time"] = None
                            cl.user_session.set("agent_status", agent_status)
                            
                        except Exception as e:
                            logger.error(f"Error in Guardrail Agent: {e}", exc_info=True)
                            guardrail_status = "REVIEW_REQUIRED"
                            guardrail_data = {"guardrail_status": guardrail_status, "issues_found": [], "summary": f"Error: {str(e)}"}
                            guardrail_step.output = f"Error: {str(e)}"
                    
                    # Step 3: Call Formatter with RAG output
                    async with cl.Step(name="Formatter Agent", type="tool") as formatter_step:
                        formatter_step.input = f"Content to format: {rag_output.answer_text[:100]}..."
                        
                        try:
                            formatter_agent = get_formatter_agent()
                            formatted_text = formatter_agent.format_with_tags(rag_output.answer_text)
                            
                            formatter_step.output = "Formatting completed"
                            
                            # Parse formatted text for HTML display
                            parsed_html = parse_formatted_text(formatted_text)
                            
                            # Truncate very long content to prevent payload errors
                            max_content_length = 50000  # Limit to ~50KB
                            display_html = parsed_html
                            if len(display_html) > max_content_length:
                                display_html = display_html[:max_content_length] + "\n\n... (content truncated)"
                            
                            # Display formatted result (without dashboard to reduce payload size)
                            await cl.Message(
                                content=display_html,
                                author="RAG Agent"
                            ).send()
                            
                            # Send dashboard separately and less frequently
                            import time
                            agent_list = []
                            current_time = time.time()
                            for agent_name_key, agent_data in agent_status.items():
                                agent_info = {
                                    "name": agent_data.get("name", agent_name_key),
                                    "active": agent_data.get("active", False),
                                    "duration": agent_data.get("duration", 0),
                                    "description": agent_data.get("description", "")[:100]  # Limit description length
                                }
                                
                                if agent_info["active"] and agent_data.get("start_time"):
                                    elapsed_ms = int((current_time - agent_data["start_time"]) * 1000)
                                    agent_info["duration"] = agent_data.get("duration", 0) + elapsed_ms
                                else:
                                    agent_info["duration"] = agent_data.get("duration", 0)
                                agent_list.append(agent_info)
                            
                            # Update stats
                            stats = cl.user_session.get("stats", {})
                            stats["totalMessages"] = stats.get("totalMessages", 0) + 1
                            cl.user_session.set("stats", stats)
                            
                            # Send dashboard update separately
                            agent_dashboard = cl.CustomElement(
                                name="MultiAgentDashboard",
                                props={
                                    "agents": agent_list,
                                    "stats": stats
                                }
                            )
                            await cl.Message(
                                content="",
                                elements=[agent_dashboard]
                            ).send()
                            
                            # Display Guardrail status if not approved
                            if guardrail_status != "APPROVED":
                                await cl.Message(
                                    content=f"## 🛡️ Guardrail Status: {guardrail_status}\n\n{guardrail_data.get('summary', '')}",
                                    author="Guardrail Agent",
                            ).send()
                            
                            # Display references
                            if rag_output.KB or rag_output.KG:
                                ref_text = "## 📎 References\n\n"
                                if rag_output.KB:
                                    ref_text += f"### 📚 Knowledge Base ({len(rag_output.KB)} docs)\n\n"
                                    for ref in rag_output.KB[:10]:  # Limit to top 10
                                        if isinstance(ref, dict):
                                            doc_id = ref.get("id", "N/A")
                                            # Just show doc_id without score
                                            ref_text += f"- `{doc_id}`\n"
                                        else:
                                            # Backward compatibility: if it's still a string
                                            ref_text += f"- `{ref}`\n"
                                    ref_text += "\n"
                                if rag_output.KG:
                                    ref_text += f"### 🕸️ Knowledge Graph ({len(rag_output.KG)} nodes)\n\n"
                                    for ref in rag_output.KG[:10]:
                                        if isinstance(ref, dict):
                                            node_id = ref.get("id", "N/A")
                                            # Just show node_id without score
                                            ref_text += f"- `{node_id}`\n"
                                        else:
                                            # Backward compatibility: if it's still a string
                                            ref_text += f"- `{ref}`\n"
                                
                                await cl.Message(
                                    content=ref_text,
                                    author="RAG Agent",
                                ).send()
                                
                        except Exception as e:
                            logger.error(f"Error in Formatter Agent: {e}", exc_info=True)
                            # Fallback: display unformatted text
                            await cl.Message(
                                content=rag_output.answer_text,
                                author="RAG Agent",
                                elements=[agent_dashboard]
                            ).send()
                
                # Fallback: Show raw output if not RAGOutput
                elif output:
                    await cl.Message(
                        content=output,
                        author="RAG Agent",
                    ).send()
        
        elif flow == "business_analysis":
            # Business Analysis Flow
            async with cl.Step(name="Business Analysis Agent", type="tool") as ba_step:
                ba_step.input = user_query
                
                # Update agent status
                import time
                agent_status = cl.user_session.get("agent_status", {})
                if "Business Analysis Agent" in agent_status:
                    agent_status["Business Analysis Agent"]["active"] = True
                    # Track start time for duration calculation
                    if "start_time" not in agent_status["Business Analysis Agent"] or agent_status["Business Analysis Agent"]["start_time"] is None:
                        agent_status["Business Analysis Agent"]["start_time"] = time.time()
                cl.user_session.set("agent_status", agent_status)
                
                # Parse output (should be JSON string)
                try:
                    output_data = json.loads(output)
                    # Set step output to a simple text summary (not JSON)
                    ba_step.output = f"✓ Analysis completed successfully"
                    
                    # Display analysis result
                    ba_answer = output_data.get("BA_answer_text", "")
                    kb_refs = output_data.get("KB", [])
                    kg_refs = output_data.get("KG", [])
                    
                    # Ensure ba_answer is a string and not JSON
                    if not ba_answer or not isinstance(ba_answer, str):
                        ba_answer = "No analysis result available"
                    elif ba_answer.strip().startswith("{") and '"BA_answer_text"' in ba_answer:
                        # If somehow we got the full JSON, try to parse it again
                        try:
                            nested_data = json.loads(ba_answer)
                            ba_answer = nested_data.get("BA_answer_text", ba_answer)
                        except:
                            pass
                    
                    # Mark agent as inactive
                    if "Business Analysis Agent" in agent_status:
                        agent_status["Business Analysis Agent"]["active"] = False
                        # Calculate duration
                        if agent_status["Business Analysis Agent"].get("start_time"):
                            import time
                            elapsed = time.time() - agent_status["Business Analysis Agent"]["start_time"]
                            agent_status["Business Analysis Agent"]["duration"] = agent_status["Business Analysis Agent"].get("duration", 0) + int(elapsed * 1000)
                            agent_status["Business Analysis Agent"]["start_time"] = None
                    cl.user_session.set("agent_status", agent_status)
                    
                    # Create updated dashboard with current durations
                    import time
                    agent_list = []
                    current_time = time.time()
                    for agent_name, agent_data in agent_status.items():
                        agent_info = agent_data.copy()
                        # Calculate current duration if active
                        if agent_info.get("active") and agent_info.get("start_time"):
                            elapsed_ms = int((current_time - agent_info["start_time"]) * 1000)
                            agent_info["duration"] = agent_info.get("duration", 0) + elapsed_ms
                        else:
                            agent_info["duration"] = agent_info.get("duration", 0)
                        agent_list.append(agent_info)
                    
                    # Update stats
                    stats = cl.user_session.get("stats", {})
                    stats["totalMessages"] = stats.get("totalMessages", 0) + 1
                    cl.user_session.set("stats", stats)
                    
                    # Display BA result as markdown (Chainlit supports markdown natively)
                    # Ensure ba_answer is valid markdown text, not JSON
                    if ba_answer and ba_answer.strip() and not ba_answer.strip().startswith("{") and '"BA_answer_text"' not in ba_answer:
                        # Truncate very long content to prevent payload errors
                        max_content_length = 100000  # Limit to ~100KB for markdown
                        display_text = ba_answer
                        if len(display_text) > max_content_length:
                            display_text = display_text[:max_content_length] + "\n\n... (content truncated)"
                    else:
                        display_text = "⚠️ Error: Unable to parse Business Analysis output"
                    
                    # Send dashboard update separately
                    agent_dashboard = cl.CustomElement(
                        name="MultiAgentDashboard",
                        props={
                            "agents": agent_list,
                            "stats": stats
                        }
                    )
                    
                    # Display BA result as markdown (Chainlit will render it)
                    await cl.Message(
                        content=display_text,
                        author="Business Analysis Agent"
                    ).send()
                    
                    await cl.Message(
                        content="",
                        elements=[agent_dashboard]
                    ).send()
                    
                    # Step 2: Call Suggestion_agent with BA output
                    async with cl.Step(name="Suggestion Agent", type="tool") as suggestion_step:
                        suggestion_step.input = f"BA answer: {ba_answer[:100]}..."
                        
                        # Update agent status
                        agent_status = cl.user_session.get("agent_status", {})
                        if "Suggestion Agent" in agent_status:
                            agent_status["Suggestion Agent"]["active"] = True
                            if "start_time" not in agent_status["Suggestion Agent"] or agent_status["Suggestion Agent"]["start_time"] is None:
                                import time
                                agent_status["Suggestion Agent"]["start_time"] = time.time()
                        cl.user_session.set("agent_status", agent_status)
                        
                        try:
                            # Prepare KB and KG as comma-separated strings
                            kb_str = ",".join([str(ref) if isinstance(ref, str) else str(ref.get("id", ref.get("doc_id", ""))) for ref in kb_refs])
                            kg_str = ",".join([str(ref) if isinstance(ref, str) else str(ref.get("id", ref.get("node_id", ""))) for ref in kg_refs])
                            
                            # Call Suggestion_agent
                            suggestion_result_json = generate_suggestion(
                                ba_answer_text=ba_answer,
                                kb=kb_str,
                                kg=kg_str
                            )
                            suggestion_data = json.loads(suggestion_result_json)
                            suggestion_text = suggestion_data.get("answer_text", "No suggestions generated")
                            suggestion_sources = suggestion_data.get("source", [])
                            
                            suggestion_step.output = f"Generated suggestions with {len(suggestion_sources)} sources"
                            
                            # Mark Suggestion Agent as inactive
                            if "Suggestion Agent" in agent_status:
                                agent_status["Suggestion Agent"]["active"] = False
                                if agent_status["Suggestion Agent"].get("start_time"):
                                    import time
                                    elapsed = time.time() - agent_status["Suggestion Agent"]["start_time"]
                                    agent_status["Suggestion Agent"]["duration"] = agent_status["Suggestion Agent"].get("duration", 0) + int(elapsed * 1000)
                                    agent_status["Suggestion Agent"]["start_time"] = None
                            cl.user_session.set("agent_status", agent_status)
                            
                        except Exception as e:
                            logger.error(f"Error in Suggestion Agent: {e}", exc_info=True)
                            suggestion_text = f"Error generating suggestions: {str(e)}"
                            suggestion_data = {"answer_text": suggestion_text, "source": [], "KB": kb_refs, "KG": kg_refs}
                            suggestion_sources = []
                            suggestion_step.output = f"Error: {str(e)}"
                    
                    # Step 3: Call Guardrail with Suggestion output
                    async with cl.Step(name="Guardrail Agent", type="tool") as guardrail_step:
                        guardrail_step.input = f"Suggestion: {suggestion_text[:100]}..."
                        
                        # Update agent status
                        agent_status = cl.user_session.get("agent_status", {})
                        if "Guardrail Agent" in agent_status:
                            agent_status["Guardrail Agent"]["active"] = True
                            if "start_time" not in agent_status["Guardrail Agent"] or agent_status["Guardrail Agent"]["start_time"] is None:
                                import time
                                agent_status["Guardrail Agent"]["start_time"] = time.time()
                        cl.user_session.set("agent_status", agent_status)
                        
                        try:
                            # Prepare sources as comma-separated string
                            source_str = ",".join(suggestion_sources) if isinstance(suggestion_sources, list) else str(suggestion_sources)
                            
                            # Call Guardrail
                            guardrail_result_json = evaluate_content(
                                user_input=user_query,
                                answer_text=suggestion_text,
                                source=source_str,
                                kb=kb_str,
                                kg=kg_str
                            )
                            guardrail_data = json.loads(guardrail_result_json)
                            guardrail_status = guardrail_data.get("guardrail_status", "REVIEW_REQUIRED")
                            
                            guardrail_step.output = f"Status: {guardrail_status}"
                            
                            # Mark Guardrail Agent as inactive
                            if "Guardrail Agent" in agent_status:
                                agent_status["Guardrail Agent"]["active"] = False
                                if agent_status["Guardrail Agent"].get("start_time"):
                                    import time
                                    elapsed = time.time() - agent_status["Guardrail Agent"]["start_time"]
                                    agent_status["Guardrail Agent"]["duration"] = agent_status["Guardrail Agent"].get("duration", 0) + int(elapsed * 1000)
                                    agent_status["Guardrail Agent"]["start_time"] = None
                            cl.user_session.set("agent_status", agent_status)
                            
                        except Exception as e:
                            logger.error(f"Error in Guardrail Agent: {e}", exc_info=True)
                            guardrail_status = "REVIEW_REQUIRED"
                            guardrail_data = {"guardrail_status": guardrail_status, "issues_found": [], "summary": f"Error: {str(e)}"}
                            guardrail_step.output = f"Error: {str(e)}"
                    
                    # Step 4: Call Formatter with Guardrail output
                    async with cl.Step(name="Formatter Agent", type="tool") as formatter_step:
                        formatter_step.input = f"Content to format: {suggestion_text[:100]}..."
                        
                        try:
                            formatter_agent = get_formatter_agent()
                            formatted_text = formatter_agent.format_with_tags(suggestion_text)
                            
                            formatter_step.output = "Formatting completed"
                            
                            # Display suggestion result as markdown
                            # Truncate very long content to prevent payload errors
                            max_content_length = 100000  # Limit to ~100KB for markdown
                            display_text = suggestion_text
                            if len(display_text) > max_content_length:
                                display_text = display_text[:max_content_length] + "\n\n... (content truncated)"
                            
                            # Display formatted result as markdown (Chainlit will render it)
                            await cl.Message(
                                content=display_text,
                                author="Business Analysis Agent"
                            ).send()
                            
                            # Send dashboard separately and less frequently
                            import time
                            agent_list = []
                            current_time = time.time()
                            for agent_name_key, agent_data in agent_status.items():
                                agent_info = {
                                    "name": agent_data.get("name", agent_name_key),
                                    "active": agent_data.get("active", False),
                                    "duration": agent_data.get("duration", 0),
                                    "description": agent_data.get("description", "")[:100]  # Limit description length
                                }
                                
                                if agent_info["active"] and agent_data.get("start_time"):
                                    elapsed_ms = int((current_time - agent_data["start_time"]) * 1000)
                                    agent_info["duration"] = agent_data.get("duration", 0) + elapsed_ms
                                else:
                                    agent_info["duration"] = agent_data.get("duration", 0)
                                agent_list.append(agent_info)
                            
                            # Update stats
                            stats = cl.user_session.get("stats", {})
                            stats["totalMessages"] = stats.get("totalMessages", 0) + 1
                            cl.user_session.set("stats", stats)
                            
                            # Send dashboard update separately
                            agent_dashboard = cl.CustomElement(
                                name="MultiAgentDashboard",
                                props={
                                    "agents": agent_list,
                                    "stats": stats
                                }
                            )
                            await cl.Message(
                                content="",
                                elements=[agent_dashboard]
                            ).send()
                            
                            # Display Guardrail status if not approved
                            if guardrail_status != "APPROVED":
                                await cl.Message(
                                    content=f"## 🛡️ Guardrail Status: {guardrail_status}\n\n{guardrail_data.get('summary', '')}",
                                    author="Guardrail Agent",
                                ).send()
                            
                            # Display references
                            if kb_refs or kg_refs:
                                ref_text = "## 📎 References\n\n"
                                if kb_refs:
                                    ref_text += f"### 📚 Knowledge Base ({len(kb_refs)} docs)\n\n"
                                    for ref in kb_refs[:10]:
                                        if isinstance(ref, dict):
                                            doc_id = ref.get("id", ref.get("doc_id", "N/A"))
                                            ref_text += f"- `{doc_id}`\n"
                                        else:
                                            ref_text += f"- `{ref}`\n"
                                    ref_text += "\n"
                                if kg_refs:
                                    ref_text += f"### 🕸️ Knowledge Graph ({len(kg_refs)} nodes)\n\n"
                                    for ref in kg_refs[:10]:
                                        if isinstance(ref, dict):
                                            node_id = ref.get("id", ref.get("node_id", "N/A"))
                                            ref_text += f"- `{node_id}`\n"
                                        else:
                                            ref_text += f"- `{ref}`\n"
                                
                                await cl.Message(
                                    content=ref_text,
                                    author="Business Analysis Agent",
                                ).send()
                            
                            # Display sources if available
                            if suggestion_sources:
                                sources_text = "## 🌐 Sources\n\n"
                                for source in suggestion_sources[:5]:
                                    sources_text += f"- {source}\n"
                                await cl.Message(
                                    content=sources_text,
                                    author="Suggestion Agent",
                                ).send()
                                
                        except Exception as e:
                            logger.error(f"Error in Formatter Agent: {e}", exc_info=True)
                            # Fallback: display BA output as markdown
                            display_text = suggestion_text if suggestion_text else ba_answer
                            max_content_length = 100000
                            if len(display_text) > max_content_length:
                                display_text = display_text[:max_content_length] + "\n\n... (content truncated)"
                            await cl.Message(
                                content=display_text,
                                author="Business Analysis Agent"
                        ).send()
                except json.JSONDecodeError:
                    # Not JSON, display as markdown
                    ba_step.output = "✓ Analysis completed"
                    # Truncate very long content
                    max_content_length = 100000
                    display_text = output
                    if len(display_text) > max_content_length:
                        display_text = display_text[:max_content_length] + "\n\n... (content truncated)"
                    await cl.Message(
                        content=f"## 📊 Business Analysis Result\n\n{display_text}",
                        author="Business Analysis Agent",
                    ).send()
        
        else:
            # Unknown flow or error
            await cl.Message(
                content=f"**Result:**\n\n{output}",
                author=agent_name,
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"Error processing request: {str(e)}\n\n**Raw Output:**\n\n{output}",
            author="Orchestrator Agent",
        ).send()


if __name__ == "__main__":
    # Run Chainlit app
    # This will be handled by chainlit command
    # To run: chainlit run app.py --port 7777
    pass
