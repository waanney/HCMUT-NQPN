"""Test Ingest Agent with Runner."""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 1. Import Runner from openai-agents package FIRST (before adding src to path)
# This ensures we get Runner from the installed package, not local agents module
_temp_agents = sys.modules.pop('agents', None)
try:
    from agents import Runner  # This imports from openai-agents package
except ImportError:
    import importlib
    agents_module = importlib.import_module("agents")
    Runner = getattr(agents_module, "Runner", None)
    if Runner is None:
        raise ImportError("Could not import Runner from openai-agents package")
finally:
    # Don't restore agents module yet - we need to clear it for local imports
    pass

# 2. Now add src directory to path for local imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# 3. Clear agents module from sys.modules so we can import from local src/agents
if 'agents' in sys.modules:
    del sys.modules['agents']
if 'agents.ingest_agent' in sys.modules:
    del sys.modules['agents.ingest_agent']

# 4. Import from local agents module (after src is in path and modules cleared)
from agents.ingest_agent import create_ingest_agent

# Tạo agent
agent = create_ingest_agent()

# Sample data for testing
sample_documents = [
    {"id": "doc1", "text": "Sample document text about Project Alpha", "source": "test"},
    {"id": "doc2", "text": "Another document about requirements", "source": "test"}
]

sample_projects = [
    {
        "project_id": "PROJ-001",
        "name": "Project Alpha",
        "description": "Sample project for testing",
        "status": "active",
        "stakeholders": ["Product Owner", "Developer"]
    }
]

sample_requirements = [
    {
        "req_id": "REQ-001",
        "title": "User Authentication",
        "description": "System must provide secure user authentication",
        "type": "functional",
        "priority": "high",
        "status": "draft",
        "project_id": "PROJ-001"
    }
]

sample_user_stories = [
    {
        "story_id": "US-001",
        "title": "User Registration",
        "as_a": "new customer",
        "i_want": "to create an account",
        "so_that": "I can make purchases",
        "priority": "high",
        "status": "backlog",
        "project_id": "PROJ-001"
    }
]

# Sử dụng với Runner (synchronous version) - Tạo cả Milvus và Neo4j JSON
import json

milvus_input = f"Create a Milvus JSON file with these documents: {json.dumps(sample_documents)}"
neo4j_input = f"Create a Neo4j JSON file with projects: {json.dumps(sample_projects)}, requirements: {json.dumps(sample_requirements)}, user_stories: {json.dumps(sample_user_stories)}"

full_input = f"""Please create both JSON files:

1. {milvus_input}

2. {neo4j_input}
"""

print("Running Ingest Agent to create both Milvus and Neo4j JSON files...")
result = Runner.run_sync(
    starting_agent=agent,
    input=full_input
)
print("\nAgent Response:")
print(result.final_output)