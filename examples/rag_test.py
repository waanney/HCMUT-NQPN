import sys
from pathlib import Path

# Add src directory to path so imports work correctly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agents.rag_agent import RAGAgent

agent = RAGAgent()
response = agent.query("Your question here")
print(response.text)
print(response.references["KB"])  # KB references
print(response.references["KG"])  # KG references