"""Test RAG Agent with Pydantic JSON output."""

import json
import sys
from pathlib import Path

# Add src directory to path so imports work correctly
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agents.rag_agent import run_rag_query

# Run RAG query and get structured output
output = run_rag_query(
    query="What features does Project Alpha include?",
    top_k=15,
)

# Print as JSON
print(output)
print("\n\n")
print(output.model_dump())
print("\n\n")
print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))