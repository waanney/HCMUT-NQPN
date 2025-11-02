# Test RAG Agent

## Prerequisites

1. **Milvus** running (default: `localhost:19530`)
   - Make sure Milvus is running and accessible
   - Collection `gsoft_docs` should exist (will be created automatically if not)

2. **Neo4j** running (default: `bolt://localhost:7687`)
   - Make sure Neo4j is running
   - Database should have Project, Requirement, UserStory nodes

3. **Environment Variables** (optional):
   - `OPENAI_API_KEY`: For LLM integration (optional, agent will work without it)

## Quick Test

Run the simple test script:

```bash
python examples/test_rag_simple.py
```

## Full Test Options

Run with different modes:

```bash
# Basic test (default)
python examples/test_rag_agent.py --mode basic

# Interactive mode
python examples/test_rag_agent.py --mode interactive

# Test Milvus only
python examples/test_rag_agent.py --mode milvus

# Test Neo4j only
python examples/test_rag_agent.py --mode neo4j
```

## What Gets Tested

1. **Agent Initialization**: Tests connection to Milvus and Neo4j
2. **Milvus Search**: Searches document chunks in Knowledge Base
3. **Neo4j Search**: Searches Project, Requirement, UserStory nodes in Knowledge Graph
4. **LLM Integration**: Generates answers from retrieved context
5. **Response Format**: Validates response structure with text and references

## Expected Output

- **Text**: Generated answer from LLM
- **References**: 
  - `KB`: Array of Milvus document references
  - `KG`: Array of Neo4j node references with relationships

## Troubleshooting

1. **Connection Errors**: Check if Milvus and Neo4j are running
2. **No Results**: Make sure you have data in both Milvus and Neo4j
3. **LLM Errors**: Set `OPENAI_API_KEY` environment variable if using LLM

