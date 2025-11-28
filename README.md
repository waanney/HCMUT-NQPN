# sAInapse Multi-Agent RAG Platform

A collaborative workspace for business analysts and solution engineers to ingest documents, detect requirement conflicts, and generate implementation artifacts. The platform chains multiple specialized agents (Ingest, BA, Suggestion, RAG, CodeGen, Formatter, Guardrail…) so that Retrieval-Augmented Generation and knowledge-graph reasoning stay aligned throughout the delivery cycle.

## High-Level Architecture

1. **Input Data** – Users provide requirements, specs, and reference documents. The Orchestrator inspects the payload and chooses the right workflow.
2. **Main Processor** – `Ingest_Agent` chunks documents, builds embeddings, and writes to Milvus (vector DB) and Neo4j (knowledge graph). Milvus powers semantic search, while Neo4j stores Projects, Requirements, and User Stories with their relationships.
3. **Deconfliction & Generation** – `BA_Agent` plus `Suggestion Agent` flag duplicates or contradictions. `CodeGen Agent` and `APIGen Agent` can emit code snippets, API blueprints, or remediation steps.
4. **Postprocessor** – `Formatter Agent` applies `<g>/<y>/<r>` tags to highlight add/modify/delete suggestions, and `Guardrail Agent` enforces safety policies before responses reach the Chainlit UI.

`RAG_Agent` is callable at any stage to merge Milvus (KB) and Neo4j (KG) evidence inside the same answer.

## Key Components

- **Orchestrator Agent** (`src/agents/orchestrator_agent.py`) – routes every request into the proper pipeline.
- **Ingest Agent** (`src/agents/ingest_agent.py`) – handles chunking, embedding, and persistence across Milvus and Neo4j.
- **Business Analysis & Suggestion Agents** – analyze requirements, dependencies, and remediation steps.
- **RAG Agent** (`src/agents/rag_agent.py`) – fuses vector and graph retrieval results.
- **CodeGen / APIGen / Web Generator Agents** – create PoC code, API skeletons, and UI assets.
- **Formatter + Guardrail Agents** – normalize outbound messages and enforce compliance.
- **Chainlit UI** (`app.py`) – interactive dashboard that exposes agent status and conversations.

## System Requirements

- Python 3.12+
- Docker (recommended) for Milvus and Neo4j
- Optional GPU when using `intfloat/multilingual-e5-large`
- Core environment variables:
  - `OPENAI_API_KEY` (or compatible provider key)
  - `MILVUS_URI`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` if you override defaults

## Installation & Startup

### 1. Bring up data services

```bash
cd /home/waanne/Documents/02_Projects/sAInapse-Seal-Hackathon/docker
docker network create milvus
docker compose up -d
```

- Neo4j Browser: `http://localhost:7474` (neo4j/password123)  
- Milvus Attu UI: `http://localhost:8000`  
More details live in `examples/README_RUN_SERVICES.md`.

### 2. Install dependencies

```bash
cd /home/waanne/Documents/02_Projects/sAInapse-Seal-Hackathon
pip install -e .
```

`uv` or `pipx` work the same if you prefer isolated environments.

### 3. Launch Chainlit UI

```bash
export OPENAI_API_KEY=sk-***
chainlit run app.py --port 7777
```

Keep `.chainlit/config.toml` with `unsafe_allow_html = true` so Formatter Agent colors render correctly (see `FORMATTER_README.md`).

## Typical Workflow

1. **Ingest documents** via the UI or `python examples/create_mock_data.py`.
2. **Monitor agents** in Chainlit to see Ingest/RAG/BA progress live.
3. **Ask questions or submit stories**; the Orchestrator will pick the right agent mix.
4. **Generate insights/code** using CodeGen/APIGen plus Suggestion Agent for next actions.
5. **Review formatted output** with color-coded edits; Guardrail validates content before final delivery.

## Quick Test Scripts

- `python examples/test_rag_simple.py` – smoke test for Milvus + Neo4j connectivity.
- `python examples/test_rag_agent.py --mode interactive` – chat directly with the RAG Agent.
- `python examples/clear_neo4j_data.py` & `examples/recreate_milvus_collection.py` – cleanup utilities.
- Additional diagnostics (embedder, GPT-5, ingest, etc.) are inside `examples/`.

## Repository Layout

- `app.py` – Chainlit application entrypoint.
- `src/agents/` – agent implementations and helper tools.
- `src/data_pipeline/` – chunker and embedder logic.
- `src/db/` – Milvus, Neo4j, and Redis clients.
- `src/tools/` – callable tool functions (context builder, formatter, web search…).
- `docker/` – docker-compose manifests for infrastructure.
- `examples/` – mock data generation, tests, debugging scripts.
- `public/` – Chainlit custom elements (React) like the multi-agent dashboard.
- `fixed_website/` – static site used in architecture demos.

## Supporting Docs

- `chainlit.md` – UI overview.
- `FORMATTER_README.md` – Formatter Agent details and color-tag scheme.
- `examples/README_MOCK_DATA.md` – mock data instructions.
- `examples/README_TEST_RAG.md` – RAG validation checklist.
- `docker/Docker.md` – operations tips for the container stack.

## Suggested Roadmap

- Add telemetry and tracing for each agent hop.
- Expand Guardrail coverage (PII, compliance, red-team scenarios).
- Provide a workflow editor so teams can customize pipelines.
- Support additional ingestion sources (Confluence, Jira, CAD files, etc.).

---

> Built for the Seal Hackathon to showcase multi-agent orchestration in enterprise requirement analysis. Use the Chainlit UI to watch agents collaborate in real time and extend the system with new agents as your use cases grow.