"""Ingest Agent for handling embeddings and document ingestion.

This agent is responsible for:
- Generating embeddings for text/document chunks
- Encoding queries for search
- Managing embedding model operations
- Creating JSON files with proper schema for Milvus and Neo4j
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from core.config import load_config
from data_pipeline.embedder import VietnameseE5Embedder
from db.milvus_client import connect_to_milvus, ensure_gsoft_docs_collection, DOC_COLLECTION_NAME
from db.neo4j_client import (
    RequirementsNeo4jClient,
    Project,
    Requirement,
    UserStory,
    RequirementType,
    Priority,
    RequirementStatus,
    StoryStatus,
    Meeting,
    Participant,
    Event,
    init_schema,
)
from openai import OpenAI
from neo4j import GraphDatabase
from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException
import re
import uuid
from collections import Counter
import math

logger = logging.getLogger(__name__)


class IngestAgent:
    """Agent for handling embedding and ingestion operations."""
    
    def __init__(self, embedder: Optional[VietnameseE5Embedder] = None):
        """Initialize Ingest Agent.
        
        Args:
            embedder: VietnameseE5Embedder instance (optional, will create if not provided)
        """
        self.config = load_config()
        
        # Initialize Embedder
        self.embedder = embedder or VietnameseE5Embedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
            normalize=self.config.embedding.normalize_embeddings,
        )
        
        # Initialize OpenAI client for LLM-based extraction
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.llm_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for LLM-based extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.llm_client = None
        else:
            self.llm_client = None
            logger.warning("OPENAI_API_KEY not set. LLM-based extraction will be unavailable.")
        
        logger.info(f"Ingest Agent initialized with model: {self.config.embedding.model_name}")
    
    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode query texts into embeddings.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of embedding vectors (list of floats)
        """
        try:
            if not queries:
                return []
            
            embeddings = self.embedder.encode_queries(queries)
            logger.debug(f"Encoded {len(queries)} queries into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding queries: {e}", exc_info=True)
            return []
    
    def encode_passages(self, passages: List[str]) -> List[List[float]]:
        """Encode passage/document texts into embeddings.
        
        Args:
            passages: List of passage/document strings
            
        Returns:
            List of embedding vectors (list of floats)
        """
        try:
            if not passages:
                return []
            
            embeddings = self.embedder.encode_passages(passages)
            logger.debug(f"Encoded {len(passages)} passages into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding passages: {e}", exc_info=True)
            return []
    
    def encode_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Encode a batch of texts (queries or passages).
        
        Args:
            texts: List of text strings to encode
            is_query: If True, treat as queries; if False, treat as passages
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if is_query:
            return self.encode_queries(texts)
        else:
            return self.encode_passages(texts)
    
    def create_sparse_vector(self, text: str, vocabulary: Optional[Dict[str, int]] = None) -> Dict[int, float]:
        """Create sparse vector from text using TF-IDF-like approach.
        
        Sparse vector format for Milvus: {index: weight, ...}
        Uses term frequency (TF) with normalization.
        
        Args:
            text: Input text to convert to sparse vector
            vocabulary: Optional vocabulary mapping (word -> index). If None, uses hash-based approach.
            
        Returns:
            Dictionary with {index: weight} format for Milvus sparse vector
        """
        if not text or not text.strip():
            return {}
        
        # Tokenize text (simple word-based tokenization)
        # For Vietnamese, we can use simple whitespace/special char splitting
        words = re.findall(r'\w+', text.lower())
        
        if not words:
            return {}
        
        # Count term frequencies
        term_counts = Counter(words)
        total_terms = len(words)
        
        # Create sparse vector using hash-based vocabulary (if no vocabulary provided)
        sparse_vec = {}
        
        if vocabulary:
            # Use provided vocabulary
            for word, count in term_counts.items():
                if word in vocabulary:
                    index = vocabulary[word]
                    # TF normalization: term_freq / total_terms
                    weight = count / total_terms
                    sparse_vec[index] = float(weight)
        else:
            # Hash-based approach: use hash(word) % max_dimension as index
            # Max dimension for sparse vector (typical: 30,000 - 100,000)
            max_dim = 30000
            
            for word, count in term_counts.items():
                # Use hash to get consistent index
                index = hash(word) % max_dim
                # Ensure positive index
                if index < 0:
                    index = abs(index)
                
                # TF normalization with sqrt to reduce impact of very frequent terms
                weight = math.sqrt(count / total_terms)
                
                # Accumulate weights if multiple words hash to same index
                sparse_vec[index] = sparse_vec.get(index, 0.0) + weight
        
        # Normalize the sparse vector (L2 normalization)
        norm = math.sqrt(sum(w * w for w in sparse_vec.values()))
        if norm > 0:
            sparse_vec = {idx: w / norm for idx, w in sparse_vec.items()}
        
        return sparse_vec
    
    def create_sparse_vectors(self, texts: List[str], vocabulary: Optional[Dict[str, int]] = None) -> List[Dict[int, float]]:
        """Create sparse vectors from multiple texts.
        
        Args:
            texts: List of text strings
            vocabulary: Optional vocabulary mapping (word -> index)
            
        Returns:
            List of sparse vectors in format {index: weight}
        """
        return [self.create_sparse_vector(text, vocabulary) for text in texts]
    
    def create_milvus_json(
        self,
        documents: List[Dict[str, Any]],
        output_file: str,
    ) -> str:
        """Create JSON file with proper Milvus schema.
        
        Milvus schema (gsoft_docs collection):
        - id (VARCHAR, primary key, max 64)
        - original_doc_id (VARCHAR, max 64)
        - permission (INT8)
        - source (VARCHAR, max 16)
        - url (VARCHAR, max 512)
        - updated_at (INT64)
        - text_preview (VARCHAR, max 1024)
        - dense_vec (FLOAT_VECTOR)
        - sparse_vec (SPARSE_FLOAT_VECTOR)
        
        Args:
            documents: List of document dicts with text content
            output_file: Path to output JSON file
            
        Returns:
            Path to created JSON file
        """
        try:
            # Encode all passages for dense embeddings
            texts = [doc.get("text", "") for doc in documents]
            dense_embeddings = self.encode_passages(texts)
            
            # Create sparse vectors for all texts
            sparse_vectors = self.create_sparse_vectors(texts)
            
            # Prepare data in Milvus schema format
            milvus_data = []
            current_timestamp = int(datetime.now().timestamp())
            
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"doc_{i:06d}")
                text = doc.get("text", "")
                text_preview = text[:1024] if len(text) > 1024 else text
                
                # Prepare Milvus record
                milvus_record = {
                    "id": str(doc_id)[:64],  # Max length 64
                    "original_doc_id": str(doc.get("original_doc_id", doc_id))[:64],
                    "permission": int(doc.get("permission", 0)),
                    "source": str(doc.get("source", "unknown"))[:16],
                    "url": str(doc.get("url", ""))[:512],
                    "updated_at": int(doc.get("updated_at", current_timestamp)),
                    "text_preview": text_preview[:1024],
                    "dense_vec": dense_embeddings[i] if i < len(dense_embeddings) else [],
                    "sparse_vec": sparse_vectors[i] if i < len(sparse_vectors) else {},  # Sparse vector for hybrid search
                }
                
                milvus_data.append(milvus_record)
            
            logger.info(f"Created {len(milvus_data)} documents with dense and sparse vectors")
            
            # Write to JSON file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(milvus_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created Milvus JSON file with {len(milvus_data)} documents: {output_file}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating Milvus JSON: {e}", exc_info=True)
            raise
    
    def create_neo4j_json(
        self,
        projects: Optional[List[Dict[str, Any]]] = None,
        requirements: Optional[List[Dict[str, Any]]] = None,
        user_stories: Optional[List[Dict[str, Any]]] = None,
        meetings: Optional[List[Dict[str, Any]]] = None,
        participants: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        output_file: str = "neo4j_data.json",
    ) -> str:
        """Create JSON file with proper Neo4j schema.
        
        Neo4j schema:
        - Project: project_id, name, description, version, created_date, updated_date, status, stakeholders[]
        - Requirement: req_id, title, description, type, priority, status, created_date, updated_date, version, source, acceptance_criteria[], constraints[], assumptions[]
        - UserStory: story_id, title, as_a, i_want, so_that, description, priority, story_points, status, sprint, acceptance_criteria[], created_date, updated_date
        - Meeting: meeting_id, title, description, date, start_time, end_time, location, meeting_type, meeting_status, agenda, action_items, decisions, next_meeting_date, next_meeting_time, project_id
        - Participant: participant_id, name, role, email, attendance_status, notes
        - Event: event_id, name, description, event_type, date, time, priority, status, assigned_to, due_date, meeting_id
        
        Args:
            projects: List of project dicts
            requirements: List of requirement dicts
            user_stories: List of user story dicts
            meetings: List of meeting dicts
            participants: List of participant dicts
            events: List of event dicts
            output_file: Path to output JSON file
            
        Returns:
            Path to created JSON file
        """
        try:
            current_date = datetime.now().isoformat()
            
            neo4j_data = {
                "projects": [],
                "requirements": [],
                "user_stories": [],
                "meetings": [],
                "participants": [],
                "events": [],
            }
            
            # Process projects
            if projects:
                for proj in projects:
                    project_data = {
                        "project_id": proj.get("project_id", ""),
                        "name": proj.get("name", ""),
                        "description": proj.get("description"),
                        "version": proj.get("version", "1.0"),
                        "created_date": proj.get("created_date", current_date),
                        "updated_date": proj.get("updated_date", current_date),
                        "status": proj.get("status", "active"),
                        "stakeholders": proj.get("stakeholders", []),
                    }
                    neo4j_data["projects"].append(project_data)
            
            # Process requirements
            if requirements:
                for req in requirements:
                    requirement_data = {
                        "req_id": req.get("req_id", ""),
                        "title": req.get("title", ""),
                        "description": req.get("description", ""),
                        "type": req.get("type", "functional"),
                        "priority": req.get("priority", "medium"),
                        "status": req.get("status", "draft"),
                        "created_date": req.get("created_date", current_date),
                        "updated_date": req.get("updated_date", current_date),
                        "version": req.get("version", "1.0"),
                        "source": req.get("source"),
                        "acceptance_criteria": req.get("acceptance_criteria", []),
                        "constraints": req.get("constraints", []),
                        "assumptions": req.get("assumptions", []),
                        "project_id": req.get("project_id"),  # For linking
                    }
                    neo4j_data["requirements"].append(requirement_data)
            
            # Process user stories
            if user_stories:
                for story in user_stories:
                    story_data = {
                        "story_id": story.get("story_id", ""),
                        "title": story.get("title", ""),
                        "as_a": story.get("as_a", ""),
                        "i_want": story.get("i_want", ""),
                        "so_that": story.get("so_that", ""),
                        "description": story.get("description"),
                        "priority": story.get("priority", "medium"),
                        "story_points": story.get("story_points"),
                        "status": story.get("status", "backlog"),
                        "sprint": story.get("sprint"),
                        "acceptance_criteria": story.get("acceptance_criteria", []),
                        "created_date": story.get("created_date", current_date),
                        "updated_date": story.get("updated_date", current_date),
                        "project_id": story.get("project_id"),  # For linking
                    }
                    neo4j_data["user_stories"].append(story_data)
            
            # Process meetings
            if meetings:
                for meeting in meetings:
                    meeting_data = {
                        "meeting_id": meeting.get("meeting_id", ""),
                        "title": meeting.get("title", ""),
                        "description": meeting.get("description"),
                        "date": meeting.get("date"),
                        "start_time": meeting.get("start_time"),
                        "end_time": meeting.get("end_time"),
                        "location": meeting.get("location"),
                        "meeting_type": meeting.get("meeting_type", "general"),
                        "meeting_status": meeting.get("meeting_status", "completed"),
                        "agenda": meeting.get("agenda", []),
                        "action_items": meeting.get("action_items", []),
                        "decisions": meeting.get("decisions", []),
                        "next_meeting_date": meeting.get("next_meeting_date"),
                        "next_meeting_time": meeting.get("next_meeting_time"),
                        "project_id": meeting.get("project_id"),
                        "created_date": meeting.get("created_date", current_date),
                        "updated_date": meeting.get("updated_date", current_date),
                    }
                    neo4j_data["meetings"].append(meeting_data)
            
            # Process participants
            if participants:
                for participant in participants:
                    participant_data = {
                        "participant_id": participant.get("participant_id", ""),
                        "name": participant.get("name", ""),
                        "role": participant.get("role"),
                        "email": participant.get("email"),
                        "attendance_status": participant.get("attendance_status", "present"),
                        "notes": participant.get("notes"),
                    }
                    neo4j_data["participants"].append(participant_data)
            
            # Process events
            if events:
                for event in events:
                    event_data = {
                        "event_id": event.get("event_id", ""),
                        "name": event.get("name", ""),
                        "description": event.get("description"),
                        "event_type": event.get("event_type", "discussion"),
                        "date": event.get("date"),
                        "time": event.get("time"),
                        "priority": event.get("priority", "medium"),
                        "status": event.get("status", "open"),
                        "assigned_to": event.get("assigned_to"),
                        "due_date": event.get("due_date"),
                        "meeting_id": event.get("meeting_id"),
                        "related_entity_ids": event.get("related_entity_ids", []),
                    }
                    neo4j_data["events"].append(event_data)
            
            # Write to JSON file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(neo4j_data, f, indent=2, ensure_ascii=False)
            
            total_items = (
                len(neo4j_data["projects"]) + len(neo4j_data["requirements"]) + 
                len(neo4j_data["user_stories"]) + len(neo4j_data["meetings"]) + 
                len(neo4j_data["participants"]) + len(neo4j_data["events"])
            )
            logger.info(f"Created Neo4j JSON file with {total_items} items: {output_file}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating Neo4j JSON: {e}", exc_info=True)
            raise
    
    def feed_milvus_from_json(self, json_file: str) -> str:
        """Feed Milvus JSON file into Milvus collection.
        
        Args:
            json_file: Path to Milvus JSON file
            
        Returns:
            Success message with number of documents inserted
        """
        try:
            # Read JSON file
            json_path = Path(json_file)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_file}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                milvus_data = json.load(f)
            
            if not milvus_data:
                return "No data found in JSON file."
            
            # Connect to Milvus and get collection
            alias = connect_to_milvus()
            collection = ensure_gsoft_docs_collection(alias=alias, dense_dim=self.config.milvus.doc_dense_dim)
            
            # Check if collection is offline
            if not hasattr(collection, 'insert'):
                return "Milvus is offline or unavailable. Cannot insert data."
            
            # Validate embedding dimensions
            expected_dim = self.config.milvus.doc_dense_dim
            dense_vectors = [doc.get("dense_vec", []) for doc in milvus_data]
            
            # Check if we have any embeddings
            if not dense_vectors or len(dense_vectors) == 0:
                logger.warning("No dense vectors found in data. Skipping Milvus insertion.")
                return "No dense vectors found in data. Cannot insert into Milvus."
            
            # Validate dimension of first vector
            first_vector = dense_vectors[0]
            if not first_vector:
                logger.warning("Empty dense vector found. Skipping Milvus insertion.")
                return "Empty dense vector found. Cannot insert into Milvus."
            
            actual_dim = len(first_vector)
            if actual_dim != expected_dim:
                error_msg = f"Dimension mismatch: Collection expects {expected_dim} dimensions but embeddings have {actual_dim}. Please recreate collection or use correct embedding model."
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Prepare data for insertion
            ids = [doc.get("id", "") for doc in milvus_data]
            original_doc_ids = [doc.get("original_doc_id", "") for doc in milvus_data]
            permissions = [doc.get("permission", 0) for doc in milvus_data]
            sources = [doc.get("source", "unknown") for doc in milvus_data]
            urls = [doc.get("url", "") for doc in milvus_data]
            updated_ats = [doc.get("updated_at", int(datetime.now().timestamp())) for doc in milvus_data]
            text_previews = [doc.get("text_preview", "") for doc in milvus_data]
            sparse_vectors = [doc.get("sparse_vec", {}) for doc in milvus_data]
            
            # Verify all vectors have correct dimension
            for i, vec in enumerate(dense_vectors):
                if len(vec) != expected_dim:
                    error_msg = f"Document {i} has incorrect dimension: {len(vec)} (expected {expected_dim})"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
            
            # Load collection if needed
            try:
                if hasattr(collection, 'load'):
                    collection.load()
                    logger.info(f"Collection loaded. Current dimension: {expected_dim}")
            except Exception as e:
                logger.warning(f"Could not load collection: {e}")
                # Continue anyway - might still work
            
            # Insert data
            logger.info(f"Inserting {len(milvus_data)} documents into Milvus with dimension {expected_dim}...")
            try:
                collection.insert([
                    ids,
                    original_doc_ids,
                    permissions,
                    sources,
                    urls,
                    updated_ats,
                    text_previews,
                    dense_vectors,
                    sparse_vectors,
                ])
                logger.info(f"Successfully inserted {len(milvus_data)} documents into Milvus collection")
            except Exception as insert_error:
                error_msg = f"Failed to insert data into Milvus: {str(insert_error)}"
                logger.error(error_msg, exc_info=True)
                # Check if it's a dimension mismatch error
                if "dimension" in str(insert_error).lower() or "dim" in str(insert_error).lower():
                    error_msg += f"\nCollection dimension mismatch: collection might be using {expected_dim} dimensions but embeddings have different size."
                    error_msg += f"\nPlease recreate the collection or use the correct embedding model."
                return f"Error: {error_msg}"
            
            # Flush data to disk
            collection.flush()
            
            # Create indexes if they don't exist (for hybrid search)
            try:
                from pymilvus import Index
                existing_indexes = collection.indexes
                index_names = [idx.field_name for idx in existing_indexes]
                
                # Create index for dense_vec if not exists
                if "dense_vec" not in index_names:
                    try:
                        dense_index_params = {
                            "metric_type": "L2",
                            "index_type": "IVF_FLAT",
                            "params": {"nlist": 128}
                        }
                        collection.create_index(
                            field_name="dense_vec",
                            index_params=dense_index_params
                        )
                        logger.info("Created index for dense_vec")
                    except Exception as e:
                        logger.warning(f"Could not create index for dense_vec: {e}")
                
                # Create index for sparse_vec if not exists
                if "sparse_vec" not in index_names:
                    try:
                        sparse_index_params = {
                            "metric_type": "IP",
                            "index_type": "SPARSE_INVERTED_INDEX",
                            "params": {}
                        }
                        collection.create_index(
                            field_name="sparse_vec",
                            index_params=sparse_index_params
                        )
                        logger.info("Created index for sparse_vec")
                    except Exception as e:
                        logger.warning(f"Could not create index for sparse_vec: {e}. This is expected if collection is empty or Milvus version doesn't support it.")
            except Exception as e:
                logger.warning(f"Could not check/create indexes: {e}")
            
            logger.info(f"Successfully inserted {len(milvus_data)} documents into Milvus collection '{DOC_COLLECTION_NAME}' with hybrid search support")
            return f"Successfully inserted {len(milvus_data)} documents into Milvus collection '{DOC_COLLECTION_NAME}' with hybrid search support (dense + sparse vectors)"
            
        except Exception as e:
            logger.error(f"Error feeding Milvus from JSON: {e}", exc_info=True)
            return f"Error feeding Milvus from JSON: {str(e)}"
    
    def feed_neo4j_from_json(self, json_file: str) -> str:
        """Feed Neo4j JSON file into Neo4j database.
        
        Args:
            json_file: Path to Neo4j JSON file
            
        Returns:
            Success message with number of nodes created
        """
        try:
            # Read JSON file
            json_path = Path(json_file)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_file}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                neo4j_data = json.load(f)
            
            # Initialize Neo4j client
            client = RequirementsNeo4jClient()
            
            # Initialize schema
            client.initialize_schema()
            
            projects_count = 0
            requirements_count = 0
            user_stories_count = 0
            
            # Create projects (use merge to avoid duplicates)
            if neo4j_data.get("projects"):
                for proj_data in neo4j_data["projects"]:
                    project = Project(
                        project_id=proj_data.get("project_id", ""),
                        name=proj_data.get("name", ""),
                        description=proj_data.get("description"),
                        version=proj_data.get("version", "1.0"),
                        created_date=proj_data.get("created_date"),
                        updated_date=proj_data.get("updated_date"),
                        status=proj_data.get("status", "active"),
                        stakeholders=proj_data.get("stakeholders", []),
                    )
                    # Use merge to avoid duplicates based on name
                    merged_id = client.merge_project_by_name(project)
                    if merged_id:
                        projects_count += 1
                        logger.debug(f"Created/merged project: {merged_id}")
            
            # Create requirements
            if neo4j_data.get("requirements"):
                for req_data in neo4j_data["requirements"]:
                    # Convert string enums to proper enum types
                    req_type = req_data.get("type", "functional")
                    if isinstance(req_type, str):
                        req_type = RequirementType(req_type) if req_type in [e.value for e in RequirementType] else RequirementType.FUNCTIONAL
                    
                    priority = req_data.get("priority", "medium")
                    if isinstance(priority, str):
                        priority = Priority(priority) if priority in [e.value for e in Priority] else Priority.MEDIUM
                    
                    status = req_data.get("status", "draft")
                    if isinstance(status, str):
                        status = RequirementStatus(status) if status in [e.value for e in RequirementStatus] else RequirementStatus.DRAFT
                    
                    requirement = Requirement(
                        req_id=req_data.get("req_id", ""),
                        title=req_data.get("title", ""),
                        description=req_data.get("description", ""),
                        type=req_type,
                        priority=priority,
                        status=status,
                        created_date=req_data.get("created_date"),
                        updated_date=req_data.get("updated_date"),
                        version=req_data.get("version", "1.0"),
                        source=req_data.get("source"),
                        acceptance_criteria=req_data.get("acceptance_criteria", []),
                        constraints=req_data.get("constraints", []),
                        assumptions=req_data.get("assumptions", []),
                    )
                    project_id = req_data.get("project_id")
                    if project_id:
                        client.create_requirement(project_id, requirement)
                        requirements_count += 1
                        logger.debug(f"Created requirement: {requirement.req_id}")
            
            # Create user stories
            if neo4j_data.get("user_stories"):
                for story_data in neo4j_data["user_stories"]:
                    # Convert string enums to proper enum types
                    priority = story_data.get("priority", "medium")
                    if isinstance(priority, str):
                        priority = Priority(priority) if priority in [e.value for e in Priority] else Priority.MEDIUM
                    
                    status = story_data.get("status", "backlog")
                    if isinstance(status, str):
                        status = StoryStatus(status) if status in [e.value for e in StoryStatus] else StoryStatus.BACKLOG
                    
                    story = UserStory(
                        story_id=story_data.get("story_id", ""),
                        title=story_data.get("title", ""),
                        as_a=story_data.get("as_a", ""),
                        i_want=story_data.get("i_want", ""),
                        so_that=story_data.get("so_that", ""),
                        description=story_data.get("description"),
                        priority=priority,
                        story_points=story_data.get("story_points"),
                        status=status,
                        sprint=story_data.get("sprint"),
                        acceptance_criteria=story_data.get("acceptance_criteria", []),
                        created_date=story_data.get("created_date"),
                        updated_date=story_data.get("updated_date"),
                    )
                    project_id = story_data.get("project_id")
                    if project_id:
                        client.create_user_story(project_id, story)
                        user_stories_count += 1
                        logger.debug(f"Created user story: {story.story_id}")
            
            # Process meetings, participants, and events
            meetings_count = 0
            participants_count = 0
            events_count = 0
            
            # First, create/merge participants (need to be done before linking to meetings)
            participant_id_mapping = {}  # Map original participant_id to merged_id
            if neo4j_data.get("participants"):
                seen_participant_ids = set()
                for participant_data in neo4j_data["participants"]:
                    participant_id = participant_data.get("participant_id")
                    if participant_id and participant_id not in seen_participant_ids:
                        participant = Participant(
                            participant_id=participant_id,
                            name=participant_data.get("name", ""),
                            role=participant_data.get("role"),
                            email=participant_data.get("email"),
                            attendance_status=participant_data.get("attendance_status", "present"),
                            notes=participant_data.get("notes"),
                        )
                        # Use merge to avoid duplicates based on name or email
                        merged_id = client.merge_participant_by_name_or_email(participant)
                        if merged_id:
                            participants_count += 1
                            seen_participant_ids.add(merged_id)
                            participant_id_mapping[participant_id] = merged_id
                            logger.debug(f"Created/merged participant: {merged_id}")
            
            # Create meetings (use merge to avoid duplicates)
            meeting_id_mapping = {}  # Map original meeting_id to merged_id
            if neo4j_data.get("meetings"):
                for meeting_data in neo4j_data["meetings"]:
                    meeting = Meeting(
                        meeting_id=meeting_data.get("meeting_id", ""),
                        title=meeting_data.get("title", ""),
                        description=meeting_data.get("description"),
                        date=meeting_data.get("date"),
                        start_time=meeting_data.get("start_time"),
                        end_time=meeting_data.get("end_time"),
                        location=meeting_data.get("location"),
                        meeting_type=meeting_data.get("meeting_type", "general"),
                        meeting_status=meeting_data.get("meeting_status", "completed"),
                        agenda=meeting_data.get("agenda", []),
                        action_items=meeting_data.get("action_items", []),
                        decisions=meeting_data.get("decisions", []),
                        next_meeting_date=meeting_data.get("next_meeting_date"),
                        next_meeting_time=meeting_data.get("next_meeting_time"),
                        project_id=meeting_data.get("project_id"),
                        created_date=meeting_data.get("created_date"),
                        updated_date=meeting_data.get("updated_date"),
                    )
                    # Use merge to avoid duplicates based on title and date
                    merged_id = client.merge_meeting_by_title_date(meeting)
                    if merged_id:
                        meetings_count += 1
                        original_meeting_id = meeting.meeting_id
                        meeting_id_mapping[original_meeting_id] = merged_id
                        logger.debug(f"Created/merged meeting: {merged_id}")
                        
                        # Link participants to meeting (use merged_id)
                        for original_participant_id, merged_participant_id in participant_id_mapping.items():
                            try:
                                client.link_participant_to_meeting(merged_participant_id, merged_id)
                            except Exception as e:
                                logger.warning(f"Could not link participant {merged_participant_id} to meeting {merged_id}: {e}")
            
            # Create events (use merge to avoid duplicates)
            if neo4j_data.get("events"):
                for event_data in neo4j_data["events"]:
                    # Convert priority string to Priority enum
                    priority_str = event_data.get("priority", "medium")
                    if isinstance(priority_str, str):
                        priority = Priority(priority_str) if priority_str in [e.value for e in Priority] else Priority.MEDIUM
                    else:
                        priority = Priority.MEDIUM
                    
                    # Get merged meeting_id if original meeting_id exists
                    original_meeting_id = event_data.get("meeting_id")
                    merged_meeting_id = meeting_id_mapping.get(original_meeting_id, original_meeting_id) if original_meeting_id else None
                    
                    event = Event(
                        event_id=event_data.get("event_id", ""),
                        name=event_data.get("name", ""),
                        description=event_data.get("description"),
                        event_type=event_data.get("event_type", "discussion"),
                        date=event_data.get("date"),
                        time=event_data.get("time"),
                        priority=priority,
                        status=event_data.get("status", "open"),
                        assigned_to=event_data.get("assigned_to"),
                        due_date=event_data.get("due_date"),
                        meeting_id=merged_meeting_id,  # Use merged meeting_id
                        related_entity_ids=event_data.get("related_entity_ids", []),
                    )
                    # Use merge to avoid duplicates based on name, date, and meeting_id
                    merged_id = client.merge_event_by_name_date_meeting(event)
                    if merged_id:
                        events_count += 1
                        logger.debug(f"Created/merged event: {merged_id}")
                    
                    # Link event to participant if assigned_to is provided
                    # Use merged_id from merge operation
                    if merged_id and event.assigned_to:
                        # Try to find participant by name or ID in participant_id_mapping
                        merged_participant_id = None
                        for original_participant_id, mapped_id in participant_id_mapping.items():
                            # Find by matching name or original participant_id
                            participant_data = next(
                                (p for p in neo4j_data.get("participants", []) 
                                 if p.get("participant_id") == original_participant_id),
                                None
                            )
                            if participant_data:
                                if (participant_data.get("name") == event.assigned_to or 
                                    participant_data.get("participant_id") == event.assigned_to or
                                    original_participant_id == event.assigned_to):
                                    merged_participant_id = mapped_id
                                    break
                        
                        if merged_participant_id:
                            try:
                                client.link_event_to_participant(merged_id, merged_participant_id)
                            except Exception as e:
                                logger.warning(f"Could not link event {merged_id} to participant {merged_participant_id}: {e}")
            
            # Close connection
            client.close()
            
            total_count = projects_count + requirements_count + user_stories_count + meetings_count + participants_count + events_count
            logger.info(f"Successfully fed Neo4j from JSON: {projects_count} projects, {requirements_count} requirements, {user_stories_count} user stories, {meetings_count} meetings, {participants_count} participants, {events_count} events")
            return f"Successfully fed Neo4j from JSON: {projects_count} projects, {requirements_count} requirements, {user_stories_count} user stories, {meetings_count} meetings, {participants_count} participants, {events_count} events (total: {total_count} nodes)"
            
        except Exception as e:
            logger.error(f"Error feeding Neo4j from JSON: {e}", exc_info=True)
            return f"Error feeding Neo4j from JSON: {str(e)}"
    
    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50, use_semantic: bool = True) -> List[str]:
        """Chunk text into smaller pieces using semantic chunking.
        
        Uses semantic chunking by:
        1. Splitting text into sentences/paragraphs
        2. Creating embeddings for each segment
        3. Grouping semantically similar segments together
        4. Maintaining semantic coherence within chunks
        
        Args:
            text: Text to chunk
            chunk_size: Target characters per chunk (approximate)
            chunk_overlap: Number of characters to overlap between chunks
            use_semantic: Whether to use semantic chunking (default: True)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
        
        if not use_semantic:
            # Fallback to simple chunking
            return self._simple_chunk(text, chunk_size, chunk_overlap)
        
        # Semantic chunking using embeddings
        return self._semantic_chunk(text, chunk_size, chunk_overlap)
    
    def _simple_chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Simple character-based chunking (fallback)."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < text_length:
                for punct in ['.', '!', '?', '\n']:
                    last_punct = chunk.rfind(punct)
                    if last_punct > chunk_size * 0.7:
                        chunk = chunk[:last_punct + 1]
                        end = start + last_punct + 1
                        break
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
        
        return [chunk for chunk in chunks if chunk]
    
    def _semantic_chunk(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Semantic chunking using embeddings to find semantic boundaries.
        
        Strategy:
        1. Split text into sentences/paragraphs
        2. Calculate embeddings for segments
        3. Group semantically similar segments up to chunk_size
        4. Use cosine similarity to find natural breakpoints
        """
        # Step 1: Split into sentences and paragraphs
        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Then split long paragraphs into sentences
        segments = []
        for para in paragraphs:
            if len(para.strip()) < chunk_size * 0.5:
                # Short paragraph - keep as is
                segments.append(para.strip())
            else:
                # Long paragraph - split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_segment = ""
                for sent in sentences:
                    if len(current_segment) + len(sent) < chunk_size * 0.8:
                        current_segment += " " + sent if current_segment else sent
                    else:
                        if current_segment:
                            segments.append(current_segment.strip())
                        current_segment = sent
                if current_segment:
                    segments.append(current_segment.strip())
        
        # Step 2: If we have embeddings available, use them for better grouping
        # Otherwise, group by size with semantic awareness
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for i, segment in enumerate(segments):
            segment_len = len(segment)
            
            # If adding this segment would exceed chunk_size significantly
            if current_length + segment_len > chunk_size * 1.5 and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # Get last chunk_overlap characters as overlap
                    overlap_text = current_chunk[-chunk_overlap:]
                    # Try to start overlap at sentence boundary
                    overlap_start = max(0, len(current_chunk) - chunk_overlap * 2)
                    overlap_segment = current_chunk[overlap_start:]
                    # Find sentence boundary in overlap
                    for punct in ['.', '!', '?', '\n']:
                        punct_pos = overlap_segment.find(punct)
                        if punct_pos > 0:
                            overlap_text = overlap_segment[punct_pos + 1:].strip()
                            break
                    current_chunk = overlap_text + " " + segment if overlap_text else segment
                else:
                    current_chunk = segment
                current_length = len(current_chunk)
            else:
                # Add to current chunk
                current_chunk = current_chunk + " " + segment if current_chunk else segment
                current_length = len(current_chunk)
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Step 3: Use embeddings to refine chunks if possible
        # Group similar segments together for better semantic coherence
        if len(segments) > 1:
            refined_chunks = self._refine_chunks_with_embeddings(chunks, chunk_size)
            if refined_chunks:
                return refined_chunks
        
        return [chunk for chunk in chunks if chunk]
    
    def _refine_chunks_with_embeddings(self, chunks: List[str], target_size: int) -> Optional[List[str]]:
        """Refine chunks using embeddings for better semantic grouping.
        
        This method uses embeddings to find semantically similar segments
        and group them together, ensuring each chunk maintains semantic coherence.
        """
        try:
            # Create embeddings for chunks
            if len(chunks) <= 1:
                return chunks
            
            # Get embeddings for all chunks
            embeddings = self.encode_passages(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                # Embedding failed, return original chunks
                return chunks
            
            # Refine: merge chunks that are semantically similar and together fit size
            import numpy as np
            from numpy.linalg import norm
            
            refined = []
            i = 0
            
            while i < len(chunks):
                current_chunk = chunks[i]
                current_embedding = np.array(embeddings[i])
                j = i + 1
                
                # Try to merge with next chunks if semantically similar
                while j < len(chunks):
                    next_chunk = chunks[j]
                    combined_length = len(current_chunk) + len(next_chunk)
                    
                    # Check if merging would be too large
                    if combined_length > target_size * 1.3:
                        break
                    
                    # Calculate semantic similarity (cosine similarity)
                    next_embedding = np.array(embeddings[j])
                    similarity = np.dot(current_embedding, next_embedding) / (
                        norm(current_embedding) * norm(next_embedding)
                    )
                    
                    # If semantically similar (similarity > 0.7), merge
                    if similarity > 0.7:
                        current_chunk += " " + next_chunk
                        # Update embedding to average of merged chunks
                        current_embedding = (current_embedding + next_embedding) / 2
                        j += 1
                    else:
                        # Not similar enough, stop merging
                        break
                
                refined.append(current_chunk.strip())
                i = j
            
            return [chunk for chunk in refined if chunk]
        
        except Exception as e:
            logger.warning(f"Error refining chunks with embeddings: {e}, using original chunks")
            return chunks
    
    def parse_file(self, file_path: str) -> str:
        """Parse file content to text.
        
        Supports:
        - .txt: Plain text
        - .docx: Word documents (if python-docx available)
        - .pdf: PDF files (if PyPDF2 available)
        - Other: Tries to read as text
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif suffix == '.docx':
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return '\n'.join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not installed, trying to read as text")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
            
            elif suffix == '.pdf':
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    logger.warning("PyPDF2 not installed, trying to read as text")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
            
            else:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise
    
    def extract_requirements_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract requirements and user stories from text using pattern matching and heuristics.
        
        This function:
        1. Tries to find specific requirements/user stories using patterns
        2. If no patterns found, creates a default project and requirement from the text
        3. Always ensures at least one project and requirement are created
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with 'projects', 'requirements', 'user_stories' lists
        """
        # Simple pattern matching for requirements
        # REQ-XXX, REQ_XXX, Requirement XXX, Yêu cầu XXX, etc.
        req_pattern = r'(?:REQ[-_]?\d+|Requirement\s+\d+|Yêu\s+cầu\s+\d+|yêu\s+cầu\s+\d+)'
        
        # User story patterns: "As a... I want... So that..." or Vietnamese equivalent
        story_pattern = r'(?:As\s+(?:a|an)\s+[\w\s]+,\s+I\s+want\s+[\w\s]+(?:,\s+)?so\s+that\s+[\w\s]+|Là\s+[\w\s]+,\s+tôi\s+muốn\s+[\w\s]+(?:,\s+)?để\s+[\w\s]+)'
        
        requirements = []
        user_stories = []
        projects = []
        
        # Find requirements using patterns
        req_matches = re.finditer(req_pattern, text, re.IGNORECASE)
        for match in req_matches:
            start = match.start()
            # Extract context around requirement (next 500 chars or until next requirement)
            end = min(start + 500, len(text))
            next_req = text.find('REQ', start + 10)
            if next_req > start:
                end = min(end, next_req)
            
            req_text = text[start:end]
            req_id_match = re.search(req_pattern, req_text, re.IGNORECASE)
            if req_id_match:
                req_id = req_id_match.group(0).upper().replace('_', '-').replace(' ', '-')
                # Extract title (first line or sentence)
                lines = req_text.split('\n')
                title = lines[0].strip()[:100] if lines else req_id
                description = req_text[:500].strip()
                
                requirements.append({
                    "req_id": req_id,
                    "title": title,
                    "description": description,
                    "type": "functional",
                    "priority": "medium",
                    "status": "draft",
                    "project_id": "DEFAULT-PROJ",
                })
        
        # Find user stories using patterns
        story_matches = re.finditer(story_pattern, text, re.IGNORECASE | re.DOTALL)
        for match in story_matches:
            story_text = match.group(0)
            # Extract components
            as_a_match = re.search(r'(?:As\s+(?:a|an)\s+|Là\s+)([\w\s]+)', story_text, re.IGNORECASE)
            i_want_match = re.search(r'(?:I\s+want\s+|tôi\s+muốn\s+)([\w\s]+?)(?:,|so\s+that|để)', story_text, re.IGNORECASE)
            so_that_match = re.search(r'(?:so\s+that\s+|để\s+)([\w\s]+)', story_text, re.IGNORECASE)
            
            if as_a_match or i_want_match:
                story_id = f"US-{len(user_stories) + 1:03d}"
                user_stories.append({
                    "story_id": story_id,
                    "title": story_text[:100],
                    "as_a": as_a_match.group(1).strip() if as_a_match else "",
                    "i_want": i_want_match.group(1).strip() if i_want_match else "",
                    "so_that": so_that_match.group(1).strip() if so_that_match else "",
                    "description": story_text,
                    "priority": "medium",
                    "status": "backlog",
                    "project_id": "DEFAULT-PROJ",
                })
        
        # If no requirements or user stories found, create default ones from the text
        if not requirements and not user_stories:
            # Extract project name from text (look for common patterns)
            project_name = "Document Project"
            project_id = f"PROJ-{uuid.uuid4().hex[:8].upper()}"
            
            # Try to find project name in text
            # Look for patterns like "Project: ...", "Dự án: ...", "Project Name: ..."
            project_name_patterns = [
                r'(?:Project|Dự án|Dự Án)\s*:?\s*([^\n]{1,100})',
                r'(?:Project Name|Tên dự án|Tên Dự Án)\s*:?\s*([^\n]{1,100})',
                r'(?:Tên|Name)\s*:?\s*([^\n]{1,100})',
            ]
            
            for pattern in project_name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    potential_name = match.group(1).strip()
                    # Clean up the name (remove common suffixes)
                    potential_name = re.sub(r'\s*(?:Project|Dự án|Dự Án).*$', '', potential_name, flags=re.IGNORECASE)
                    if len(potential_name) > 3 and len(potential_name) < 100:
                        project_name = potential_name
                        break
            
            # Create default project
            projects.append({
                "project_id": project_id,
                "name": project_name,
                "description": text[:500] if len(text) > 500 else text,  # First 500 chars as description
                "status": "active",
            })
            
            # Create a default requirement from the text
            # Use first few sentences or lines as requirement
            lines = text.split('\n')
            first_sentences = []
            char_count = 0
            
            for line in lines[:10]:  # First 10 lines
                if char_count + len(line) > 300:
                    break
                first_sentences.append(line.strip())
                char_count += len(line)
            
            title = lines[0].strip()[:100] if lines else "Document Requirement"
            description = '\n'.join(first_sentences)[:1000] if first_sentences else text[:1000]
            
            requirements.append({
                "req_id": f"REQ-{uuid.uuid4().hex[:6].upper()}",
                "title": title,
                "description": description,
                "type": "functional",
                "priority": "medium",
                "status": "draft",
                "project_id": project_id,
            })
            
            logger.info(f"Created default project '{project_name}' and requirement from text (no specific patterns found)")
        else:
            # Create default project if we found requirements or stories but no project
            if not projects:
                projects.append({
                    "project_id": "DEFAULT-PROJ",
                    "name": "Default Project",
                    "description": "Auto-created project from document ingestion",
                    "status": "active",
                })
        
        return {
            "projects": projects,
            "requirements": requirements,
            "user_stories": user_stories,
        }
    
    def extract_meeting_information(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract meeting information from text using LLM.
        
        This function uses OpenAI LLM to extract structured meeting information including:
        - Meeting details (title, date, time, location, etc.)
        - Participants (names, roles, attendance)
        - Events (decisions, action items, deadlines, milestones)
        - Relationships between entities
        
        Args:
            text: Text content of meeting minutes/document
            
        Returns:
            Dictionary with 'meetings', 'participants', 'events' lists
        """
        if not self.llm_client:
            logger.warning("OpenAI client not available. Cannot extract meeting information.")
            return {
                "meetings": [],
                "participants": [],
                "events": [],
            }
        
        try:
            # Create LLM prompt for meeting extraction
            prompt = f"""Bạn là trợ lý AI để trích xuất thông tin từ biên bản họp.

Hãy phân tích văn bản sau và trích xuất thông tin về:
1. **Cuộc họp**: Tiêu đề, ngày giờ (date, start_time, end_time), địa điểm (location), loại họp (meeting_type), trạng thái (meeting_status), agenda, action_items, decisions, next_meeting_date, next_meeting_time, project_id
2. **Người tham dự**: Tên (name), vai trò (role), email (nếu có), trạng thái tham dự (attendance_status: present/absent/late/excused), ghi chú (notes)
3. **Sự kiện**: Tên (name), mô tả (description), loại (event_type: discussion/decision/action_item/deadline/milestone), ngày giờ (date, time), độ ưu tiên (priority: critical/high/medium/low), trạng thái (status: open/in_progress/completed/cancelled), người được giao (assigned_to), deadline (due_date)

Trả về JSON với format:
{{
  "meetings": [
    {{
      "meeting_id": "MEET-<random_id>",
      "title": "...",
      "description": "...",
      "date": "YYYY-MM-DD",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "location": "...",
      "meeting_type": "general|kickoff|review|planning|retrospective",
      "meeting_status": "completed",
      "agenda": ["..."],
      "action_items": ["..."],
      "decisions": ["..."],
      "next_meeting_date": "YYYY-MM-DD",
      "next_meeting_time": "HH:MM",
      "project_id": "PROJ-<id>"
    }}
  ],
  "participants": [
    {{
      "participant_id": "PART-<random_id>",
      "name": "...",
      "role": "...",
      "email": "...",
      "attendance_status": "present|absent|late|excused",
      "notes": "..."
    }}
  ],
  "events": [
    {{
      "event_id": "EVENT-<random_id>",
      "name": "...",
      "description": "...",
      "event_type": "discussion|decision|action_item|deadline|milestone",
      "date": "YYYY-MM-DD",
      "time": "HH:MM",
      "priority": "critical|high|medium|low",
      "status": "open|in_progress|completed|cancelled",
      "assigned_to": "...",
      "due_date": "YYYY-MM-DD",
      "meeting_id": "MEET-<id>"
    }}
  ]
}}

**Lưu ý:**
- Nếu không tìm thấy thông tin, trả về mảng rỗng []
- Đảm bảo tất cả ID là unique
- Parse ngày giờ từ văn bản và chuyển sang format ISO (YYYY-MM-DD) và HH:MM
- Nếu có project name, tạo project_id dạng "PROJ-<hash>"
- Liên kết participants và events với meeting_id tương ứng

Văn bản cần phân tích:
{text[:8000]}

Trả về CHỈ JSON, không có text thêm."""

            # Call OpenAI API
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý AI chuyên trích xuất thông tin từ biên bản họp. Trả về JSON hợp lệ."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"} if hasattr(self.llm_client.chat.completions, 'create') else None,
            )
            
            # Extract JSON from response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                # Try to parse JSON (may be wrapped in code blocks)
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                # Ensure all required fields are present
                meetings = result.get("meetings", [])
                participants = result.get("participants", [])
                events = result.get("events", [])
                
                # Generate IDs if missing
                for meeting in meetings:
                    if "meeting_id" not in meeting or not meeting.get("meeting_id"):
                        meeting["meeting_id"] = f"MEET-{uuid.uuid4().hex[:8].upper()}"
                    if "meeting_status" not in meeting:
                        meeting["meeting_status"] = "completed"
                    if "meeting_type" not in meeting:
                        meeting["meeting_type"] = "general"
                
                for participant in participants:
                    if "participant_id" not in participant or not participant.get("participant_id"):
                        participant["participant_id"] = f"PART-{uuid.uuid4().hex[:8].upper()}"
                    if "attendance_status" not in participant:
                        participant["attendance_status"] = "present"
                
                for event in events:
                    if "event_id" not in event or not event.get("event_id"):
                        event["event_id"] = f"EVENT-{uuid.uuid4().hex[:8].upper()}"
                    if "event_type" not in event:
                        event["event_type"] = "discussion"
                    if "priority" not in event:
                        event["priority"] = "medium"
                    if "status" not in event:
                        event["status"] = "open"
                
                logger.info(f"Extracted {len(meetings)} meetings, {len(participants)} participants, {len(events)} events")
                
                return {
                    "meetings": meetings,
                    "participants": participants,
                    "events": events,
                }
            else:
                logger.warning("OpenAI response has no choices")
                return {
                    "meetings": [],
                    "participants": [],
                    "events": [],
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                "meetings": [],
                "participants": [],
                "events": [],
            }
        except Exception as e:
            logger.error(f"Error extracting meeting information with LLM: {e}", exc_info=True)
            return {
                "meetings": [],
                "participants": [],
                "events": [],
            }
    
    def process_file_automatically(
        self,
        file_path: str,
        chunk_size: int = 500,
        extract_requirements: bool = True,
    ) -> Dict[str, Any]:
        """Automatically process file: parse, chunk, create embeddings, and create JSON files.
        
        This function:
        1. Parses the file to extract text
        2. Chunks the text for Milvus
        3. Creates embeddings
        4. Extracts requirements/user stories for Neo4j (if enabled)
        5. Creates JSON files for both Milvus and Neo4j
        6. Automatically feeds into databases
        
        Args:
            file_path: Path to file to process
            chunk_size: Size of chunks for Milvus (characters)
            extract_requirements: Whether to extract requirements/user stories for Neo4j
            
        Returns:
            Dictionary with results including file paths, counts, and status
        """
        try:
            # Step 1: Parse file
            logger.info(f"Parsing file: {file_path}")
            text = self.parse_file(file_path)
            
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "File is empty or could not be parsed",
                }
            
            # Step 2: Chunk text for Milvus
            logger.info(f"Chunking text into chunks of size {chunk_size}")
            chunks = self.chunk_text(text, chunk_size=chunk_size)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Prepare documents for Milvus
            file_name = Path(file_path).stem
            file_id = str(uuid.uuid4())[:8]
            current_timestamp = int(datetime.now().timestamp())
            
            # Truncate file_name to fit VARCHAR(64) constraint for original_doc_id
            original_doc_id = file_name[:64] if len(file_name) > 64 else file_name
            # Truncate file_path to fit VARCHAR(512) constraint for url
            url = file_path[:512] if len(file_path) > 512 else file_path
            
            milvus_documents = []
            for i, chunk in enumerate(chunks):
                # ID format: {file_id}_{i:04d} - max 8 + 1 + 4 = 13 chars, safe for VARCHAR(64)
                milvus_documents.append({
                    "id": f"{file_id}_{i:04d}",
                    "text": chunk,
                    "original_doc_id": original_doc_id,  # Truncated to 64 chars
                    "source": "file_upload",
                    "url": url,  # Truncated to 512 chars
                    "permission": 0,
                    "updated_at": current_timestamp,
                })
            
            # Step 4: Create Milvus JSON and feed
            milvus_json_path = f"milvus_{file_name}_{file_id}.json"
            logger.info(f"Creating Milvus JSON file: {milvus_json_path}")
            try:
                milvus_json = self.create_milvus_json(milvus_documents, milvus_json_path)
                logger.info(f"Feeding Milvus JSON into collection")
                milvus_result = self.feed_milvus_from_json(milvus_json)
                logger.info(f"Milvus result: {milvus_result}")
            except Exception as e:
                logger.error(f"Error creating/feeding Milvus JSON: {e}", exc_info=True)
                milvus_result = f"Error: {str(e)}"
                milvus_json = milvus_json_path  # Keep path even on error
            
            # Step 5: Extract requirements/user stories and meeting information for Neo4j (if enabled)
            neo4j_result = None
            neo4j_json_path = None
            extracted_data = None
            meeting_data = None
            if extract_requirements:
                logger.info("Extracting requirements and user stories from text")
                extracted_data = self.extract_requirements_from_text(text)
                
                # Also extract meeting information using LLM
                logger.info("Extracting meeting information from text using LLM")
                meeting_data = self.extract_meeting_information(text)
                
                # Combine both extracted data
                has_data = (
                    (extracted_data and (extracted_data.get("projects") or extracted_data.get("requirements") or extracted_data.get("user_stories"))) or
                    (meeting_data and (meeting_data.get("meetings") or meeting_data.get("participants") or meeting_data.get("events")))
                )
                
                if has_data:
                    neo4j_json_path = f"neo4j_{file_name}_{file_id}.json"
                    logger.info(f"Creating Neo4j JSON file: {neo4j_json_path}")
                    neo4j_json = self.create_neo4j_json(
                        projects=extracted_data.get("projects", []) if extracted_data else [],
                        requirements=extracted_data.get("requirements", []) if extracted_data else [],
                        user_stories=extracted_data.get("user_stories", []) if extracted_data else [],
                        meetings=meeting_data.get("meetings", []) if meeting_data else [],
                        participants=meeting_data.get("participants", []) if meeting_data else [],
                        events=meeting_data.get("events", []) if meeting_data else [],
                        output_file=neo4j_json_path,
                    )
                    logger.info(f"Feeding Neo4j JSON into database")
                    neo4j_result = self.feed_neo4j_from_json(neo4j_json)
                else:
                    logger.info("No requirements, user stories, or meeting information found in text")
            
            return {
                "success": True,
                "file_path": file_path,
                "text_length": len(text),
                "chunks_count": len(chunks),
                "milvus": {
                    "json_file": milvus_json,
                    "result": milvus_result,
                    "documents_count": len(milvus_documents),
                },
                "neo4j": {
                    "json_file": neo4j_json_path,
                    "result": neo4j_result,
                    "projects_count": len(extracted_data["projects"]) if extracted_data else 0,
                    "requirements_count": len(extracted_data["requirements"]) if extracted_data else 0,
                    "user_stories_count": len(extracted_data["user_stories"]) if extracted_data else 0,
                    "meetings_count": len(meeting_data["meetings"]) if meeting_data else 0,
                    "participants_count": len(meeting_data["participants"]) if meeting_data else 0,
                    "events_count": len(meeting_data["events"]) if meeting_data else 0,
                } if extract_requirements else None,
            }
        
        except Exception as e:
            logger.error(f"Error processing file automatically: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
            }
    
    def process_text_automatically(
        self,
        text: str,
        source: str = "user_input",
        chunk_size: int = 500,
        extract_requirements: bool = True,
    ) -> Dict[str, Any]:
        """Automatically process text: chunk, create embeddings, and create JSON files.
        
        Similar to process_file_automatically but works directly with text.
        
        Args:
            text: Text content to process
            source: Source identifier for the text
            chunk_size: Size of chunks for Milvus (characters)
            extract_requirements: Whether to extract requirements/user stories for Neo4j
            
        Returns:
            Dictionary with results including file paths, counts, and status
        """
        try:
            if not text or not text.strip():
                return {
                    "success": False,
                    "error": "Text is empty",
                }
            
            # Step 1: Chunk text for Milvus
            logger.info(f"Chunking text into chunks of size {chunk_size}")
            chunks = self.chunk_text(text, chunk_size=chunk_size)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 2: Prepare documents for Milvus
            text_id = str(uuid.uuid4())[:8]
            current_timestamp = int(datetime.now().timestamp())
            
            # Truncate source to fit VARCHAR(64) constraint for original_doc_id
            original_doc_id = source[:64] if len(source) > 64 else source
            
            milvus_documents = []
            for i, chunk in enumerate(chunks):
                # ID format: {text_id}_{i:04d} - make it shorter to avoid length issues
                # Use text_id only instead of {source}_{text_id}_{i:04d} to keep it under 64 chars
                doc_id = f"{text_id}_{i:04d}"
                # Ensure ID doesn't exceed 64 chars
                if len(doc_id) > 64:
                    doc_id = doc_id[:64]
                
                milvus_documents.append({
                    "id": doc_id,
                    "text": chunk,
                    "original_doc_id": original_doc_id,  # Truncated to 64 chars
                    "source": source[:16] if len(source) > 16 else source,  # Truncate to 16 chars for source field
                    "url": "",  # Empty for text input
                    "permission": 0,
                    "updated_at": current_timestamp,
                })
            
            # Step 3: Create Milvus JSON and feed
            milvus_json_path = f"milvus_{source}_{text_id}.json"
            logger.info(f"Creating Milvus JSON file: {milvus_json_path}")
            try:
                milvus_json = self.create_milvus_json(milvus_documents, milvus_json_path)
                logger.info(f"Feeding Milvus JSON into collection")
                milvus_result = self.feed_milvus_from_json(milvus_json)
                logger.info(f"Milvus result: {milvus_result}")
            except Exception as e:
                logger.error(f"Error creating/feeding Milvus JSON: {e}", exc_info=True)
                milvus_result = f"Error: {str(e)}"
                milvus_json = milvus_json_path  # Keep path even on error
            
            # Step 4: Extract requirements/user stories and meeting information for Neo4j (if enabled)
            neo4j_result = None
            neo4j_json_path = None
            extracted_data = None
            meeting_data = None
            if extract_requirements:
                logger.info("Extracting requirements and user stories from text")
                extracted_data = self.extract_requirements_from_text(text)
                
                # Also extract meeting information using LLM
                logger.info("Extracting meeting information from text using LLM")
                meeting_data = self.extract_meeting_information(text)
                
                # Combine both extracted data
                has_data = (
                    (extracted_data and (extracted_data.get("projects") or extracted_data.get("requirements") or extracted_data.get("user_stories"))) or
                    (meeting_data and (meeting_data.get("meetings") or meeting_data.get("participants") or meeting_data.get("events")))
                )
                
                if has_data:
                    neo4j_json_path = f"neo4j_{source}_{text_id}.json"
                    logger.info(f"Creating Neo4j JSON file: {neo4j_json_path}")
                    neo4j_json = self.create_neo4j_json(
                        projects=extracted_data.get("projects", []) if extracted_data else [],
                        requirements=extracted_data.get("requirements", []) if extracted_data else [],
                        user_stories=extracted_data.get("user_stories", []) if extracted_data else [],
                        meetings=meeting_data.get("meetings", []) if meeting_data else [],
                        participants=meeting_data.get("participants", []) if meeting_data else [],
                        events=meeting_data.get("events", []) if meeting_data else [],
                        output_file=neo4j_json_path,
                    )
                    logger.info(f"Feeding Neo4j JSON into database")
                    neo4j_result = self.feed_neo4j_from_json(neo4j_json)
                else:
                    logger.info("No requirements, user stories, or meeting information found in text")
            
            return {
                "success": True,
                "text_length": len(text),
                "chunks_count": len(chunks),
                "milvus": {
                    "json_file": milvus_json,
                    "result": milvus_result,
                    "documents_count": len(milvus_documents),
                },
                "neo4j": {
                    "json_file": neo4j_json_path,
                    "result": neo4j_result,
                    "projects_count": len(extracted_data["projects"]) if extracted_data else 0,
                    "requirements_count": len(extracted_data["requirements"]) if extracted_data else 0,
                    "user_stories_count": len(extracted_data["user_stories"]) if extracted_data else 0,
                    "meetings_count": len(meeting_data["meetings"]) if meeting_data else 0,
                    "participants_count": len(meeting_data["participants"]) if meeting_data else 0,
                    "events_count": len(meeting_data["events"]) if meeting_data else 0,
                } if extract_requirements else None,
            }
        
        except Exception as e:
            logger.error(f"Error processing text automatically: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


# Import tools from tools directory
from tools.milvus_ingest_tool import (
    create_milvus_json_file,
    feed_milvus_from_json_file,
    process_file_for_ingestion,
    process_text_for_ingestion,
)
from tools.neo4j_ingest_tool import create_neo4j_json_file


def create_ingest_agent(
    name: str = "Ingest Agent",
    instructions: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    top_p: Optional[float] = None,
) -> Agent:
    """Create an Ingest Agent using OpenAI Agents pattern.
    
    Args:
        name: Agent name identifier
        instructions: Agent instructions/system prompt. If None, uses default.
        model: LLM model to use (default: "gpt-4o")
        temperature: Model temperature for randomness (default: 0.7)
        max_tokens: Maximum tokens in response (default: 2000)
        top_p: Model top_p parameter (optional)
        
    Returns:
        Agent instance configured for ingestion with Milvus and Neo4j JSON creation tools
        
    Example:
        from agents import Runner
        from agents.ingest_agent import create_ingest_agent
        
    agent = create_ingest_agent()
        result = await Runner.run(agent, "Create a Milvus JSON file with these documents: ...")
        print(result.final_output)
    """
    if instructions is None:
        instructions = (
            "You are an Ingest Agent that automatically processes files and text for ingestion into Milvus (Knowledge Base) and Neo4j (Knowledge Graph).\n\n"
            "You are also a Chunking Agent that automatically chunks documents into smaller chunks for embedding and storage.\n\n"
            "**AUTOMATIC PROCESSING CAPABILITIES:**\n\n"
            "When users upload files or send text:\n"
            "1. **For Files**: Use `process_file_for_ingestion` tool which automatically:\n"
            "   - Parses the file (.txt, .docx, .pdf)\n"
            "   - Chunks the text into smaller pieces\n"
            "   - Creates embeddings automatically\n"
            "   - Extracts requirements/user stories (if found)\n"
            "   - Creates JSON files with proper schemas\n"
            "   - Automatically feeds into Milvus and Neo4j\n\n"
            "2. **For Text**: Use `process_text_for_ingestion` tool which automatically:\n"
            "   - Chunks the text into smaller pieces\n"
            "   - Creates embeddings automatically\n"
            "   - Extracts requirements/user stories (if found)\n"
            "   - Creates JSON files with proper schemas\n"
            "   - Automatically feeds into Milvus and Neo4j\n\n"
            "**MANUAL TOOLS** (for advanced users):\n"
            "- `create_milvus_json_file`: Create Milvus JSON manually with specific documents\n"
            "- `create_neo4j_json_file`: Create Neo4j JSON manually with projects/requirements/user stories\n"
            "- `feed_milvus_from_json_file`: Feed existing Milvus JSON file into collection\n\n"
            "**WORKFLOW:**\n"
            "- When user uploads a file: Use `process_file_for_ingestion` with file path\n"
            "- When user sends text/message: Use `process_text_for_ingestion` with the text\n"
            "- The tools automatically handle chunking, embedding, and database ingestion\n"
            "- All processing happens automatically - no manual steps needed\n\n"
            "**IMPORTANT:**\n"
            "- Files support: .txt, .docx, .pdf\n"
            "- Chunking is automatic with configurable size (default: 500 characters)\n"
            "- Embeddings are generated automatically using intfloat/multilingual-e5-large\n"
            "- Requirements/user stories are extracted using pattern matching\n"
            "- All JSON files are automatically created with proper schemas\n"
            "- All data is automatically fed into databases\n"
            "- If user doesn't provide fields, fill them automatically\n"
            "- Format dates in ISO format (e.g., 22/10/2025 -> 2025-10-22)"
        )
    
    # Create model settings
    model_settings_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        model_settings_kwargs["top_p"] = top_p
    
    model_settings = ModelSettings(**model_settings_kwargs)
    
    # Create agent with tools (automatic processing tools + manual tools)
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=model_settings,
        tools=[
            # Automatic processing tools (preferred)
            process_file_for_ingestion,      # Process file automatically: parse, chunk, embed, ingest
            process_text_for_ingestion,      # Process text automatically: chunk, embed, ingest
            # Manual tools (for advanced users)
            create_milvus_json_file,         # Create Milvus JSON manually (automatically feeds into Milvus)
            create_neo4j_json_file,          # Create Neo4j JSON manually (automatically feeds into Neo4j)
            feed_milvus_from_json_file,      # Feed existing Milvus JSON file into collection
        ],
    )
    
    logger.info(f"Created Ingest Agent '{name}' with model {model}")
    return agent


__all__ = [
    "IngestAgent",
    "create_ingest_agent",
]
