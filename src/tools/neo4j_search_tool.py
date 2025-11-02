"""Neo4j search tool for function calling.

This tool provides a function-callable interface for searching Neo4j Knowledge Graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, Field

from core.config import load_config

logger = logging.getLogger(__name__)

# Neo4j queries for Project/Requirement/UserStory schema
SEARCH_PROJECTS_QUERY = """
MATCH (p:Project)
WHERE $term = '' OR 
      toLower(coalesce(p.name, '')) CONTAINS $term OR
      toLower(coalesce(p.description, '')) CONTAINS $term OR
      toLower(coalesce(p.project_id, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.description, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.project_id, '')) CONTAINS keyword)
RETURN id(p) AS node_id, labels(p) AS labels, p.project_id AS identifier,
       p.name AS name, p.description AS description,
       p.status AS status, p.version AS version,
       p.created_date AS created_date, p.updated_date AS updated_date,
       p.stakeholders AS stakeholders
ORDER BY 
  CASE WHEN toLower(coalesce(p.name, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) THEN 2
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.project_id, '')) CONTAINS keyword) THEN 3
       ELSE 4 END,
  p.created_date DESC
LIMIT $limit
"""

SEARCH_REQUIREMENTS_QUERY = """
MATCH (r:Requirement)
WHERE $term = '' OR 
      toLower(coalesce(r.title, '')) CONTAINS $term OR
      toLower(coalesce(r.description, '')) CONTAINS $term OR
      toLower(coalesce(r.req_id, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.title, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.description, '')) CONTAINS keyword)
RETURN id(r) AS node_id, labels(r) AS labels, r.req_id AS identifier,
       r.title AS title, r.description AS description,
       r.type AS type, r.priority AS priority, r.status AS status,
       r.version AS version, r.source AS source,
       r.acceptance_criteria AS acceptance_criteria,
       r.constraints AS constraints, r.assumptions AS assumptions,
       r.created_date AS created_date, r.updated_date AS updated_date
ORDER BY 
  CASE 
    WHEN r.priority = 'critical' THEN 1
    WHEN r.priority = 'high' THEN 2
    WHEN r.priority = 'medium' THEN 3
    ELSE 4
  END,
  CASE WHEN toLower(coalesce(r.title, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(r.title, '')) CONTAINS keyword) THEN 2
       ELSE 3 END
LIMIT $limit
"""

SEARCH_USER_STORIES_QUERY = """
MATCH (us:UserStory)
WHERE $term = '' OR 
      toLower(coalesce(us.title, '')) CONTAINS $term OR
      toLower(coalesce(us.description, '')) CONTAINS $term OR
      toLower(coalesce(us.story_id, '')) CONTAINS $term OR
      toLower(coalesce(us.i_want, '')) CONTAINS $term OR
      toLower(coalesce(us.as_a, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.title, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.description, '')) CONTAINS keyword)
RETURN id(us) AS node_id, labels(us) AS labels, us.story_id AS identifier,
       us.title AS title, 
       us.as_a AS as_a,
       us.i_want AS i_want,
       us.so_that AS so_that,
       us.description AS description,
       us.status AS status, us.priority AS priority,
       us.story_points AS story_points, us.sprint AS sprint,
       us.acceptance_criteria AS acceptance_criteria,
       us.created_date AS created_date, us.updated_date AS updated_date
ORDER BY 
  CASE 
    WHEN us.priority = 'critical' THEN 1
    WHEN us.priority = 'high' THEN 2
    WHEN us.priority = 'medium' THEN 3
    ELSE 4
  END,
  CASE WHEN toLower(coalesce(us.title, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(us.title, '')) CONTAINS keyword) THEN 2
       ELSE 3 END
LIMIT $limit
"""

SEARCH_MEETINGS_QUERY = """
MATCH (m:Meeting)
WHERE $term = '' OR 
      toLower(coalesce(m.title, '')) CONTAINS $term OR
      toLower(coalesce(m.description, '')) CONTAINS $term OR
      toLower(coalesce(m.meeting_id, '')) CONTAINS $term OR
      toLower(coalesce(m.meeting_type, '')) CONTAINS $term OR
      toLower(coalesce(m.location, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(m.title, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(m.description, '')) CONTAINS keyword) OR
      ANY(item IN m.agenda WHERE toLower(item) CONTAINS $term) OR
      ANY(item IN m.action_items WHERE toLower(item) CONTAINS $term) OR
      ANY(item IN m.decisions WHERE toLower(item) CONTAINS $term)
RETURN id(m) AS node_id, labels(m) AS labels, m.meeting_id AS identifier,
       m.title AS title, m.description AS description,
       m.date AS date, m.start_time AS start_time, m.end_time AS end_time,
       m.location AS location, m.meeting_type AS meeting_type,
       m.meeting_status AS meeting_status, m.agenda AS agenda,
       m.action_items AS action_items, m.decisions AS decisions,
       m.next_meeting_date AS next_meeting_date, m.next_meeting_time AS next_meeting_time,
       m.project_id AS project_id,
       m.created_date AS created_date, m.updated_date AS updated_date
ORDER BY 
  CASE WHEN toLower(coalesce(m.title, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(m.title, '')) CONTAINS keyword) THEN 2
       ELSE 3 END,
  m.date DESC
LIMIT $limit
"""

SEARCH_PARTICIPANTS_QUERY = """
MATCH (p:Participant)
WHERE $term = '' OR 
      toLower(coalesce(p.name, '')) CONTAINS $term OR
      toLower(coalesce(p.participant_id, '')) CONTAINS $term OR
      toLower(coalesce(p.role, '')) CONTAINS $term OR
      toLower(coalesce(p.email, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.role, '')) CONTAINS keyword)
RETURN id(p) AS node_id, labels(p) AS labels, p.participant_id AS identifier,
       p.name AS name, p.role AS role, p.email AS email,
       p.attendance_status AS attendance_status, p.notes AS notes
ORDER BY 
  CASE WHEN toLower(coalesce(p.name, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(p.name, '')) CONTAINS keyword) THEN 2
       ELSE 3 END,
  p.name ASC
LIMIT $limit
"""

SEARCH_EVENTS_QUERY = """
MATCH (e:Event)
WHERE $term = '' OR 
      toLower(coalesce(e.name, '')) CONTAINS $term OR
      toLower(coalesce(e.description, '')) CONTAINS $term OR
      toLower(coalesce(e.event_id, '')) CONTAINS $term OR
      toLower(coalesce(e.event_type, '')) CONTAINS $term OR
      toLower(coalesce(e.assigned_to, '')) CONTAINS $term OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(e.name, '')) CONTAINS keyword) OR
      ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(e.description, '')) CONTAINS keyword)
RETURN id(e) AS node_id, labels(e) AS labels, e.event_id AS identifier,
       e.name AS name, e.description AS description,
       e.event_type AS event_type, e.date AS date, e.time AS time,
       e.priority AS priority, e.status AS status,
       e.assigned_to AS assigned_to, e.due_date AS due_date,
       e.meeting_id AS meeting_id, e.related_entity_ids AS related_entity_ids
ORDER BY 
  CASE 
    WHEN e.priority = 'critical' THEN 1
    WHEN e.priority = 'high' THEN 2
    WHEN e.priority = 'medium' THEN 3
    ELSE 4
  END,
  CASE WHEN toLower(coalesce(e.name, '')) CONTAINS $term THEN 1 
       WHEN ANY(keyword IN split($term, ' ') WHERE toLower(coalesce(e.name, '')) CONTAINS keyword) THEN 2
       ELSE 3 END,
  e.date DESC, e.time DESC
LIMIT $limit
"""

GET_NODE_CONTEXT_QUERY = """
MATCH (n) WHERE id(n) = $node_id
OPTIONAL MATCH (n)-[r]->(m)
RETURN 'outgoing' AS direction, type(r) AS rel_type, id(m) AS target_id, null AS source_id,
       labels(m) AS target_labels, null AS source_labels,
       CASE 
         WHEN 'Project' IN labels(m) THEN m.project_id
         WHEN 'Requirement' IN labels(m) THEN m.req_id
         WHEN 'UserStory' IN labels(m) THEN m.story_id
         WHEN 'Meeting' IN labels(m) THEN m.meeting_id
         WHEN 'Participant' IN labels(m) THEN m.participant_id
         WHEN 'Event' IN labels(m) THEN m.event_id
         ELSE null
       END AS target_identifier, null AS source_identifier,
       properties(m) AS target_props, null AS source_props,
       properties(r) AS rel_props
UNION ALL
MATCH (m)-[r]->(n) WHERE id(n) = $node_id
RETURN 'incoming' AS direction, type(r) AS rel_type, null AS target_id, id(m) AS source_id,
       null AS target_labels, labels(m) AS source_labels,
       null AS target_identifier,
       CASE 
         WHEN 'Project' IN labels(m) THEN m.project_id
         WHEN 'Requirement' IN labels(m) THEN m.req_id
         WHEN 'UserStory' IN labels(m) THEN m.story_id
         WHEN 'Meeting' IN labels(m) THEN m.meeting_id
         WHEN 'Participant' IN labels(m) THEN m.participant_id
         WHEN 'Event' IN labels(m) THEN m.event_id
         ELSE null
       END AS source_identifier,
       null AS target_props, properties(m) AS source_props,
       properties(r) AS rel_props
LIMIT $limit
"""


class Neo4jSearchInput(BaseModel):
    """Input model for Neo4j search tool.
    
    Format:
    {
        "query": "user question or search text",
        "top_k": 50  # Optional, default is 50
    }
    """
    
    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=50, description="Number of top results to return", ge=1, le=1000)


class Neo4jSearchOutput(BaseModel):
    """Output model for Neo4j search tool.
    
    Format:
    {
        "identifiers": ["PROJ-001", "REQ-002", ...],  # Node identifiers
        "results": [
            {
                "node_id": 123,
                "identifier": "PROJ-001",
                "labels": ["Project"],
                "properties": {...},
                "relationships": [...]
            },
            ...
        ]
    }
    """
    
    identifiers: List[str] = Field(default_factory=list, description="Array of node identifiers")
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed search results with node properties and relationships"
    )


class Neo4jSearchTool:
    """Tool for searching Neo4j Knowledge Graph."""
    
    def __init__(self, driver: Optional[Any] = None):
        """Initialize Neo4j search tool.
        
        Args:
            driver: Neo4j driver instance (optional, will create if not provided)
        """
        self.config = load_config()
        
        # Initialize Neo4j
        if driver is None:
            neo4j_uri = self.config.neo4j.uri
            # Convert http/https to bolt if needed
            if neo4j_uri.startswith("http://"):
                neo4j_uri = neo4j_uri.replace("http://", "bolt://")
            elif neo4j_uri.startswith("https://"):
                neo4j_uri = neo4j_uri.replace("https://", "bolt://")
            elif not neo4j_uri.startswith("bolt://"):
                neo4j_uri = f"bolt://{neo4j_uri}"
            
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(self.config.neo4j.user, self.config.neo4j.password),
            )
        else:
            self.driver = driver
    
    def _extract_project_keywords(self, query: str) -> str:
        """Extract project-specific keywords from query for better matching."""
        term = query.lower().strip()
        keywords = [k.strip() for k in term.split() if len(k.strip()) > 2]
        
        project_keywords = []
        for keyword in keywords:
            if keyword in ["alpha", "beta", "gamma", "delta", "epsilon"]:
                project_keywords.append(keyword)
            if keyword.startswith("proj"):
                query_words = term.split()
                for i, word in enumerate(query_words):
                    if word == "project" and i + 1 < len(query_words):
                        next_word = query_words[i + 1].strip()
                        if next_word in ["alpha", "beta", "gamma", "delta", "epsilon"]:
                            project_keywords.append(next_word)
        
        if project_keywords:
            return project_keywords[0]
        return term if term else "project"
    
    def _get_node_relationships(self, node_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get relationships for a specific node."""
        try:
            if self.driver is None:
                return []
            
            with self.driver.session() as session:
                records = session.run(
                    GET_NODE_CONTEXT_QUERY,
                    {"node_id": node_id, "limit": limit},
                ).data()

                relationships = []
                for rec in records:
                    rel_info = {
                        "direction": rec.get("direction"),
                        "type": rec.get("rel_type"),
                        "rel_properties": rec.get("rel_props") or {},
                    }
                    
                    if rec.get("direction") == "outgoing":
                        rel_info["target_id"] = rec.get("target_id")
                        rel_info["target_labels"] = rec.get("target_labels") or []
                        rel_info["target_identifier"] = rec.get("target_identifier")
                        rel_info["target_properties"] = rec.get("target_props") or {}
                    else:
                        rel_info["source_id"] = rec.get("source_id")
                        rel_info["source_labels"] = rec.get("source_labels") or []
                        rel_info["source_identifier"] = rec.get("source_identifier")
                        rel_info["source_properties"] = rec.get("source_props") or {}

                    relationships.append(rel_info)

                return relationships

        except ServiceUnavailable as e:
            logger.warning(f"Neo4j connection unavailable when getting relationships: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting node relationships: {e}", exc_info=True)
            return []
    
    def search(self, input_data: Neo4jSearchInput) -> Neo4jSearchOutput:
        """Search Neo4j for relevant Project, Requirement, and UserStory nodes.
        
        This is the main function for function calling.
        
        Args:
            input_data: Neo4jSearchInput with query and top_k
            
        Returns:
            Neo4jSearchOutput with identifiers and detailed results
        """
        try:
            # Verify driver is available
            if self.driver is None:
                logger.warning("Neo4j driver is not initialized. Skipping search.")
                return Neo4jSearchOutput()
            
            # Test connection
            try:
                self.driver.verify_connectivity()
                logger.debug("Neo4j connection verified successfully")
            except ServiceUnavailable as e:
                logger.error(f"Neo4j server is not available: {e}")
                logger.error(f"Neo4j URI: {self.config.neo4j.uri}, User: {self.config.neo4j.user}")
                return Neo4jSearchOutput()
            except Exception as e:
                logger.error(f"Neo4j connection error: {e}")
                logger.error(f"Neo4j URI: {self.config.neo4j.uri}, User: {self.config.neo4j.user}")
                return Neo4jSearchOutput()
            
            # Extract keywords from query for better matching
            term = input_data.query.lower().strip()
            search_term = self._extract_project_keywords(input_data.query)
            
            logger.info(f"Searching Neo4j with query: '{input_data.query}', term: '{term}', search_term: '{search_term}'")
            
            identifiers = []
            results = []
            seen_node_ids = set()  # Track seen node_ids to avoid duplicates

            with self.driver.session() as session:
                # Search Projects
                project_records = session.run(
                    SEARCH_PROJECTS_QUERY,
                    {"term": search_term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(project_records)} projects")
                
                # Debug: Log first raw record to see what query returns
                if project_records and len(project_records) > 0:
                    logger.debug(f"First Project Raw Record: {project_records[0]}")
                    logger.debug(f"First Project Record Keys: {list(project_records[0].keys())}")

                for i, rec in enumerate(project_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)
                    
                    # Log raw record first to debug
                    logger.debug(f"Raw Project Record {i}: {rec}")

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["name", "description", "status", "version", 
                                "created_date", "updated_date", "stakeholders"]:
                        value = rec.get(key)
                        if value is not None:
                            properties[key] = value
                        else:
                            logger.debug(f"  - Field '{key}' is None or missing in record")

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    # Log project search results with ALL properties
                    logger.info(f"Neo4j Project Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Node ID: {node_id}")
                    logger.info(f"  - Labels: {rec.get('labels', [])}")
                    logger.info(f"  - Name: {properties.get('name', 'N/A')}")
                    if properties.get('description'):
                        desc = properties['description'][:200] + "..." if len(properties['description']) > 200 else properties['description']
                        logger.info(f"  - Description: {desc}")
                    logger.info(f"  - Status: {properties.get('status', 'N/A')}")
                    logger.info(f"  - Version: {properties.get('version', 'N/A')}")
                    logger.info(f"  - Created Date: {properties.get('created_date', 'N/A')}")
                    logger.info(f"  - Updated Date: {properties.get('updated_date', 'N/A')}")
                    logger.info(f"  - Stakeholders: {properties.get('stakeholders', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")
                    
                    # Log detailed relationships
                    if relationships:
                        for j, rel in enumerate(relationships, 1):
                            direction = rel.get("direction", "N/A")
                            rel_type = rel.get("type", "N/A")
                            if direction == "outgoing":
                                target_id = rel.get("target_identifier") or rel.get("target_id")
                                target_labels = rel.get("target_labels", [])
                                logger.info(f"    Relationship {j}: {identifier} --[{rel_type}]--> {target_id} ({', '.join(target_labels)})")
                            elif direction == "incoming":
                                source_id = rel.get("source_identifier") or rel.get("source_id")
                                source_labels = rel.get("source_labels", [])
                                logger.info(f"    Relationship {j}: {source_id} ({', '.join(source_labels)}) --[{rel_type}]--> {identifier}")
                    
                    # Log all properties for debugging
                    logger.debug(f"  - All Properties: {properties}")
                    logger.debug(f"  - All Relationships: {relationships}")

                # Search Requirements
                req_records = session.run(
                    SEARCH_REQUIREMENTS_QUERY,
                    {"term": term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(req_records)} requirements")
                
                # If searching for REQ-XXX format, try exact match too
                if input_data.query.strip().upper().startswith("REQ-"):
                    req_id = input_data.query.strip().upper()
                    exact_req_records = session.run(
                        "MATCH (r:Requirement) WHERE r.req_id = $req_id RETURN id(r) AS node_id, labels(r) AS labels, r.req_id AS identifier, r.title AS title, r.description AS description, r.type AS type, r.priority AS priority, r.status AS status, r.version AS version, r.source AS source, r.acceptance_criteria AS acceptance_criteria, r.constraints AS constraints, r.assumptions AS assumptions, r.created_date AS created_date, r.updated_date AS updated_date LIMIT 1",
                        {"req_id": req_id},
                    ).data()
                    logger.debug(f"Found {len(exact_req_records)} requirements with exact req_id match: {req_id}")
                    if exact_req_records and len(exact_req_records) > 0:
                        req_records = exact_req_records + req_records  # Prepend exact match

                for i, rec in enumerate(req_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    if isinstance(node_id, str):
                        try:
                            node_id = int(node_id)
                        except:
                            continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["title", "description", "type", "priority", "status",
                                "version", "source", "acceptance_criteria", "constraints",
                                "assumptions", "created_date", "updated_date"]:
                        if rec.get(key) is not None:
                            properties[key] = rec.get(key)

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    # Log requirement search results with ALL properties
                    logger.info(f"Neo4j Requirement Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Node ID: {node_id}")
                    logger.info(f"  - Labels: {rec.get('labels', [])}")
                    logger.info(f"  - Title: {properties.get('title', 'N/A')}")
                    if properties.get('description'):
                        desc = properties['description'][:200] + "..." if len(properties['description']) > 200 else properties['description']
                        logger.info(f"  - Description: {desc}")
                    logger.info(f"  - Type: {properties.get('type', 'N/A')}")
                    logger.info(f"  - Priority: {properties.get('priority', 'N/A')}")
                    logger.info(f"  - Status: {properties.get('status', 'N/A')}")
                    logger.info(f"  - Version: {properties.get('version', 'N/A')}")
                    logger.info(f"  - Created Date: {properties.get('created_date', 'N/A')}")
                    logger.info(f"  - Updated Date: {properties.get('updated_date', 'N/A')}")
                    logger.info(f"  - Source: {properties.get('source', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")
                    
                    # Log detailed relationships
                    if relationships:
                        for j, rel in enumerate(relationships, 1):
                            direction = rel.get("direction", "N/A")
                            rel_type = rel.get("type", "N/A")
                            if direction == "outgoing":
                                target_id = rel.get("target_identifier") or rel.get("target_id")
                                target_labels = rel.get("target_labels", [])
                                logger.info(f"    Relationship {j}: {identifier} --[{rel_type}]--> {target_id} ({', '.join(target_labels)})")
                            elif direction == "incoming":
                                source_id = rel.get("source_identifier") or rel.get("source_id")
                                source_labels = rel.get("source_labels", [])
                                logger.info(f"    Relationship {j}: {source_id} ({', '.join(source_labels)}) --[{rel_type}]--> {identifier}")
                    
                    # Log all properties for debugging
                    logger.debug(f"  - All Properties: {properties}")
                    logger.debug(f"  - All Relationships: {relationships}")
                    logger.debug(f"  - Raw Record: {rec}")

                # Search User Stories
                story_records = session.run(
                    SEARCH_USER_STORIES_QUERY,
                    {"term": search_term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(story_records)} user stories")

                for i, rec in enumerate(story_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    if isinstance(node_id, str):
                        try:
                            node_id = int(node_id)
                        except:
                            continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["title", "as_a", "i_want", "so_that", "description",
                                "status", "priority", "story_points", "sprint",
                                "acceptance_criteria", "created_date", "updated_date"]:
                        if rec.get(key) is not None:
                            properties[key] = rec.get(key)

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    # Log user story search results with ALL properties
                    logger.info(f"Neo4j UserStory Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Node ID: {node_id}")
                    logger.info(f"  - Labels: {rec.get('labels', [])}")
                    logger.info(f"  - Title: {properties.get('title', 'N/A')}")
                    logger.info(f"  - As a: {properties.get('as_a', 'N/A')}")
                    logger.info(f"  - I want: {properties.get('i_want', 'N/A')}")
                    logger.info(f"  - So that: {properties.get('so_that', 'N/A')}")
                    logger.info(f"  - Status: {properties.get('status', 'N/A')}")
                    logger.info(f"  - Priority: {properties.get('priority', 'N/A')}")
                    logger.info(f"  - Created Date: {properties.get('created_date', 'N/A')}")
                    logger.info(f"  - Updated Date: {properties.get('updated_date', 'N/A')}")
                    logger.info(f"  - Story Points: {properties.get('story_points', 'N/A')}")
                    logger.info(f"  - Sprint: {properties.get('sprint', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")
                    
                    # Log detailed relationships
                    if relationships:
                        for j, rel in enumerate(relationships, 1):
                            direction = rel.get("direction", "N/A")
                            rel_type = rel.get("type", "N/A")
                            if direction == "outgoing":
                                target_id = rel.get("target_identifier") or rel.get("target_id")
                                target_labels = rel.get("target_labels", [])
                                logger.info(f"    Relationship {j}: {identifier} --[{rel_type}]--> {target_id} ({', '.join(target_labels)})")
                            elif direction == "incoming":
                                source_id = rel.get("source_identifier") or rel.get("source_id")
                                source_labels = rel.get("source_labels", [])
                                logger.info(f"    Relationship {j}: {source_id} ({', '.join(source_labels)}) --[{rel_type}]--> {identifier}")
                    
                    # Log all properties for debugging
                    logger.debug(f"  - All Properties: {properties}")
                    logger.debug(f"  - All Relationships: {relationships}")
                    logger.debug(f"  - Raw Record: {rec}")

                # Search Meetings
                meeting_records = session.run(
                    SEARCH_MEETINGS_QUERY,
                    {"term": term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(meeting_records)} meetings")

                for i, rec in enumerate(meeting_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    if isinstance(node_id, str):
                        try:
                            node_id = int(node_id)
                        except:
                            continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["title", "description", "date", "start_time", "end_time",
                                "location", "meeting_type", "meeting_status", "agenda",
                                "action_items", "decisions", "next_meeting_date",
                                "next_meeting_time", "project_id", "created_date", "updated_date"]:
                        if rec.get(key) is not None:
                            properties[key] = rec.get(key)

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    logger.info(f"Neo4j Meeting Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Node ID: {node_id}")
                    logger.info(f"  - Title: {properties.get('title', 'N/A')}")
                    if properties.get('description'):
                        desc = properties['description'][:200] + "..." if len(properties['description']) > 200 else properties['description']
                        logger.info(f"  - Description: {desc}")
                    logger.info(f"  - Date: {properties.get('date', 'N/A')}")
                    logger.info(f"  - Meeting Type: {properties.get('meeting_type', 'N/A')}")
                    logger.info(f"  - Status: {properties.get('meeting_status', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")

                # Search Participants
                participant_records = session.run(
                    SEARCH_PARTICIPANTS_QUERY,
                    {"term": term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(participant_records)} participants")

                for i, rec in enumerate(participant_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    if isinstance(node_id, str):
                        try:
                            node_id = int(node_id)
                        except:
                            continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["name", "role", "email", "attendance_status", "notes"]:
                        if rec.get(key) is not None:
                            properties[key] = rec.get(key)

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    logger.info(f"Neo4j Participant Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Name: {properties.get('name', 'N/A')}")
                    logger.info(f"  - Role: {properties.get('role', 'N/A')}")
                    logger.info(f"  - Email: {properties.get('email', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")

                # Search Events
                event_records = session.run(
                    SEARCH_EVENTS_QUERY,
                    {"term": term, "limit": input_data.top_k},
                ).data()
                logger.debug(f"Found {len(event_records)} events")

                for i, rec in enumerate(event_records, 1):
                    node_id = rec.get("node_id")
                    if node_id is None:
                        continue
                    
                    if isinstance(node_id, str):
                        try:
                            node_id = int(node_id)
                        except:
                            continue
                    
                    # Skip if already seen (no merge - just skip duplicates)
                    if node_id in seen_node_ids:
                        continue
                    seen_node_ids.add(node_id)

                    relationships = self._get_node_relationships(node_id, limit=10)

                    properties = {}
                    for key in ["name", "description", "event_type", "date", "time",
                                "priority", "status", "assigned_to", "due_date",
                                "meeting_id", "related_entity_ids"]:
                        if rec.get(key) is not None:
                            properties[key] = rec.get(key)

                    identifier = rec.get("identifier")
                    if identifier:
                        identifiers.append(identifier)
                    
                    result = {
                        "node_id": node_id,
                        "identifier": identifier,
                        "labels": rec.get("labels", []),
                        "properties": properties,
                        "relationships": relationships,
                    }
                    results.append(result)
                    
                    logger.info(f"Neo4j Event Result {i}:")
                    logger.info(f"  - Identifier: {identifier}")
                    logger.info(f"  - Name: {properties.get('name', 'N/A')}")
                    if properties.get('description'):
                        desc = properties['description'][:200] + "..." if len(properties['description']) > 200 else properties['description']
                        logger.info(f"  - Description: {desc}")
                    logger.info(f"  - Event Type: {properties.get('event_type', 'N/A')}")
                    logger.info(f"  - Priority: {properties.get('priority', 'N/A')}")
                    logger.info(f"  - Status: {properties.get('status', 'N/A')}")
                    logger.info(f"  - Assigned To: {properties.get('assigned_to', 'N/A')}")
                    logger.info(f"  - Relationships: {len(relationships)}")

            output = Neo4jSearchOutput(
                identifiers=identifiers[:input_data.top_k * 50],  # Limit identifiers
                results=results[:input_data.top_k * 50],  # Limit results
            )
            logger.info(f"Neo4j search completed: {len(output.identifiers)} identifiers, {len(output.results)} results")
            logger.info(f"Neo4j identifiers: {output.identifiers}")
            return output

        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection unavailable: {e}")
            logger.error(f"Please check: 1) Neo4j server is running, 2) URI is correct (bolt://localhost:7687), 3) Credentials are correct")
            return Neo4jSearchOutput()
        except Exception as e:
            logger.error(f"Error searching Neo4j: {e}", exc_info=True)
            logger.error(f"Neo4j URI: {self.config.neo4j.uri}, User: {self.config.neo4j.user}")
            return Neo4jSearchOutput()
    
    def close(self):
        """Close Neo4j driver connection."""
        try:
            if hasattr(self, "driver") and self.driver:
                self.driver.close()
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {e}")


# Function for OpenAI function calling format
def get_neo4j_search_function_schema() -> Dict[str, Any]:
    """Get OpenAI function calling schema for Neo4j search tool.
    
    Returns:
        Dictionary with function schema for OpenAI function calling
    """
    return {
        "type": "function",
        "function": {
            "name": "search_neo4j",
            "description": "Search the Knowledge Graph (Neo4j) for relevant nodes including Projects, Requirements, UserStories, Meetings, Participants, and Events",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (default: 50, max: 1000)",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 1000
                    }
                },
                "required": ["query"]
            }
        }
    }


__all__ = ["Neo4jSearchTool", "Neo4jSearchInput", "Neo4jSearchOutput", "get_neo4j_search_function_schema"]

