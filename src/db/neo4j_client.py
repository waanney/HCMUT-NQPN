"""
Requirements Engineering Neo4j Client
Manages schema and operations for SRS/User Stories analysis, contradiction detection, and suggestions.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)


# ========================
# Enums for type safety
# ========================
class RequirementType(str, Enum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non-functional"
    BUSINESS = "business"
    TECHNICAL = "technical"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RequirementStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    DEPRECATED = "deprecated"


class StoryStatus(str, Enum):
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class ContradictionType(str, Enum):
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    SCOPE = "scope"
    PRIORITY = "priority"


class ContradictionSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class SuggestionType(str, Enum):
    CLARIFICATION = "clarification"
    MERGE = "merge"
    SPLIT = "split"
    REFORMULATION = "reformulation"
    ADD_CONSTRAINT = "add_constraint"


# ========================
# Pydantic Models
# ========================
class Project(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0"
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    status: str = "active"
    stakeholders: List[str] = Field(default_factory=list)


class Requirement(BaseModel):
    req_id: str
    title: str
    description: str
    type: RequirementType = RequirementType.FUNCTIONAL
    priority: Priority = Priority.MEDIUM
    status: RequirementStatus = RequirementStatus.DRAFT
    created_date: Optional[str] = None
    updated_date: Optional[str] = None
    version: str = "1.0"
    source: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class UserStory(BaseModel):
    story_id: str
    title: str
    as_a: str  # Role/persona
    i_want: str  # Goal/action
    so_that: str  # Benefit/value
    description: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    story_points: Optional[int] = None
    status: StoryStatus = StoryStatus.BACKLOG
    sprint: Optional[str] = None
    acceptance_criteria: List[str] = Field(default_factory=list)
    created_date: Optional[str] = None
    updated_date: Optional[str] = None


class Actor(BaseModel):
    actor_id: str
    name: str
    type: str = "user"  # user, system, external_service, stakeholder
    description: Optional[str] = None
    role: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    contradiction_id: str
    type: ContradictionType
    severity: ContradictionSeverity
    description: str
    detected_date: Optional[str] = None
    resolution_status: str = "open"  # open, resolved, accepted, escalated
    detection_method: str = "llm_analysis"  # rule_based, llm_analysis, manual
    confidence_score: Optional[float] = None


class Suggestion(BaseModel):
    suggestion_id: str
    type: SuggestionType
    description: str
    rationale: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    status: str = "pending"  # pending, accepted, rejected, implemented
    created_date: Optional[str] = None
    ai_confidence: Optional[float] = None


class Entity(BaseModel):
    entity_id: str
    name: str
    type: str = "data_object"  # data_object, system_component, business_concept
    description: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)


class Constraint(BaseModel):
    constraint_id: str
    type: str  # technical, business, legal, temporal, resource
    description: str
    impact_level: Optional[str] = None
    enforced: bool = True


class UseCase(BaseModel):
    usecase_id: str
    name: str
    description: Optional[str] = None
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    main_flow: List[str] = Field(default_factory=list)
    alternative_flows: List[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM


class Meeting(BaseModel):
    meeting_id: str
    title: str
    description: Optional[str] = None
    date: Optional[str] = None  # ISO format date
    start_time: Optional[str] = None  # Time in HH:MM format
    end_time: Optional[str] = None  # Time in HH:MM format
    location: Optional[str] = None
    meeting_type: str = "general"  # general, kickoff, review, planning, retrospective, etc.
    meeting_status: str = "completed"  # scheduled, in_progress, completed, cancelled
    agenda: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    next_meeting_date: Optional[str] = None  # ISO format date
    next_meeting_time: Optional[str] = None  # Time in HH:MM format
    project_id: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None


class Participant(BaseModel):
    participant_id: str
    name: str
    role: Optional[str] = None  # Project Lead, Data Engineer, UX Designer, etc.
    email: Optional[str] = None
    attendance_status: str = "present"  # present, absent, late, excused
    notes: Optional[str] = None


class Event(BaseModel):
    event_id: str
    name: str
    description: Optional[str] = None
    event_type: str = "discussion"  # discussion, decision, action_item, deadline, milestone
    date: Optional[str] = None  # ISO format date
    time: Optional[str] = None  # Time in HH:MM format
    priority: Priority = Priority.MEDIUM
    status: str = "open"  # open, in_progress, completed, cancelled
    assigned_to: Optional[str] = None  # participant_id or name
    due_date: Optional[str] = None  # ISO format date
    meeting_id: Optional[str] = None
    related_entity_ids: List[str] = Field(default_factory=list)  # IDs of related requirements, stories, etc.


# ========================
# Neo4j Configuration
# ========================
@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str


def get_driver(cfg: Neo4jConfig):
    return GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))


# ========================
# Schema Initialization
# ========================
SCHEMA_CYPHER = [
    # Constraints
    "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE",
    "CREATE CONSTRAINT req_id_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.req_id IS UNIQUE",
    "CREATE CONSTRAINT story_id_unique IF NOT EXISTS FOR (s:UserStory) REQUIRE s.story_id IS UNIQUE",
    "CREATE CONSTRAINT actor_id_unique IF NOT EXISTS FOR (a:Actor) REQUIRE a.actor_id IS UNIQUE",
    "CREATE CONSTRAINT contradiction_id_unique IF NOT EXISTS FOR (c:Contradiction) REQUIRE c.contradiction_id IS UNIQUE",
    "CREATE CONSTRAINT suggestion_id_unique IF NOT EXISTS FOR (s:Suggestion) REQUIRE s.suggestion_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT constraint_id_unique IF NOT EXISTS FOR (c:Constraint) REQUIRE c.constraint_id IS UNIQUE",
    "CREATE CONSTRAINT usecase_id_unique IF NOT EXISTS FOR (u:UseCase) REQUIRE u.usecase_id IS UNIQUE",
    "CREATE CONSTRAINT meeting_id_unique IF NOT EXISTS FOR (m:Meeting) REQUIRE m.meeting_id IS UNIQUE",
    "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.participant_id IS UNIQUE",
    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    
    # Indexes
    "CREATE INDEX req_type_idx IF NOT EXISTS FOR (r:Requirement) ON (r.type)",
    "CREATE INDEX req_status_idx IF NOT EXISTS FOR (r:Requirement) ON (r.status)",
    "CREATE INDEX req_priority_idx IF NOT EXISTS FOR (r:Requirement) ON (r.priority)",
    "CREATE INDEX story_status_idx IF NOT EXISTS FOR (s:UserStory) ON (s.status)",
    "CREATE INDEX story_priority_idx IF NOT EXISTS FOR (s:UserStory) ON (s.priority)",
    "CREATE INDEX contradiction_severity_idx IF NOT EXISTS FOR (c:Contradiction) ON (c.severity)",
    "CREATE INDEX contradiction_status_idx IF NOT EXISTS FOR (c:Contradiction) ON (c.resolution_status)",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX project_status_idx IF NOT EXISTS FOR (p:Project) ON (p.status)",
    "CREATE INDEX meeting_date_idx IF NOT EXISTS FOR (m:Meeting) ON (m.date)",
    "CREATE INDEX meeting_status_idx IF NOT EXISTS FOR (m:Meeting) ON (m.meeting_status)",
    "CREATE INDEX participant_name_idx IF NOT EXISTS FOR (p:Participant) ON (p.name)",
    "CREATE INDEX event_date_idx IF NOT EXISTS FOR (e:Event) ON (e.date)",
    "CREATE INDEX event_type_idx IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
]

# Fulltext indexes (separate handling)
FULLTEXT_INDEXES = [
    """
    CREATE FULLTEXT INDEX requirement_text_search IF NOT EXISTS
    FOR (r:Requirement) ON EACH [r.title, r.description]
    """,
    """
    CREATE FULLTEXT INDEX story_text_search IF NOT EXISTS
    FOR (s:UserStory) ON EACH [s.title, s.description, s.as_a, s.i_want, s.so_that]
    """,
    """
    CREATE FULLTEXT INDEX meeting_text_search IF NOT EXISTS
    FOR (m:Meeting) ON EACH [m.title, m.description, m.agenda, m.decisions, m.action_items]
    """,
    """
    CREATE FULLTEXT INDEX event_text_search IF NOT EXISTS
    FOR (e:Event) ON EACH [e.name, e.description]
    """,
]


def init_schema(driver):
    """Initialize Neo4j schema with constraints and indexes."""
    with driver.session() as session:
        for cypher in SCHEMA_CYPHER:
            try:
                session.run(cypher)
            except Exception as e:
                print(f"Warning creating constraint/index: {e}")
        
        for cypher in FULLTEXT_INDEXES:
            try:
                session.run(cypher)
            except Exception as e:
                print(f"Warning creating fulltext index: {e}")


# ========================
# CRUD Operations
# ========================

# Project Operations
def create_project(driver, project: Project):
    """Create a new project node."""
    with driver.session() as session:
        session.run(
            """
            CREATE (p:Project {
                project_id: $project_id,
                name: $name,
                description: $description,
                version: $version,
                created_date: $created_date,
                updated_date: $updated_date,
                status: $status,
                stakeholders: $stakeholders
            })
            """,
            project_id=project.project_id,
            name=project.name,
            description=project.description,
            version=project.version,
            created_date=project.created_date or datetime.now().isoformat(),
            updated_date=project.updated_date or datetime.now().isoformat(),
            status=project.status,
            stakeholders=project.stakeholders,
        )


def create_requirement(driver, project_id: str, requirement: Requirement):
    """Create a requirement and link to project."""
    with driver.session() as session:
        session.run(
            """
            MATCH (p:Project {project_id: $project_id})
            CREATE (r:Requirement {
                req_id: $req_id,
                title: $title,
                description: $description,
                type: $type,
                priority: $priority,
                status: $status,
                created_date: $created_date,
                updated_date: $updated_date,
                version: $version,
                source: $source,
                acceptance_criteria: $acceptance_criteria,
                constraints: $constraints,
                assumptions: $assumptions
            })
            CREATE (p)-[:CONTAINS_REQUIREMENT]->(r)
            """,
            project_id=project_id,
            req_id=requirement.req_id,
            title=requirement.title,
            description=requirement.description,
            type=requirement.type.value,
            priority=requirement.priority.value,
            status=requirement.status.value,
            created_date=requirement.created_date or datetime.now().isoformat(),
            updated_date=requirement.updated_date or datetime.now().isoformat(),
            version=requirement.version,
            source=requirement.source,
            acceptance_criteria=requirement.acceptance_criteria,
            constraints=requirement.constraints,
            assumptions=requirement.assumptions,
        )


def create_user_story(driver, project_id: str, story: UserStory):
    """Create a user story and link to project."""
    with driver.session() as session:
        session.run(
            """
            MATCH (p:Project {project_id: $project_id})
            CREATE (s:UserStory {
                story_id: $story_id,
                title: $title,
                as_a: $as_a,
                i_want: $i_want,
                so_that: $so_that,
                description: $description,
                priority: $priority,
                story_points: $story_points,
                status: $status,
                sprint: $sprint,
                acceptance_criteria: $acceptance_criteria,
                created_date: $created_date,
                updated_date: $updated_date
            })
            CREATE (p)-[:CONTAINS_STORY]->(s)
            """,
            project_id=project_id,
            story_id=story.story_id,
            title=story.title,
            as_a=story.as_a,
            i_want=story.i_want,
            so_that=story.so_that,
            description=story.description,
            priority=story.priority.value,
            story_points=story.story_points,
            status=story.status.value,
            sprint=story.sprint,
            acceptance_criteria=story.acceptance_criteria,
            created_date=story.created_date or datetime.now().isoformat(),
            updated_date=story.updated_date or datetime.now().isoformat(),
        )


def create_actor(driver, project_id: str, actor: Actor):
    """Create an actor and link to project."""
    with driver.session() as session:
        session.run(
            """
            MATCH (p:Project {project_id: $project_id})
            MERGE (a:Actor {actor_id: $actor_id})
            SET a.name = $name,
                a.type = $type,
                a.description = $description,
                a.role = $role,
                a.responsibilities = $responsibilities,
                a.goals = $goals
            MERGE (p)-[:INVOLVES_ACTOR]->(a)
            """,
            project_id=project_id,
            actor_id=actor.actor_id,
            name=actor.name,
            type=actor.type,
            description=actor.description,
            role=actor.role,
            responsibilities=actor.responsibilities,
            goals=actor.goals,
        )


def create_contradiction(driver, contradiction: Contradiction, involved_req_ids: List[str] = None, involved_story_ids: List[str] = None):
    """Create a contradiction and link to requirements/stories."""
    with driver.session() as session:
        # Create contradiction
        session.run(
            """
            CREATE (c:Contradiction {
                contradiction_id: $contradiction_id,
                type: $type,
                severity: $severity,
                description: $description,
                detected_date: $detected_date,
                resolution_status: $resolution_status,
                detection_method: $detection_method,
                confidence_score: $confidence_score
            })
            """,
            contradiction_id=contradiction.contradiction_id,
            type=contradiction.type.value,
            severity=contradiction.severity.value,
            description=contradiction.description,
            detected_date=contradiction.detected_date or datetime.now().isoformat(),
            resolution_status=contradiction.resolution_status,
            detection_method=contradiction.detection_method,
            confidence_score=contradiction.confidence_score,
        )
        
        # Link to requirements
        if involved_req_ids:
            for req_id in involved_req_ids:
                session.run(
                    """
                    MATCH (c:Contradiction {contradiction_id: $contradiction_id})
                    MATCH (r:Requirement {req_id: $req_id})
                    MERGE (c)-[:INVOLVES]->(r)
                    """,
                    contradiction_id=contradiction.contradiction_id,
                    req_id=req_id,
                )
        
        # Link to user stories
        if involved_story_ids:
            for story_id in involved_story_ids:
                session.run(
                    """
                    MATCH (c:Contradiction {contradiction_id: $contradiction_id})
                    MATCH (s:UserStory {story_id: $story_id})
                    MERGE (c)-[:INVOLVES]->(s)
                    """,
                    contradiction_id=contradiction.contradiction_id,
                    story_id=story_id,
                )


def create_suggestion(driver, suggestion: Suggestion, contradiction_id: Optional[str] = None, target_req_ids: List[str] = None, target_story_ids: List[str] = None):
    """Create a suggestion and link to contradictions/requirements/stories."""
    with driver.session() as session:
        # Create suggestion
        session.run(
            """
            CREATE (s:Suggestion {
                suggestion_id: $suggestion_id,
                type: $type,
                description: $description,
                rationale: $rationale,
                priority: $priority,
                status: $status,
                created_date: $created_date,
                ai_confidence: $ai_confidence
            })
            """,
            suggestion_id=suggestion.suggestion_id,
            type=suggestion.type.value,
            description=suggestion.description,
            rationale=suggestion.rationale,
            priority=suggestion.priority.value,
            status=suggestion.status,
            created_date=suggestion.created_date or datetime.now().isoformat(),
            ai_confidence=suggestion.ai_confidence,
        )
        
        # Link to contradiction
        if contradiction_id:
            session.run(
                """
                MATCH (c:Contradiction {contradiction_id: $contradiction_id})
                MATCH (s:Suggestion {suggestion_id: $suggestion_id})
                MERGE (c)-[:HAS_SUGGESTION]->(s)
                MERGE (s)-[:RESOLVES]->(c)
                """,
                contradiction_id=contradiction_id,
                suggestion_id=suggestion.suggestion_id,
            )
        
        # Link to requirements
        if target_req_ids:
            for req_id in target_req_ids:
                session.run(
                    """
                    MATCH (s:Suggestion {suggestion_id: $suggestion_id})
                    MATCH (r:Requirement {req_id: $req_id})
                    MERGE (s)-[:TARGETS]->(r)
                    """,
                    suggestion_id=suggestion.suggestion_id,
                    req_id=req_id,
                )
        
        # Link to user stories
        if target_story_ids:
            for story_id in target_story_ids:
                session.run(
                    """
                    MATCH (s:Suggestion {suggestion_id: $suggestion_id})
                    MATCH (st:UserStory {story_id: $story_id})
                    MERGE (s)-[:TARGETS]->(st)
                    """,
                    suggestion_id=suggestion.suggestion_id,
                    story_id=story_id,
                )


def create_entity(driver, entity: Entity):
    """Create an entity node."""
    with driver.session() as session:
        session.run(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.name = $name,
                e.type = $type,
                e.description = $description,
                e.attributes = $attributes,
                e.validation_rules = $validation_rules
            """,
            entity_id=entity.entity_id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            attributes=entity.attributes,
            validation_rules=entity.validation_rules,
        )


def create_constraint(driver, constraint: Constraint):
    """Create a constraint node."""
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Constraint {constraint_id: $constraint_id})
            SET c.type = $type,
                c.description = $description,
                c.impact_level = $impact_level,
                c.enforced = $enforced
            """,
            constraint_id=constraint.constraint_id,
            type=constraint.type,
            description=constraint.description,
            impact_level=constraint.impact_level,
            enforced=constraint.enforced,
        )


# ========================
# Relationship Operations
# ========================
def link_requirement_dependency(driver, source_req_id: str, target_req_id: str):
    """Create DEPENDS_ON relationship between requirements."""
    with driver.session() as session:
        session.run(
            """
            MATCH (r1:Requirement {req_id: $source_req_id})
            MATCH (r2:Requirement {req_id: $target_req_id})
            MERGE (r1)-[:DEPENDS_ON]->(r2)
            """,
            source_req_id=source_req_id,
            target_req_id=target_req_id,
        )


def link_requirement_conflict(driver, req1_id: str, req2_id: str, reason: str, severity: str):
    """Create CONFLICTS_WITH relationship between requirements."""
    with driver.session() as session:
        session.run(
            """
            MATCH (r1:Requirement {req_id: $req1_id})
            MATCH (r2:Requirement {req_id: $req2_id})
            MERGE (r1)-[c:CONFLICTS_WITH]->(r2)
            SET c.reason = $reason, c.severity = $severity
            """,
            req1_id=req1_id,
            req2_id=req2_id,
            reason=reason,
            severity=severity,
        )


def link_story_to_requirement(driver, story_id: str, req_id: str):
    """Link user story to requirement (decomposition)."""
    with driver.session() as session:
        session.run(
            """
            MATCH (s:UserStory {story_id: $story_id})
            MATCH (r:Requirement {req_id: $req_id})
            MERGE (s)-[:DECOMPOSES_INTO]->(r)
            MERGE (r)-[:DERIVED_FROM]->(s)
            """,
            story_id=story_id,
            req_id=req_id,
        )


def link_requirement_to_entity(driver, req_id: str, entity_id: str):
    """Link requirement to entity."""
    with driver.session() as session:
        session.run(
            """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (r)-[:INVOLVES_ENTITY]->(e)
            """,
            req_id=req_id,
            entity_id=entity_id,
        )


def link_requirement_to_constraint(driver, req_id: str, constraint_id: str):
    """Link requirement to constraint."""
    with driver.session() as session:
        session.run(
            """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (c:Constraint {constraint_id: $constraint_id})
            MERGE (r)-[:HAS_CONSTRAINT]->(c)
            """,
            req_id=req_id,
            constraint_id=constraint_id,
        )


def link_requirement_to_actor(driver, req_id: str, actor_id: str):
    """Link requirement to actor (assignment)."""
    with driver.session() as session:
        session.run(
            """
            MATCH (r:Requirement {req_id: $req_id})
            MATCH (a:Actor {actor_id: $actor_id})
            MERGE (r)-[:ASSIGNED_TO]->(a)
            """,
            req_id=req_id,
            actor_id=actor_id,
        )


def link_story_blocks_story(driver, blocking_story_id: str, blocked_story_id: str):
    """Create BLOCKS relationship between user stories."""
    with driver.session() as session:
        session.run(
            """
            MATCH (s1:UserStory {story_id: $blocking_story_id})
            MATCH (s2:UserStory {story_id: $blocked_story_id})
            MERGE (s1)-[:BLOCKS]->(s2)
            """,
            blocking_story_id=blocking_story_id,
            blocked_story_id=blocked_story_id,
        )


# ========================
# Meeting Operations
# ========================
def create_meeting(driver, meeting: Meeting):
    """Create a meeting node."""
    with driver.session() as session:
        session.run(
            """
            CREATE (m:Meeting {
                meeting_id: $meeting_id,
                title: $title,
                description: $description,
                date: $date,
                start_time: $start_time,
                end_time: $end_time,
                location: $location,
                meeting_type: $meeting_type,
                meeting_status: $meeting_status,
                agenda: $agenda,
                action_items: $action_items,
                decisions: $decisions,
                next_meeting_date: $next_meeting_date,
                next_meeting_time: $next_meeting_time,
                project_id: $project_id,
                created_date: $created_date,
                updated_date: $updated_date
            })
            """,
            meeting_id=meeting.meeting_id,
            title=meeting.title,
            description=meeting.description,
            date=meeting.date,
            start_time=meeting.start_time,
            end_time=meeting.end_time,
            location=meeting.location,
            meeting_type=meeting.meeting_type,
            meeting_status=meeting.meeting_status,
            agenda=meeting.agenda,
            action_items=meeting.action_items,
            decisions=meeting.decisions,
            next_meeting_date=meeting.next_meeting_date,
            next_meeting_time=meeting.next_meeting_time,
            project_id=meeting.project_id,
            created_date=meeting.created_date or datetime.now().isoformat(),
            updated_date=meeting.updated_date or datetime.now().isoformat(),
        )
        
        # Link to project if project_id is provided
        if meeting.project_id:
            session.run(
                """
                MATCH (m:Meeting {meeting_id: $meeting_id})
                MATCH (p:Project {project_id: $project_id})
                MERGE (p)-[:HAS_MEETING]->(m)
                """,
                meeting_id=meeting.meeting_id,
                project_id=meeting.project_id,
            )


def create_participant(driver, participant: Participant):
    """Create a participant node."""
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Participant {participant_id: $participant_id})
            SET p.name = $name,
                p.role = $role,
                p.email = $email,
                p.attendance_status = $attendance_status,
                p.notes = $notes
            """,
            participant_id=participant.participant_id,
            name=participant.name,
            role=participant.role,
            email=participant.email,
            attendance_status=participant.attendance_status,
            notes=participant.notes,
        )


def create_event(driver, event: Event):
    """Create an event node."""
    with driver.session() as session:
        session.run(
            """
            CREATE (e:Event {
                event_id: $event_id,
                name: $name,
                description: $description,
                event_type: $event_type,
                date: $date,
                time: $time,
                priority: $priority,
                status: $status,
                assigned_to: $assigned_to,
                due_date: $due_date,
                meeting_id: $meeting_id
            })
            """,
            event_id=event.event_id,
            name=event.name,
            description=event.description,
            event_type=event.event_type,
            date=event.date,
            time=event.time,
            priority=event.priority.value,
            status=event.status,
            assigned_to=event.assigned_to,
            due_date=event.due_date,
            meeting_id=event.meeting_id,
        )
        
        # Link to meeting if meeting_id is provided
        if event.meeting_id:
            session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                MATCH (m:Meeting {meeting_id: $meeting_id})
                MERGE (m)-[:HAS_EVENT]->(e)
                """,
                event_id=event.event_id,
                meeting_id=event.meeting_id,
            )


def link_participant_to_meeting(driver, participant_id: str, meeting_id: str):
    """Link participant to meeting."""
    with driver.session() as session:
        session.run(
            """
            MATCH (p:Participant {participant_id: $participant_id})
            MATCH (m:Meeting {meeting_id: $meeting_id})
            MERGE (p)-[:ATTENDED]->(m)
            """,
            participant_id=participant_id,
            meeting_id=meeting_id,
        )


def link_event_to_participant(driver, event_id: str, participant_id: str):
    """Link event to participant (assignment)."""
    with driver.session() as session:
        session.run(
            """
            MATCH (e:Event {event_id: $event_id})
            MATCH (p:Participant {participant_id: $participant_id})
            MERGE (e)-[:ASSIGNED_TO]->(p)
            """,
            event_id=event_id,
            participant_id=participant_id,
        )


def link_event_to_requirement(driver, event_id: str, req_id: str):
    """Link event to requirement."""
    with driver.session() as session:
        session.run(
            """
            MATCH (e:Event {event_id: $event_id})
            MATCH (r:Requirement {req_id: $req_id})
            MERGE (e)-[:RELATES_TO]->(r)
            """,
            event_id=event_id,
            req_id=req_id,
        )


# ========================
# Merge Operations (merge similar nodes)
# ========================
def merge_participant_by_name_or_email(driver, participant: Participant) -> Optional[str]:
    """Merge participant by name or email if exists, otherwise create new.
    
    Returns the participant_id (existing or newly created).
    """
    with driver.session() as session:
        # First, try to find by email if provided
        if participant.email:
            result = session.run(
                """
                MATCH (p:Participant)
                WHERE p.email = $email AND p.email IS NOT NULL AND p.email <> ''
                RETURN p.participant_id AS participant_id
                LIMIT 1
                """,
                email=participant.email,
            )
            record = result.single()
            if record:
                existing_id = record["participant_id"]
                # Update existing participant with new data
                session.run(
                    """
                    MATCH (p:Participant {participant_id: $participant_id})
                    SET p.name = $name,
                        p.role = COALESCE($role, p.role),
                        p.email = $email,
                        p.attendance_status = COALESCE($attendance_status, p.attendance_status),
                        p.notes = COALESCE($notes, p.notes)
                    """,
                    participant_id=existing_id,
                    name=participant.name,
                    role=participant.role,
                    email=participant.email,
                    attendance_status=participant.attendance_status,
                    notes=participant.notes,
                )
                logger.info(f"Merged participant by email: {existing_id} (email: {participant.email})")
                return existing_id
        
        # If no email match, try by name (normalized)
        name_normalized = participant.name.lower().strip()
        result = session.run(
            """
            MATCH (p:Participant)
            WHERE toLower(trim(p.name)) = $name_normalized
            RETURN p.participant_id AS participant_id
            LIMIT 1
            """,
            name_normalized=name_normalized,
        )
        record = result.single()
        if record:
            existing_id = record["participant_id"]
            # Update existing participant
            session.run(
                """
                MATCH (p:Participant {participant_id: $participant_id})
                SET p.name = $name,
                    p.role = COALESCE($role, p.role),
                    p.email = COALESCE($email, p.email),
                    p.attendance_status = COALESCE($attendance_status, p.attendance_status),
                    p.notes = COALESCE($notes, p.notes)
                """,
                participant_id=existing_id,
                name=participant.name,
                role=participant.role,
                email=participant.email,
                attendance_status=participant.attendance_status,
                notes=participant.notes,
            )
            logger.info(f"Merged participant by name: {existing_id} (name: {participant.name})")
            return existing_id
        
        # No match found, create new
        create_participant(driver, participant)
        logger.info(f"Created new participant: {participant.participant_id} (name: {participant.name})")
        return participant.participant_id


def merge_meeting_by_title_date(driver, meeting: Meeting) -> Optional[str]:
    """Merge meeting by title and date if exists, otherwise create new.
    
    Returns the meeting_id (existing or newly created).
    """
    with driver.session() as session:
        if meeting.title and meeting.date:
            # Try to find by title and date
            result = session.run(
                """
                MATCH (m:Meeting)
                WHERE m.title = $title AND m.date = $date
                RETURN m.meeting_id AS meeting_id
                LIMIT 1
                """,
                title=meeting.title,
                date=meeting.date,
            )
            record = result.single()
            if record:
                existing_id = record["meeting_id"]
                # Update existing meeting with new data
                session.run(
                    """
                    MATCH (m:Meeting {meeting_id: $meeting_id})
                    SET m.title = $title,
                        m.description = COALESCE($description, m.description),
                        m.date = $date,
                        m.start_time = COALESCE($start_time, m.start_time),
                        m.end_time = COALESCE($end_time, m.end_time),
                        m.location = COALESCE($location, m.location),
                        m.meeting_type = COALESCE($meeting_type, m.meeting_type),
                        m.meeting_status = COALESCE($meeting_status, m.meeting_status),
                        m.agenda = COALESCE($agenda, m.agenda),
                        m.action_items = COALESCE($action_items, m.action_items),
                        m.decisions = COALESCE($decisions, m.decisions),
                        m.next_meeting_date = COALESCE($next_meeting_date, m.next_meeting_date),
                        m.next_meeting_time = COALESCE($next_meeting_time, m.next_meeting_time),
                        m.project_id = COALESCE($project_id, m.project_id),
                        m.updated_date = $updated_date
                    """,
                    meeting_id=existing_id,
                    title=meeting.title,
                    description=meeting.description,
                    date=meeting.date,
                    start_time=meeting.start_time,
                    end_time=meeting.end_time,
                    location=meeting.location,
                    meeting_type=meeting.meeting_type,
                    meeting_status=meeting.meeting_status,
                    agenda=meeting.agenda,
                    action_items=meeting.action_items,
                    decisions=meeting.decisions,
                    next_meeting_date=meeting.next_meeting_date,
                    next_meeting_time=meeting.next_meeting_time,
                    project_id=meeting.project_id,
                    updated_date=meeting.updated_date or datetime.now().isoformat(),
                )
                logger.info(f"Merged meeting by title and date: {existing_id} (title: {meeting.title}, date: {meeting.date})")
                return existing_id
        
        # No match found, create new
        create_meeting(driver, meeting)
        logger.info(f"Created new meeting: {meeting.meeting_id} (title: {meeting.title})")
        return meeting.meeting_id


def merge_event_by_name_date_meeting(driver, event: Event) -> Optional[str]:
    """Merge event by name, date, and meeting_id if exists, otherwise create new.
    
    This function:
    1. First tries to merge by name + meeting_id (date optional) - allows similar events to be merged
    2. Checks for duplicate event_id and generates new one if needed
    3. Creates new event if no match found
    
    Returns the event_id (existing or newly created).
    """
    with driver.session() as session:
        # First, ensure event_id is unique - check if it already exists
        if event.event_id:
            existing_id_check = session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                RETURN e.event_id AS event_id
                LIMIT 1
                """,
                event_id=event.event_id,
            )
            if existing_id_check.single():
                # Event ID already exists, we'll try to find by name+meeting instead
                logger.warning(f"Event ID {event.event_id} already exists, will try to merge by name+meeting")
        
        # Try to find existing event by name and meeting_id (date is optional for merge)
        # This allows similar events to be merged even if date is slightly different
        if event.name and event.meeting_id:
            # Try to find by name + meeting_id (prioritize exact match with date)
            result = session.run(
                """
                MATCH (e:Event)
                WHERE toLower(trim(e.name)) = toLower(trim($name))
                  AND e.meeting_id = $meeting_id
                WITH e
                ORDER BY 
                    CASE WHEN e.date = $date THEN 0 ELSE 1 END,
                    e.date DESC
                RETURN e.event_id AS event_id
                LIMIT 1
                """,
                name=event.name,
                date=event.date,
                meeting_id=event.meeting_id,
            )
            record = result.single()
            if record:
                existing_id = record["event_id"]
                # Update existing event (merge properties)
                session.run(
                    """
                    MATCH (e:Event {event_id: $event_id})
                    SET e.name = $name,
                        e.description = COALESCE($description, e.description),
                        e.event_type = COALESCE($event_type, e.event_type),
                        e.date = COALESCE($date, e.date),
                        e.time = COALESCE($time, e.time),
                        e.priority = COALESCE($priority, e.priority),
                        e.status = COALESCE($status, e.status),
                        e.assigned_to = COALESCE($assigned_to, e.assigned_to),
                        e.due_date = COALESCE($due_date, e.due_date),
                        e.meeting_id = $meeting_id
                    """,
                    event_id=existing_id,
                    name=event.name,
                    description=event.description,
                    event_type=event.event_type,
                    date=event.date,
                    time=event.time,
                    priority=event.priority.value if hasattr(event.priority, 'value') else event.priority,
                    status=event.status,
                    assigned_to=event.assigned_to,
                    due_date=event.due_date,
                    meeting_id=event.meeting_id,
                )
                logger.info(f"Merged event by name and meeting: {existing_id} (name: {event.name}, meeting: {event.meeting_id})")
                return existing_id
        
        # No match found, create new
        # Ensure event_id is unique - generate new one if it already exists
        if event.event_id:
            existing_check = session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                RETURN e.event_id AS event_id
                LIMIT 1
                """,
                event_id=event.event_id,
            )
            if existing_check.single():
                # Event ID already exists, generate new unique one
                import uuid
                event.event_id = f"EVENT-{uuid.uuid4().hex[:12].upper()}"
                logger.warning(f"Event ID conflict detected, generated new ID: {event.event_id}")
        elif not event.event_id:
            # Generate event_id if not provided
            import uuid
            event.event_id = f"EVENT-{uuid.uuid4().hex[:12].upper()}"
            logger.info(f"Generated new event_id: {event.event_id}")
        
        create_event(driver, event)
        logger.info(f"Created new event: {event.event_id} (name: {event.name})")
        return event.event_id


def merge_project_by_name(driver, project: Project) -> Optional[str]:
    """Merge project by name if exists, otherwise create new.
    
    Returns the project_id (existing or newly created).
    """
    with driver.session() as session:
        name_normalized = project.name.lower().strip()
        result = session.run(
            """
            MATCH (p:Project)
            WHERE toLower(trim(p.name)) = $name_normalized
            RETURN p.project_id AS project_id
            LIMIT 1
            """,
            name_normalized=name_normalized,
        )
        record = result.single()
        if record:
            existing_id = record["project_id"]
            # Update existing project
            session.run(
                """
                MATCH (p:Project {project_id: $project_id})
                SET p.name = $name,
                    p.description = COALESCE($description, p.description),
                    p.version = COALESCE($version, p.version),
                    p.status = COALESCE($status, p.status),
                    p.stakeholders = COALESCE($stakeholders, p.stakeholders),
                    p.updated_date = $updated_date
                """,
                project_id=existing_id,
                name=project.name,
                description=project.description,
                version=project.version,
                status=project.status,
                stakeholders=project.stakeholders,
                updated_date=project.updated_date or datetime.now().isoformat(),
            )
            logger.info(f"Merged project by name: {existing_id} (name: {project.name})")
            return existing_id
        
        # No match found, create new
        create_project(driver, project)
        logger.info(f"Created new project: {project.project_id} (name: {project.name})")
        return project.project_id


def merge_actor_by_name_type(driver, actor: Actor) -> Optional[str]:
    """Merge actor by name and type if exists, otherwise create new.
    
    Returns the actor_id (existing or newly created).
    """
    with driver.session() as session:
        name_normalized = actor.name.lower().strip()
        result = session.run(
            """
            MATCH (a:Actor)
            WHERE toLower(trim(a.name)) = $name_normalized 
              AND a.type = $type
            RETURN a.actor_id AS actor_id
            LIMIT 1
            """,
            name_normalized=name_normalized,
            type=actor.type,
        )
        record = result.single()
        if record:
            existing_id = record["actor_id"]
            # Update existing actor
            session.run(
                """
                MATCH (a:Actor {actor_id: $actor_id})
                SET a.name = $name,
                    a.type = $type,
                    a.description = COALESCE($description, a.description),
                    a.role = COALESCE($role, a.role),
                    a.responsibilities = COALESCE($responsibilities, a.responsibilities),
                    a.goals = COALESCE($goals, a.goals)
                """,
                actor_id=existing_id,
                name=actor.name,
                type=actor.type,
                description=actor.description,
                role=actor.role,
                responsibilities=actor.responsibilities,
                goals=actor.goals,
            )
            logger.info(f"Merged actor by name and type: {existing_id} (name: {actor.name}, type: {actor.type})")
            return existing_id
        
        # No match found, create new (need project_id, but merge doesn't require it)
        # If actor has project_id in actor_id, try to extract it
        project_id = None
        if hasattr(actor, 'project_id') and actor.project_id:
            project_id = actor.project_id
        elif "-" in actor.actor_id:
            # Try to extract from actor_id format
            parts = actor.actor_id.split("-")
            if len(parts) > 0:
                project_id = parts[0]
        
        if project_id:
            create_actor(driver, project_id=project_id, actor=actor)
        else:
            # Create without project link if no project_id available
            with driver.session() as session:
                session.run(
                    """
                    MERGE (a:Actor {actor_id: $actor_id})
                    SET a.name = $name,
                        a.type = $type,
                        a.description = $description,
                        a.role = $role,
                        a.responsibilities = $responsibilities,
                        a.goals = $goals
                    """,
                    actor_id=actor.actor_id,
                    name=actor.name,
                    type=actor.type,
                    description=actor.description,
                    role=actor.role,
                    responsibilities=actor.responsibilities,
                    goals=actor.goals,
                )
        logger.info(f"Created new actor: {actor.actor_id} (name: {actor.name})")
        return actor.actor_id


def merge_entity_by_name_type(driver, entity: Entity) -> Optional[str]:
    """Merge entity by name and type if exists, otherwise create new.
    
    Returns the entity_id (existing or newly created).
    """
    with driver.session() as session:
        name_normalized = entity.name.lower().strip()
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE toLower(trim(e.name)) = $name_normalized 
              AND e.type = $type
            RETURN e.entity_id AS entity_id
            LIMIT 1
            """,
            name_normalized=name_normalized,
            type=entity.type,
        )
        record = result.single()
        if record:
            existing_id = record["entity_id"]
            # Update existing entity
            session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.type = $type,
                    e.description = COALESCE($description, e.description),
                    e.attributes = COALESCE($attributes, e.attributes),
                    e.validation_rules = COALESCE($validation_rules, e.validation_rules)
                """,
                entity_id=existing_id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                attributes=entity.attributes,
                validation_rules=entity.validation_rules,
            )
            logger.info(f"Merged entity by name and type: {existing_id} (name: {entity.name}, type: {entity.type})")
            return existing_id
        
        # No match found, create new
        create_entity(driver, entity)
        logger.info(f"Created new entity: {entity.entity_id} (name: {entity.name})")
        return entity.entity_id


# ========================
# Query Operations
# ========================
def get_project_requirements(driver, project_id: str) -> List[Dict[str, Any]]:
    """Get all requirements for a project."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_REQUIREMENT]->(r:Requirement)
            RETURN r
            ORDER BY r.priority DESC, r.created_date ASC
            """,
            project_id=project_id,
        )
        return [dict(record["r"]) for record in result]


def get_project_user_stories(driver, project_id: str) -> List[Dict[str, Any]]:
    """Get all user stories for a project."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_STORY]->(s:UserStory)
            RETURN s
            ORDER BY s.priority DESC, s.created_date ASC
            """,
            project_id=project_id,
        )
        return [dict(record["s"]) for record in result]


def get_open_contradictions(driver, project_id: str) -> List[Dict[str, Any]]:
    """Get all open contradictions for a project with involved requirements."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_REQUIREMENT]->(r:Requirement)
            MATCH (c:Contradiction)-[:INVOLVES]->(r)
            WHERE c.resolution_status = 'open'
            WITH c, collect(DISTINCT r.req_id) AS involved_reqs
            OPTIONAL MATCH (c)-[:INVOLVES]->(s:UserStory)
            WITH c, involved_reqs, collect(DISTINCT s.story_id) AS involved_stories
            RETURN c.contradiction_id AS contradiction_id,
                   c.type AS type,
                   c.severity AS severity,
                   c.description AS description,
                   c.detected_date AS detected_date,
                   c.confidence_score AS confidence_score,
                   involved_reqs,
                   involved_stories
            ORDER BY c.severity DESC, c.detected_date DESC
            """,
            project_id=project_id,
        )
        return [dict(record) for record in result]


def get_suggestions_for_contradiction(driver, contradiction_id: str) -> List[Dict[str, Any]]:
    """Get all suggestions for a specific contradiction."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Contradiction {contradiction_id: $contradiction_id})-[:HAS_SUGGESTION]->(s:Suggestion)
            RETURN s
            ORDER BY s.priority DESC, s.ai_confidence DESC
            """,
            contradiction_id=contradiction_id,
        )
        return [dict(record["s"]) for record in result]


def find_circular_dependencies(driver, project_id: str) -> List[Dict[str, Any]]:
    """Detect circular dependencies in requirements."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_REQUIREMENT]->(r:Requirement)
            MATCH path = (r)-[:DEPENDS_ON*]->(r)
            RETURN [node IN nodes(path) | node.req_id] AS cycle,
                   length(path) AS cycle_length
            LIMIT 50
            """,
            project_id=project_id,
        )
        return [dict(record) for record in result]


def find_requirements_without_acceptance_criteria(driver, project_id: str) -> List[Dict[str, Any]]:
    """Find requirements missing acceptance criteria (quality check)."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_REQUIREMENT]->(r:Requirement)
            WHERE size(r.acceptance_criteria) = 0
            RETURN r.req_id AS req_id,
                   r.title AS title,
                   r.status AS status,
                   r.priority AS priority
            ORDER BY r.priority DESC
            """,
            project_id=project_id,
        )
        return [dict(record) for record in result]


def find_orphan_user_stories(driver, project_id: str) -> List[Dict[str, Any]]:
    """Find user stories not linked to any requirement."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})-[:CONTAINS_STORY]->(s:UserStory)
            WHERE NOT (s)-[:DECOMPOSES_INTO]->(:Requirement)
            RETURN s.story_id AS story_id,
                   s.title AS title,
                   s.status AS status,
                   s.priority AS priority
            ORDER BY s.priority DESC
            """,
            project_id=project_id,
        )
        return [dict(record) for record in result]


def search_requirements_fulltext(driver, search_text: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Full-text search on requirements."""
    with driver.session() as session:
        if project_id:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('requirement_text_search', $search_text)
                YIELD node, score
                MATCH (p:Project {project_id: $project_id})-[:CONTAINS_REQUIREMENT]->(node)
                RETURN node.req_id AS req_id,
                       node.title AS title,
                       node.description AS description,
                       node.priority AS priority,
                       node.status AS status,
                       score
                ORDER BY score DESC
                LIMIT 20
                """,
                search_text=search_text,
                project_id=project_id,
            )
        else:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('requirement_text_search', $search_text)
                YIELD node, score
                RETURN node.req_id AS req_id,
                       node.title AS title,
                       node.description AS description,
                       node.priority AS priority,
                       node.status AS status,
                       score
                ORDER BY score DESC
                LIMIT 20
                """,
                search_text=search_text,
            )
        return [dict(record) for record in result]


def get_requirement_impact_analysis(driver, req_id: str) -> Dict[str, Any]:
    """Analyze impact of a requirement (dependencies, conflicts, related stories)."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (r:Requirement {req_id: $req_id})
            OPTIONAL MATCH (r)-[:DEPENDS_ON]->(dep:Requirement)
            OPTIONAL MATCH (dependent:Requirement)-[:DEPENDS_ON]->(r)
            OPTIONAL MATCH (r)-[:CONFLICTS_WITH]-(conflict:Requirement)
            OPTIONAL MATCH (r)<-[:DECOMPOSES_INTO]-(story:UserStory)
            OPTIONAL MATCH (r)-[:INVOLVES_ENTITY]->(entity:Entity)
            OPTIONAL MATCH (r)-[:HAS_CONSTRAINT]->(constraint:Constraint)
            OPTIONAL MATCH (contradiction:Contradiction)-[:INVOLVES]->(r)
            RETURN r,
                   collect(DISTINCT dep.req_id) AS depends_on,
                   collect(DISTINCT dependent.req_id) AS dependent_on_this,
                   collect(DISTINCT conflict.req_id) AS conflicts_with,
                   collect(DISTINCT story.story_id) AS related_stories,
                   collect(DISTINCT entity.name) AS involved_entities,
                   collect(DISTINCT constraint.constraint_id) AS constraints,
                   collect(DISTINCT contradiction.contradiction_id) AS contradictions
            """,
            req_id=req_id,
        )
        record = result.single()
        if record:
            return {
                "requirement": dict(record["r"]),
                "depends_on": record["depends_on"],
                "dependent_on_this": record["dependent_on_this"],
                "conflicts_with": record["conflicts_with"],
                "related_stories": record["related_stories"],
                "involved_entities": record["involved_entities"],
                "constraints": record["constraints"],
                "contradictions": record["contradictions"],
            }
        return {}


def get_project_quality_metrics(driver, project_id: str) -> Dict[str, Any]:
    """Calculate quality metrics for a project."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Project {project_id: $project_id})
            OPTIONAL MATCH (p)-[:CONTAINS_REQUIREMENT]->(r:Requirement)
            OPTIONAL MATCH (p)-[:CONTAINS_STORY]->(s:UserStory)
            OPTIONAL MATCH (c:Contradiction)-[:INVOLVES]->(r)
            WHERE c.resolution_status = 'open'
            WITH p,
                 count(DISTINCT r) AS total_requirements,
                 count(DISTINCT s) AS total_stories,
                 count(DISTINCT c) AS open_contradictions,
                 sum(CASE WHEN size(r.acceptance_criteria) = 0 THEN 1 ELSE 0 END) AS reqs_without_criteria,
                 sum(CASE WHEN NOT (s)-[:DECOMPOSES_INTO]->(:Requirement) THEN 1 ELSE 0 END) AS orphan_stories
            OPTIONAL MATCH (p)-[:CONTAINS_REQUIREMENT]->(r1:Requirement)-[:DEPENDS_ON*]->(r1)
            RETURN total_requirements,
                   total_stories,
                   open_contradictions,
                   reqs_without_criteria,
                   orphan_stories,
                   count(DISTINCT r1) AS circular_dependencies,
                   CASE WHEN total_requirements > 0
                        THEN toFloat(total_requirements - reqs_without_criteria) / total_requirements
                        ELSE 1.0 END AS completeness_score
            """,
            project_id=project_id,
        )
        record = result.single()
        if record:
            return dict(record)
        return {}


# ========================
# Main class wrapper
# ========================
class RequirementsNeo4jClient:
    """Neo4j client for Requirements Engineering operations."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.config = Neo4jConfig(
            uri=uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=user or os.getenv("NEO4J_USER", "neo4j"),
            password=password or os.getenv("NEO4J_PASSWORD", "password"),
        )
        self.driver = get_driver(self.config)
    
    def close(self):
        """Close the driver connection."""
        self.driver.close()
    
    def initialize_schema(self):
        """Initialize database schema."""
        init_schema(self.driver)
    
    def create_project(self, project: Project):
        create_project(self.driver, project)
    
    def create_requirement(self, project_id: str, requirement: Requirement):
        create_requirement(self.driver, project_id, requirement)
    
    def create_user_story(self, project_id: str, story: UserStory):
        create_user_story(self.driver, project_id, story)
    
    def create_actor(self, project_id: str, actor: Actor):
        create_actor(self.driver, project_id, actor)
    
    def create_contradiction(self, contradiction: Contradiction, involved_req_ids: List[str] = None, involved_story_ids: List[str] = None):
        create_contradiction(self.driver, contradiction, involved_req_ids, involved_story_ids)
    
    def create_suggestion(self, suggestion: Suggestion, contradiction_id: str = None, target_req_ids: List[str] = None, target_story_ids: List[str] = None):
        create_suggestion(self.driver, suggestion, contradiction_id, target_req_ids, target_story_ids)
    
    def create_entity(self, entity: Entity):
        create_entity(self.driver, entity)
    
    def create_constraint(self, constraint: Constraint):
        create_constraint(self.driver, constraint)
    
    def link_requirement_dependency(self, source_req_id: str, target_req_id: str):
        link_requirement_dependency(self.driver, source_req_id, target_req_id)
    
    def link_requirement_conflict(self, req1_id: str, req2_id: str, reason: str, severity: str):
        link_requirement_conflict(self.driver, req1_id, req2_id, reason, severity)
    
    def link_story_to_requirement(self, story_id: str, req_id: str):
        link_story_to_requirement(self.driver, story_id, req_id)
    
    def link_requirement_to_entity(self, req_id: str, entity_id: str):
        link_requirement_to_entity(self.driver, req_id, entity_id)
    
    def link_requirement_to_constraint(self, req_id: str, constraint_id: str):
        link_requirement_to_constraint(self.driver, req_id, constraint_id)
    
    def link_requirement_to_actor(self, req_id: str, actor_id: str):
        link_requirement_to_actor(self.driver, req_id, actor_id)
    
    def link_story_blocks_story(self, blocking_story_id: str, blocked_story_id: str):
        link_story_blocks_story(self.driver, blocking_story_id, blocked_story_id)
    
    def get_project_requirements(self, project_id: str) -> List[Dict[str, Any]]:
        return get_project_requirements(self.driver, project_id)
    
    def get_project_user_stories(self, project_id: str) -> List[Dict[str, Any]]:
        return get_project_user_stories(self.driver, project_id)
    
    def get_open_contradictions(self, project_id: str) -> List[Dict[str, Any]]:
        return get_open_contradictions(self.driver, project_id)
    
    def get_suggestions_for_contradiction(self, contradiction_id: str) -> List[Dict[str, Any]]:
        return get_suggestions_for_contradiction(self.driver, contradiction_id)
    
    def find_circular_dependencies(self, project_id: str) -> List[Dict[str, Any]]:
        return find_circular_dependencies(self.driver, project_id)
    
    def find_requirements_without_acceptance_criteria(self, project_id: str) -> List[Dict[str, Any]]:
        return find_requirements_without_acceptance_criteria(self.driver, project_id)
    
    def find_orphan_user_stories(self, project_id: str) -> List[Dict[str, Any]]:
        return find_orphan_user_stories(self.driver, project_id)
    
    def search_requirements_fulltext(self, search_text: str, project_id: str = None) -> List[Dict[str, Any]]:
        return search_requirements_fulltext(self.driver, search_text, project_id)
    
    def get_requirement_impact_analysis(self, req_id: str) -> Dict[str, Any]:
        return get_requirement_impact_analysis(self.driver, req_id)
    
    def get_project_quality_metrics(self, project_id: str) -> Dict[str, Any]:
        return get_project_quality_metrics(self.driver, project_id)
    
    def create_meeting(self, meeting: Meeting):
        create_meeting(self.driver, meeting)
    
    def create_participant(self, participant: Participant):
        create_participant(self.driver, participant)
    
    def create_event(self, event: Event):
        create_event(self.driver, event)
    
    def link_participant_to_meeting(self, participant_id: str, meeting_id: str):
        link_participant_to_meeting(self.driver, participant_id, meeting_id)
    
    def link_event_to_participant(self, event_id: str, participant_id: str):
        link_event_to_participant(self.driver, event_id, participant_id)
    
    def link_event_to_requirement(self, event_id: str, req_id: str):
        link_event_to_requirement(self.driver, event_id, req_id)
    
    def merge_participant_by_name_or_email(self, participant: Participant) -> Optional[str]:
        """Merge participant by name or email if exists, otherwise create new."""
        return merge_participant_by_name_or_email(self.driver, participant)
    
    def merge_meeting_by_title_date(self, meeting: Meeting) -> Optional[str]:
        """Merge meeting by title and date if exists, otherwise create new."""
        return merge_meeting_by_title_date(self.driver, meeting)
    
    def merge_event_by_name_date_meeting(self, event: Event) -> Optional[str]:
        """Merge event by name, date, and meeting_id if exists, otherwise create new."""
        return merge_event_by_name_date_meeting(self.driver, event)
    
    def merge_project_by_name(self, project: Project) -> Optional[str]:
        """Merge project by name if exists, otherwise create new."""
        return merge_project_by_name(self.driver, project)
    
    def merge_actor_by_name_type(self, actor: Actor) -> Optional[str]:
        """Merge actor by name and type if exists, otherwise create new."""
        return merge_actor_by_name_type(self.driver, actor)
    
    def merge_entity_by_name_type(self, entity: Entity) -> Optional[str]:
        """Merge entity by name and type if exists, otherwise create new."""
        return merge_entity_by_name_type(self.driver, entity)


# ========================
# Demo/Test Usage
# ========================
if __name__ == "__main__":
    # Initialize client
    client = RequirementsNeo4jClient()
    
    # Initialize schema
    print("Initializing schema...")
    client.initialize_schema()
    
    # Create a sample project
    print("\nCreating sample project...")
    project = Project(
        project_id="PROJ-001",
        name="E-commerce Platform",
        description="Online shopping platform development",
        stakeholders=["Product Owner", "Business Analyst", "Development Team"]
    )
    client.create_project(project)
    
    # Create sample requirements
    print("Creating sample requirements...")
    req1 = Requirement(
        req_id="REQ-001",
        title="User Authentication",
        description="System must provide secure user authentication with email and password",
        type=RequirementType.FUNCTIONAL,
        priority=Priority.CRITICAL,
        acceptance_criteria=[
            "User can register with email",
            "User can login with credentials",
            "Password must be encrypted"
        ]
    )
    client.create_requirement("PROJ-001", req1)
    
    req2 = Requirement(
        req_id="REQ-002",
        title="Shopping Cart",
        description="Users can add products to shopping cart",
        type=RequirementType.FUNCTIONAL,
        priority=Priority.HIGH,
        acceptance_criteria=[
            "Add product to cart",
            "Remove product from cart",
            "Update quantity"
        ]
    )
    client.create_requirement("PROJ-001", req2)
    
    # Create sample user story
    print("Creating sample user story...")
    story1 = UserStory(
        story_id="US-001",
        title="User Registration",
        as_a="new customer",
        i_want="to create an account",
        so_that="I can make purchases",
        priority=Priority.HIGH,
        story_points=5,
        acceptance_criteria=[
            "Registration form is available",
            "Email verification is sent",
            "Account is created successfully"
        ]
    )
    client.create_user_story("PROJ-001", story1)
    
    # Link story to requirement
    print("Linking story to requirement...")
    client.link_story_to_requirement("US-001", "REQ-001")
    
    # Create a contradiction
    print("Creating sample contradiction...")
    contradiction = Contradiction(
        contradiction_id="CONT-001",
        type=ContradictionType.LOGICAL,
        severity=ContradictionSeverity.MAJOR,
        description="REQ-001 requires email authentication but REQ-003 specifies phone-only login",
        confidence_score=0.85
    )
    client.create_contradiction(contradiction, involved_req_ids=["REQ-001"])
    
    # Create a suggestion
    print("Creating sample suggestion...")
    suggestion = Suggestion(
        suggestion_id="SUG-001",
        type=SuggestionType.CLARIFICATION,
        description="Update REQ-001 to support both email and phone authentication",
        rationale="Resolves authentication method contradiction",
        priority=Priority.HIGH,
        ai_confidence=0.92
    )
    client.create_suggestion(suggestion, contradiction_id="CONT-001", target_req_ids=["REQ-001"])
    
    # Query data
    print("\n=== Project Requirements ===")
    requirements = client.get_project_requirements("PROJ-001")
    for req in requirements:
        print(f"- {req['req_id']}: {req['title']} [{req['priority']}]")
    
    print("\n=== Open Contradictions ===")
    contradictions = client.get_open_contradictions("PROJ-001")
    for cont in contradictions:
        print(f"- {cont['contradiction_id']}: {cont['description']} [Severity: {cont['severity']}]")
        print(f"  Involved: {cont['involved_reqs']}")
    
    print("\n=== Quality Metrics ===")
    metrics = client.get_project_quality_metrics("PROJ-001")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
    
    print("\n=== Requirements without Acceptance Criteria ===")
    missing_criteria = client.find_requirements_without_acceptance_criteria("PROJ-001")
    for req in missing_criteria:
        print(f"- {req['req_id']}: {req['title']}")
    
    # Close connection
    client.close()
    print("\n✅ Demo completed successfully!")
