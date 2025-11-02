"""
Script to print Milvus and Neo4j schemas
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from db.milvus_client import (
    _build_docs_schema,
    _build_faq_schema,
    DOC_DENSE_DIM_DEFAULT,
    FAQ_DENSE_DIM_DEFAULT,
)
from db.neo4j_client import (
    SCHEMA_CYPHER,
    FULLTEXT_INDEXES,
    Project,
    Requirement,
    UserStory,
    Actor,
    Contradiction,
    Suggestion,
    Entity,
    Constraint,
    UseCase,
)


def print_milvus_schema():
    """Print Milvus collection schemas"""
    print("=" * 80)
    print("MILVUS SCHEMA")
    print("=" * 80)
    
    print("\n📚 Collection: gsoft_docs")
    print("-" * 80)
    docs_schema = _build_docs_schema(DOC_DENSE_DIM_DEFAULT)
    print(f"Description: {docs_schema.description}")
    print(f"\nFields:")
    for field in docs_schema.fields:
        field_info = f"  - {field.name}: {field.dtype.value}"
        if hasattr(field, 'max_length') and field.max_length:
            field_info += f" (max_length={field.max_length})"
        if hasattr(field, 'dim') and field.dim:
            field_info += f" (dim={field.dim})"
        if field.is_primary:
            field_info += " [PRIMARY KEY]"
        if field.auto_id:
            field_info += " [AUTO_ID]"
        print(field_info)
    
    print("\n📚 Collection: gsoft_faq")
    print("-" * 80)
    faq_schema = _build_faq_schema(FAQ_DENSE_DIM_DEFAULT)
    print(f"Description: {faq_schema.description}")
    print(f"\nFields:")
    for field in faq_schema.fields:
        field_info = f"  - {field.name}: {field.dtype.value}"
        if hasattr(field, 'max_length') and field.max_length:
            field_info += f" (max_length={field.max_length})"
        if hasattr(field, 'dim') and field.dim:
            field_info += f" (dim={field.dim})"
        if field.is_primary:
            field_info += " [PRIMARY KEY]"
        if field.auto_id:
            field_info += " [AUTO_ID]"
        print(field_info)
    
    print(f"\n⚙️  Configuration:")
    print(f"  - DOC_DENSE_DIM: {DOC_DENSE_DIM_DEFAULT}")
    print(f"  - FAQ_DENSE_DIM: {FAQ_DENSE_DIM_DEFAULT}")


def print_neo4j_schema():
    """Print Neo4j schema"""
    print("\n" + "=" * 80)
    print("NEO4J SCHEMA")
    print("=" * 80)
    
    print("\n📊 Node Types:")
    print("-" * 80)
    
    # Project
    print("\n1. Project:")
    print("   Properties:")
    for field_name, field_info in Project.model_fields.items():
        field_type = field_info.annotation
        if hasattr(field_type, '__origin__') and hasattr(field_type.__origin__, '__name__'):
            if field_type.__origin__.__name__ == 'list':
                field_type_str = f"List[{field_type.__args__[0].__name__ if hasattr(field_type.__args__[0], '__name__') else str(field_type.__args__[0])}]"
            else:
                field_type_str = str(field_type)
        else:
            field_type_str = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
        required = "" if field_info.is_required() else " (optional)"
        default = f" = {field_info.default}" if field_info.default is not None and field_info.default != ... else ""
        print(f"     - {field_name}: {field_type_str}{required}{default}")
    
    # Requirement
    print("\n2. Requirement:")
    print("   Properties:")
    for field_name, field_info in Requirement.model_fields.items():
        field_type = field_info.annotation
        if hasattr(field_type, '__origin__') and hasattr(field_type.__origin__, '__name__'):
            if field_type.__origin__.__name__ == 'list':
                field_type_str = f"List[{field_type.__args__[0].__name__ if hasattr(field_type.__args__[0], '__name__') else str(field_type.__args__[0])}]"
            else:
                field_type_str = str(field_type)
        else:
            field_type_str = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
        required = "" if field_info.is_required() else " (optional)"
        default = f" = {field_info.default}" if field_info.default is not None and field_info.default != ... else ""
        print(f"     - {field_name}: {field_type_str}{required}{default}")
    
    # UserStory
    print("\n3. UserStory:")
    print("   Properties:")
    for field_name, field_info in UserStory.model_fields.items():
        field_type = field_info.annotation
        if hasattr(field_type, '__origin__') and hasattr(field_type.__origin__, '__name__'):
            if field_type.__origin__.__name__ == 'list':
                field_type_str = f"List[{field_type.__args__[0].__name__ if hasattr(field_type.__args__[0], '__name__') else str(field_type.__args__[0])}]"
            else:
                field_type_str = str(field_type)
        else:
            field_type_str = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
        required = "" if field_info.is_required() else " (optional)"
        default = f" = {field_info.default}" if field_info.default is not None and field_info.default != ... else ""
        print(f"     - {field_name}: {field_type_str}{required}{default}")
    
    # Other node types (simplified)
    other_nodes = [
        ("Actor", Actor),
        ("Contradiction", Contradiction),
        ("Suggestion", Suggestion),
        ("Entity", Entity),
        ("Constraint", Constraint),
        ("UseCase", UseCase),
    ]
    
    for node_name, node_model in other_nodes:
        print(f"\n{other_nodes.index((node_name, node_model)) + 4}. {node_name}:")
        print("   Properties:")
        for field_name, field_info in node_model.model_fields.items():
            field_type = field_info.annotation
            if hasattr(field_type, '__origin__') and hasattr(field_type.__origin__, '__name__'):
                if field_type.__origin__.__name__ == 'list':
                    field_type_str = f"List[{field_type.__args__[0].__name__ if hasattr(field_type.__args__[0], '__name__') else str(field_type.__args__[0])}]"
                else:
                    field_type_str = str(field_type)
            else:
                field_type_str = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
            required = "" if field_info.is_required() else " (optional)"
            default = f" = {field_info.default}" if field_info.default is not None and field_info.default != ... else ""
            print(f"     - {field_name}: {field_type_str}{required}{default}")
    
    print("\n🔗 Relationship Types:")
    print("-" * 80)
    relationships = [
        "CONTAINS_REQUIREMENT: (Project) -> (Requirement)",
        "CONTAINS_STORY: (Project) -> (UserStory)",
        "DEPENDS_ON: (Requirement) -> (Requirement)",
        "CONFLICTS_WITH: (Requirement) <-> (Requirement)",
        "DECOMPOSES_INTO: (UserStory) -> (Requirement)",
        "DERIVED_FROM: (Requirement) -> (UserStory)",
        "INVOLVES: (Contradiction) -> (Requirement|UserStory)",
        "HAS_SUGGESTION: (Contradiction) -> (Suggestion)",
        "RESOLVES: (Suggestion) -> (Contradiction)",
        "TARGETS: (Suggestion) -> (Requirement|UserStory)",
        "INVOLVES_ACTOR: (Project) -> (Actor)",
        "ASSIGNED_TO: (Requirement) -> (Actor)",
        "INVOLVES_ENTITY: (Requirement) -> (Entity)",
        "HAS_CONSTRAINT: (Requirement) -> (Constraint)",
        "BLOCKS: (UserStory) -> (UserStory)",
    ]
    for rel in relationships:
        print(f"  - {rel}")
    
    print("\n🔑 Constraints & Indexes:")
    print("-" * 80)
    print("Unique Constraints:")
    for cypher in SCHEMA_CYPHER:
        if "CONSTRAINT" in cypher and "UNIQUE" in cypher:
            constraint_name = cypher.split("FOR")[1].split("REQUIRE")[0].strip()
            field = cypher.split("REQUIRE")[1].split("IS UNIQUE")[0].strip()
            print(f"  - {constraint_name}: {field}")
    
    print("\nIndexes:")
    for cypher in SCHEMA_CYPHER:
        if "CREATE INDEX" in cypher:
            index_info = cypher.replace("CREATE INDEX ", "").replace(" IF NOT EXISTS", "").split("FOR")[0].strip()
            field_info = cypher.split("ON")[1].strip() if "ON" in cypher else ""
            print(f"  - {index_info}: {field_info}")
    
    print("\nFulltext Indexes:")
    for cypher in FULLTEXT_INDEXES:
        if "CREATE FULLTEXT INDEX" in cypher:
            index_name = cypher.split("IF NOT EXISTS")[0].replace("CREATE FULLTEXT INDEX", "").strip()
            node_and_fields = cypher.split("ON EACH")[1] if "ON EACH" in cypher else ""
            print(f"  - {index_name}: {node_and_fields.strip()}")


if __name__ == "__main__":
    print_milvus_schema()
    print_neo4j_schema()
    print("\n" + "=" * 80)
    print("✅ Schema printing completed!")
    print("=" * 80)

