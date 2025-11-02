"""Script to clear Neo4j data for testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from core.config import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_neo4j_data():
    """Clear all Project, Requirement, and UserStory nodes from Neo4j."""
    config = load_config()
    
    # Connect to Neo4j
    neo4j_uri = config.neo4j.uri
    if neo4j_uri.startswith("http://"):
        neo4j_uri = neo4j_uri.replace("http://", "bolt://")
    elif neo4j_uri.startswith("https://"):
        neo4j_uri = neo4j_uri.replace("https://", "bolt://")
    elif not neo4j_uri.startswith("bolt://"):
        neo4j_uri = f"bolt://{neo4j_uri}"
    
    try:
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )
        
        # Verify connectivity
        try:
            driver.verify_connectivity()
        except ServiceUnavailable:
            logger.error("Neo4j server is not available. Cannot clear data.")
            return
        
        with driver.session() as session:
            # Get count before deletion
            result = session.run("MATCH (n) WHERE n:Project OR n:Requirement OR n:UserStory RETURN count(n) AS count")
            count_before = result.single()["count"]
            
            if count_before == 0:
                logger.info("No data to clear. Neo4j is already empty.")
                driver.close()
                return
            
            logger.info(f"Found {count_before} nodes to delete.")
            
            # Delete all Project, Requirement, and UserStory nodes with their relationships
            result = session.run(
                "MATCH (n) WHERE n:Project OR n:Requirement OR n:UserStory DETACH DELETE n RETURN count(n) AS deleted"
            )
            deleted = result.single()["deleted"]
            
            logger.info(f"Successfully deleted {deleted} nodes and their relationships.")
            
            # Optionally clear all data (uncomment if you want to clear everything)
            # result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
            # deleted = result.single()["deleted"]
            # logger.info(f"Deleted all {deleted} nodes from Neo4j.")
        
        driver.close()
        logger.info("Data cleared successfully!")
        
    except Exception as e:
        logger.error(f"Error clearing Neo4j data: {e}", exc_info=True)


if __name__ == "__main__":
    print("=" * 80)
    print("Clear Neo4j Data")
    print("=" * 80)
    print()
    response = input("This will delete all Project, Requirement, and UserStory nodes. Continue? (y/N): ")
    if response.lower() == "y":
        clear_neo4j_data()
    else:
        print("Cancelled.")



