"""Quick script to clear Neo4j data without confirmation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from core.config import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def clear_neo4j_data(clear_all=False):
    """Clear Neo4j data."""
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
        
        try:
            driver.verify_connectivity()
        except ServiceUnavailable:
            logger.error("Neo4j server is not available.")
            return
        
        with driver.session() as session:
            if clear_all:
                # Delete all nodes
                result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
                deleted = result.single()["deleted"]
                logger.info(f"Deleted all {deleted} nodes from Neo4j.")
            else:
                # Delete only Project, Requirement, UserStory nodes
                result = session.run(
                    "MATCH (n) WHERE n:Project OR n:Requirement OR n:UserStory DETACH DELETE n RETURN count(n) AS deleted"
                )
                deleted = result.single()["deleted"]
                logger.info(f"Deleted {deleted} Project/Requirement/UserStory nodes.")
        
        driver.close()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clear Neo4j data")
    parser.add_argument("--all", action="store_true", help="Clear all nodes (not just Project/Requirement/UserStory)")
    args = parser.parse_args()
    
    clear_neo4j_data(clear_all=args.all)



