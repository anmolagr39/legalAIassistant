"""
Neo4j database configuration and connection management
"""
from neo4j import GraphDatabase
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import logging

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Neo4j database connection manager"""
    
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = "neo4j"  # Community edition only supports 'neo4j'
        self.driver = None
    
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            logger.info(f"[OK] Connected to Neo4j at {self.uri} (database: {self.database})")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query, parameters=None):
        """Execute a Cypher query"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
    
    def execute_write(self, query, parameters=None):
        """Execute a write transaction"""
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )
            return result
    
    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write(query)
        logger.warning("Database cleared")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        indexes = [
            "CREATE INDEX ipc_section_number IF NOT EXISTS FOR (n:IPCSection) ON (n.section_number)",
            "CREATE INDEX case_name IF NOT EXISTS FOR (n:Case) ON (n.case_name)",
            "CREATE INDEX case_number IF NOT EXISTS FOR (n:Case) ON (n.case_number)",
            "CREATE INDEX court_name IF NOT EXISTS FOR (n:Court) ON (n.name)",
            "CREATE INDEX judge_name IF NOT EXISTS FOR (n:Judge) ON (n.name)",
            "CREATE INDEX article_number IF NOT EXISTS FOR (n:Article) ON (n.article_number)",
            "CREATE INDEX legal_act_name IF NOT EXISTS FOR (n:LegalAct) ON (n.name)",
            "CREATE INDEX offense_type IF NOT EXISTS FOR (n:Offense) ON (n.offense_type)",
        ]
        
        for index_query in indexes:
            try:
                self.execute_write(index_query)
                logger.info(f"Index created: {index_query[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        constraints = [
            "CREATE CONSTRAINT ipc_section_unique IF NOT EXISTS FOR (n:IPCSection) REQUIRE n.section_number IS UNIQUE",
            "CREATE CONSTRAINT case_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
            "CREATE CONSTRAINT article_unique IF NOT EXISTS FOR (n:Article) REQUIRE n.article_number IS UNIQUE",
        ]
        
        for constraint_query in constraints:
            try:
                self.execute_write(constraint_query)
                logger.info(f"Constraint created: {constraint_query[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")
    
    def get_stats(self):
        """Get database statistics"""
        queries = {
            "nodes": "MATCH (n) RETURN count(n) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_types": "MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC"
        }
        
        stats = {}
        for key, query in queries.items():
            result = self.execute_query(query)
            if key == "node_types":
                stats[key] = [(r["type"], r["count"]) for r in result]
            else:
                stats[key] = result[0]["count"] if result else 0
        
        return stats


def test_connection():
    """Test Neo4j connection"""
    conn = Neo4jConnection()
    try:
        conn.connect()
        stats = conn.get_stats()
        print(f"\n✓ Neo4j Connection Successful!")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Relationships: {stats['relationships']}")
        conn.close()
        return True
    except Exception as e:
        print(f"\n✗ Neo4j Connection Failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
