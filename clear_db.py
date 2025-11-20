"""Clear Neo4j database"""
from config.neo4j_config import Neo4jConnection

if __name__ == "__main__":
    conn = Neo4jConnection()
    conn.connect()
    
    print("Clearing database...")
    conn.clear_database()
    print("✓ Database cleared!")
    
    conn.close()
