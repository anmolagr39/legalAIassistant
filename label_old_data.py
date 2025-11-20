"""
Add source labels to existing nodes before running new pipeline
"""
from config.neo4j_config import Neo4jConnection

def label_existing_data():
    """Add 'OldData' label to all existing nodes"""
    conn = Neo4jConnection()
    conn.connect()
    
    print("Labeling existing data...")
    
    # Add OldData label to all existing nodes
    query = """
    MATCH (n)
    WHERE NOT 'NewData' IN labels(n)
    SET n:OldData
    RETURN count(n) as labeled_count
    """
    
    result = conn.execute_query(query)
    count = result[0]['labeled_count'] if result else 0
    
    print(f"✓ Labeled {count} existing nodes with 'OldData'")
    print("\nNow new pipeline will add 'NewData' label to distinguish.")
    
    conn.close()

if __name__ == "__main__":
    label_existing_data()
