"""Check the actual graph structure"""
from config.neo4j_config import Neo4jConnection

conn = Neo4jConnection()
conn.connect()

# Check if Judge nodes exist
print("\n1. Judge nodes:")
result = conn.execute_query("MATCH (j:Judge) RETURN count(j) as count")
print(f"Total judges: {result[0]['count']}")

# Check relationships FROM Case
print("\n2. Relationships FROM Case:")
result = conn.execute_query("""
MATCH (c:Case)-[r]->()
RETURN type(r) as rel_type, count(*) as count
ORDER BY count DESC
""")
for r in result:
    print(f"  {r['rel_type']}: {r['count']}")

# Check relationships TO Judge
print("\n3. Relationships TO Judge:")
result = conn.execute_query("""
MATCH ()-[r]->(j:Judge)
RETURN type(r) as rel_type, count(*) as count
ORDER BY count DESC
""")
for r in result:
    print(f"  {r['rel_type']}: {r['count']}")

# Check if PRESIDED_BY exists
print("\n4. PRESIDED_BY relationships:")
result = conn.execute_query("""
MATCH ()-[r:PRESIDED_BY]->()
RETURN count(r) as count
""")
print(f"Total PRESIDED_BY: {result[0]['count']}")

# Sample judge with cases
print("\n5. Sample judge connections:")
result = conn.execute_query("""
MATCH (j:Judge)
OPTIONAL MATCH (c:Case)-[r]->(j)
RETURN j.name, type(r) as rel_type, count(c) as case_count
LIMIT 5
""")
for r in result:
    print(f"  {r['j.name']}: {r['rel_type']} - {r['case_count']} cases")

conn.close()
