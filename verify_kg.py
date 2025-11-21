"""Quick verification of Neo4j IPC sections"""
from config.neo4j_config import Neo4jConnection

conn = Neo4jConnection()
conn.connect()

query = """
MATCH (s:IPCSection) 
WHERE s.section_number IN ['302', '303', '304', '127', '140']
RETURN s.section_number as section, s.offense as offense
ORDER BY toInteger(s.section_number)
"""

sections = conn.execute_query(query)

print("="*80)
print("IPC SECTIONS IN NEO4J DATABASE")
print("="*80)

for s in sections:
    print(f"\nSection {s['section']}:")
    print(f"  Offense: {s['offense']}")

conn.close()
