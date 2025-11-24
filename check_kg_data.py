"""Check KG structure and design queries"""
from config.neo4j_config import Neo4jConnection

conn = Neo4jConnection()
conn.connect()

print("\n" + "="*80)
print("KNOWLEDGE GRAPH STRUCTURE ANALYSIS")
print("="*80)

print("\n=== NODE TYPES ===")
nodes = conn.execute_query('MATCH (n) RETURN DISTINCT labels(n)[0] as label, count(*) as count ORDER BY count DESC')
for r in nodes:
    print(f"  {r['label']}: {r['count']}")

print("\n=== RELATIONSHIPS ===")
rels = conn.execute_query('MATCH ()-[r]-() RETURN DISTINCT type(r) as rel_type, count(*) as count ORDER BY count DESC')
for r in rels:
    print(f"  {r['rel_type']}: {r['count']}")

print("\n=== SAMPLE IPC SECTIONS ===")
ipcs = conn.execute_query('MATCH (s:IPCSection) RETURN s.section_number, s.offense LIMIT 10')
for r in ipcs:
    print(f"  Section {r['s.section_number']}: {r['s.offense']}")

print("\n=== SAMPLE ARTICLES (Constitution) ===")
articles = conn.execute_query('MATCH (a:Article) RETURN a.article_number, a.title ORDER BY toInteger(a.article_number) LIMIT 10')
for r in articles:
    print(f"  Article {r['a.article_number']}: {r['a.title']}")

print("\n=== SAMPLE CASES ===")
cases = conn.execute_query('MATCH (c:Case) RETURN c.case_name LIMIT 10')
for r in cases:
    print(f"  {r['c.case_name']}")

print("\n=== SAMPLE JUDGES ===")
judges = conn.execute_query('MATCH (j:Judge) RETURN j.name LIMIT 10')
for r in judges:
    print(f"  {r['j.name']}")

print("\n=== KEY RELATIONSHIPS BREAKDOWN ===")

# IPC to Case relationships
print("\n1. IPC Sections -> Cases:")
result = conn.execute_query("""
MATCH (s:IPCSection)<-[r:GOVERNED_BY]-(c:Case)
WITH s, count(c) as case_count
RETURN s.section_number, case_count
ORDER BY case_count DESC
LIMIT 10
""")
for r in result:
    print(f"  Section {r['s.section_number']}: {r['case_count']} cases")

# Articles to Cases
print("\n2. Constitutional Articles -> Cases:")
result = conn.execute_query("""
MATCH (a:Article)<-[r:REFERS_TO]-(c:Case)
WITH a, count(c) as case_count
RETURN a.article_number, a.title, case_count
ORDER BY case_count DESC
LIMIT 10
""")
for r in result:
    print(f"  Article {r['a.article_number']} ({r['a.title']}): {r['case_count']} cases")

# Judge to Cases
print("\n3. Judges -> Cases:")
result = conn.execute_query("""
MATCH (j:Judge)<-[r:DECIDED_BY]-(c:Case)
WITH j, count(c) as case_count
RETURN j.name, case_count
ORDER BY case_count DESC
LIMIT 10
""")
for r in result:
    print(f"  {r['j.name']}: {r['case_count']} cases")

# Case Citations
print("\n4. Case Citation Network:")
result = conn.execute_query("""
MATCH (c1:Case)-[r:CITES]->(c2:Case)
RETURN count(r) as total_citations
""")
print(f"  Total case citations: {result[0]['total_citations']}")

conn.close()
print("\n" + "="*80)
