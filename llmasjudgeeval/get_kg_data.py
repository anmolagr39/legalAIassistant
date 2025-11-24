import sys
sys.path.append('..')
from config.neo4j_config import Neo4jConnection

conn = Neo4jConnection()
conn.connect()

# Get top judges
print("\n=== TOP JUDGES BY CASE COUNT ===")
judges = conn.execute_query("""
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case) 
    RETURN j.name as judge, count(c) as cases 
    ORDER BY cases DESC LIMIT 15
""")
for r in judges:
    print(f"{r['judge']}: {r['cases']} cases")

# Get sample cases with IPC sections
print("\n=== SAMPLE CASES WITH IPC SECTIONS ===")
cases = conn.execute_query("""
    MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection) 
    RETURN c.case_name, collect(s.section_number) as sections 
    LIMIT 20
""")
for r in cases:
    print(f"{r['c.case_name']}: Sections {r['sections']}")

# Get cases by specific IPC sections
print("\n=== CASES FOR IPC 302 (Murder) ===")
cases_302 = conn.execute_query("""
    MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection {section_number: '302'}) 
    RETURN c.case_name LIMIT 10
""")
for r in cases_302:
    print(f"- {r['c.case_name']}")

print("\n=== CASES FOR IPC 376 (Rape) ===")
cases_376 = conn.execute_query("""
    MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection {section_number: '376'}) 
    RETURN c.case_name LIMIT 10
""")
for r in cases_376:
    print(f"- {r['c.case_name']}")

# Get judges with specific cases
print("\n=== SAMPLE JUDGE-CASE MAPPINGS ===")
judge_cases = conn.execute_query("""
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case)
    RETURN j.name as judge, collect(c.case_name)[0..3] as sample_cases
    LIMIT 10
""")
for r in judge_cases:
    print(f"\n{r['judge']}:")
    for case in r['sample_cases']:
        print(f"  - {case}")

# Get courts
print("\n=== COURTS ===")
courts = conn.execute_query("""
    MATCH (court:Court)<-[:HEARD_IN]-(c:Case)
    RETURN court.name, count(c) as case_count
""")
for r in courts:
    print(f"{r['court.name']}: {r['case_count']} cases")

conn.close()
