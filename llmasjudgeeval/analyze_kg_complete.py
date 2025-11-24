import sys
sys.path.append('..')
from config.neo4j_config import Neo4jConnection

conn = Neo4jConnection()
conn.connect()

print("\n=== COMPLETE KG ANALYSIS ===\n")

# 1. All relationship types
print("1. ALL RELATIONSHIP TYPES:")
rels = conn.execute_query("""
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) as rel_type, count(*) as count
    ORDER BY count DESC
""")
for r in rels:
    print(f"  {r['rel_type']}: {r['count']}")

# 2. Cases with multiple IPC sections
print("\n2. CASES WITH MULTIPLE IPC SECTIONS:")
multi_ipc = conn.execute_query("""
    MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection)
    WITH c, collect(s.section_number) as sections
    WHERE size(sections) > 1
    RETURN c.case_name, sections
    LIMIT 10
""")
for r in multi_ipc:
    print(f"  {r['c.case_name']}: {r['sections']}")

# 3. IPC sections with their offenses and punishments
print("\n3. IPC SECTIONS WITH OFFENSES AND PUNISHMENTS:")
ipc_details = conn.execute_query("""
    MATCH (s:IPCSection)
    OPTIONAL MATCH (s)-[:APPLIES_TO]->(o:Offense)
    OPTIONAL MATCH (s)-[:PRESCRIBES]->(p:Punishment)
    RETURN s.section_number, s.offense, 
           collect(DISTINCT o.offense_type) as offenses,
           collect(DISTINCT p.punishment_type) as punishments
    LIMIT 15
""")
for r in ipc_details:
    if r['offenses'] or r['punishments']:
        print(f"\n  Section {r['s.section_number']}: {r['s.offense']}")
        if r['offenses']:
            print(f"    Offenses: {r['offenses']}")
        if r['punishments']:
            print(f"    Punishments: {r['punishments']}")

# 4. Cases citing other cases
print("\n4. CASES CITING OTHER CASES:")
citations = conn.execute_query("""
    MATCH (c1:Case)-[:CITES]->(c2:Case)
    RETURN c1.case_name as citing_case, c2.case_name as cited_case
    LIMIT 15
""")
for r in citations:
    print(f"  {r['citing_case']} -> cites -> {r['cited_case']}")

# 5. Cases referring to constitutional articles
print("\n5. CASES REFERRING TO CONSTITUTIONAL ARTICLES:")
article_refs = conn.execute_query("""
    MATCH (c:Case)-[:REFERS_TO]->(a:Article)
    RETURN c.case_name, a.article_number, a.title
    LIMIT 15
""")
for r in article_refs:
    print(f"  {r['c.case_name']} -> Article {r['a.article_number']}: {r['a.title']}")

# 6. All constitutional articles
print("\n6. ALL CONSTITUTIONAL ARTICLES:")
articles = conn.execute_query("""
    MATCH (a:Article)
    RETURN a.article_number, a.title
    ORDER BY toInteger(a.article_number)
""")
for r in articles:
    print(f"  Article {r['a.article_number']}: {r['a.title']}")

# 7. Judge-Court relationships
print("\n7. JUDGES AND COURTS THEY SERVED IN:")
judge_courts = conn.execute_query("""
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case)-[:HEARD_IN]->(court:Court)
    RETURN DISTINCT j.name, court.name, count(c) as cases
    ORDER BY cases DESC
    LIMIT 15
""")
for r in judge_courts:
    print(f"  {r['j.name']} - {r['court.name']}: {r['cases']} cases")

# 8. Top cited cases
print("\n8. TOP CITED CASES:")
top_cited = conn.execute_query("""
    MATCH (c:Case)<-[:CITES]-(citing:Case)
    RETURN c.case_name, count(citing) as citation_count
    ORDER BY citation_count DESC
    LIMIT 10
""")
for r in top_cited:
    print(f"  {r['c.case_name']}: {r['citation_count']} citations")

# 9. Cases by date range
print("\n9. RECENT CASES (Sample):")
recent = conn.execute_query("""
    MATCH (c:Case)
    WHERE c.date IS NOT NULL
    RETURN c.case_name, c.date
    ORDER BY c.date DESC
    LIMIT 10
""")
for r in recent:
    print(f"  {r['c.case_name']} ({r['c.date']})")

# 10. IPC sections without case relationships
print("\n10. IPC SECTIONS NOT LINKED TO CASES:")
orphan_ipc = conn.execute_query("""
    MATCH (s:IPCSection)
    WHERE NOT (s)<-[:GOVERNED_BY]-(:Case)
    RETURN s.section_number, s.offense
    LIMIT 15
""")
for r in orphan_ipc:
    print(f"  Section {r['s.section_number']}: {r['s.offense']}")

# 11. Complex path: Judge -> Case -> IPC -> Offense
print("\n11. JUDGE -> CASE -> IPC SECTION -> OFFENSE PATHS:")
paths = conn.execute_query("""
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case)-[:GOVERNED_BY]->(s:IPCSection)-[:APPLIES_TO]->(o:Offense)
    RETURN j.name, c.case_name, s.section_number, o.offense_type
    LIMIT 10
""")
for r in paths:
    print(f"  {r['j.name']} -> {r['c.case_name']} -> Sec {r['s.section_number']} -> {r['o.offense_type']}")

# 12. Cases heard in Supreme Court with judges
print("\n12. SUPREME COURT CASES WITH JUDGES:")
sc_cases = conn.execute_query("""
    MATCH (court:Court)<-[:HEARD_IN]-(c:Case)-[:DECIDED_BY]->(j:Judge)
    WHERE court.name CONTAINS 'Supreme'
    RETURN c.case_name, j.name
    LIMIT 15
""")
for r in sc_cases:
    print(f"  {r['c.case_name']} - Judge: {r['j.name']}")

# 13. All offense types
print("\n13. ALL OFFENSE TYPES IN KG:")
offenses = conn.execute_query("""
    MATCH (o:Offense)
    RETURN DISTINCT o.offense_type
    ORDER BY o.offense_type
    LIMIT 20
""")
for r in offenses:
    print(f"  - {r['o.offense_type']}")

# 14. All punishment types
print("\n14. ALL PUNISHMENT TYPES IN KG:")
punishments = conn.execute_query("""
    MATCH (p:Punishment)
    RETURN DISTINCT p.punishment_type
    ORDER BY p.punishment_type
    LIMIT 20
""")
for r in punishments:
    print(f"  - {r['p.punishment_type']}")

# 15. Cases with article references and judges
print("\n15. CASES WITH ARTICLES AND JUDGES:")
article_judge = conn.execute_query("""
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case)-[:REFERS_TO]->(a:Article)
    RETURN c.case_name, j.name, collect(a.article_number) as articles
    LIMIT 10
""")
for r in article_judge:
    print(f"  {r['c.case_name']}")
    print(f"    Judge: {r['j.name']}")
    print(f"    Articles: {r['articles']}")

conn.close()
print("\n=== ANALYSIS COMPLETE ===")
