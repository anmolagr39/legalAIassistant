"""
Query and test the Legal Knowledge Graph
"""
import sys
sys.path.insert(0, '.')

from config.neo4j_config import Neo4jConnection
import json


def test_kg():
    """Test the built knowledge graph"""
    conn = Neo4jConnection()
    conn.connect()
    
    print("\n" + "=" * 80)
    print("TESTING LEGAL KNOWLEDGE GRAPH")
    print("=" * 80)
    
    # Query 1: Overall Statistics
    print("\n📊 1. OVERALL STATISTICS")
    print("-" * 80)
    stats = conn.get_stats()
    print(f"Total Nodes: {stats['nodes']}")
    print(f"Total Relationships: {stats['relationships']}")
    print("\nNode Distribution:")
    for node_type, count in stats['node_types']:
        print(f"  {node_type}: {count}")
    
    # Query 2: Sample IPC Sections
    print("\n⚖️  2. SAMPLE IPC SECTIONS")
    print("-" * 80)
    query = """
    MATCH (s:IPCSection)
    RETURN s.section_number, s.offense, s.punishment
    ORDER BY s.section_number
    LIMIT 10
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"Section {r['s.section_number']}: {r['s.offense']}")
        print(f"  Punishment: {r['s.punishment']}\n")
    
    # Query 3: Sample Cases
    print("\n📁 3. PROCESSED CASES")
    print("-" * 80)
    query = """
    MATCH (c:Case)
    RETURN c.case_id, c.case_name, c.date
    ORDER BY c.case_id
    LIMIT 10
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"{r['c.case_id']}: {r['c.case_name']}")
        print(f"  Date: {r['c.date']}\n")
    
    # Query 4: Courts
    print("\n🏛️  4. COURTS")
    print("-" * 80)
    query = """
    MATCH (court:Court)<-[:HEARD_IN]-(c:Case)
    RETURN court.name, count(c) as case_count
    ORDER BY case_count DESC
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"{r['court.name']}: {r['case_count']} cases")
    
    # Query 5: Judges
    print("\n👨‍⚖️  5. JUDGES")
    print("-" * 80)
    query = """
    MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case)
    RETURN j.name, count(c) as case_count
    ORDER BY case_count DESC
    LIMIT 10
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"{r['j.name']}: {r['case_count']} cases")
    
    # Query 6: IPC Sections in Cases
    print("\n🔗 6. IPC SECTIONS REFERENCED IN CASES")
    print("-" * 80)
    query = """
    MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection)
    RETURN s.section_number, s.offense, count(c) as case_count
    ORDER BY case_count DESC
    LIMIT 10
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"Section {r['s.section_number']} ({r['s.offense']}): {r['case_count']} cases")
    
    # Query 7: Constitutional Articles Referenced
    print("\n📜 7. CONSTITUTIONAL ARTICLES IN CASES")
    print("-" * 80)
    query = """
    MATCH (c:Case)-[:REFERS_TO]->(a:Article)
    RETURN a.article_number, a.title, count(c) as case_count
    ORDER BY case_count DESC
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"Article {r['a.article_number']} ({r['a.title']}): {r['case_count']} cases")
    
    # Query 8: Citation Network
    print("\n🔗 8. CITATION NETWORK")
    print("-" * 80)
    query = """
    MATCH (c1:Case)-[:CITES]->(c2:Case)
    RETURN c1.case_name, c2.case_name
    LIMIT 10
    """
    results = conn.execute_query(query)
    if results:
        for r in results:
            print(f"{r['c1.case_name'][:50]}...")
            print(f"  → cites: {r['c2.case_name'][:60]}...\n")
    else:
        print("No citations found yet")
    
    # Query 9: Most Common Offenses
    print("\n⚠️  9. MOST COMMON OFFENSES")
    print("-" * 80)
    query = """
    MATCH (o:Offense)<-[:APPLIES_TO]-(s:IPCSection)
    RETURN o.offense_type, count(s) as section_count
    ORDER BY section_count DESC
    LIMIT 10
    """
    results = conn.execute_query(query)
    for r in results:
        print(f"{r['o.offense_type'][:70]}: {r['section_count']} sections")
    
    # Query 10: Sample Case Details
    print("\n📋 10. DETAILED CASE EXAMPLE")
    print("-" * 80)
    query = """
    MATCH (c:Case {case_id: 'C1'})
    OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
    OPTIONAL MATCH (c)-[:DECIDED_BY]->(j:Judge)
    OPTIONAL MATCH (c)-[:GOVERNED_BY]->(s:IPCSection)
    OPTIONAL MATCH (c)-[:REFERS_TO]->(a:Article)
    RETURN c.case_name, c.date, court.name, 
           collect(DISTINCT j.name) as judges,
           collect(DISTINCT s.section_number) as sections,
           collect(DISTINCT a.article_number) as articles
    """
    results = conn.execute_query(query)
    if results:
        r = results[0]
        print(f"Case: {r['c.case_name']}")
        print(f"Date: {r['c.date']}")
        print(f"Court: {r['court.name']}")
        print(f"Judges: {', '.join(r['judges']) if r['judges'] else 'None'}")
        print(f"IPC Sections: {', '.join(r['sections']) if r['sections'] else 'None'}")
        print(f"Articles: {', '.join(r['articles']) if r['articles'] else 'None'}")
    
    # Query 11: Graph Density
    print("\n📊 11. GRAPH METRICS")
    print("-" * 80)
    avg_degree_query = """
    MATCH (n)
    WITH count(n) as node_count
    MATCH ()-[r]->()
    WITH node_count, count(r) as rel_count
    RETURN node_count, rel_count, 
           toFloat(rel_count) / node_count as avg_degree
    """
    results = conn.execute_query(avg_degree_query)
    if results:
        r = results[0]
        print(f"Average degree (relationships per node): {r['avg_degree']:.2f}")
        print(f"Graph density: {(r['rel_count'] / (r['node_count'] * (r['node_count'] - 1))) * 100:.4f}%")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✓ Knowledge Graph Testing Complete!")
    print("=" * 80)
    print("\n💡 Next: Open Neo4j Browser at http://localhost:7474")
    print("   Try query: MATCH (n) RETURN n LIMIT 100")
    print("\n")


if __name__ == "__main__":
    test_kg()
