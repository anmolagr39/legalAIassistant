"""
Interactive Query Interface for Legal Knowledge Graph
Test querying the existing database
"""
from config.neo4j_config import Neo4jConnection
import json


def get_db_stats(conn):
    """Get database statistics"""
    print("\n" + "="*80)
    print("📊 DATABASE STATISTICS")
    print("="*80)
    
    stats = conn.get_stats()
    print(f"\nTotal Nodes: {stats['nodes']}")
    print(f"Total Relationships: {stats['relationships']}")
    
    print("\n📋 Node Types:")
    for node_type, count in stats['node_types']:
        print(f"  • {node_type}: {count}")
    
    print("\n🔗 Relationship Types:")
    rel_query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(*) as count
    ORDER BY count DESC
    """
    results = conn.execute_query(rel_query)
    for r in results:
        print(f"  • {r['rel_type']}: {r['count']}")


def query_ipc_sections(conn):
    """Query IPC sections"""
    print("\n" + "="*80)
    print("⚖️  IPC SECTIONS")
    print("="*80)
    
    query = """
    MATCH (s:IPCSection)
    RETURN s.section_number as section, s.offense as offense, s.punishment as punishment
    ORDER BY toInteger(s.section_number)
    LIMIT 10
    """
    results = conn.execute_query(query)
    
    print("\nSample IPC Sections:")
    for r in results:
        print(f"\n  Section {r['section']}:")
        print(f"    Offense: {r['offense'][:80]}...")
        print(f"    Punishment: {r['punishment'][:80]}...")


def query_cases(conn):
    """Query cases"""
    print("\n" + "="*80)
    print("📜 CASE DOCUMENTS")
    print("="*80)
    
    query = """
    MATCH (c:Case)
    OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
    OPTIONAL MATCH (c)-[:PRESIDED_BY]->(judge:Judge)
    OPTIONAL MATCH (c)-[:GOVERNED_BY]->(section:IPCSection)
    RETURN c.case_name as case_name, 
           c.date as date,
           court.name as court,
           collect(DISTINCT judge.name)[0..3] as judges,
           collect(DISTINCT section.section_number)[0..5] as sections
    LIMIT 5
    """
    results = conn.execute_query(query)
    
    print("\nSample Cases:")
    for r in results:
        print(f"\n  Case: {r['case_name']}")
        print(f"  Date: {r['date']}")
        print(f"  Court: {r['court']}")
        print(f"  Judges: {', '.join(r['judges']) if r['judges'] else 'None'}")
        print(f"  IPC Sections: {', '.join(r['sections']) if r['sections'] else 'None'}")


def query_articles(conn):
    """Query constitutional articles"""
    print("\n" + "="*80)
    print("📖 CONSTITUTIONAL ARTICLES")
    print("="*80)
    
    query = """
    MATCH (a:Article)
    RETURN a.article_number as number, a.title as title, a.part as part
    ORDER BY toInteger(a.article_number)
    LIMIT 10
    """
    results = conn.execute_query(query)
    
    if results:
        print("\nSample Articles:")
        for r in results:
            print(f"\n  Article {r['number']}: {r['title']}")
            print(f"    Part: {r['part']}")
    else:
        print("\nNo articles found in database yet.")


def search_by_keyword(conn, keyword):
    """Search across all entities by keyword"""
    print("\n" + "="*80)
    print(f"🔍 SEARCH RESULTS FOR: '{keyword}'")
    print("="*80)
    
    # Search IPC sections
    ipc_query = """
    MATCH (s:IPCSection)
    WHERE toLower(s.offense) CONTAINS toLower($keyword)
       OR toLower(s.punishment) CONTAINS toLower($keyword)
    RETURN s.section_number as section, s.offense as offense
    LIMIT 5
    """
    results = conn.execute_query(ipc_query, {'keyword': keyword})
    
    if results:
        print("\n📌 IPC Sections:")
        for r in results:
            print(f"  • Section {r['section']}: {r['offense'][:100]}...")
    
    # Search cases
    case_query = """
    MATCH (c:Case)
    WHERE toLower(c.case_name) CONTAINS toLower($keyword)
       OR toLower(c.summary) CONTAINS toLower($keyword)
    RETURN c.case_name as case_name, c.date as date
    LIMIT 5
    """
    results = conn.execute_query(case_query, {'keyword': keyword})
    
    if results:
        print("\n📌 Cases:")
        for r in results:
            print(f"  • {r['case_name']} ({r['date']})")


def find_related_cases(conn, section_number):
    """Find cases related to an IPC section"""
    print("\n" + "="*80)
    print(f"🔗 CASES RELATED TO IPC SECTION {section_number}")
    print("="*80)
    
    query = """
    MATCH (s:IPCSection {section_number: $section})
    OPTIONAL MATCH (c:Case)-[:GOVERNED_BY]->(s)
    RETURN s.section_number as section,
           s.offense as offense,
           collect(c.case_name)[0..5] as cases,
           count(c) as case_count
    """
    results = conn.execute_query(query, {'section': section_number})
    
    if results:
        r = results[0]
        print(f"\nSection {r['section']}:")
        print(f"Offense: {r['offense'][:100]}...")
        print(f"\nRelated Cases ({r['case_count']}):")
        for case in r['cases']:
            print(f"  • {case}")
    else:
        print(f"\nNo data found for Section {section_number}")


def interactive_menu():
    """Interactive query menu"""
    conn = Neo4jConnection()
    conn.connect()
    
    while True:
        print("\n" + "="*80)
        print("🤖 LEGAL KNOWLEDGE GRAPH QUERY INTERFACE")
        print("="*80)
        print("\nOptions:")
        print("  1. Show database statistics")
        print("  2. View IPC sections")
        print("  3. View cases")
        print("  4. View constitutional articles")
        print("  5. Search by keyword")
        print("  6. Find cases by IPC section")
        print("  7. Custom Cypher query")
        print("  0. Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            get_db_stats(conn)
        elif choice == '2':
            query_ipc_sections(conn)
        elif choice == '3':
            query_cases(conn)
        elif choice == '4':
            query_articles(conn)
        elif choice == '5':
            keyword = input("Enter search keyword: ").strip()
            search_by_keyword(conn, keyword)
        elif choice == '6':
            section = input("Enter IPC section number (e.g., 302): ").strip()
            find_related_cases(conn, section)
        elif choice == '7':
            print("\nEnter Cypher query (end with semicolon):")
            query = input().strip()
            try:
                results = conn.execute_query(query)
                print("\nResults:")
                for r in results[:10]:
                    print(f"  {dict(r)}")
                if len(results) > 10:
                    print(f"  ... ({len(results) - 10} more results)")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice!")
    
    conn.close()
    print("\n✓ Session ended")


if __name__ == "__main__":
    interactive_menu()
