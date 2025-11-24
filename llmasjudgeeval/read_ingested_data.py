"""
Script to read ingested data from ChromaDB and Neo4j to create ground truth dataset
"""
import sys
import os
import json
import chromadb
from neo4j import GraphDatabase

# Neo4j configuration - read from config
import sys
sys.path.append('config')
from settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def read_ipc_rag_data():
    """Read data from IPC RAG ChromaDB"""
    print("\n" + "="*80)
    print("Reading IPC RAG Data...")
    print("="*80)
    
    try:
        client = chromadb.PersistentClient(path="3rags/ipcrag/chroma_db")
        # List available collections
        collections = client.list_collections()
        if not collections:
            print("No collections found in IPC RAG")
            return None
        print(f"Available collections: {[c.name for c in collections]}")
        collection = collections[0]  # Use first available collection
        
        # Get all data
        results = collection.get(limit=50, include=['documents', 'metadatas'])
        
        print(f"Total IPC documents: {collection.count()}")
        print(f"\nSample IPC entries (first 10):")
        
        samples = []
        for i in range(min(10, len(results['documents']))):
            doc = results['documents'][i]
            meta = results['metadatas'][i] if results['metadatas'] else {}
            
            sample = {
                'content': doc[:200] + "..." if len(doc) > 200 else doc,
                'metadata': meta
            }
            samples.append(sample)
            print(f"\n{i+1}. {sample}")
        
        return {
            'total_count': collection.count(),
            'samples': samples,
            'all_documents': results['documents'][:50],
            'all_metadata': results['metadatas'][:50] if results['metadatas'] else []
        }
    except Exception as e:
        print(f"Error reading IPC RAG: {e}")
        return None

def read_legal_rag_data():
    """Read data from Legal Acts RAG ChromaDB"""
    print("\n" + "="*80)
    print("Reading Legal Acts RAG Data...")
    print("="*80)
    
    try:
        client = chromadb.PersistentClient(path="3rags/legal-rag-system/chroma_db")
        
        # List available collections
        available_collections = client.list_collections()
        print(f"Available collections in Legal RAG: {[c.name for c in available_collections]}")
        
        # Try to get both collections
        collections_data = {}
        
        try:
            const_collection = client.get_collection(name="constitution_chunks")
            const_results = const_collection.get(limit=30, include=['documents', 'metadatas'])
            collections_data['constitution'] = {
                'total_count': const_collection.count(),
                'samples': const_results['documents'][:10],
                'metadata': const_results['metadatas'][:10] if const_results['metadatas'] else [],
                'all_documents': const_results['documents'],
                'all_metadata': const_results['metadatas'] if const_results['metadatas'] else []
            }
            print(f"Constitution documents: {const_collection.count()}")
        except Exception as e:
            print(f"Constitution collection not found or error: {e}")
        
        try:
            acts_collection = client.get_collection(name="legal_acts")
            acts_results = acts_collection.get(limit=30, include=['documents', 'metadatas'])
            collections_data['legal_acts'] = {
                'total_count': acts_collection.count(),
                'samples': acts_results['documents'][:10],
                'metadata': acts_results['metadatas'][:10] if acts_results['metadatas'] else [],
                'all_documents': acts_results['documents'],
                'all_metadata': acts_results['metadatas'] if acts_results['metadatas'] else []
            }
            print(f"Legal Acts documents: {acts_collection.count()}")
        except Exception as e:
            print(f"Legal Acts collection not found or error: {e}")
        
        # Print samples
        for coll_name, data in collections_data.items():
            print(f"\n{coll_name.upper()} - Sample entries (first 5):")
            for i, doc in enumerate(data['samples'][:5]):
                print(f"\n{i+1}. {doc[:200]}...")
        
        return collections_data
    except Exception as e:
        print(f"Error reading Legal RAG: {e}")
        return None

def read_past_cases_rag_data():
    """Read data from Past Cases RAG ChromaDB"""
    print("\n" + "="*80)
    print("Reading Past Cases RAG Data...")
    print("="*80)
    
    try:
        client = chromadb.PersistentClient(path="3rags/past_cases_rag/chroma_db")
        # List available collections
        collections = client.list_collections()
        if not collections:
            print("No collections found in Past Cases RAG")
            return None
        print(f"Available collections: {[c.name for c in collections]}")
        collection = collections[0]  # Use first available collection
        
        # Get all data
        results = collection.get(limit=50, include=['documents', 'metadatas'])
        
        print(f"Total Past Case documents: {collection.count()}")
        print(f"\nSample Case entries (first 5):")
        
        samples = []
        for i in range(min(5, len(results['documents']))):
            doc = results['documents'][i]
            meta = results['metadatas'][i] if results['metadatas'] else {}
            
            sample = {
                'content': doc[:300] + "..." if len(doc) > 300 else doc,
                'metadata': meta
            }
            samples.append(sample)
            print(f"\n{i+1}. Case: {meta.get('case_name', 'Unknown')}")
            print(f"   Court: {meta.get('court', 'Unknown')}")
            print(f"   Date: {meta.get('date', 'Unknown')}")
            print(f"   Content preview: {doc[:200]}...")
        
        return {
            'total_count': collection.count(),
            'samples': samples,
            'all_documents': results['documents'],
            'all_metadata': results['metadatas'] if results['metadatas'] else []
        }
    except Exception as e:
        print(f"Error reading Past Cases RAG: {e}")
        return None

def read_neo4j_kg_data():
    """Read data from Neo4j Knowledge Graph"""
    print("\n" + "="*80)
    print("Reading Neo4j Knowledge Graph Data...")
    print("="*80)
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            # Get node counts
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as NodeType, count(*) as Count
                ORDER BY Count DESC
            """).data()
            
            print("\nNode Counts:")
            for item in node_counts:
                print(f"  {item['NodeType']}: {item['Count']}")
            
            # Get sample IPC Sections
            ipc_sections = session.run("""
                MATCH (s:IPCSection)
                RETURN s.section_number as section, s.offense as offense, 
                       s.punishment as punishment, s.cognizable as cognizable,
                       s.bailable as bailable
                LIMIT 20
            """).data()
            
            print(f"\nSample IPC Sections (first 10):")
            for i, ipc in enumerate(ipc_sections[:10]):
                print(f"  {i+1}. Section {ipc['section']}: {ipc['offense']}")
            
            # Get sample Cases
            cases = session.run("""
                MATCH (c:Case)
                RETURN c.case_id as id, c.case_name as name, c.date as date
                LIMIT 20
            """).data()
            
            print(f"\nSample Cases (first 10):")
            for i, case in enumerate(cases[:10]):
                print(f"  {i+1}. {case['name']} ({case['date']})")
            
            # Get sample Judges
            judges = session.run("""
                MATCH (j:Judge)
                RETURN j.name as name
                LIMIT 15
            """).data()
            
            print(f"\nSample Judges (first 10):")
            for i, judge in enumerate(judges[:10]):
                print(f"  {i+1}. {judge['name']}")
            
            # Get sample Courts
            courts = session.run("""
                MATCH (c:Court)
                RETURN c.name as name
                LIMIT 10
            """).data()
            
            print(f"\nSample Courts:")
            for i, court in enumerate(courts):
                print(f"  {i+1}. {court['name']}")
            
            # Get relationship counts
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as RelationType, count(*) as Count
                ORDER BY Count DESC
            """).data()
            
            print(f"\nRelationship Counts:")
            for item in rel_counts:
                print(f"  {item['RelationType']}: {item['Count']}")
            
            # Get case-IPC relationships
            case_ipc = session.run("""
                MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection)
                RETURN c.case_name as case_name, s.section_number as section
                LIMIT 15
            """).data()
            
            print(f"\nSample Case-IPC Relationships (first 10):")
            for i, rel in enumerate(case_ipc[:10]):
                print(f"  {i+1}. {rel['case_name']} -> IPC Section {rel['section']}")
            
            # Get top cited IPC sections
            top_ipc = session.run("""
                MATCH (s:IPCSection)<-[:GOVERNED_BY]-(c:Case)
                RETURN s.section_number as section, s.offense as offense, count(c) as case_count
                ORDER BY case_count DESC
                LIMIT 15
            """).data()
            
            print(f"\nTop Cited IPC Sections:")
            for i, ipc in enumerate(top_ipc[:10]):
                print(f"  {i+1}. Section {ipc['section']}: {ipc['offense']} ({ipc['case_count']} cases)")
            
            # Get Articles
            articles = session.run("""
                MATCH (a:Article)
                RETURN a.article_number as number, a.title as title
                LIMIT 15
            """).data()
            
            print(f"\nSample Constitutional Articles (first 10):")
            for i, art in enumerate(articles[:10]):
                print(f"  {i+1}. Article {art['number']}: {art['title']}")
            
            driver.close()
            
            return {
                'node_counts': node_counts,
                'ipc_sections': ipc_sections,
                'cases': cases,
                'judges': judges,
                'courts': courts,
                'relationships': rel_counts,
                'case_ipc_relationships': case_ipc,
                'top_cited_ipc': top_ipc,
                'articles': articles
            }
    except Exception as e:
        print(f"Error reading Neo4j KG: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to read all data sources"""
    print("\n" + "#"*80)
    print("# READING INGESTED DATA FROM ALL SOURCES")
    print("#"*80)
    
    # Read all data
    ipc_data = read_ipc_rag_data()
    legal_data = read_legal_rag_data()
    cases_data = read_past_cases_rag_data()
    kg_data = read_neo4j_kg_data()
    
    # Save to JSON for reference
    output = {
        'ipc_rag': ipc_data,
        'legal_rag': legal_data,
        'past_cases_rag': cases_data,
        'knowledge_graph': kg_data
    }
    
    with open('ingested_data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ Data reading complete! Summary saved to 'ingested_data_summary.json'")
    print("="*80)

if __name__ == "__main__":
    main()
