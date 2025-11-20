"""
GraphRAG Chatbot for Legal Knowledge Graph
Natural language queries → Cypher → Neo4j → LLM-generated answers
"""
import os
from config.neo4j_config import Neo4jConnection
from extraction.gemini_extractor import GeminiExtractor
from dotenv import load_dotenv
import json
import google.generativeai as genai

load_dotenv()


class LegalGraphRAG:
    """GraphRAG system for legal knowledge graph"""
    
    def __init__(self):
        self.neo4j_conn = Neo4jConnection()
        self.neo4j_conn.connect()
        
        # Initialize LLM (Gemini)
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        self.llm = GeminiExtractor(api_key=gemini_key)
        
        # Also configure genai for direct calls
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
    def get_all_node_names(self) -> str:
        """Get ALL node names from database to include in prompt"""
        node_data = ""
        
        # Get ALL case names
        try:
            cases = self.neo4j_conn.execute_query(
                "MATCH (c:Case) RETURN c.case_name ORDER BY c.case_name"
            )
            node_data += f"\nALL CASE NAMES ({len(cases)} total):\n"
            for r in cases:
                node_data += f"  - {r['c.case_name']}\n"
        except:
            pass
        
        # Get ALL judges
        try:
            judges = self.neo4j_conn.execute_query(
                "MATCH (j:Judge) RETURN j.name ORDER BY j.name"
            )
            node_data += f"\nALL JUDGE NAMES ({len(judges)} total):\n"
            for r in judges:
                node_data += f"  - {r['j.name']}\n"
        except:
            pass
        
        # Get ALL IPC sections
        try:
            sections = self.neo4j_conn.execute_query(
                "MATCH (s:IPCSection) RETURN s.section_number ORDER BY toInteger(s.section_number)"
            )
            node_data += f"\nALL IPC SECTIONS ({len(sections)} total):\n"
            section_nums = [r['s.section_number'] for r in sections]
            node_data += f"  {', '.join(section_nums)}\n"
        except:
            pass
        
        # Get ALL articles
        try:
            articles = self.neo4j_conn.execute_query(
                "MATCH (a:Article) RETURN a.article_number, a.title ORDER BY toInteger(a.article_number)"
            )
            if articles:
                node_data += f"\nALL CONSTITUTIONAL ARTICLES ({len(articles)} total):\n"
                for r in articles:
                    node_data += f"  - Article {r['a.article_number']}: {r['a.title']}\n"
        except:
            pass
        
        # Get ALL courts
        try:
            courts = self.neo4j_conn.execute_query(
                "MATCH (c:Court) RETURN c.name ORDER BY c.name"
            )
            node_data += f"\nALL COURTS ({len(courts)} total):\n"
            for r in courts:
                node_data += f"  - {r['c.name']}\n"
        except:
            pass
        
        return node_data
    
    def generate_cypher_query(self, question: str) -> str:
        """Convert natural language question to Cypher query using LLM"""
        
        schema_info = """
        Graph Schema:
        - Nodes: Case, IPCSection, Judge, Court, Article, Offense, Punishment
        - Relationships:
          * (Case)-[:GOVERNED_BY]->(IPCSection)
          * (Case)-[:DECIDED_BY]->(Judge)
          * (Case)-[:HEARD_IN]->(Court)
          * (Case)-[:REFERS_TO]->(Article)
          * (Case)-[:CITES]->(Case)
          * (IPCSection)-[:PRESCRIBES]->(Offense)
          * (IPCSection)-[:PRESCRIBES]->(Punishment)
        
        Node Properties:
        - Case: case_name, date, summary, citations
        - IPCSection: section_number, offense, punishment, bailable, cognizable
        - Judge: name
        - Court: name
        - Article: article_number, title, provisions, part
        - Offense: offense_type
        - Punishment: punishment_type
        """
        
        # Get all actual node names
        all_nodes = self.get_all_node_names()
        schema_info += all_nodes
        
        prompt = f"""You are a Neo4j Cypher query expert for a legal knowledge graph.

{schema_info}

Convert this natural language question into a Cypher query:
"{question}"

CRITICAL: Return ONLY the raw Cypher query text, NO JSON, NO explanations, NO formatting.

Rules:
1. Return ONLY the Cypher query as plain text
2. Use LIMIT 10 for queries that might return many results
3. Use toLower() for case-insensitive text matching
4. Use CONTAINS for partial text matching
5. Always include relevant node properties in RETURN clause

Examples:
Q: "What is IPC section 302?"
A: MATCH (s:IPCSection {{section_number: "302"}}) RETURN s.section_number, s.offense, s.punishment

Q: "Which cases involve murder?"
A: MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection) WHERE toLower(s.offense) CONTAINS "murder" RETURN c.case_name, c.date, s.section_number LIMIT 10

Q: "Who are the judges?"
A: MATCH (j:Judge)<-[:DECIDED_BY]-(c:Case) RETURN j.name, count(c) as case_count ORDER BY case_count DESC LIMIT 10

Q: "Cases related to Article 21"
A: MATCH (c:Case)-[:REFERS_TO]->(a:Article {{article_number: "21"}}) RETURN c.case_name, c.date, a.title LIMIT 10

IMPORTANT: 
- Use the EXACT node names from the lists above when generating queries
- If user mentions a case/judge name, find the exact match from the lists above
- Use proper property matching with exact values from the data

Now generate ONLY the Cypher query for: "{question}"
Cypher query:"""
        
        # Use Gemini to generate Cypher query
        response = self.gemini_model.generate_content(prompt)
        cypher = response.text.strip()
        
        # Clean up the query
        if cypher.startswith("```"):
            cypher = cypher.split("```")[1]
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        cypher = cypher.strip()
        
        return cypher
    
    def execute_query(self, cypher: str) -> list:
        """Execute Cypher query and return results"""
        try:
            results = self.neo4j_conn.execute_query(cypher)
            return [dict(record) for record in results]
        except Exception as e:
            return [{"error": str(e)}]
    
    def generate_answer(self, question: str, results: list) -> str:
        """Generate natural language answer from query results"""
        
        if not results:
            return "I couldn't find any relevant information in the knowledge graph for your question."
        
        if "error" in results[0]:
            return f"Sorry, I encountered an error: {results[0]['error']}"
        
        # Convert results to readable format
        results_text = json.dumps(results, indent=2, default=str)
        
        prompt = f"""You are a legal assistant helping lawyers understand information from a legal knowledge graph.

User Question: "{question}"

Retrieved Data from Knowledge Graph:
{results_text}

Generate a clear, professional answer based ONLY on the retrieved data. 

Rules:
1. Be concise and factual
2. Cite specific case names, section numbers, or article numbers
3. If multiple results exist, summarize key patterns
4. Don't make up information not in the data
5. Use legal terminology appropriately

Answer:"""
        
        # Use Gemini to generate answer
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()
    
    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Main query pipeline:
        1. Convert question to Cypher
        2. Execute query
        3. Generate natural language answer
        """
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"❓ Question: {question}")
            print(f"{'='*80}")
        
        # Step 1: Generate Cypher query
        if verbose:
            print("\n🔄 Generating Cypher query...")
        
        cypher = self.generate_cypher_query(question)
        
        if verbose:
            print(f"\n📝 Cypher Query:\n{cypher}")
        
        # Step 2: Execute query
        if verbose:
            print("\n🔍 Querying knowledge graph...")
        
        results = self.execute_query(cypher)
        
        if verbose:
            print(f"\n📊 Retrieved {len(results)} results")
        
        # Step 3: Generate answer
        if verbose:
            print("\n💭 Generating answer...")
        
        answer = self.generate_answer(question, results)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ ANSWER:")
            print(f"{'='*80}")
            print(answer)
            print(f"{'='*80}\n")
        
        return {
            'question': question,
            'cypher': cypher,
            'results': results,
            'answer': answer
        }
    
    def close(self):
        """Close connections"""
        self.neo4j_conn.close()


def interactive_chatbot():
    """Run interactive chatbot"""
    print("\n" + "="*80)
    print("🤖 LEGAL KNOWLEDGE GRAPH CHATBOT (GraphRAG)")
    print("="*80)
    print("\nAsk questions in natural language about:")
    print("  • IPC sections and offenses")
    print("  • Court cases and judgments")
    print("  • Judges and courts")
    print("  • Constitutional articles")
    print("\nType 'exit' to quit, 'examples' for sample questions\n")
    
    rag = LegalGraphRAG()
    
    examples = [
        "What is IPC section 302?",
        "Which cases involve murder?",
        "Who are the top judges by case count?",
        "What are bailable offenses?",
        "Cases related to Article 21",
        "What punishments are prescribed for theft?",
        "Show me recent cases from 2022",
        "Which IPC sections are most commonly cited?",
    ]
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'examples':
                print("\n📋 Example Questions:")
                for i, ex in enumerate(examples, 1):
                    print(f"  {i}. {ex}")
                print()
                continue
            
            # Process query
            result = rag.query(user_input, verbose=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    rag.close()


def test_queries():
    """Test with predefined queries"""
    print("\n" + "="*80)
    print("🧪 TESTING GRAPHRAG SYSTEM")
    print("="*80)
    
    rag = LegalGraphRAG()
    
    test_questions = [
        "What is IPC section 302?",
        "Which cases involve murder?",
        "Who are the top 5 judges by case count?",
    ]
    
    for question in test_questions:
        rag.query(question, verbose=True)
        input("\nPress Enter for next question...")
    
    rag.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_queries()
    else:
        interactive_chatbot()
