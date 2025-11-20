"""
Agentic Legal Assistant Orchestrator
Routes queries to appropriate systems: Knowledge Graph and/or RAG systems
"""
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv

# Add paths for all systems
sys.path.append(os.path.join(os.path.dirname(__file__), '3rags', 'ipcrag', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '3rags', 'past_cases_rag', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '3rags', 'legal-rag-system', 'src'))

from config.neo4j_config import Neo4jConnection
from extraction.gemini_extractor import GeminiExtractor

load_dotenv()


class SystemType(Enum):
    """Available systems for query routing"""
    KNOWLEDGE_GRAPH = "knowledge_graph"
    IPC_RAG = "ipc_rag"
    PAST_CASES_RAG = "past_cases_rag"
    LEGAL_ACTS_RAG = "legal_acts_rag"


class QueryRouter:
    """Routes user queries to appropriate backend systems"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query and determine which systems to use
        Returns: {
            "systems": [SystemType, ...],
            "reasoning": str,
            "query_type": str
        }
        """
        
        prompt = f"""You are a legal query router. Analyze this query and determine which backend systems should be used.

Available Systems:
1. KNOWLEDGE_GRAPH: Neo4j graph database with:
   - 1482 legal cases with citations, judges, courts
   - 445 IPC sections with offenses and punishments
   - 16 constitutional articles
   - Relationships: cases cite each other, cases decided by judges, cases governed by IPC sections
   Use for: Finding related cases, citation networks, judge-case connections, IPC section lookups, relationship queries

2. IPC_RAG: RAG system for Indian Penal Code sections
   - 445 IPC sections with detailed descriptions
   - Offense types, punishments, bailable status
   Use for: Detailed IPC section explanations, offense definitions, punishment details, cognizable/bailable info

3. PAST_CASES_RAG: RAG system for Supreme Court case documents
   - 2914 full case judgments with detailed text
   Use for: Case law research, precedents, legal reasoning, detailed case facts, judicial interpretations

4. LEGAL_ACTS_RAG: RAG system for Constitution and Legal Acts
   - Full Constitution of India text
   - Various legal act provisions
   Use for: Constitutional provisions, fundamental rights, legal act details, constitutional interpretation

User Query: "{query}"

Analyze and respond in JSON format:
{{
    "systems": ["system_name1", "system_name2"],
    "reasoning": "Why these systems were chosen",
    "query_type": "factual|analytical|research|comparison",
    "requires_multiple": true/false
}}

System names must be from: knowledge_graph, ipc_rag, past_cases_rag, legal_acts_rag

Examples:
- "What is IPC section 302?" → knowledge_graph (quick lookup) + ipc_rag (detailed explanation)
- "Cases involving murder" → knowledge_graph (structured query) + past_cases_rag (detailed case law)
- "Article 21 right to life" → knowledge_graph (article info) + legal_acts_rag (full text)
- "Who is judge XYZ?" → knowledge_graph only (graph relationships)
- "Explain bailable vs non-bailable" → ipc_rag only (IPC concepts)
"""
        
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Convert system names to enums
        systems = []
        for sys_name in result["systems"]:
            try:
                systems.append(SystemType(sys_name))
            except ValueError:
                pass
        
        result["systems"] = systems
        return result


class KnowledgeGraphSystem:
    """Interface to Neo4j Knowledge Graph"""
    
    def __init__(self):
        self.conn = Neo4jConnection()
        self.conn.connect()
        genai_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=genai_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def query(self, user_query: str) -> Dict:
        """Query the knowledge graph"""
        # Generate Cypher query
        cypher = self._generate_cypher(user_query)
        
        # Execute query
        try:
            results = self.conn.execute_query(cypher)
            results_list = [dict(record) for record in results]
            return {
                "source": "Knowledge Graph",
                "cypher": cypher,
                "results": results_list,
                "count": len(results_list)
            }
        except Exception as e:
            return {
                "source": "Knowledge Graph",
                "error": str(e),
                "results": []
            }
    
    def _generate_cypher(self, question: str) -> str:
        """Generate Cypher query from natural language"""
        schema_info = """
        Graph Schema:
        - Nodes: Case (case_name, date), IPCSection (section_number, offense, punishment), 
                 Judge (name), Court (name), Article (article_number, title)
        - Relationships: (Case)-[:GOVERNED_BY]->(IPCSection), (Case)-[:DECIDED_BY]->(Judge),
                         (Case)-[:HEARD_IN]->(Court), (Case)-[:CITES]->(Case)
        """
        
        prompt = f"""{schema_info}

Convert this question to Cypher query: "{question}"

Return ONLY the Cypher query, no explanations.
Use LIMIT 10 for large result sets.
"""
        
        response = self.model.generate_content(prompt)
        cypher = response.text.strip()
        
        # Clean up
        if cypher.startswith("```"):
            cypher = cypher.split("```")[1]
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        
        return cypher.strip()
    
    def close(self):
        self.conn.close()


class IPCRagSystem:
    """Interface to IPC RAG system"""
    
    def __init__(self):
        from config import Config
        from rag_engine import FIRRagEngine
        import os
        
        # Set API key
        Config.set_gemini_api_key(GEMINI_API_KEY)
        
        # Initialize engine
        self.engine = FIRRagEngine(Config)
        
        # Check if data is ingested
        try:
            collection = self.engine.chroma_client.get_collection(Config.COLLECTION_NAME)
            count = collection.count()
            print(f"✅ IPC RAG initialized with {count} chunks")
        except:
            print("⚠️ IPC RAG collection empty. Please ingest data first.")
            # Auto-ingest if FIR_DATASET.csv exists
            dataset_path = os.path.join(os.path.dirname(__file__), '3rags', 'ipcrag', 'FIR_DATASET.csv')
            if os.path.exists(dataset_path):
                print("📥 Auto-ingesting IPC data...")
                self.engine.ingest_fir_data(dataset_path)
            else:
                print(f"❌ Dataset not found at {dataset_path}")
        
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """Query IPC RAG system"""
        try:
            result = self.engine.query(user_query, top_k=top_k)
            return {
                "source": "IPC RAG",
                "answer": result.get("answer", ""),
                "retrieved_docs": result.get("retrieved_docs", []),
                "count": len(result.get("retrieved_docs", []))
            }
        except Exception as e:
            return {
                "source": "IPC RAG",
                "error": str(e),
                "answer": ""
            }


class PastCasesRagSystem:
    """Interface to Past Cases RAG system"""
    
    def __init__(self):
        from rag import ChromaVectorDB, LegalRAGSystem
        
        self.vector_db = ChromaVectorDB()
        self.rag_system = LegalRAGSystem(self.vector_db)
        
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """Query Past Cases RAG system"""
        try:
            result = self.rag_system.query(user_query, top_k=top_k)
            return {
                "source": "Past Cases RAG",
                "answer": result.get("response", ""),
                "retrieved_docs": result.get("contexts", []),
                "count": len(result.get("contexts", []))
            }
        except Exception as e:
            return {
                "source": "Past Cases RAG",
                "error": str(e),
                "answer": ""
            }


class LegalActsRagSystem:
    """Interface to Legal Acts RAG system"""
    
    def __init__(self):
        from vector_store import VectorStore
        from rag_system import LegalRAG
        
        self.vector_store = VectorStore()
        self.rag_system = LegalRAG(self.vector_store)
        
    def query(self, user_query: str, top_k: int = 5) -> Dict:
        """Query Legal Acts RAG system"""
        try:
            result = self.rag_system.query(user_query, top_k=top_k)
            return {
                "source": "Legal Acts RAG",
                "answer": result.get("answer", ""),
                "retrieved_docs": result.get("sources", []),
                "count": len(result.get("sources", []))
            }
        except Exception as e:
            return {
                "source": "Legal Acts RAG",
                "error": str(e),
                "answer": ""
            }


class AgenticOrchestrator:
    """Main orchestrator that coordinates all systems"""
    
    def __init__(self):
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        print("🚀 Initializing Agentic Legal Assistant...")
        
        self.router = QueryRouter(gemini_key)
        genai.configure(api_key=gemini_key)
        self.answerer = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize systems lazily
        self.systems = {}
        print("✅ Orchestrator ready!")
        
    def _get_system(self, system_type: SystemType):
        """Lazy load systems"""
        if system_type not in self.systems:
            print(f"  Loading {system_type.value}...")
            
            if system_type == SystemType.KNOWLEDGE_GRAPH:
                self.systems[system_type] = KnowledgeGraphSystem()
            elif system_type == SystemType.IPC_RAG:
                self.systems[system_type] = IPCRagSystem()
            elif system_type == SystemType.PAST_CASES_RAG:
                self.systems[system_type] = PastCasesRagSystem()
            elif system_type == SystemType.LEGAL_ACTS_RAG:
                self.systems[system_type] = LegalActsRagSystem()
        
        return self.systems[system_type]
    
    def process_query(self, user_query: str) -> Dict:
        """
        Main processing pipeline:
        1. Route query to appropriate systems
        2. Query selected systems in parallel
        3. Synthesize results into coherent answer
        """
        
        print(f"\n{'='*80}")
        print(f"❓ User Query: {user_query}")
        print(f"{'='*80}")
        
        # Step 1: Route query
        print("\n🔀 Step 1: Routing query...")
        routing = self.router.analyze_query(user_query)
        
        print(f"  Selected systems: {[s.value for s in routing['systems']]}")
        print(f"  Reasoning: {routing['reasoning']}")
        print(f"  Query type: {routing['query_type']}")
        
        # Step 2: Query systems
        print("\n🔍 Step 2: Querying systems...")
        all_results = []
        
        for system_type in routing['systems']:
            try:
                system = self._get_system(system_type)
                result = system.query(user_query)
                all_results.append(result)
                print(f"  ✓ {system_type.value}: Retrieved {result.get('count', 0)} results")
            except Exception as e:
                print(f"  ✗ {system_type.value}: Error - {e}")
                all_results.append({
                    "source": system_type.value,
                    "error": str(e)
                })
        
        # Step 3: Synthesize answer
        print("\n💭 Step 3: Generating answer...")
        final_answer = self._synthesize_answer(user_query, all_results, routing)
        
        return {
            "query": user_query,
            "routing": routing,
            "system_results": all_results,
            "final_answer": final_answer
        }
    
    def _synthesize_answer(self, query: str, results: List[Dict], routing: Dict) -> str:
        """Synthesize final answer from all retrieved information"""
        
        # Prepare context from all sources
        context_parts = []
        
        for result in results:
            source = result.get("source", "Unknown")
            context_parts.append(f"\n--- {source} ---")
            
            if "error" in result:
                context_parts.append(f"Error: {result['error']}")
                continue
            
            if "results" in result and result["results"]:
                context_parts.append(f"Retrieved {len(result['results'])} records:")
                context_parts.append(json.dumps(result["results"][:5], indent=2, default=str))
            
            if "answer" in result and result["answer"]:
                context_parts.append(f"Answer: {result['answer']}")
            
            if "retrieved_docs" in result:
                context_parts.append(f"Documents: {len(result['retrieved_docs'])} retrieved")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert legal assistant. Synthesize a comprehensive answer based on information from multiple sources.

User Question: "{query}"

Query Type: {routing.get('query_type', 'unknown')}

Retrieved Information:
{context}

Instructions:
1. Provide a clear, accurate answer to the user's question
2. Integrate information from all sources coherently
3. Cite which source(s) the information comes from
4. Use proper legal terminology
5. If sources conflict, acknowledge it
6. If information is insufficient, say so

Answer:"""
        
        response = self.answerer.generate_content(prompt)
        return response.text.strip()
    
    def interactive_mode(self):
        """Run interactive Q&A session"""
        print("\n" + "="*80)
        print("🏛️  AGENTIC LEGAL ASSISTANT")
        print("="*80)
        print("\nMulti-System Legal Research Assistant")
        print("  • Knowledge Graph (Neo4j) - Structured legal data")
        print("  • IPC RAG - Indian Penal Code sections")
        print("  • Past Cases RAG - Supreme Court judgments") 
        print("  • Legal Acts RAG - Constitution & legal acts")
        print("\nType 'exit' to quit\n")
        
        while True:
            try:
                user_query = input("\n💬 Your question: ").strip()
                
                if not user_query:
                    continue
                
                if user_query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Process query
                result = self.process_query(user_query)
                
                # Display answer
                print(f"\n{'='*80}")
                print("✅ FINAL ANSWER:")
                print(f"{'='*80}")
                print(result["final_answer"])
                print(f"{'='*80}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def close(self):
        """Cleanup resources"""
        if SystemType.KNOWLEDGE_GRAPH in self.systems:
            self.systems[SystemType.KNOWLEDGE_GRAPH].close()


def main():
    """Main entry point"""
    try:
        orchestrator = AgenticOrchestrator()
        orchestrator.interactive_mode()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'orchestrator' in locals():
            orchestrator.close()


if __name__ == "__main__":
    main()
