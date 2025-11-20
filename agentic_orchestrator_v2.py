"""
Agentic Legal Assistant Orchestrator
Routes queries to appropriate systems: Knowledge Graph and/or RAG systems
"""
import os
import sys
import json
from typing import Dict, List, Optional
from enum import Enum
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths for all systems
root_dir = os.path.dirname(__file__)
ipcrag_path = os.path.join(root_dir, '3rags', 'ipcrag', 'src')
past_cases_path = os.path.join(root_dir, '3rags', 'past_cases_rag', 'src')
legal_acts_path = os.path.join(root_dir, '3rags', 'legal-rag-system', 'src')

# Don't add to path yet - will do lazily per system

# Import KG components from current directory
sys.path.insert(0, root_dir)
from config.neo4j_config import Neo4jConnection
from extraction.gemini_extractor import GeminiExtractor

# Get API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyA5aqm_rKXyuei5tLu26a1o4iOpMeUad_g')


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

CRITICAL: Respond with COMPLETE and VALID JSON ONLY. No markdown, no explanation, no truncation.
Ensure the JSON is properly closed with all braces and quotes.

Format (copy exactly):
{{
    "systems": ["system_name1"],
    "reasoning": "Brief reason",
    "query_type": "factual",
    "requires_multiple": false
}}

System names MUST be from: knowledge_graph, ipc_rag, past_cases_rag, legal_acts_rag

Examples:
- "What is IPC section XYZ?" → knowledge_graph + ipc_rag
- "Cases involving murder" → knowledge_graph + past_cases_rag
- "Article 21 right to life" → knowledge_graph + legal_acts_rag
- "Who is judge XYZ?" → knowledge_graph only
- "Explain bailable vs non-bailable" → ipc_rag only
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': 300,
                    'response_mime_type': 'application/json'
                }
            )
            
            # Check if response has text
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"⚠️ API returned no content (finish_reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'})")
                # Fallback: Use both KG and IPC RAG for IPC queries
                import re
                if re.search(r'\bipc\b|\bsection\b|\b\d{2,3}\b', query.lower()):
                    return {
                        "systems": [SystemType.KNOWLEDGE_GRAPH, SystemType.IPC_RAG],
                        "reasoning": "Fallback for IPC query due to API issue",
                        "query_type": "factual",
                        "requires_multiple": True
                    }
                else:
                    return {
                        "systems": [SystemType.KNOWLEDGE_GRAPH],
                        "reasoning": "Fallback due to API issue",
                        "query_type": "factual",
                        "requires_multiple": False
                    }
            
            response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Remove control characters that break JSON parsing
            import re
            # Remove control characters including newlines, tabs, etc.
            response_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', response_text)
            # Remove extra whitespace
            response_text = re.sub(r'\s+', ' ', response_text).strip()
            
            # Remove trailing commas before closing braces/brackets (common JSON error)
            # Handle comma before } or ]
            response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            
            # Fix incomplete JSON - add missing closing braces
            open_braces = response_text.count('{')
            close_braces = response_text.count('}')
            if open_braces > close_braces:
                # Add missing closing braces
                response_text += '\n' + '}' * (open_braces - close_braces)
            
            # Fix incomplete JSON - close incomplete strings
            # Count quotes to see if there's an unclosed string
            if response_text.count('"') % 2 != 0:
                response_text += '"'
                # Also add the missing closing brace if needed
                if response_text.count('{') > response_text.count('}'):
                    response_text += '\n}'
            
            result = json.loads(response_text)
            
            # Convert system names to enums
            systems = []
            for sys_name in result["systems"]:
                try:
                    systems.append(SystemType(sys_name))
                except ValueError:
                    print(f"⚠️ Unknown system: {sys_name}")
            
            result["systems"] = systems
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"Response text: {response_text}")
            # Fallback: use knowledge_graph only
            return {
                "systems": [SystemType.KNOWLEDGE_GRAPH],
                "reasoning": "Fallback due to parsing error",
                "query_type": "factual",
                "requires_multiple": False
            }


class KnowledgeGraphSystem:
    """Interface to Neo4j Knowledge Graph"""
    
    def __init__(self):
        self.conn = Neo4jConnection()
        self.conn.connect()
        self.extractor = GeminiExtractor(GEMINI_API_KEY)
        
    def query(self, user_query: str) -> Dict:
        """Query knowledge graph"""
        try:
            # Get all node names for better query generation
            cypher = self._generate_cypher(user_query)
            results = self.conn.execute_query(cypher)
            
            return {
                "source": "Knowledge Graph",
                "cypher": cypher,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "source": "Knowledge Graph",
                "error": str(e),
                "results": []
            }
    
    def _generate_cypher(self, query: str) -> str:
        """Generate Cypher query using Gemini"""
        # Get ALL node names from the database for accurate query generation
        all_cases_query = "MATCH (c:Case) RETURN c.name as name ORDER BY c.name"
        all_judges_query = "MATCH (j:Judge) RETURN j.name as name ORDER BY j.name"
        all_ipc_query = "MATCH (i:IPCSection) RETURN i.section as section, i.offense as offense ORDER BY i.section"
        all_articles_query = "MATCH (a:Article) RETURN a.number as number, a.title as title ORDER BY a.number"
        all_courts_query = "MATCH (c:Court) RETURN c.name as name"
        
        # Fetch all node names
        all_cases = self.conn.execute_query(all_cases_query)
        all_judges = self.conn.execute_query(all_judges_query)
        all_ipc_sections = self.conn.execute_query(all_ipc_query)
        all_articles = self.conn.execute_query(all_articles_query)
        all_courts = self.conn.execute_query(all_courts_query)
        
        # Format node lists for the prompt
        case_names = [c['name'] for c in all_cases[:100]]  # Limit to first 100 for context
        judge_names = [j['name'] for j in all_judges]
        ipc_sections = [f"{i['section']}: {i['offense']}" for i in all_ipc_sections[:50]]
        article_info = [f"Article {a['number']}: {a['title']}" for a in all_articles]
        court_names = [c['name'] for c in all_courts]
        
        prompt = f"""Convert this natural language query to Cypher for Neo4j.

Database Schema:
- Case nodes: name, citation, year
- Judge nodes: name
- IPCSection nodes: section, offense, punishment
- Article nodes: number, title
- Court nodes: name

EXACT NODE NAMES IN DATABASE:

Cases (showing first 100 of {len(all_cases)}):
{json.dumps(case_names, indent=2)}

Judges (all {len(all_judges)}):
{json.dumps(judge_names, indent=2)}

IPC Sections (showing first 50 of {len(all_ipc_sections)}):
{json.dumps(ipc_sections, indent=2)}

Articles (all {len(all_articles)}):
{json.dumps(article_info, indent=2)}

Courts (all {len(all_courts)}):
{json.dumps(court_names, indent=2)}

Relationships:
- (Case)-[:DECIDED_BY]->(Judge)
- (Case)-[:GOVERNED_BY]->(IPCSection)
- (Case)-[:CITES]->(Case)
- (Case)-[:HEARD_IN]->(Court)

IMPORTANT: Use the EXACT node names listed above in your Cypher query. For IPC sections, use the section number (e.g., "303", "302") from the list above.

User Query: {query}

Return ONLY the Cypher query, no explanation:"""

        # Use Gemini directly for Cypher generation
        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        cypher = response.text.strip()
        
        # Clean up markdown formatting
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
        # Import here to avoid path issues
        # Use importlib to load modules with absolute paths
        import importlib.util
        
        # Load config module from ipcrag path
        config_file = os.path.join(ipcrag_path, 'config.py')
        spec = importlib.util.spec_from_file_location("ipcrag_config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        Config = config_module.Config
        
        # Temporarily inject ipcrag's config into sys.modules so rag_engine can find it
        original_config = sys.modules.get('config')
        sys.modules['config'] = config_module
        
        try:
            # Load rag_engine module from ipcrag path
            rag_engine_file = os.path.join(ipcrag_path, 'rag_engine.py')
            spec = importlib.util.spec_from_file_location("ipcrag_rag_engine", rag_engine_file)
            rag_engine_module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, ipcrag_path)  # Needed for rag_engine's other imports
            spec.loader.exec_module(rag_engine_module)
            sys.path.pop(0)  # Remove ipcrag_path immediately after loading
            
            FIRRagEngine = rag_engine_module.FIRRagEngine
        finally:
            # Restore original config module
            if original_config:
                sys.modules['config'] = original_config
            else:
                sys.modules.pop('config', None)
        
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
            print("⚠️ IPC RAG collection empty. Checking for data...")
            # Try to find and ingest data
            dataset_path = os.path.join(os.path.dirname(__file__), '3rags', 'ipcrag', 'FIR_DATASET.csv')
            if os.path.exists(dataset_path):
                print(f"📥 Found dataset at {dataset_path}, ingesting...")
                self.engine.ingest_fir_data(dataset_path)
                print("✅ Data ingestion complete")
            else:
                print(f"❌ No dataset found at {dataset_path}")
        
    def query(self, user_query: str) -> Dict:
        """Query IPC RAG system"""
        try:
            # Enhance query for better IPC section matching
            import re
            enhanced_query = user_query
            # Extract section number if present
            section_match = re.search(r'\b(\d{2,3}[A-Z]?)\b', user_query)
            if section_match:
                section_num = section_match.group(1)
                enhanced_query = f"IPC Section {section_num} Indian Penal Code offense punishment description {user_query}"
            
            result = self.engine.query(enhanced_query)
            return {
                "source": "IPC RAG",
                "answer": result.get("answer", ""),
                "retrieved_count": result.get("retrieved_count", 0),
                "sources": result.get("sources", []),
                "count": result.get("retrieved_count", 0)
            }
        except Exception as e:
            print(f"❌ IPC RAG error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "source": "IPC RAG",
                "error": str(e),
                "answer": "",
                "count": 0
            }


class PastCasesRagSystem:
    """Interface to Past Cases RAG system"""
    
    def __init__(self):
        # Import here
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if not p.endswith('legalkg')]
        sys.path.insert(0, past_cases_path)
        
        try:
            from rag.vector_db import ChromaVectorDB
            from rag.rag_system import LegalRAGSystem
            
            # Initialize with proper paths
            chroma_path = os.path.join(os.path.dirname(__file__), '3rags', 'past_cases_rag', 'chroma_db')
            self.vector_db = ChromaVectorDB(persist_directory=chroma_path)
            self.rag = LegalRAGSystem(self.vector_db)
            
            # Check if data exists
            try:
                count = self.vector_db.collection.count()
                print(f"✅ Past Cases RAG initialized with {count} documents")
            except:
                print("⚠️ Past Cases RAG collection empty. Run ingestion first.")
        finally:
            sys.path = original_path
        
    def query(self, user_query: str) -> Dict:
        """Query Past Cases RAG system"""
        try:
            result = self.rag.query(user_query)
            return {
                "source": "Past Cases RAG",
                "answer": result.get("answer", result.get("response", "")),
                "retrieved_docs": result.get("contexts", result.get("sources", [])),
                "count": len(result.get("contexts", result.get("sources", [])))
            }
        except Exception as e:
            print(f"❌ Past Cases RAG error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "source": "Past Cases RAG",
                "error": str(e),
                "answer": "",
                "count": 0
            }


class LegalActsRagSystem:
    """Interface to Legal Acts RAG system"""
    
    def __init__(self):
        # Import here
        original_path = sys.path.copy()
        sys.path = [p for p in sys.path if not p.endswith('legalkg')]
        sys.path.insert(0, legal_acts_path)
        
        try:
            from vector_store import VectorStore
            from rag_system import LegalRAG
            
            persist_dir = os.path.join(os.path.dirname(__file__), '3rags', 'legal-rag-system', 'chroma_db')
            self.vector_store = VectorStore(persist_dir)
            self.rag = LegalRAG(
                self.vector_store,
                "constitution_articles",
                "legal_acts_chunks",
                api_key=GEMINI_API_KEY
            )
            
            # Check if collections exist
            try:
                collections = self.vector_store.client.list_collections()
                print(f"✅ Legal Acts RAG initialized with collections: {[c.name for c in collections]}")
            except Exception as e:
                print(f"⚠️ Legal Acts RAG: {e}")
        finally:
            sys.path = original_path
        
    def query(self, user_query: str) -> Dict:
        """Query Legal Acts RAG system"""
        try:
            result = self.rag.query(user_query)
            return {
                "source": "Legal Acts RAG",
                "answer": result.get("answer", ""),
                "retrieved_docs": result.get("sources", []),
                "count": len(result.get("sources", []))
            }
        except Exception as e:
            print(f"❌ Legal Acts RAG error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "source": "Legal Acts RAG",
                "error": str(e),
                "answer": "",
                "count": 0
            }


class AgenticOrchestrator:
    """Main orchestrator that coordinates all systems"""
    
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        print("🚀 Initializing Agentic Legal Assistant...")
        
        self.router = QueryRouter(GEMINI_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
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
        2. Query selected systems
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
                import traceback
                traceback.print_exc()
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
                context_parts.append(f"Retrieved {len(result['results'])} records")
                # Limit context size
                context_parts.append(json.dumps(result["results"][:3], indent=2, default=str)[:1000])
            
            if "answer" in result and result["answer"]:
                context_parts.append(f"Answer: {result['answer'][:800]}")
            
            if "retrieved_docs" in result:
                context_parts.append(f"Documents: {len(result['retrieved_docs'])} retrieved")
        
        context = "\n".join(context_parts)[:3000]  # Limit total context
        
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
7. Be concise but comprehensive

Answer:"""
        
        try:
            response = self.answerer.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 800
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
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
                
                # Display retrieved information from each system
                print(f"\n{'='*80}")
                print("📚 RETRIEVED INFORMATION:")
                print(f"{'='*80}")
                
                for sys_result in result.get("system_results", []):
                    source = sys_result.get("source", "Unknown")
                    print(f"\n🔹 {source}:")
                    
                    if "error" in sys_result:
                        print(f"   ❌ Error: {sys_result['error']}")
                        continue
                    
                    # Show count
                    count = sys_result.get("count", 0)
                    print(f"   Retrieved: {count} results")
                    
                    # Show KG results
                    if "results" in sys_result and sys_result["results"]:
                        print(f"   📊 Knowledge Graph Results:")
                        for i, record in enumerate(sys_result["results"][:3], 1):
                            print(f"      {i}. {json.dumps(record, default=str)[:200]}...")
                    
                    # Show RAG answer/chunks
                    if "answer" in sys_result and sys_result["answer"]:
                        print(f"   📝 RAG Response:")
                        answer_preview = sys_result["answer"][:300]
                        print(f"      {answer_preview}...")
                    
                    # Show retrieved documents
                    if "retrieved_docs" in sys_result and sys_result["retrieved_docs"]:
                        print(f"   📄 Retrieved Documents/Chunks:")
                        for i, doc in enumerate(sys_result["retrieved_docs"][:2], 1):
                            if isinstance(doc, dict):
                                content = doc.get("content", doc.get("content_preview", str(doc)[:150]))
                                print(f"      {i}. {content[:150]}...")
                            else:
                                print(f"      {i}. {str(doc)[:150]}...")
                
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
