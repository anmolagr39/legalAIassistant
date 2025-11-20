"""
Main pipeline for building Legal Knowledge Graph
"""
import os
import logging
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from config.settings import (
    GEMINI_API_KEY, MAX_CASES, BATCH_SIZE,
    FIR_DATASET, CASE_DOCS_DIR, CONSTITUTION_PDF
)
from config.neo4j_config import Neo4jConnection
from data_processing.preprocessor import (
    FIRDatasetLoader, CaseDocumentLoader, LegalTextPreprocessor
)
from data_processing.pdf_processor import ConstitutionProcessor
from extraction.gemini_extractor import GeminiExtractor
from extraction.ollama_extractor import OllamaExtractor
from extraction.openai_extractor import OpenAIExtractor
from extraction.groq_extractor import GroqExtractor
from kg_construction.kg_builder import KnowledgeGraphBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kg_construction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LegalKGPipeline:
    """Main pipeline for Legal Knowledge Graph construction"""
    
    def __init__(self, use_local_llm: bool = False):
        logger.info("=" * 80)
        logger.info("Initializing Legal Knowledge Graph Pipeline")
        logger.info("=" * 80)
        
        # Initialize components
        self.neo4j_conn = None
        self.kg_builder = None
        self.gemini_extractor = None
        self.use_local_llm = use_local_llm
        
        # Statistics
        self.stats = {
            'ipc_sections_processed': 0,
            'cases_processed': 0,
            'articles_processed': 0,
            'total_nodes_created': 0,
            'total_relationships_created': 0,
            'errors': 0
        }
    
    def initialize(self):
        """Initialize all components"""
        try:
            # Connect to Neo4j
            logger.info("Connecting to Neo4j...")
            self.neo4j_conn = Neo4jConnection()
            self.neo4j_conn.connect()
            
            # Create indexes and constraints
            logger.info("Creating indexes and constraints...")
            self.neo4j_conn.create_indexes()
            self.neo4j_conn.create_constraints()
            
            # Initialize KG builder
            self.kg_builder = KnowledgeGraphBuilder(self.neo4j_conn)
            
            # Initialize LLM extractor
            if self.use_local_llm:
                logger.info("Initializing Local LLM (Ollama)...")
                self.gemini_extractor = OllamaExtractor()
            else:
                logger.info("Initializing Groq API (Llama 3.1)...")
                groq_key = os.getenv('GROQ_API_KEY')
                groq_key_2 = os.getenv('GROQ_API_KEY_2')
                if not groq_key:
                    raise ValueError("GROQ_API_KEY not found in environment")
                self.gemini_extractor = GroqExtractor(api_key=groq_key, api_key_2=groq_key_2)
            
            logger.info("✓ All components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"✗ Initialization failed: {e}")
            return False
    
    def process_fir_dataset(self):
        """Process FIR dataset (IPC Sections)"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Processing FIR Dataset (IPC Sections)")
        logger.info("=" * 80)
        
        try:
            # Load and preprocess FIR dataset
            fir_loader = FIRDatasetLoader(FIR_DATASET)
            df = fir_loader.load()
            sections = fir_loader.preprocess(df)
            
            logger.info(f"Processing {len(sections)} IPC sections...")
            
            # Process each section
            for section in tqdm(sections, desc="IPC Sections"):
                try:
                    # Create IPC Section node
                    success = self.kg_builder.create_ipc_section(section)
                    
                    if success:
                        self.stats['ipc_sections_processed'] += 1
                        
                        # Create offense relationship if offense exists
                        if section.get('offense'):
                            self.kg_builder.create_ipc_offense_relationship(
                                section['section_number'],
                                section['offense'][:100]  # Limit length
                            )
                        
                        # Create punishment relationship if punishment exists
                        if section.get('punishment'):
                            self.kg_builder.create_ipc_punishment_relationship(
                                section['section_number'],
                                section['punishment'][:100]  # Limit length
                            )
                
                except Exception as e:
                    logger.error(f"Error processing section {section.get('section_number')}: {e}")
                    self.stats['errors'] += 1
            
            logger.info(f"✓ Processed {self.stats['ipc_sections_processed']} IPC sections")
        
        except Exception as e:
            logger.error(f"✗ FIR dataset processing failed: {e}")
    
    def process_constitution(self):
        """Process Constitution of India"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Processing Constitution of India")
        logger.info("=" * 80)
        
        try:
            if not CONSTITUTION_PDF.exists():
                logger.warning(f"Constitution PDF not found at {CONSTITUTION_PDF}")
                return
            
            # Extract text from Constitution
            processor = ConstitutionProcessor(CONSTITUTION_PDF)
            full_text = processor.extract_text()
            
            logger.info(f"Extracted {len(full_text)} characters from Constitution PDF")
            
            # Step 1: Use LLM to identify all article numbers in chunks
            chunk_size = 6000  # Reduced to avoid token limits
            overlap = 500
            all_article_numbers = set()
            
            total_chunks = (len(full_text) + chunk_size - overlap - 1) // (chunk_size - overlap)
            logger.info(f"Step 1: Identifying all article numbers using LLM...")
            logger.info(f"Total chunks to process: {total_chunks}")
            
            chunk_num = 0
            for i in range(0, len(full_text), chunk_size - overlap):
                chunk = full_text[i:i + chunk_size]
                chunk_num += 1
                
                # Ask LLM to extract article numbers from this chunk
                prompt = f"""Find PRIMARY article numbers in this Constitution text.
Extract numbers from "Article X" patterns only (not clauses/sub-sections).
Return unique numbers as JSON array.

Text:
{chunk}

Format: {{"articles": ["1", "14"]}}"""
                
                try:
                    response = self.gemini_extractor._call_groq(prompt)
                    if response:
                        result = self.gemini_extractor._parse_json_response(response)
                        if result and 'articles' in result:
                            chunk_articles = result['articles']
                            for num in chunk_articles:
                                all_article_numbers.add(str(num))
                            logger.info(f"✓ Chunk {chunk_num}/{total_chunks}: Found {len(chunk_articles)} articles (Total unique: {len(all_article_numbers)})")
                        else:
                            logger.info(f"✓ Chunk {chunk_num}/{total_chunks}: No articles found")
                    time.sleep(1.5)  # Increased delay to avoid rate limits
                except Exception as e:
                    logger.warning(f"✗ Chunk {chunk_num}/{total_chunks}: Error - {e}")
            
            # Sort article numbers
            sorted_articles = sorted(all_article_numbers, key=lambda x: int(re.sub(r'[A-Z]', '', x) or 0))
            logger.info(f"Found {len(sorted_articles)} unique article numbers: {sorted_articles[:20]}...")
            
            # Step 2: Process each article number
            logger.info(f"\nStep 2: Extracting detailed information for each article...")
            logger.info(f"Total articles to extract: {len(sorted_articles)}")
            
            for idx, article_num in enumerate(tqdm(sorted_articles, desc="Constitutional Articles"), 1):
                try:
                    # Find article text in full_text
                    # Search for this article in context
                    search_pattern = rf'(?i)article\s+{re.escape(article_num)}[\.:\s—\-–]'
                    match = re.search(search_pattern, full_text)
                    
                    if match:
                        # Extract surrounding context (5000 chars)
                        start = max(0, match.start() - 500)
                        end = min(len(full_text), match.start() + 5000)
                        article_context = full_text[start:end]
                        
                        # Use LLM to extract structured information
                        article_info = self.gemini_extractor.extract_article(article_context)
                        
                        if article_info:
                            article_info['article_number'] = article_num
                            
                            # Create Article node
                            success = self.kg_builder.create_article(article_info)
                            
                            if success:
                                self.stats['articles_processed'] += 1
                                
                                # Log progress every 20 articles
                                if idx % 20 == 0:
                                    progress_pct = (idx / len(sorted_articles)) * 100
                                    logger.info(f"Progress: {idx}/{len(sorted_articles)} articles ({progress_pct:.1f}% complete)")
                        
                        time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing article {article_num}: {e}")
                    self.stats['errors'] += 1
            
            logger.info(f"✓ Processed {self.stats['articles_processed']} constitutional articles")
        
        except Exception as e:
            logger.error(f"✗ Constitution processing failed: {e}")
    
    def process_case_documents(self, max_cases: int = MAX_CASES):
        """Process case documents"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 3: Processing Case Documents (max {max_cases} cases)")
        logger.info("=" * 80)
        
        try:
            # Load case documents
            case_loader = CaseDocumentLoader(CASE_DOCS_DIR, max_cases=max_cases)
            cases = case_loader.load()
            processed_cases = case_loader.preprocess(cases)
            
            logger.info(f"Processing {len(processed_cases)} case documents...")
            
            # Process each case
            for case in tqdm(processed_cases, desc="Cases"):
                try:
                    # Extract entities from first chunk (main content)
                    main_chunk = case['chunks'][0] if case['chunks'] else case['content'][:10000]
                    
                    # Get case summary from Gemini
                    case_summary = self.gemini_extractor.summarize_case(main_chunk)
                    
                    if not case_summary:
                        logger.warning(f"Could not summarize case {case['case_id']}")
                        continue
                    
                    # Create Case node
                    case_node_data = {
                        'case_id': case['case_id'],
                        'case_name': case_summary.get('case_name', case['case_name'])[:200],
                        'case_number': case_summary.get('case_number', case['case_number']) or '',
                        'date': case_summary.get('date', case['date']) or '',
                        'content': case['content'][:5000]  # Store first 5000 chars
                    }
                    
                    success = self.kg_builder.create_case(case_node_data)
                    
                    if success:
                        self.stats['cases_processed'] += 1
                        
                        # Create relationships
                        
                        # Court
                        court = case_summary.get('court') or case.get('court')
                        if court:
                            self.kg_builder.create_case_court_relationship(
                                case['case_id'], court
                            )
                        
                        # Judges
                        judges = case_summary.get('judges', [])
                        for judge in judges[:5]:  # Limit to 5 judges
                            if judge:
                                self.kg_builder.create_case_judge_relationship(
                                    case['case_id'], judge[:100]
                                )
                        
                        # IPC Sections mentioned
                        sections = case_summary.get('sections_mentioned', [])
                        for section in sections[:10]:  # Limit to 10 sections
                            if section:
                                self.kg_builder.create_case_ipc_relationship(
                                    case['case_id'], section
                                )
                        
                        # Articles mentioned
                        articles = case_summary.get('articles_mentioned', [])
                        for article in articles[:10]:  # Limit to 10 articles
                            if article:
                                self.kg_builder.create_case_article_relationship(
                                    case['case_id'], article
                                )
                        
                        # Extract entities and relations for richer connections
                        entities = self.gemini_extractor.extract_entities(main_chunk)
                        
                        if entities:
                            # Process citations
                            citations = entities.get('citations', [])
                            for cited_case in citations[:5]:  # Limit citations
                                if cited_case and len(cited_case) > 5:
                                    self.kg_builder.create_case_citation_relationship(
                                        case['case_id'], cited_case[:200]
                                    )
                    
                    # Rate limiting (only for Gemini API)
                    if not self.use_local_llm:
                        time.sleep(1.5)
                    else:
                        time.sleep(0.1)  # Minimal delay for local LLM
                
                except Exception as e:
                    logger.error(f"Error processing case {case.get('case_id')}: {e}")
                    self.stats['errors'] += 1
                    time.sleep(2)  # Extra delay after error
            
            logger.info(f"✓ Processed {self.stats['cases_processed']} cases")
        
        except Exception as e:
            logger.error(f"✗ Case processing failed: {e}")
    
    def generate_final_report(self):
        """Generate final statistics report"""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL REPORT")
        logger.info("=" * 80)
        
        # Get Neo4j statistics
        db_stats = self.neo4j_conn.get_stats()
        
        logger.info(f"\nProcessing Statistics:")
        logger.info(f"  IPC Sections Processed: {self.stats['ipc_sections_processed']}")
        logger.info(f"  Cases Processed: {self.stats['cases_processed']}")
        logger.info(f"  Articles Processed: {self.stats['articles_processed']}")
        logger.info(f"  Errors Encountered: {self.stats['errors']}")
        
        logger.info(f"\nKnowledge Graph Statistics:")
        logger.info(f"  Total Nodes: {db_stats['nodes']}")
        logger.info(f"  Total Relationships: {db_stats['relationships']}")
        
        logger.info(f"\nNode Types Distribution:")
        for node_type, count in db_stats.get('node_types', []):
            logger.info(f"    {node_type}: {count}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Knowledge Graph Construction Complete!")
        logger.info("=" * 80)
    
    def run(self, skip_constitution: bool = False, max_cases: int = MAX_CASES):
        """Run the complete pipeline"""
        start_time = time.time()
        
        try:
            # Initialize
            if not self.initialize():
                logger.error("Pipeline initialization failed")
                return False
            
            # Step 1: Process IPC Sections
            self.process_fir_dataset()
            
            # Step 2: Process Constitution (optional)
            if not skip_constitution:
                self.process_constitution()
            else:
                logger.info("\nSkipping Constitution processing (as requested)")
            
            # Step 3: Process Case Documents
            self.process_case_documents(max_cases=max_cases)
            
            # Generate final report
            self.generate_final_report()
            
            # Calculate time taken
            elapsed_time = time.time() - start_time
            logger.info(f"\nTotal time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            return True
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.neo4j_conn:
                self.neo4j_conn.close()
    
    def cleanup_database(self):
        """Clear the database (use with caution!)"""
        logger.warning("⚠ Clearing database...")
        if self.neo4j_conn:
            self.neo4j_conn.clear_database()
            logger.info("✓ Database cleared")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Legal Knowledge Graph')
    parser.add_argument('--max-cases', type=int, default=MAX_CASES,
                       help=f'Maximum number of cases to process (default: {MAX_CASES})')
    parser.add_argument('--skip-constitution', action='store_true',
                       help='Skip constitution processing')
    parser.add_argument('--clear-db', action='store_true',
                       help='Clear database before starting')
    parser.add_argument('--local-llm', action='store_true',
                       help='Use local LLM (Ollama) instead of Gemini API')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = LegalKGPipeline(use_local_llm=args.local_llm)
    
    # Clear database if requested
    if args.clear_db:
        pipeline.neo4j_conn = Neo4jConnection()
        pipeline.neo4j_conn.connect()
        response = input("⚠ Are you sure you want to clear the database? (yes/no): ")
        if response.lower() == 'yes':
            pipeline.cleanup_database()
        pipeline.neo4j_conn.close()
        pipeline.neo4j_conn = None
    
    # Run pipeline
    success = pipeline.run(
        skip_constitution=args.skip_constitution,
        max_cases=args.max_cases
    )
    
    if success:
        print("\n✓ Knowledge Graph construction completed successfully!")
        print("\nNext steps:")
        print("  1. Open Neo4j Browser at http://localhost:7474")
        print("  2. Run query: MATCH (n) RETURN n LIMIT 100")
        print("  3. Explore the knowledge graph!")
    else:
        print("\n✗ Knowledge Graph construction failed. Check logs for details.")


if __name__ == "__main__":
    main()
