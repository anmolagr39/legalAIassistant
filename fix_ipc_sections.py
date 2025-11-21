"""
Fix IPC sections in Neo4j by clearing and reloading with correct section numbers
"""
from config.neo4j_config import Neo4jConnection
from data_processing.preprocessor import FIRDatasetLoader
from kg_construction.kg_builder import KnowledgeGraphBuilder
from config.settings import FIR_DATASET
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_ipc_sections():
    """Clear and reload IPC sections with correct mappings"""
    
    # Connect to Neo4j
    conn = Neo4jConnection()
    conn.connect()
    
    # Step 1: Delete all IPC Section nodes
    logger.info("Step 1: Deleting existing IPC Section nodes...")
    delete_query = """
    MATCH (s:IPCSection)
    DETACH DELETE s
    """
    conn.execute_write(delete_query, {})
    logger.info("✓ Deleted all IPC Section nodes")
    
    # Step 2: Load correct data
    logger.info("\nStep 2: Loading FIR dataset with correct section numbers...")
    fir_loader = FIRDatasetLoader(FIR_DATASET)
    df = fir_loader.load()
    sections = fir_loader.preprocess(df)
    
    logger.info(f"Loaded {len(sections)} IPC sections")
    
    # Show first few sections to verify
    print("\n=== Sample sections to be loaded ===")
    for section in sections[:5]:
        print(f"Section {section['section_number']}: {section['offense'][:80]}...")
    
    # Step 3: Create IPC Section nodes with correct data
    logger.info("\nStep 3: Creating IPC Section nodes with correct data...")
    kg_builder = KnowledgeGraphBuilder(conn)
    
    created = 0
    errors = 0
    
    for section in tqdm(sections, desc="Creating IPC sections"):
        try:
            success = kg_builder.create_ipc_section(section)
            if success:
                created += 1
                
                # Create relationships
                if section.get('offense'):
                    kg_builder.create_ipc_offense_relationship(
                        section['section_number'],
                        section['offense'][:100]
                    )
                
                if section.get('punishment'):
                    kg_builder.create_ipc_punishment_relationship(
                        section['section_number'],
                        section['punishment'][:100]
                    )
        except Exception as e:
            logger.error(f"Error creating section {section.get('section_number')}: {e}")
            errors += 1
    
    logger.info(f"\n✓ Created {created} IPC sections")
    if errors > 0:
        logger.warning(f"⚠️ {errors} errors occurred")
    
    # Step 4: Verify the fix
    logger.info("\nStep 4: Verifying the fix...")
    verify_query = """
    MATCH (s:IPCSection) 
    WHERE s.section_number IN ["302", "303", "127", "140"]
    RETURN s.section_number as section, s.offense as offense
    ORDER BY s.section_number
    """
    results = conn.execute_query(verify_query)
    
    print("\n=== Verification - Sample sections ===")
    for r in results:
        print(f"Section {r['section']}: {r['offense'][:80]}...")
    
    conn.close()
    logger.info("\n✅ IPC sections fixed successfully!")

if __name__ == "__main__":
    print("="*80)
    print("FIX IPC SECTIONS IN NEO4J")
    print("="*80)
    print("\nThis will:")
    print("  1. Delete all existing IPC Section nodes")
    print("  2. Reload them with correct section numbers from FIR_DATASET.csv")
    print("  3. Recreate all relationships")
    print("\n⚠️  WARNING: This will delete and recreate all IPC Section nodes!")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() == 'yes':
        fix_ipc_sections()
    else:
        print("Aborted.")
