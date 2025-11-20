"""
Configuration settings for Legal Knowledge Graph
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
CASE_DOCS_DIR = DATA_DIR / "Object_casedocs"
FIR_DATASET = DATA_DIR / "FIR_DATASET.csv"
LEGAL_ACTS_CSV = DATA_DIR / "legal_acts_chunks.csv"
CONSTITUTION_PDF = DATA_DIR / "the_constitution_of_india.pdf"

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # Using Gemini 2.0 Flash
GEMINI_TEMPERATURE = 0.1  # Low temperature for consistent extraction
GEMINI_MAX_TOKENS = 8192

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Processing Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_CASES = int(os.getenv("MAX_CASES", "300"))
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"

# Embeddings Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Schema Configuration
NODE_TYPES = [
    "IPCSection",
    "LegalAct", 
    "Case",
    "Court",
    "Judge",
    "LegalPrinciple",
    "Offense",
    "Punishment",
    "Party",
    "Citation",
    "LegalConcept",
    "Article",
    "ConstitutionalProvision"
]

RELATIONSHIP_TYPES = [
    "CITES",
    "GOVERNED_BY",
    "APPLIES_TO",
    "PRESCRIBES",
    "HEARD_IN",
    "DECIDED_BY",
    "INVOLVES_PARTY",
    "ESTABLISHES",
    "OVERRULES",
    "FOLLOWS",
    "REFERS_TO",
    "RELATED_TO",
    "INTERPRETS",
    "AMENDS"
]

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY not set in .env file")
    
    if not NEO4J_PASSWORD:
        errors.append("NEO4J_PASSWORD not set in .env file")
    
    if not FIR_DATASET.exists():
        errors.append(f"FIR_DATASET not found at {FIR_DATASET}")
    
    if not CASE_DOCS_DIR.exists():
        errors.append(f"CASE_DOCS_DIR not found at {CASE_DOCS_DIR}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

if __name__ == "__main__":
    validate_config()
    print("✓ Configuration validated successfully!")
