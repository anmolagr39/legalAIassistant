# Configuration for Legal RAG System

# ChromaDB Settings
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
CONSTITUTION_COLLECTION_NAME = "constitution_articles"
LEGAL_ACTS_COLLECTION_NAME = "legal_acts"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini API (Add your API key to .env file)
GEMINI_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 8192

# Chunking Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Parameters
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7

# Data Paths
CONSTITUTION_PDF_PATH = "../the_constitution_of_india (1).pdf"
LEGAL_ACTS_CSV_PATH = "../legal-acts.csv"