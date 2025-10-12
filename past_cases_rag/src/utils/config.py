"""
Configuration settings for the Legal Assistant RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Database Configuration
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")
COLLECTION_NAME = "supreme_court_cases"

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Text Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "5"))
SIMILARITY_THRESHOLD = 0.7

# Data Paths
CASE_DOCS_PATH = "./Object_casedocs"

# Legal Document Structure
LEGAL_SECTIONS = [
    "Facts", "Issues", "Held", "Reasoning", "Judgment", "Orders",
    "Petitioner", "Respondent", "Court", "Date", "Citation"
]

# Gemini Model Configuration
GEMINI_MODEL = "gemini-2.0-flash-exp"
GENERATION_CONFIG = {
    "temperature": 0.3,
    "max_output_tokens": 2048,
    "candidate_count": 1
}