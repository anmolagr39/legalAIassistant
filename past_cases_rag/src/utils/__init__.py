"""
Utility functions for the Legal Assistant RAG system.
"""

from .config import *
from .logger import setup_logging, get_logger

__all__ = [
    'GOOGLE_API_KEY', 'CHROMADB_PATH', 'COLLECTION_NAME', 'EMBEDDING_MODEL',
    'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MAX_RETRIEVAL_DOCS', 'SIMILARITY_THRESHOLD',
    'CASE_DOCS_PATH', 'LEGAL_SECTIONS', 'GEMINI_MODEL', 'GENERATION_CONFIG',
    'setup_logging', 'get_logger'
]