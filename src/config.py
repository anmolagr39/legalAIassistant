import os
from typing import Optional

class Config:
    """Configuration class for FIR RAG system"""
    
    # API Configuration
    GEMINI_API_KEY: Optional[str] = None
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "fir_cases"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text Processing Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    
    # File paths
    FIR_DATASET_PATH: str = "./data/FIR_dataset.csv"
    
    @classmethod
    def set_gemini_api_key(cls, api_key: str):
        """Set the Gemini API key"""
        cls.GEMINI_API_KEY = api_key
        os.environ["GEMINI_API_KEY"] = api_key
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("Gemini API key is required. Use Config.set_gemini_api_key()")
        return True
    
    @classmethod
    def get_absolute_path(cls, relative_path: str) -> str:
        """Convert relative path to absolute path"""
        # Get the directory where this config file is located
        config_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root
        project_root = os.path.dirname(config_dir)
        # Join with the relative path, removing any leading './'
        clean_path = relative_path.lstrip('./')
        return os.path.join(project_root, clean_path)