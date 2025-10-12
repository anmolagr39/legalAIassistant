#!/usr/bin/env python3
"""
Quick setup script to load processed data into vector store for SUPERT evaluation
"""

import sys
import os
import pandas as pd

# Add src and config to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

try:
    from data_processor import DataProcessor
    from vector_store import VectorStore
    from rag_system import LegalRAG
except ImportError as e:
    print(f"Error importing RAG components: {e}")
    sys.exit(1)

def setup_vector_store():
    """Load processed data into vector store if not already done."""
    print("🔄 Setting up vector store for SUPERT evaluation...")
    
    # Initialize components
    data_processor = DataProcessor()
    vector_store = VectorStore(persist_directory="./data/chroma_db")
    
    # Check if collections exist
    collections = vector_store.list_collections()
    print(f"Existing collections: {collections}")
    
    if "constitution_chunks" not in collections:
        print("📚 Loading constitution chunks...")
        # Load processed constitution data
        const_df = pd.read_csv("./data/processed/constitution_chunks.csv")
        const_chunks = []
        for _, row in const_df.iterrows():
            const_chunks.append({
                'content': row['content'],
                'metadata': {
                    'source': 'constitution',
                    'article': row.get('article', 'Unknown'),
                    'chunk_id': len(const_chunks)
                }
            })
        
        # Add to vector store
        vector_store.add_documents("constitution_chunks", const_chunks)
        print(f"✅ Added {len(const_chunks)} constitution chunks")
    
    if "legal_acts_chunks" not in collections:
        print("⚖️ Loading legal acts chunks...")
        # Load processed legal acts data
        acts_df = pd.read_csv("./data/processed/legal_acts_chunks.csv")
        acts_chunks = []
        for _, row in acts_df.iterrows():
            acts_chunks.append({
                'content': row['content'],
                'metadata': {
                    'source': 'legal_acts',
                    'act_name': row.get('act_name', 'Unknown Act'),
                    'chunk_id': len(acts_chunks)
                }
            })
        
        # Add to vector store (only first 100 for testing)
        vector_store.add_documents("legal_acts_chunks", acts_chunks[:100])
        print(f"✅ Added {min(len(acts_chunks), 100)} legal acts chunks")
    
    print("✅ Vector store setup complete!")
    return vector_store

if __name__ == "__main__":
    setup_vector_store()