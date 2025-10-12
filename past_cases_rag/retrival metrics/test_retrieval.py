#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag.vector_db import ChromaVectorDB
from src.rag.rag_system import LegalRAGSystem

def test_retrieval():
    """Test the retrieval system with a simple query."""
    
    print("Testing retrieval system...")
    
    # Initialize components
    try:
        # Use correct path to chroma_db
        db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
        vector_db = ChromaVectorDB(persist_directory=db_path)
        print("✓ Vector DB initialized")
        
        rag_system = LegalRAGSystem(vector_db)
        print("✓ RAG System initialized")
        
        # Get collection stats
        stats = vector_db.get_collection_stats()
        print(f"✓ Collection has {stats['total_documents']} documents")
        
        # Test direct vector search
        print("\n--- Testing direct vector search ---")
        direct_results = vector_db.search("fundamental rights", n_results=3)
        print(f"Direct search returned {len(direct_results)} results")
        if direct_results:
            print(f"First result preview: {direct_results[0]['document'][:100]}...")
        
        # Test RAG system retrieval
        print("\n--- Testing RAG system retrieval ---")
        rag_results = rag_system.retrieve_relevant_documents("fundamental rights", num_docs=3)
        print(f"RAG search returned: {rag_results}")
        print(f"RAG search type: {type(rag_results)}")
        print(f"RAG search length: {len(rag_results) if rag_results else 'None/0'}")
        
        if rag_results:
            print(f"First RAG result keys: {list(rag_results[0].keys())}")
            print(f"First RAG result preview: {rag_results[0]['document'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()