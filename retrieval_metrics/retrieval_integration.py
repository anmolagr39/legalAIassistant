"""
Integration script for evaluating retrieval performance of the FIR RAG system.

This script connects the RetrievalEvaluator with the FIR RAG engine
to evaluate how well the system retrieves relevant documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.config import Config
    from src.rag_engine import FIRRagEngine
    SRC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import src modules: {e}")
    print("   This is OK for testing - we'll use mock implementations")
    SRC_AVAILABLE = False
    
    # Create mock classes for testing
    class Config:
        CHROMA_DB_PATH = "./chroma_db"
        COLLECTION_NAME = "fir_documents"
    
    class FIRRagEngine:
        def __init__(self, config):
            self.config = config
        def get_retriever(self):
            return None
from retrieval_metrics.retrieval_evaluator import RetrievalEvaluator, RetrievalTestCase, create_retrieval_test_cases
import pandas as pd
from typing import List, Dict
import numpy as np

# Import IPC queries from create_ipc_queries
try:
    from create_ipc_queries import create_ipc_specific_queries_and_test_cases, get_expected_ipc_sections_for_queries
except ImportError:
    print("⚠️ Could not import IPC queries. Using predefined queries.")


def create_ipc_specific_test_cases() -> List[RetrievalTestCase]:
    """
    Create 10 representative test cases specific to IPC sections in the FIR dataset.
    These are based on the most common legal queries related to IPC sections.
    """
    test_cases = [
        # Specific IPC Section queries (3 test cases)
        RetrievalTestCase(
            query="What is the punishment for IPC Section 140?",
            relevant_document_ids=set(),  # Will be filled by semantic matching
            query_category="specific_section",
            expected_ipc_sections=["140"],
            description="Query about IPC Section 140 military uniform"
        ),
        RetrievalTestCase(
            query="What does IPC Section 127 say about receiving stolen property?",
            relevant_document_ids=set(),
            query_category="specific_section",
            expected_ipc_sections=["127", "125", "126"],
            description="Query about IPC Section 127 receiving stolen property"
        ),
        RetrievalTestCase(
            query="Explain IPC Section 128 about prisoner escape by public servant",
            relevant_document_ids=set(),
            query_category="specific_section",
            expected_ipc_sections=["128"],
            description="Query about IPC Section 128 prisoner escape"
        ),
        
        # Category-based queries (3 test cases)
        RetrievalTestCase(
            query="What are the IPC sections for theft cases?",
            relevant_document_ids=set(),
            query_category="category_based",
            expected_ipc_sections=["378", "379", "380", "381", "382"],
            description="Query about theft-related IPC sections"
        ),
        RetrievalTestCase(
            query="Which IPC sections deal with fraud and cheating?",
            relevant_document_ids=set(),
            query_category="category_based",
            expected_ipc_sections=["415", "416", "417", "418", "419", "420"],
            description="Query about fraud-related IPC sections"
        ),
        RetrievalTestCase(
            query="What IPC sections cover assault and violence?",
            relevant_document_ids=set(),
            query_category="category_based",
            expected_ipc_sections=["319", "320", "321", "322", "351", "352"],
            description="Query about assault-related IPC sections"
        ),
        
        # Legal procedure queries (2 test cases)
        RetrievalTestCase(
            query="Which IPC offenses are cognizable?",
            relevant_document_ids=set(),
            query_category="legal_procedure",
            description="Query about cognizable IPC offenses"
        ),
        RetrievalTestCase(
            query="What IPC sections have non-bailable offenses?",
            relevant_document_ids=set(),
            query_category="legal_procedure",
            description="Query about non-bailable IPC offenses"
        ),
        
        # Punishment inquiry query (1 test case)
        RetrievalTestCase(
            query="What is the punishment for wearing military uniform illegally?",
            relevant_document_ids=set(),
            query_category="punishment_inquiry",
            expected_ipc_sections=["140"],
            description="Query about illegal military uniform punishment"
        ),
        
        # Procedural query (1 test case)
        RetrievalTestCase(
            query="How to file FIR for IPC Section 140 violation?",
            relevant_document_ids=set(),
            query_category="procedural",
            expected_ipc_sections=["140"],
            description="Procedural query about IPC 140 FIR filing"
        )
    ]
    
    return test_cases


def get_documents_from_chroma(rag_engine: FIRRagEngine) -> tuple[List[str], List[str]]:
    """
    Extract all documents and their IDs from ChromaDB for ground truth creation.
    
    Returns:
        Tuple of (document_texts, document_ids)
    """
    try:
        collection = rag_engine.chroma_client.get_collection(rag_engine.config.COLLECTION_NAME)
        
        # Get all documents
        all_docs = collection.get()
        
        if not all_docs or not all_docs.get('documents'):
            raise Exception("No documents found in ChromaDB collection")
        
        document_texts = all_docs['documents']
        document_ids = all_docs['ids']
        
        print(f"📚 Found {len(document_texts)} documents in ChromaDB")
        return document_texts, document_ids
        
    except Exception as e:
        print(f"❌ Error accessing ChromaDB: {e}")
        return [], []


def create_retrieval_function(rag_engine: FIRRagEngine):
    """
    Create a retrieval function compatible with the evaluator.
    
    Returns:
        Function that takes (query, top_k) and returns document IDs
    """
    def retrieve_documents(query: str, top_k: int = 5) -> List[str]:
        """Retrieve documents and return their IDs."""
        try:
            results = rag_engine.retrieve_relevant_cases(query, top_k)
            
            if results and 'ids' in results and results['ids']:
                return results['ids'][0]  # ChromaDB returns nested list
            else:
                return []
                
        except Exception as e:
            print(f"❌ Retrieval error for query '{query}': {e}")
            return []
    
    return retrieve_documents


def evaluate_ipc_rag_retrieval(top_k: int = 10, 
                              similarity_threshold: float = 0.6,
                              k_values: List[int] = [5, 10],
                              output_file: str = "ipc_retrieval_evaluation.csv") -> Dict:
    """
    Evaluate the retrieval performance of the IPC RAG system with Precision@k and Recall@k.
    
    Args:
        top_k: Number of documents to retrieve for each query (should be >= max(k_values))
        similarity_threshold: Threshold for semantic ground truth creation
        k_values: List of k values to calculate Precision@k and Recall@k for
        output_file: File to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    print("🚀 Starting IPC RAG Retrieval Evaluation with Precision@k and Recall@k...")
    
    if not SRC_AVAILABLE:
        print("❌ src modules not available. Cannot run real RAG evaluation.")
        print("   Use test_standalone_retrieval.py for mock testing instead.")
        return {
            "error": "src modules not available",
            "suggestion": "Use test_standalone_retrieval.py for testing"
        }
    
    # Initialize components
    config = Config()
    rag_engine = FIRRagEngine(config)
    evaluator = RetrievalEvaluator(similarity_threshold=similarity_threshold)
    
    # Get documents from ChromaDB
    document_texts, document_ids = get_documents_from_chroma(rag_engine)
    
    if not document_texts:
        raise Exception("No documents found. Please ingest data first.")
    
    # Create IPC-specific test cases using imported queries
    print("📝 Creating IPC-specific test cases...")
    
    try:
        # Use queries from create_ipc_queries.py
        ipc_queries, query_categories = create_ipc_specific_queries_and_test_cases()
        expected_sections = get_expected_ipc_sections_for_queries()
        
        print(f"📚 Loaded {len(ipc_queries)} IPC-specific queries from create_ipc_queries.py")
        test_queries = ipc_queries
        
    except (NameError, Exception) as e:
        print(f"⚠️ Using predefined test cases: {e}")
        predefined_test_cases = create_ipc_specific_test_cases()
        test_queries = [tc.query for tc in predefined_test_cases]
    
    # Create semantic ground truth
    test_cases = evaluator.create_semantic_ground_truth(
        queries=test_queries,
        document_texts=document_texts,
        document_ids=document_ids,
        relevance_threshold=similarity_threshold
    )
    
    # Create retrieval function
    retrieval_function = create_retrieval_function(rag_engine)
    
    # Evaluate retrieval performance
    print(f"🔍 Evaluating retrieval with top_k={top_k}, k_values={k_values}...")
    results = evaluator.evaluate_batch(
        test_cases=test_cases,
        retrieval_function=retrieval_function,
        top_k=top_k,
        k_values=k_values
    )
    
    # Print summary
    print("\n📊 IPC RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    
    aggregate = results['aggregate_metrics']
    
    print(f"🎯 FINAL EVALUATION SCORES (Single Values):")
    print(f"   Overall Metrics:")
    print(f"     Precision: {aggregate['precision']:.3f}")
    print(f"     Recall:    {aggregate['recall']:.3f}")
    print(f"     F1-Score:  {aggregate['f1_score']:.3f}")
    
    print(f"\n   Precision@k and Recall@k:")
    for k in k_values:
        print(f"     Precision@{k}: {aggregate[f'precision_at_{k}']:.3f}")
        print(f"     Recall@{k}:    {aggregate[f'recall_at_{k}']:.3f}")
        print(f"     F1@{k}:        {aggregate[f'f1_at_{k}']:.3f}")
    
    print(f"\n� Summary Statistics:")
    print(f"   Total Queries Evaluated: {aggregate['total_queries']}")
    print(f"   Successful Queries:      {results['successful_queries']}")
    
    print(f"\n🎯 KEY PERFORMANCE INDICATORS:")
    print(f"   Best F1 Score: F1@{k_values[0]} = {aggregate[f'f1_at_{k_values[0]}']:.3f}")
    if len(k_values) > 1:
        print(f"   F1@{k_values[1]} = {aggregate[f'f1_at_{k_values[1]}']:.3f}")
    
    # Determine system performance level
    best_f1 = max([aggregate[f'f1_at_{k}'] for k in k_values] + [aggregate['f1_score']])
    
    if best_f1 >= 0.8:
        performance = "🌟 EXCELLENT"
    elif best_f1 >= 0.6:
        performance = "👍 GOOD"
    elif best_f1 >= 0.4:
        performance = "⚠️ MODERATE"
    else:
        performance = "❌ NEEDS IMPROVEMENT"
    
    print(f"   System Performance: {performance} (Best F1: {best_f1:.3f})")
    
    # Show per-category performance with single values
    category_performance = {}
    for result in results['per_query_results']:
        category = result['category']
        if category not in category_performance:
            category_performance[category] = []
        category_performance[category].append(result)
    
    print(f"\n📋 Performance by Category (Single Average Values):")
    for category, cat_results in category_performance.items():
        avg_f1 = np.mean([r['f1_score'] for r in cat_results])
        avg_f1_5 = np.mean([r.get('f1_at_5', 0) for r in cat_results])
        avg_f1_10 = np.mean([r.get('f1_at_10', 0) for r in cat_results])
        
        print(f"   {category.upper():20} - F1: {avg_f1:.3f}, F1@5: {avg_f1_5:.3f}, F1@10: {avg_f1_10:.3f}")
    
    # Save results
    evaluator.save_results(results, output_file)
    
    return results


def analyze_retrieval_errors(results: Dict, top_n: int = 5):
    """
    Analyze the worst performing queries to identify retrieval issues.
    
    Args:
        results: Results from evaluate_fir_rag_retrieval
        top_n: Number of worst queries to analyze
    """
    print(f"\n🔍 Analyzing Top {top_n} Worst Performing Queries:")
    print("=" * 60)
    
    per_query = results['per_query_results']
    
    # Sort by F1 score (ascending)
    worst_queries = sorted(per_query, key=lambda x: x['f1_score'])[:top_n]
    
    for i, result in enumerate(worst_queries, 1):
        print(f"\n{i}. Query: {result['query']}")
        print(f"   Category: {result['category']}")
        print(f"   F1: {result['f1_score']:.3f}, Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
        print(f"   Retrieved: {result['total_retrieved']}, Relevant: {result['total_relevant']}")
        print(f"   TP: {result['true_positives']}, FP: {result['false_positives']}, FN: {result['false_negatives']}")
        
        if result['f1_score'] == 0:
            print("   🚨 NO RELEVANT DOCUMENTS RETRIEVED")
        elif result['precision'] < 0.3:
            print("   ⚠️  Low precision - many irrelevant documents retrieved")
        elif result['recall'] < 0.3:
            print("   ⚠️  Low recall - many relevant documents missed")


if __name__ == "__main__":
    # Run evaluation
    try:
        results = evaluate_ipc_rag_retrieval(
            top_k=10,
            similarity_threshold=0.6,
            k_values=[5, 10],
            output_file="ipc_retrieval_evaluation.csv"
        )
        
        # Analyze errors
        analyze_retrieval_errors(results)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()