"""
Simple ROUGE-C evaluation using the same exact approach as the working retrieval metrics.
This bypasses the import issues by using the verified working RAG connection.
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any
import json
import time

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from rouge_c_pastcases import LegalPastCasesRougeC
from legal_queries_pastcases import get_simple_legal_queries

# Import using the exact same method as working retrieval metrics
from src.rag.vector_db import ChromaVectorDB
from src.rag.rag_system import LegalRAGSystem

def initialize_rag_system_simple():
    """Initialize using the exact same method as working retrieval metrics."""
    try:
        # Use the correct path to the main chroma_db directory
        chroma_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        vector_db = ChromaVectorDB(persist_directory=chroma_db_path)
        rag_system = LegalRAGSystem(vector_db)
        
        # Check if system is working
        stats = vector_db.get_collection_stats()
        print("✓ RAG system initialized successfully!")
        print(f"Total documents in collection: {stats['total_documents']}")
        
        if stats['total_documents'] > 0:
            return rag_system
        else:
            print("Warning: Database appears to be empty")
            return rag_system
        
    except Exception as e:
        print(f"✗ Failed to initialize RAG system: {e}")
        return None

def evaluate_query_simple(rag_system, rouge_evaluator: LegalPastCasesRougeC, 
                         query: str, query_num: int) -> Dict[str, Any]:
    """Evaluate a single query by running the full RAG system."""
    try:
        print(f"  Running RAG system for query {query_num}...")
        
        # Use the answer_question method to get full RAG response
        rag_response = rag_system.answer_question(query, num_docs=5)
        
        # Debug: Print the entire RAG response structure
        print(f"  RAG response keys: {list(rag_response.keys())}")
        print(f"  Sources type: {type(rag_response.get('sources', []))}")
        print(f"  Sources length: {len(rag_response.get('sources', []))}")
        
        # Extract the generated answer
        generated_answer = rag_response.get('answer', 'No answer generated.')
        
        # Extract contexts from sources
        sources = rag_response.get('sources', [])
        contexts = []
        
        print(f"  Processing {len(sources)} sources...")
        for i, source in enumerate(sources):
            print(f"    Source {i}: type={type(source)}, keys={list(source.keys()) if isinstance(source, dict) else 'N/A'}")
            if isinstance(source, dict):
                # Try to get content directly
                content = source.get('content') or source.get('text') or source.get('document')
                
                # If no content but we have chunk_id, try to fetch it from the vector DB
                if not content and 'chunk_id' in source:
                    try:
                        # Get the actual document content using chunk_id
                        chunk_id = source['chunk_id']
                        vector_db = rag_system.vector_db
                        collection = vector_db.collection
                        
                        # Query for the specific chunk
                        result = collection.get(ids=[chunk_id], include=['documents'])
                        if result and result['documents'] and len(result['documents']) > 0:
                            content = result['documents'][0]
                            print(f"    -> Fetched content from chunk_id {chunk_id} (length: {len(content)})")
                        else:
                            print(f"    -> No content found for chunk_id {chunk_id}")
                    except Exception as e:
                        print(f"    -> Error fetching content for chunk_id: {e}")
                
                if content and len(content.strip()) > 0:
                    contexts.append(content)
                    print(f"    -> Added context (length: {len(content)})")
                else:
                    print(f"    -> No content available")
        
        # Print debug information
        print(f"  Generated answer length: {len(generated_answer)} characters")
        print(f"  Retrieved {len(contexts)} contexts")
        
        # If no contexts but we have an answer, create a simple context from answer
        if not contexts and generated_answer != 'No answer generated.':
            # Use the answer as context for ROUGE-C evaluation
            contexts = [generated_answer]
        
        # Evaluate using ROUGE-C
        rouge_results = rouge_evaluator.evaluate_single(
            question=query,
            answer=generated_answer,
            contexts=contexts
        )
        
        return {
            "query": query,
            "retrieved_contexts": len(contexts),
            "generated_answer": generated_answer,
            "rag_metadata": rag_response.get('metadata', {}),
            "rouge_c_results": rouge_results
        }
        
    except Exception as e:
        print(f"Error evaluating query {query_num}: {e}")
        return {
            "query": query,
            "error": str(e),
            "rouge_c_results": None
        }

def display_simple_results(result: Dict[str, Any], query_num: int):
    """Display results in simple format."""
    print(f"\n{'='*80}")
    print(f"Query {query_num}: {result['query']}")
    print(f"{'='*80}")
    print(f"Retrieved contexts: {result['retrieved_contexts']}")
    print(f"Generated answer: {result['generated_answer'][:150]}...")
    
    if result.get("rouge_c_results"):
        rouge_results = result["rouge_c_results"]
        
        print(f"\nROUGE-C Metrics:")
        print(f"  ROUGE-L     - P: {rouge_results['rouge_l']['precision']:.3f}, R: {rouge_results['rouge_l']['recall']:.3f}, F1: {rouge_results['rouge_l']['f1']:.3f}")
        print(f"  Token Overlap - P: {rouge_results['token_overlap']['precision']:.3f}, R: {rouge_results['token_overlap']['recall']:.3f}, F1: {rouge_results['token_overlap']['f1']:.3f}")
        print(f"  Legal Terms - P: {rouge_results['legal_term_overlap']['precision']:.3f}, R: {rouge_results['legal_term_overlap']['recall']:.3f}, F1: {rouge_results['legal_term_overlap']['f1']:.3f}")
        
        semantic = rouge_results['semantic_similarity']
        print(f"  Semantic Similarity - Combined: {semantic['combined_similarity']:.3f}, Max: {semantic['max_similarity']:.3f}, Avg: {semantic['avg_similarity']:.3f}")
        print(f"  Overall Score: {rouge_results['overall_score']:.3f}")
    else:
        print(f"ERROR: {result.get('error', 'Unknown error')}")

def run_simple_rouge_c_evaluation():
    """Run ROUGE-C evaluation with simple, working approach."""
    
    print("="*80)
    print("ROUGE-C Evaluation for Legal Past Cases RAG")
    print("Using simple approach that matches working retrieval metrics")
    print("="*80)
    
    # Initialize systems
    print("\nInitializing systems...")
    rag_system = initialize_rag_system_simple()
    
    if not rag_system:
        print("Failed to initialize RAG system. Exiting.")
        return
    
    # Initialize ROUGE-C evaluator
    try:
        rouge_evaluator = LegalPastCasesRougeC(
            embedding_model="all-MiniLM-L6-v2",
            use_stemming=False,
            remove_stopwords=True
        )
        print("✓ ROUGE-C evaluator initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize ROUGE-C evaluator: {e}")
        return
    
    # Get queries
    queries = get_simple_legal_queries()
    print(f"\nEvaluating {len(queries)} legal queries...")
    
    # Run evaluation
    evaluation_results = []
    rouge_c_results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nProcessing query {i}/{len(queries)}...")
        
        # Evaluate single query
        result = evaluate_query_simple(rag_system, rouge_evaluator, query, i)
        evaluation_results.append(result)
        
        if result.get("rouge_c_results"):
            rouge_c_results.append(result["rouge_c_results"])
            display_simple_results(result, i)
    
    # Compute and display batch statistics
    if rouge_c_results:
        print(f"\n{'='*80}")
        print("BATCH STATISTICS")
        print(f"{'='*80}")
        
        batch_stats = rouge_evaluator.evaluate_batch(rouge_c_results)
        
        print(f"Number of evaluations: {batch_stats['num_evaluations']}")
        print(f"\nROUGE-L F1        - Mean: {batch_stats['rouge_l']['mean_f1']:.3f} ± {batch_stats['rouge_l']['std_f1']:.3f}")
        print(f"Token Overlap F1  - Mean: {batch_stats['token_overlap']['mean_f1']:.3f} ± {batch_stats['token_overlap']['std_f1']:.3f}")
        print(f"Legal Terms F1    - Mean: {batch_stats['legal_term_overlap']['mean_f1']:.3f} ± {batch_stats['legal_term_overlap']['std_f1']:.3f}")
        print(f"Semantic Sim      - Mean: {batch_stats['semantic_similarity']['mean']:.3f} ± {batch_stats['semantic_similarity']['std']:.3f}")
        print(f"Overall Score     - Mean: {batch_stats['overall_score']['mean']:.3f} ± {batch_stats['overall_score']['std']:.3f}")
        
        # Save results
        try:
            results_file = os.path.join(os.path.dirname(__file__), "rouge_c_simple_results.json")
            
            output_data = {
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "individual_results": evaluation_results,
                "batch_statistics": batch_stats
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
        
        # Display summary
        overall_mean = batch_stats["overall_score"]["mean"]
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Overall ROUGE-C Score: {overall_mean:.3f}")
        
        if overall_mean > 0.6:
            print("✓ System shows good context-answer alignment")
        elif overall_mean > 0.4:
            print("⚠ System shows moderate alignment - consider improving answer generation")
        else:
            print("✗ System needs significant improvement in context utilization")
            
    else:
        print("No successful evaluations completed.")

def main():
    """Main execution function."""
    try:
        run_simple_rouge_c_evaluation()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()