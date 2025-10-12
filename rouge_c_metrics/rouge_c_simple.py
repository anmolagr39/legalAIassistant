"""
Simple ROUGE-C evaluation for IPC RAG system.
Automated evaluation script for the legal AI assistant.
"""

import os
import sys
import json
from typing import List, Dict, Any
from rouge_c import RougeC

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine
from create_ipc_queries import IPC_EVALUATION_QUERIES


def generate_qa_pairs_from_rag(rag_engine: RAGEngine, queries: List[str], 
                              num_contexts: int = 5) -> List[Dict[str, Any]]:
    """
    Generate Q-A pairs using the RAG engine for ROUGE-C evaluation.
    
    Args:
        rag_engine: Initialized RAG engine
        queries: List of questions to evaluate
        num_contexts: Number of contexts to retrieve per query
        
    Returns:
        List of Q-A dictionaries with retrieved contexts
    """
    qa_pairs = []
    
    print(f"🔍 Generating Q-A pairs for {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"   Processing query {i}/{len(queries)}: {query[:50]}...")
        
        try:
            # Get answer and contexts from RAG engine
            answer = rag_engine.query(query)
            
            # Retrieve contexts separately to get the actual retrieved documents
            retrieved_docs = rag_engine.retriever.query(query, n_results=num_contexts)
            contexts = retrieved_docs['documents'][0] if retrieved_docs['documents'] else []
            
            qa_pair = {
                'question': query,
                'generated_answer': answer,
                'retrieved_contexts': contexts
            }
            
            qa_pairs.append(qa_pair)
            
        except Exception as e:
            print(f"   ❌ Error processing query {i}: {e}")
            continue
    
    print(f"✅ Generated {len(qa_pairs)} Q-A pairs")
    return qa_pairs


def run_simple_evaluation(embedding_model: str = "all-MiniLM-L6-v2", 
                         num_contexts: int = 5,
                         save_qa_data: bool = True,
                         save_results: bool = True) -> Dict[str, Any]:
    """
    Run simple ROUGE-C evaluation on IPC RAG system.
    
    Args:
        embedding_model: Model for semantic similarity computation
        num_contexts: Number of contexts to retrieve per query
        save_qa_data: Whether to save generated Q-A pairs
        save_results: Whether to save evaluation results
        
    Returns:
        ROUGE-C evaluation results
    """
    print("\n" + "="*80)
    print("🎯 ROUGE-C EVALUATION FOR IPC RAG SYSTEM")
    print("="*80)
    
    # Initialize RAG engine
    print("\n1️⃣ Initializing RAG engine...")
    try:
        rag_engine = RAGEngine()
        print("✅ RAG engine initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG engine: {e}")
        return {}
    
    # Generate Q-A pairs from evaluation queries
    print("\n2️⃣ Generating Q-A pairs from IPC evaluation queries...")
    qa_pairs = generate_qa_pairs_from_rag(rag_engine, IPC_EVALUATION_QUERIES, num_contexts)
    
    if not qa_pairs:
        print("❌ No Q-A pairs generated. Cannot proceed with evaluation.")
        return {}
    
    # Save Q-A data if requested
    if save_qa_data:
        # Save to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        qa_filepath = os.path.join(project_root, "ipc_qa_data.json")
        print(f"\n💾 Saving Q-A data to: {qa_filepath}")
        
        with open(qa_filepath, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Q-A data saved with {len(qa_pairs)} pairs")
    
    # Initialize ROUGE-C evaluator
    print(f"\n3️⃣ Initializing ROUGE-C evaluator with model: {embedding_model}")
    rouge_c = RougeC(embedding_model=embedding_model, use_stemming=False, remove_stopwords=True)
    
    # Run ROUGE-C evaluation
    print("\n4️⃣ Running ROUGE-C evaluation...")
    results = rouge_c.compute_rouge_c_batch(qa_pairs)
    
    if not results:
        print("❌ No evaluation results generated")
        return {}
    
    # Display results
    print("\n5️⃣ Displaying evaluation results...")
    rouge_c.display_results(results)
    
    # Save results if requested
    if save_results:
        # Save to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        results_filepath = os.path.join(project_root, "ipc_rouge_c_results.json")
        print(f"\n💾 Saving evaluation results to: {results_filepath}")
        rouge_c.save_results(results, results_filepath)
    
    return results


def main():
    """Main function for simple evaluation."""
    print("🚀 Starting simple ROUGE-C evaluation...")
    
    # Configuration
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',  # Fast model for quick evaluation
        'num_contexts': 5,                       # Number of contexts to retrieve
        'save_qa_data': True,                   # Save generated Q-A pairs
        'save_results': True                    # Save evaluation results
    }
    
    print(f"⚙️ Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Run evaluation
    results = run_simple_evaluation(**config)
    
    if results and 'aggregate_metrics' in results:
        # Extract key metrics for summary
        metrics = results['aggregate_metrics']
        
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   ROUGE-L F1:     {metrics['rouge_l']['avg_f1']:.3f}")
        print(f"   Token F1:       {metrics['token_overlap']['avg_f1']:.3f}")
        print(f"   Semantic Sim:   {metrics['semantic_similarity']['avg_combined']:.3f}")
        
        overall_score = (
            metrics['rouge_l']['avg_f1'] + 
            metrics['token_overlap']['avg_f1'] + 
            metrics['semantic_similarity']['avg_combined']
        ) / 3
        
        print(f"   Overall Score:  {overall_score:.3f}")
        
        # Performance indicator
        if overall_score > 0.6:
            print("   🟢 EXCELLENT performance!")
        elif overall_score > 0.4:
            print("   🟡 GOOD performance")
        elif overall_score > 0.2:
            print("   🟠 FAIR performance")
        else:
            print("   🔴 POOR performance - needs improvement")
    
    print(f"\n✅ ROUGE-C evaluation completed!")
    return results


if __name__ == "__main__":
    main()