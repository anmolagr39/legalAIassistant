"""
RAG Performance Optimization Script
This script will help improve the F1, Precision, and Recall scores
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import config
    import rag_engine
    from retrieval_metrics.retrieval_evaluator import RetrievalEvaluator, RetrievalTestCase
    from create_ipc_queries import create_ipc_specific_queries_and_test_cases
    import chromadb
    import json
    import numpy as np
    
    print("🚀 RAG Performance Optimization")
    print("=" * 50)
    
    # Setup
    config_obj = config.Config()
    rag_engine_obj = rag_engine.FIRRagEngine(config_obj)
    
    # Load improved ground truth
    with open('improved_ground_truth.json', 'r') as f:
        improved_ground_truth = json.load(f)
    
    # Get test queries
    ipc_queries, query_categories = create_ipc_specific_queries_and_test_cases()
    
    # Test different retrieval strategies
    def evaluate_strategy(top_k, description):
        print(f"\n🧪 Testing Strategy: {description}")
        print(f"   Parameters: top_k={top_k}")
        
        # Create test cases
        test_cases = []
        for query in ipc_queries:
            category = "general"
            for cat, cat_queries in query_categories.items():
                if query in cat_queries:
                    category = cat
                    break
            
            relevant_doc_ids = set(improved_ground_truth.get(query, []))
            test_case = RetrievalTestCase(
                query=query,
                relevant_document_ids=relevant_doc_ids,
                query_category=category
            )
            test_cases.append(test_case)
        
        # Initialize evaluator
        evaluator = RetrievalEvaluator(similarity_threshold=0.6)
        
        # Retrieval function
        def rag_retrieval_function(query: str, k: int = 10):
            try:
                results = rag_engine_obj.retrieve_relevant_cases(query, k)
                if not results or 'ids' not in results or not results['ids']:
                    return []
                return results['ids'][0] if results['ids'] else []
            except Exception as e:
                return []
        
        # Evaluate
        results = evaluator.evaluate_batch(
            test_cases=test_cases,
            retrieval_function=rag_retrieval_function,
            k_values=[5, 10],
            top_k=top_k
        )
        
        aggregate = results['aggregate_metrics']
        return {
            'f1_score': aggregate['f1_score'],
            'f1_at_5': aggregate['f1_at_5'],
            'f1_at_10': aggregate['f1_at_10'],
            'precision_at_5': aggregate['precision_at_5'],
            'recall_at_5': aggregate['recall_at_5'],
            'precision_at_10': aggregate['precision_at_10'],
            'recall_at_10': aggregate['recall_at_10'],
        }
    
    # Test different configurations
    strategies = [
        (10, "Standard retrieval (top_k=10)"),
        (15, "Expanded retrieval (top_k=15)"),
        (20, "Wide retrieval (top_k=20)"),
        (5, "Focused retrieval (top_k=5)"),
    ]
    
    best_results = {}
    best_f1_5 = 0
    best_strategy = None
    
    for top_k, description in strategies:
        results = evaluate_strategy(top_k, description)
        
        print(f"   Results:")
        print(f"     F1@5:  {results['f1_at_5']:.3f}")
        print(f"     F1@10: {results['f1_at_10']:.3f}")
        print(f"     P@5:   {results['precision_at_5']:.3f}")
        print(f"     R@5:   {results['recall_at_5']:.3f}")
        
        if results['f1_at_5'] > best_f1_5:
            best_f1_5 = results['f1_at_5']
            best_strategy = (top_k, description)
            best_results = results
    
    print(f"\n🏆 BEST PERFORMING STRATEGY:")
    print(f"   Strategy: {best_strategy[1]}")
    print(f"   F1@5: {best_results['f1_at_5']:.3f}")
    print(f"   F1@10: {best_results['f1_at_10']:.3f}")
    print(f"   Precision@5: {best_results['precision_at_5']:.3f}")
    print(f"   Recall@5: {best_results['recall_at_5']:.3f}")
    print(f"   Precision@10: {best_results['precision_at_10']:.3f}")
    print(f"   Recall@10: {best_results['recall_at_10']:.3f}")
    
    # Performance analysis
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    
    if best_results['f1_at_5'] >= 0.3:
        print(f"✅ Good performance! F1@5 = {best_results['f1_at_5']:.3f}")
    elif best_results['f1_at_5'] >= 0.15:
        print(f"⚠️ Moderate performance. F1@5 = {best_results['f1_at_5']:.3f}")
    else:
        print(f"❌ Needs improvement. F1@5 = {best_results['f1_at_5']:.3f}")
    
    # Precision vs Recall analysis
    if best_results['precision_at_5'] > best_results['recall_at_5']:
        print(f"📈 System has high precision but lower recall")
        print(f"   → Retrieves fewer documents but they are more relevant")
        print(f"   → Consider increasing top_k to improve recall")
    else:
        print(f"📈 System has high recall but lower precision")
        print(f"   → Retrieves many documents but some are less relevant")
        print(f"   → Consider improving ranking algorithm")
    
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if best_results['f1_at_5'] < 0.2:
        print(f"   1. 🔧 CRITICAL: Check document chunking strategy")
        print(f"      - Current chunks might be too small/large")
        print(f"      - Consider overlapping chunks for better coverage")
        
        print(f"   2. 🎯 IMPORTANT: Improve query preprocessing")
        print(f"      - Add query expansion for IPC section queries")
        print(f"      - Handle synonyms and related legal terms")
        
        print(f"   3. 📊 CONSIDER: Different embedding model")
        print(f"      - Try legal domain-specific embeddings")
        print(f"      - Consider fine-tuning on legal text")
    
    elif best_results['f1_at_5'] < 0.4:
        print(f"   1. 🔧 Optimize retrieval parameters")
        print(f"      - Fine-tune similarity thresholds")
        print(f"      - Experiment with different top_k values")
        
        print(f"   2. 📝 Enhance query processing")
        print(f"      - Add query expansion techniques")
        print(f"      - Handle legal terminology better")
    
    else:
        print(f"   1. ✅ System is performing well!")
        print(f"   2. 🚀 Consider advanced techniques:")
        print(f"      - Query expansion")
        print(f"      - Re-ranking mechanisms")
        print(f"      - Hybrid retrieval (dense + sparse)")
    
    # Save optimization results
    optimization_results = {
        'best_strategy': best_strategy[1],
        'best_top_k': best_strategy[0],
        'best_metrics': best_results,
        'all_strategies': dict(zip([desc for _, desc in strategies], 
                                 [evaluate_strategy(k, desc) for k, desc in strategies]))
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\n💾 Optimization results saved to 'optimization_results.json'")
    print(f"✅ Optimization analysis complete!")
    
except Exception as e:
    print(f"❌ Error during optimization: {e}")
    import traceback
    traceback.print_exc()