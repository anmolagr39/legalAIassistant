"""
RAG Retrieval Metrics Evaluation Script

This script evaluates the performance of the FIR Legal AI Assistant's RAG system
using comprehensive retrieval metrics including:
- F1-Score, Precision, Recall @5 and @10
- Query-by-query performance analysis  
- Category-based performance breakdown

Usage: python evaluate_retrieval_metrics.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the actual RAG components
    import config
    import rag_engine
    from retrieval_metrics.retrieval_evaluator import RetrievalEvaluator, RetrievalTestCase
    from create_ipc_queries import create_ipc_specific_queries_and_test_cases, get_expected_ipc_sections_for_queries
    import chromadb
    
    print("✅ All imports successful!")
    print("🧪 Testing IPC Retrieval Evaluation with Real RAG System...")
    
    # Test RAG system setup
    try:
        print("🔧 Setting up RAG system...")
        config_obj = config.Config()
        rag_engine_obj = rag_engine.FIRRagEngine(config_obj)
        
        # Test ChromaDB connection
        print("🗄️ Testing ChromaDB connection...")
        client = chromadb.PersistentClient(path=config_obj.CHROMA_DB_PATH)
        collection = client.get_collection(name=config_obj.COLLECTION_NAME)
        doc_count = collection.count()
        print(f"📊 Found {doc_count} documents in ChromaDB")
        
        if doc_count == 0:
            print("❌ No documents found in ChromaDB. Please ingest data first.")
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = RetrievalEvaluator(similarity_threshold=0.6)
        
        # Get test queries
        print("📝 Loading IPC test queries...")
        ipc_queries, query_categories = create_ipc_specific_queries_and_test_cases()
        expected_sections = get_expected_ipc_sections_for_queries()
        
        print(f"📚 Loaded {len(ipc_queries)} IPC-specific queries")
        
        # Load improved ground truth if available
        try:
            import json
            with open('improved_ground_truth.json', 'r') as f:
                improved_ground_truth = json.load(f)
            print("✅ Using improved ground truth mapping")
            use_improved_ground_truth = True
        except FileNotFoundError:
            print("⚠️ Improved ground truth not found. Run analyze_documents.py first for better results.")
            use_improved_ground_truth = False
        
        # Create test cases
        test_cases = []
        for i, query in enumerate(ipc_queries):
            category = query_categories.get(query, "general")
            
            if use_improved_ground_truth:
                # Use improved ground truth with actual document IDs
                relevant_doc_ids = set(improved_ground_truth.get(query, []))
            else:
                # Fallback to original approach (which gives poor results)
                expected = expected_sections.get(query, [])
                relevant_doc_ids = set(expected)
            
            test_case = RetrievalTestCase(
                query=query,
                relevant_document_ids=relevant_doc_ids,
                query_category=category
            )
            test_cases.append(test_case)
        
        print(f"🎯 Created {len(test_cases)} test cases")
        
        # Function to retrieve documents using actual RAG system
        def rag_retrieval_function(query: str, top_k: int = 10):
            try:
                # Use the RAG engine to retrieve documents directly
                results = rag_engine_obj.retrieve_relevant_cases(query, top_k)
                
                # Check if results are valid
                if not results or 'ids' not in results or not results['ids']:
                    print(f"⚠️ No results returned for query: {query}")
                    return []
                
                # Extract document IDs from ChromaDB results
                # ChromaDB returns results in nested list format
                doc_ids = results['ids'][0] if results['ids'] else []
                
                # Return just the document IDs (as required by evaluator)
                return doc_ids
                
            except Exception as e:
                print(f"❌ Error in retrieval for query '{query}': {e}")
                import traceback
                traceback.print_exc()
                return []
        
        # Run evaluation
        print("\n🚀 Starting retrieval evaluation...")
        print("=" * 60)
        
        results = evaluator.evaluate_batch(
            test_cases=test_cases,
            retrieval_function=rag_retrieval_function,
            k_values=[5, 10],
            top_k=10
        )
        
        # Display results
        print("\n✅ Evaluation completed successfully!")
        print(f"📊 SINGLE VALUE RESULTS FOR YOUR RAG SYSTEM:")
        
        aggregate = results['aggregate_metrics']
        
        print(f"\n🎯 FINAL SCORES (averaged across {len(test_cases)} queries):")
        print(f"   F1-Score:     {aggregate['f1_score']:.3f}")
        print(f"   F1@5:         {aggregate['f1_at_5']:.3f}")
        print(f"   F1@10:        {aggregate['f1_at_10']:.3f}")
        print(f"   Precision@5:  {aggregate['precision_at_5']:.3f}")
        print(f"   Recall@5:     {aggregate['recall_at_5']:.3f}")
        print(f"   Precision@10: {aggregate['precision_at_10']:.3f}")
        print(f"   Recall@10:    {aggregate['recall_at_10']:.3f}")
        
        print(f"\n📈 Query Statistics:")
        print(f"   Total Queries: {results['total_queries']}")
        print(f"   Successful Queries: {results['successful_queries']}")
        
        # Determine best metric
        best_f1 = max(aggregate['f1_score'], aggregate['f1_at_5'], aggregate['f1_at_10'])
        if best_f1 == aggregate['f1_at_5']:
            best_metric = "F1@5"
        elif best_f1 == aggregate['f1_at_10']:
            best_metric = "F1@10"
        else:
            best_metric = "F1-Score"
        
        print(f"\n🏆 Best Performance: {best_metric} = {best_f1:.3f}")
        
        # Performance assessment
        if best_f1 >= 0.8:
            assessment = "🌟 EXCELLENT - Your RAG system is performing very well!"
        elif best_f1 >= 0.6:
            assessment = "👍 GOOD - Your RAG system shows solid performance"
        elif best_f1 >= 0.4:
            assessment = "⚠️ MODERATE - Consider improving document retrieval"
        else:
            assessment = "❌ NEEDS IMPROVEMENT - Focus on better document matching"
        
        print(f"\n{assessment}")
        
        # Show category breakdown if available
        if 'category_results' in results:
            print(f"\n📋 PERFORMANCE BY CATEGORY:")
            for category, metrics in results['category_results'].items():
                print(f"   {category.upper():20} - F1: {metrics.get('f1_score', 0):.3f}, F1@5: {metrics.get('f1_at_5', 0):.3f}")
        
        print(f"\n💾 Results saved for further analysis")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory and the modules are available.")
    print("\nRequired dependencies:")
    print("- src/config.py")
    print("- src/rag_engine.py") 
    print("- ChromaDB with documents")
    print("- create_ipc_queries.py")