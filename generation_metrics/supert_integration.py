"""
Main evaluation script for FIR RAG Legal Assistant.
Run this from the evaluation_metrics directory to evaluate your RAG system.
"""

import sys
import os

# Fix path calculation - go up one level to project root, then into src
current_dir = os.path.dirname(os.path.abspath(__file__))  # evaluation_metrics/
project_root = os.path.dirname(current_dir)              # legalAIassistant/
src_path = os.path.join(project_root, 'src')             # legalAIassistant/src/

# Add src to Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Debug: Print paths to verify
print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Src path exists: {os.path.exists(src_path)}")

# Import from src directory
try:
    from config import Config
    from rag_engine import FIRRagEngine
    print("✅ Successfully imported from src")
except ImportError as e:
    print(f"❌ Failed to import from src: {e}")
    print(f"Available files in src: {os.listdir(src_path) if os.path.exists(src_path) else 'Path does not exist'}")
    sys.exit(1)

# Import from local evaluation_metrics directory
try:
    from supert import SUPERTEvaluator, create_evaluation_examples_from_rag, get_test_queries
    print("✅ Successfully imported SUPERT modules")
except ImportError as e:
    print(f"❌ Failed to import SUPERT modules: {e}")
    print(f"Available files in evaluation_metrics: {os.listdir(current_dir)}")
    print("Make sure supert.py exists and has the required functions")
    sys.exit(1)

def evaluate_fir_rag_system():
    """
    Complete evaluation script for your FIR RAG system using SUPERT metric.
    """
    print("\n🚀 Starting FIR RAG System Evaluation with SUPERT")
    print("="*60)
    
    try:
        # Initialize your RAG system
        print("📊 Initializing RAG system...")
        config = Config()
        rag_engine = FIRRagEngine(config)
        
        # Initialize SUPERT evaluator
        print("🔍 Initializing SUPERT evaluator...")
        evaluator = SUPERTEvaluator()
        
        # Get test queries
        test_queries = get_test_queries()
        print(f"📝 Loaded {len(test_queries)} test queries")
        
        # Create evaluation examples by running queries through RAG
        print("🔄 Generating answers for test queries...")
        examples = create_evaluation_examples_from_rag(rag_engine, test_queries)
        
        # Evaluate with SUPERT
        print("📊 Running SUPERT evaluation...")
        results_df = evaluator.evaluate_batch(
            examples, 
            output_path=os.path.join(project_root, 'fir_rag_supert_evaluation.csv')
        )
        
        # Additional analysis
        if not results_df.empty:
            print("\n🎯 TOP PERFORMING QUERIES:")
            top_queries = results_df.nlargest(3, 'supert_f1')[['query', 'supert_f1', 'supert_precision', 'supert_recall']]
            print(top_queries.to_string(index=False))
            
            print("\n⚠️ LOWEST PERFORMING QUERIES:")
            low_queries = results_df.nsmallest(3, 'supert_f1')[['query', 'supert_f1', 'supert_precision', 'supert_recall']]
            print(low_queries.to_string(index=False))
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔍 FIR RAG System Evaluation")
    print("="*50)
    
    # Run SUPERT evaluation
    results = evaluate_fir_rag_system()
    
    if results is not None and not results.empty:
        print(f"\n📊 Evaluation completed! Results saved to CSV.")
        print(f"📈 Overall F1 Score: {results['supert_f1'].mean():.4f}")
        print(f"📈 Overall Precision: {results['supert_precision'].mean():.4f}")
        print(f"📈 Overall Recall: {results['supert_recall'].mean():.4f}")
    else:
        print("\n❌ Evaluation failed. Please check the logs above.")