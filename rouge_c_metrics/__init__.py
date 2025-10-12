"""
ROUGE-C Metrics Package for IPC RAG System Evaluation

Reference-free evaluation metrics that measure the quality of generated text 
against retrieved context without requiring reference answers.

Key Features:
- ROUGE-L: Longest Common Subsequence overlap
- Token Overlap: Precision, recall, F1 of token matches  
- Semantic Similarity: Cosine similarity between embeddings
- Batch evaluation support
- CLI interface for interactive evaluation
- Simple evaluation function for automated testing

Components:
- RougeC: Core evaluation class with all metrics
- RougeCCLI: Command-line interface for interactive evaluation
- run_simple_evaluation: Automated evaluation function for IPC RAG system

Usage Examples:
    # Simple automated evaluation
    from rouge_c_metrics import run_simple_evaluation
    results = run_simple_evaluation()
    
    # Direct evaluation
    from rouge_c_metrics import RougeC
    evaluator = RougeC()
    results = evaluator.compute_rouge_c_single(answer, contexts)
    
    # Interactive CLI
    from rouge_c_metrics import RougeCCLI
    cli = RougeCCLI()
    cli.interactive_evaluation()
"""

from .rouge_c import RougeC
from .rouge_c_cli import RougeCCLI
from .rouge_c_simple import run_simple_evaluation, generate_qa_pairs_from_rag
from .results_storage import ResultsManager, quick_store_results

__version__ = "1.0.0"
__author__ = "Legal AI Assistant Team"

__all__ = [
    "RougeC",
    "RougeCCLI", 
    "run_simple_evaluation",
    "generate_qa_pairs_from_rag",
    "ResultsManager",
    "quick_store_results"
]