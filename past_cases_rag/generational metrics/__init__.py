"""
Evaluation module for Legal Assistant RAG system.
"""

from .supert_metric import SUPERTEvaluator, evaluate_legal_rag_with_supert, get_legal_test_queries

__all__ = [
    'SUPERTEvaluator',
    'evaluate_legal_rag_with_supert', 
    'get_legal_test_queries'
]
