"""
Evaluation module for FIR RAG Legal Assistant.

This module contains evaluation metrics and tools for assessing
the performance of the RAG system.

Available metrics:
- SUPERT: Semantic similarity-based evaluation using optimal bipartite matching
- BLANC: Quality-based evaluation measuring language model prediction help
"""

from .supert import SUPERTEvaluator, create_evaluation_examples_from_rag, get_test_queries

__all__ = [
    'SUPERTEvaluator',
    'create_evaluation_examples_from_rag', 
    'get_test_queries',
]