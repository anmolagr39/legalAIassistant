"""
Retrieval Metrics module for FIR RAG Legal Assistant.

This module contains metrics specifically for evaluating the retrieval
component of the RAG system.

Available metrics:
- Precision: Portion of retrieved documents that are actually relevant
- Recall: Portion of relevant documents that were successfully retrieved  
- F1-Score: Harmonic mean of precision and recall
"""

from .retrieval_evaluator import RetrievalEvaluator, create_retrieval_test_cases

__all__ = [
    'RetrievalEvaluator',
    'create_retrieval_test_cases',
]