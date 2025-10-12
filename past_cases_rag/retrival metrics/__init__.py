"""
Legal RAG Retrieval Metrics Module

This module provides retrieval-based evaluation metrics for the Legal Past Cases RAG system,
including precision, recall, and F1-score calculations without requiring ground truth.
"""

from .legal_rag_metrics import RetrievalMetrics
from .legal_queries import get_legal_case_queries

__version__ = "1.0.0"
__author__ = "Legal RAG Team"

__all__ = [
    'RetrievalMetrics',
    'get_legal_case_queries'
]