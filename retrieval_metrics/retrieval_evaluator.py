"""
Retrieval Evaluation Metrics for RAG System

This module implements precision, recall, and F1-score metrics
specifically for evaluating the retrieval component of the RAG system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import json
import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class RetrievalTestCase:
    """Test case for retrieval evaluation"""
    query: str
    relevant_document_ids: Set[str]  # Ground truth relevant documents
    query_category: str = "general"
    expected_ipc_sections: List[str] = None
    description: str = ""


class RetrievalEvaluator:
    """
    Evaluates retrieval performance using Precision, Recall, and F1-Score.
    
    Metrics:
    - Precision: TP / (TP + FP) - Portion of retrieved docs that are relevant
    - Recall: TP / (TP + FN) - Portion of relevant docs that were retrieved
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the retrieval evaluator.
        
        Args:
            similarity_threshold: Minimum similarity to consider a document relevant
        """
        self.similarity_threshold = similarity_threshold
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def evaluate_single_query(self, 
                             test_case: RetrievalTestCase,
                             retrieved_document_ids: List[str],
                             k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a single query with Precision@k and Recall@k.
        
        Args:
            test_case: The test case containing query and ground truth
            retrieved_document_ids: List of document IDs retrieved by the system (in ranked order)
            k_values: List of k values to calculate Precision@k and Recall@k for
            
        Returns:
            Dictionary containing precision, recall, f1_score, precision@k, recall@k, and counts
        """
        # Convert to sets for easier computation
        relevant_docs = test_case.relevant_document_ids
        retrieved_docs = set(retrieved_document_ids)
        
        # Calculate True Positives, False Positives, False Negatives for all retrieved docs
        true_positives = len(relevant_docs.intersection(retrieved_docs))
        false_positives = len(retrieved_docs - relevant_docs)
        false_negatives = len(relevant_docs - retrieved_docs)
        
        # Calculate overall metrics (using all retrieved documents)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate Precision@k and Recall@k
        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_retrieved': len(retrieved_docs),
            'total_relevant': len(relevant_docs),
            'query': test_case.query,
            'category': test_case.query_category
        }
        
        # Calculate Precision@k and Recall@k for different k values
        for k in k_values:
            # Get top-k retrieved documents
            top_k_retrieved = retrieved_document_ids[:k] if len(retrieved_document_ids) >= k else retrieved_document_ids
            top_k_retrieved_set = set(top_k_retrieved)
            
            # Calculate metrics for top-k
            tp_at_k = len(relevant_docs.intersection(top_k_retrieved_set))
            
            # Precision@k = relevant docs in top-k / k
            precision_at_k = tp_at_k / k if k > 0 else 0.0
            
            # Recall@k = relevant docs in top-k / total relevant docs
            recall_at_k = tp_at_k / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
            
            # F1@k
            f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
            
            # Add to results
            result[f'precision_at_{k}'] = precision_at_k
            result[f'recall_at_{k}'] = recall_at_k
            result[f'f1_at_{k}'] = f1_at_k
            result[f'tp_at_{k}'] = tp_at_k
        
        return result
    
    def evaluate_batch(self, 
                      test_cases: List[RetrievalTestCase],
                      retrieval_function,
                      top_k: int = 10,
                      k_values: List[int] = [5, 10]) -> Dict[str, any]:
        """
        Evaluate retrieval performance across multiple test cases with Precision@k and Recall@k.
        
        Args:
            test_cases: List of test cases to evaluate
            retrieval_function: Function that takes (query, top_k) and returns document IDs
            top_k: Number of documents to retrieve (should be >= max(k_values))
            k_values: List of k values to calculate Precision@k and Recall@k for
            
        Returns:
            Dictionary containing aggregated metrics and per-query results
        """
        results = []
        
        # Ensure top_k is at least as large as the maximum k value
        top_k = max(top_k, max(k_values) if k_values else top_k)
        
        print(f"🔍 Evaluating retrieval performance on {len(test_cases)} test cases...")
        print(f"📊 Computing Precision@k and Recall@k for k = {k_values}")
        
        for i, test_case in enumerate(test_cases):
            try:
                # Get retrieved documents from the RAG system
                retrieved_docs = retrieval_function(test_case.query, top_k)
                
                # Extract document IDs (assuming they're in 'ids' key)
                if isinstance(retrieved_docs, dict) and 'ids' in retrieved_docs:
                    doc_ids = retrieved_docs['ids'][0] if retrieved_docs['ids'] else []
                elif isinstance(retrieved_docs, list):
                    doc_ids = retrieved_docs
                else:
                    doc_ids = []
                
                # Evaluate this query with k-values
                result = self.evaluate_single_query(test_case, doc_ids, k_values)
                results.append(result)
                
                # Print progress with @k metrics
                k_metrics = []
                for k in k_values:
                    p_k = result.get(f'precision_at_{k}', 0)
                    r_k = result.get(f'recall_at_{k}', 0)
                    k_metrics.append(f"P@{k}={p_k:.3f}/R@{k}={r_k:.3f}")
                
                print(f"Query {i+1}/{len(test_cases)}: F1={result['f1_score']:.3f}, {', '.join(k_metrics)}")
                
            except Exception as e:
                print(f"❌ Error evaluating query {i+1}: {e}")
                # Add zero scores for failed queries
                error_result = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': len(test_case.relevant_document_ids),
                    'total_retrieved': 0,
                    'total_relevant': len(test_case.relevant_document_ids),
                    'query': test_case.query,
                    'category': test_case.query_category,
                    'error': str(e)
                }
                
                # Add zero @k metrics
                for k in k_values:
                    error_result[f'precision_at_{k}'] = 0.0
                    error_result[f'recall_at_{k}'] = 0.0
                    error_result[f'f1_at_{k}'] = 0.0
                    error_result[f'tp_at_{k}'] = 0
                
                results.append(error_result)
        
        # Calculate aggregate metrics including @k metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results, k_values)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'per_query_results': results,
            'total_queries': len(test_cases),
            'successful_queries': len([r for r in results if 'error' not in r]),
            'k_values': k_values
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict], k_values: List[int] = [5, 10]) -> Dict[str, float]:
        """Calculate consolidated average metrics across all queries."""
        if not results:
            return {}
        
        # Use macro-averaged metrics as the single representative value
        # (Average of individual query scores - treats each query equally)
        macro_precision = np.mean([r['precision'] for r in results])
        macro_recall = np.mean([r['recall'] for r in results])
        macro_f1 = np.mean([r['f1_score'] for r in results])
        
        aggregate_metrics = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1,
        }
        
        # Calculate single @k metrics (macro-averaged)
        for k in k_values:
            precision_at_k_values = [r.get(f'precision_at_{k}', 0) for r in results]
            recall_at_k_values = [r.get(f'recall_at_{k}', 0) for r in results]
            f1_at_k_values = [r.get(f'f1_at_{k}', 0) for r in results]
            
            aggregate_metrics[f'precision_at_{k}'] = np.mean(precision_at_k_values)
            aggregate_metrics[f'recall_at_{k}'] = np.mean(recall_at_k_values)
            aggregate_metrics[f'f1_at_{k}'] = np.mean(f1_at_k_values)
        
        # Add count statistics for reference
        total_tp = sum(r['true_positives'] for r in results)
        total_fp = sum(r['false_positives'] for r in results)
        total_fn = sum(r['false_negatives'] for r in results)
        
        aggregate_metrics.update({
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_queries': len(results)
        })
        
        # Add @k count statistics
        for k in k_values:
            total_tp_at_k = sum(r.get(f'tp_at_{k}', 0) for r in results)
            aggregate_metrics[f'total_tp_at_{k}'] = total_tp_at_k
        
        return aggregate_metrics
    
    def create_semantic_ground_truth(self, 
                                   queries: List[str],
                                   document_texts: List[str],
                                   document_ids: List[str],
                                   relevance_threshold: float = 0.6) -> List[RetrievalTestCase]:
        """
        Create ground truth using semantic similarity when manual labels aren't available.
        
        Args:
            queries: List of test queries
            document_texts: List of document texts
            document_ids: List of document IDs
            relevance_threshold: Minimum similarity to consider relevant
            
        Returns:
            List of test cases with semantic ground truth
        """
        print(f"🧠 Creating semantic ground truth for {len(queries)} queries...")
        
        # Encode all documents and queries
        doc_embeddings = self.embedder.encode(document_texts)
        query_embeddings = self.embedder.encode(queries)
        
        test_cases = []
        
        for i, query in enumerate(queries):
            # Calculate similarities between query and all documents
            similarities = cosine_similarity([query_embeddings[i]], doc_embeddings)[0]
            
            # Find documents above threshold
            relevant_indices = np.where(similarities >= relevance_threshold)[0]
            relevant_doc_ids = {document_ids[idx] for idx in relevant_indices}
            
            # Extract query category and IPC sections if possible
            category = self._categorize_query(query)
            ipc_sections = self._extract_ipc_sections(query)
            
            test_case = RetrievalTestCase(
                query=query,
                relevant_document_ids=relevant_doc_ids,
                query_category=category,
                expected_ipc_sections=ipc_sections,
                description=f"Semantic ground truth (threshold={relevance_threshold})"
            )
            
            test_cases.append(test_case)
            
        print(f"✅ Created {len(test_cases)} test cases")
        return test_cases
    
    def _categorize_query(self, query: str) -> str:
        """Categorize query based on content."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['theft', 'stealing', 'stolen', 'robbery']):
            return 'theft'
        elif any(word in query_lower for word in ['assault', 'attack', 'violence', 'hurt']):
            return 'assault'
        elif any(word in query_lower for word in ['fraud', 'cheating', 'deception', 'scam']):
            return 'fraud'
        elif any(word in query_lower for word in ['murder', 'kill', 'homicide', 'death']):
            return 'murder'
        elif any(word in query_lower for word in ['drug', 'narcotic', 'substance', 'addiction']):
            return 'drugs'
        else:
            return 'general'
    
    def _extract_ipc_sections(self, text: str) -> List[str]:
        """Extract IPC sections from query text."""
        ipc_patterns = [
            r'(?:IPC|Indian Penal Code)\s*(?:Section|Sec\.?|§)?\s*(\d+[A-Z]?)',
            r'Section\s*(\d+[A-Z]?)',
            r'Sec\.?\s*(\d+[A-Z]?)',
            r'§\s*(\d+[A-Z]?)',
            r'(\d{2,3}[A-Z]?)\s*(?:IPC|Indian Penal Code)'
        ]
        
        sections = []
        for pattern in ipc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend(matches)
        
        return list(set(sections))
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file."""
        # Save detailed results as JSON
        json_path = output_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            # Convert sets to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save summary as CSV
        if 'per_query_results' in results:
            df = pd.DataFrame(results['per_query_results'])
            df.to_csv(output_path, index=False)
            
        print(f"✅ Results saved to {output_path} and {json_path}")
    
    def _make_serializable(self, obj):
        """Convert sets and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


def create_retrieval_test_cases() -> List[RetrievalTestCase]:
    """
    Create sample test cases for retrieval evaluation.
    In a real scenario, these would be manually annotated or derived from user queries.
    """
    test_cases = [
        RetrievalTestCase(
            query="What are the penalties for theft under IPC?",
            relevant_document_ids={"doc_theft_1", "doc_theft_2", "doc_ipc_379", "doc_ipc_380"},
            query_category="theft",
            expected_ipc_sections=["379", "380"],
            description="Query about theft penalties"
        ),
        RetrievalTestCase(
            query="How to file FIR for fraud cases?",
            relevant_document_ids={"doc_fraud_1", "doc_fir_process", "doc_ipc_420"},
            query_category="fraud",
            expected_ipc_sections=["420"],
            description="Query about fraud FIR filing"
        ),
        RetrievalTestCase(
            query="What constitutes assault under Indian law?",
            relevant_document_ids={"doc_assault_1", "doc_assault_2", "doc_ipc_351", "doc_ipc_352"},
            query_category="assault",
            expected_ipc_sections=["351", "352"],
            description="Query about assault definition"
        ),
        RetrievalTestCase(
            query="Procedures for domestic violence cases",
            relevant_document_ids={"doc_domestic_violence", "doc_protection_act", "doc_legal_aid"},
            query_category="domestic_violence",
            expected_ipc_sections=["498A"],
            description="Query about domestic violence procedures"
        ),
        RetrievalTestCase(
            query="What are cognizable and non-cognizable offenses?",
            relevant_document_ids={"doc_cognizable", "doc_crpc_definitions", "doc_police_powers"},
            query_category="general",
            description="Query about offense classifications"
        )
    ]
    
    return test_cases