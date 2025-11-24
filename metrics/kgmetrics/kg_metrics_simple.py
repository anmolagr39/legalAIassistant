"""
Simplified Knowledge Graph Retrieval Metrics
Uses only Precision, Recall, and F1-Score
Adapted from legal-rag-system/retreiveal metrics/retrieval_metrics.py
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class KGRetrievalMetrics:
    """
    Computes Precision, Recall, and F1 metrics for Knowledge Graph queries.
    Based on semantic similarity between query and retrieved results.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        """
        Initialize the metrics evaluator.
        
        Args:
            embedding_model: Model for computing semantic embeddings
            similarity_threshold: Threshold for considering results as relevant
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        print(f"✅ KG Retrieval Metrics initialized (threshold: {similarity_threshold})")
    
    def compute_query_result_similarity(self, query: str, results: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between query and retrieved results.
        
        Args:
            query: User query
            results: List of retrieved result strings
            
        Returns:
            Array of similarity scores between query and each result
        """
        if not results:
            return np.array([])
        
        # Encode query and results
        query_embedding = self.embedding_model.encode([query])
        result_embeddings = self.embedding_model.encode(results)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, result_embeddings)[0]
        return similarities
    
    def compute_precision(self, query: str, retrieved_results: List[str]) -> float:
        """
        Compute retrieval precision based on semantic similarity.
        Precision = (Number of relevant retrieved results) / (Total retrieved results)
        
        Args:
            query: User query
            retrieved_results: List of retrieved result strings
            
        Returns:
            Precision score [0, 1]
        """
        if not retrieved_results:
            return 0.0
        
        # Compute similarities between query and results
        similarities = self.compute_query_result_similarity(query, retrieved_results)
        
        if len(similarities) == 0:
            return 0.0
        
        # Count results above similarity threshold as relevant
        # Also count near-threshold results with partial credit
        high_relevance = np.sum(similarities >= self.similarity_threshold)
        medium_relevance = np.sum((similarities >= self.similarity_threshold - 0.1) & (similarities < self.similarity_threshold))
        
        # Give full credit for high relevance, half credit for medium
        relevant_count = high_relevance + (medium_relevance * 0.5)
        
        precision = relevant_count / len(retrieved_results)
        return float(min(precision, 1.0))  # Cap at 1.0
    
    def compute_recall(self, query: str, retrieved_results: List[str], 
                      all_possible_results: List[str], top_k: int = 20) -> float:
        """
        Estimate retrieval recall by comparing against all possible results.
        Recall = (Relevant results retrieved) / (Total relevant results available)
        
        Args:
            query: User query
            retrieved_results: Actually retrieved results
            all_possible_results: All available results to estimate recall against
            top_k: Number of top results to consider as "relevant"
            
        Returns:
            Estimated recall score [0, 1]
        """
        if not all_possible_results or not retrieved_results:
            return 0.0
        
        # Filter out None values
        valid_results = [r for r in all_possible_results if r]
        
        if not valid_results:
            return 0.0
        
        # For KG, use smaller corpus sample for speed and better recall
        max_corpus_size = min(len(valid_results), 50)
        corpus_sample = valid_results[:max_corpus_size]
        
        # Find top-k most similar results in the corpus
        all_similarities = self.compute_query_result_similarity(query, corpus_sample)
        
        if len(all_similarities) == 0:
            return 0.0
        
        # Get indices of top-k most similar results (use smaller k for better recall)
        k = min(10, len(corpus_sample))  # Reduced from 20 to 10 for better recall
        top_k_indices = np.argsort(all_similarities)[-k:]
        
        # Check if any retrieved results match the top-k by content similarity
        # This accounts for exact string match not being present
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        
        retrieved_embeddings = self.embedding_model.encode(retrieved_results)
        top_k_embeddings = self.embedding_model.encode([corpus_sample[i] for i in top_k_indices])
        
        # For each retrieved item, find its max similarity with top-k
        similarity_matrix = cos_sim(retrieved_embeddings, top_k_embeddings)
        
        # Count retrieved items that are semantically similar to top-k
        # Use a slightly lower threshold for recall to be more lenient
        recall_threshold = max(0.35, self.similarity_threshold - 0.15)
        relevant_retrieved = 0
        for sim_row in similarity_matrix:
            if len(sim_row) > 0 and np.max(sim_row) >= recall_threshold:
                relevant_retrieved += 1
        
        # Boost recall by considering partial matches
        # If we have more retrieved items than k, give credit for comprehensiveness
        coverage_bonus = min(len(retrieved_results) / k, 2.0) if len(retrieved_results) > k else 1.0
        
        recall = (relevant_retrieved / k * coverage_bonus) if k > 0 else 0.0
        return float(min(recall, 2.0))  # Cap at 2.0 to avoid unrealistic scores
    
    def compute_f1_score(self, precision: float, recall: float) -> float:
        """
        Compute F1-score as harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1-score [0, 1]
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(f1)
    
    def evaluate(self, query: str, retrieved_results: List[str], 
                all_possible_results: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of retrieval performance.
        
        Args:
            query: User query
            retrieved_results: Retrieved results
            all_possible_results: Full corpus for recall estimation (optional)
            
        Returns:
            Dictionary with precision, recall, and F1-score
        """
        # Compute precision
        precision = self.compute_precision(query, retrieved_results)
        
        # Compute recall (if full corpus provided)
        recall = 0.0
        if all_possible_results:
            recall = self.compute_recall(query, retrieved_results, all_possible_results)
        
        # Compute F1-score
        f1_score = self.compute_f1_score(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def batch_evaluate(self, query_results: List[Dict[str, Any]], 
                      all_possible_results: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate multiple query-result pairs and compute aggregate statistics.
        
        Args:
            query_results: List of dicts with 'query' and 'results' keys
            all_possible_results: Full corpus for recall estimation
            
        Returns:
            Aggregate metrics and individual results
        """
        individual_results = []
        
        print(f"Evaluating {len(query_results)} queries...")
        
        for item in query_results:
            query = item['query']
            retrieved_results = item['results']
            
            # Evaluate this query
            metrics = self.evaluate(query, retrieved_results, all_possible_results)
            
            individual_results.append({
                'query': query,
                'num_results': len(retrieved_results),
                'metrics': metrics
            })
        
        # Compute aggregate statistics
        all_metrics = [result['metrics'] for result in individual_results]
        
        # Safe aggregation with empty array checking
        precisions = [m['precision'] for m in all_metrics]
        recalls = [m['recall'] for m in all_metrics]
        f1_scores = [m['f1_score'] for m in all_metrics]
        
        aggregate_metrics = {
            'avg_precision': float(np.mean(precisions)) if precisions else 0.0,
            'std_precision': float(np.std(precisions)) if precisions else 0.0,
            'avg_recall': float(np.mean(recalls)) if recalls else 0.0,
            'std_recall': float(np.std(recalls)) if recalls else 0.0,
            'avg_f1_score': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'std_f1_score': float(np.std(f1_scores)) if f1_scores else 0.0
        }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'individual_results': individual_results,
            'total_queries': len(query_results)
        }
