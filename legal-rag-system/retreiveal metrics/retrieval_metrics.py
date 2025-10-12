"""
Retrieval-based metrics for RAG systems without ground truth.
Based on semantic similarity and retrieval consistency approaches.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

class RetrievalMetrics:
    """
    Implements retrieval-based metrics for RAG systems without requiring ground truth.
    Uses semantic similarity and consistency-based approaches.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        """
        Initialize the retrieval metrics evaluator.
        
        Args:
            embedding_model: Model for computing semantic embeddings
            similarity_threshold: Threshold for considering documents as relevant
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        console.print(f"[green]Retrieval Metrics initialized with threshold: {similarity_threshold}[/green]")
    
    def compute_query_document_similarity(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between query and retrieved documents.
        
        Args:
            query: User query
            documents: List of retrieved document texts
            
        Returns:
            Array of similarity scores between query and each document
        """
        # Encode query and documents
        query_embedding = self.embedding_model.encode([query])
        doc_embeddings = self.embedding_model.encode(documents)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        return similarities
    
    def compute_document_coherence(self, documents: List[str]) -> float:
        """
        Compute coherence among retrieved documents.
        High coherence indicates consistent retrieval.
        
        Args:
            documents: List of retrieved document texts
            
        Returns:
            Average pairwise similarity among documents
        """
        if len(documents) < 2:
            return 1.0
        
        # Encode all documents
        doc_embeddings = self.embedding_model.encode(documents)
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(doc_embeddings)
        
        # Get upper triangular part (excluding diagonal)
        n = len(documents)
        pairwise_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_similarities.append(similarity_matrix[i][j])
        
        return np.mean(pairwise_similarities) if pairwise_similarities else 1.0
    
    def compute_retrieval_precision(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Compute retrieval precision based on semantic similarity.
        Precision = (Number of relevant retrieved docs) / (Total retrieved docs)
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document texts
            
        Returns:
            Precision score [0, 1]
        """
        if not retrieved_docs:
            return 0.0
        
        # Compute similarities between query and documents
        similarities = self.compute_query_document_similarity(query, retrieved_docs)
        
        # Count documents above similarity threshold as relevant
        relevant_count = np.sum(similarities >= self.similarity_threshold)
        
        precision = relevant_count / len(retrieved_docs)
        return precision
    
    def compute_retrieval_recall(self, query: str, retrieved_docs: List[str], 
                               all_available_docs: List[str], top_k_for_recall: int = 20) -> float:
        """
        Estimate retrieval recall by comparing against a larger set of potentially relevant documents.
        Recall = (Relevant docs retrieved) / (Total relevant docs in corpus)
        
        Args:
            query: User query
            retrieved_docs: Actually retrieved documents
            all_available_docs: Larger set of documents to estimate recall against
            top_k_for_recall: Number of top documents to consider as "relevant" in corpus
            
        Returns:
            Estimated recall score [0, 1]
        """
        if not all_available_docs or not retrieved_docs:
            return 0.0
        
        # Limit the corpus size to avoid memory issues and improve performance
        max_corpus_size = min(len(all_available_docs), 1000)
        corpus_sample = all_available_docs[:max_corpus_size] if len(all_available_docs) > max_corpus_size else all_available_docs
        
        # Find top-k most similar documents in the sampled corpus
        all_similarities = self.compute_query_document_similarity(query, corpus_sample)
        
        # Get indices of top-k most similar documents
        k = min(top_k_for_recall, len(corpus_sample))
        top_k_indices = np.argsort(all_similarities)[-k:]
        top_k_docs = [corpus_sample[i] for i in top_k_indices]
        
        # Count how many of these top-k docs were actually retrieved
        retrieved_set = set(retrieved_docs)
        relevant_retrieved = sum(1 for doc in top_k_docs if doc in retrieved_set)
        
        recall = relevant_retrieved / k
        return recall
    
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
        return f1
    
    def compute_retrieval_diversity(self, retrieved_docs: List[str]) -> float:
        """
        Compute diversity of retrieved documents.
        Higher diversity indicates varied relevant content.
        
        Args:
            retrieved_docs: List of retrieved document texts
            
        Returns:
            Diversity score [0, 1] (1 - average pairwise similarity)
        """
        coherence = self.compute_document_coherence(retrieved_docs)
        diversity = 1 - coherence
        return max(0.0, diversity)  # Ensure non-negative
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], 
                          all_available_docs: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of retrieval performance.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            all_available_docs: Full corpus for recall estimation (optional)
            
        Returns:
            Dictionary with all computed metrics
        """
        console.print(f"[blue]Evaluating retrieval for query: '{query[:50]}...'[/blue]")
        
        # Compute precision
        precision = self.compute_retrieval_precision(query, retrieved_docs)
        
        # Compute recall (if full corpus provided)
        recall = 0.0
        if all_available_docs:
            recall = self.compute_retrieval_recall(query, retrieved_docs, all_available_docs)
        
        # Compute F1-score
        f1_score = self.compute_f1_score(precision, recall)
        
        # Additional metrics
        coherence = self.compute_document_coherence(retrieved_docs)
        diversity = self.compute_retrieval_diversity(retrieved_docs)
        
        # Average similarity to query
        if retrieved_docs:
            similarities = self.compute_query_document_similarity(query, retrieved_docs)
            avg_similarity = np.mean(similarities)
        else:
            avg_similarity = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'coherence': coherence,
            'diversity': diversity,
            'avg_query_similarity': avg_similarity,
            'num_retrieved': len(retrieved_docs)
        }
        
        return metrics
    
    def batch_evaluate(self, query_results: List[Dict[str, Any]], 
                      all_available_docs: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate multiple query-result pairs and compute aggregate statistics.
        
        Args:
            query_results: List of dicts with 'query' and 'retrieved_docs' keys
            all_available_docs: Full corpus for recall estimation
            
        Returns:
            Aggregate metrics and individual results
        """
        individual_results = []
        
        with Progress() as progress:
            task = progress.add_task("[green]Evaluating queries...", total=len(query_results))
            
            for item in query_results:
                query = item['query']
                retrieved_docs = item['retrieved_docs']
                
                # Evaluate this query
                metrics = self.evaluate_retrieval(query, retrieved_docs, all_available_docs)
                
                individual_results.append({
                    'query': query,
                    'metrics': metrics
                })
                
                progress.advance(task)
        
        # Compute aggregate statistics
        all_metrics = [result['metrics'] for result in individual_results]
        
        aggregate_metrics = {}
        metric_names = ['precision', 'recall', 'f1_score', 'coherence', 'diversity', 'avg_query_similarity']
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            aggregate_metrics[f'avg_{metric_name}'] = np.mean(values)
            aggregate_metrics[f'std_{metric_name}'] = np.std(values)
            aggregate_metrics[f'min_{metric_name}'] = np.min(values)
            aggregate_metrics[f'max_{metric_name}'] = np.max(values)
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'individual_results': individual_results,
            'total_queries': len(query_results)
        }
    
    def display_results(self, results: Dict[str, Any]):
        """
        Display evaluation results in a formatted table.
        
        Args:
            results: Results from batch_evaluate or single evaluate_retrieval
        """
        if 'aggregate_metrics' in results:
            # Display aggregate results
            console.print("\n[bold green]Aggregate Retrieval Metrics[/bold green]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Average", justify="right")
            table.add_column("Std Dev", justify="right")
            table.add_column("Min", justify="right")
            table.add_column("Max", justify="right")
            
            metrics = results['aggregate_metrics']
            metric_names = ['precision', 'recall', 'f1_score', 'coherence', 'diversity', 'avg_query_similarity']
            
            for metric in metric_names:
                table.add_row(
                    metric.replace('_', ' ').title(),
                    f"{metrics[f'avg_{metric}']:.3f}",
                    f"{metrics[f'std_{metric}']:.3f}",
                    f"{metrics[f'min_{metric}']:.3f}",
                    f"{metrics[f'max_{metric}']:.3f}"
                )
            
            console.print(table)
            console.print(f"\n[blue]Total queries evaluated: {results['total_queries']}[/blue]")
            
        else:
            # Display single result
            console.print("\n[bold green]Retrieval Metrics[/bold green]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Score", justify="right")
            
            for metric_name, value in results.items():
                if isinstance(value, (int, float)):
                    table.add_row(
                        metric_name.replace('_', ' ').title(),
                        f"{value:.3f}"
                    )
            
            console.print(table)
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Results saved to: {filepath}[/green]")