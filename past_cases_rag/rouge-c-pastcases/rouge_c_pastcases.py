"""
ROUGE-C: Reference-free Context-based ROUGE metric for Legal Past Cases RAG system.
Measures quality of generated text against retrieved legal case contexts without requiring reference answers.

Based on: Lin, 'ROUGE: A Package for Automatic Evaluation of Summaries', ACL 2004 
(adapted for reference-free evaluation of legal case retrieval)
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Add parent directory to path for importing RAG system
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

console = Console()

class LegalPastCasesRougeC:
    """
    ROUGE-C: Context-based reference-free ROUGE metric for Legal Past Cases RAG.
    Evaluates generated legal responses against retrieved case contexts using:
    1. ROUGE-L: Longest Common Subsequence overlap
    2. Semantic similarity: Cosine similarity between embeddings
    3. Token overlap: Precision, recall, and F1 of token matches
    4. Legal terminology overlap: Specialized for legal domain
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 use_stemming: bool = False, remove_stopwords: bool = True):
        """
        Initialize Legal Past Cases ROUGE-C evaluator.
        
        Args:
            embedding_model: Model for semantic similarity computation
            use_stemming: Whether to apply stemming (requires nltk)
            remove_stopwords: Whether to remove stopwords
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Legal-specific and common stopwords
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'said', 'also',
            'case', 'court', 'law', 'legal'  # Some legal terms kept as they're important
        }
        
        # Legal terminology weights (higher weight for important legal terms)
        self.legal_terms = {
            'constitutional', 'amendment', 'article', 'supreme', 'high', 'tribunal', 'judgment',
            'petition', 'writ', 'mandamus', 'certiorari', 'habeas', 'corpus', 'fundamental',
            'rights', 'liberty', 'equality', 'justice', 'procedure', 'appeal', 'criminal',
            'civil', 'administrative', 'judicial', 'review', 'constitution', 'violation',
            'breach', 'defendant', 'plaintiff', 'appellant', 'respondent', 'magistrate',
            'jurisdiction', 'precedent', 'ratio', 'decidendi', 'obiter', 'dicta'
        }
        
        console.print(f"[green]Legal Past Cases ROUGE-C initialized with embedding model: {embedding_model}[/green]")
        
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                console.print("[blue]Stemming enabled[/blue]")
            except ImportError:
                console.print("[yellow]Warning: NLTK not available, stemming disabled[/yellow]")
                self.use_stemming = False
                self.stemmer = None
        else:
            self.stemmer = None
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, lowercasing, and optionally removing stopwords/stemming.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        # Basic tokenization and normalization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove stopwords if enabled (but keep legal terms)
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords or token in self.legal_terms]
        
        # Apply stemming if enabled
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of Longest Common Subsequence between two token sequences.
        
        Args:
            seq1: First token sequence
            seq2: Second token sequence
            
        Returns:
            Length of LCS
        """
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_rouge_l(self, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE-L scores between answer and contexts.
        
        Args:
            answer: Generated answer text
            contexts: List of retrieved context texts
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        answer_tokens = self._preprocess_text(answer)
        
        if not contexts or not answer_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        context_tokens = self._preprocess_text(combined_context)
        
        if not context_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Compute LCS
        lcs_len = self._lcs_length(answer_tokens, context_tokens)
        
        # Calculate precision, recall, F1
        precision = lcs_len / len(answer_tokens) if answer_tokens else 0.0
        recall = lcs_len / len(context_tokens) if context_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_token_overlap(self, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Compute token overlap metrics between answer and contexts.
        
        Args:
            answer: Generated answer text
            contexts: List of retrieved context texts
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        answer_tokens = set(self._preprocess_text(answer))
        
        if not contexts or not answer_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        context_tokens = set(self._preprocess_text(combined_context))
        
        if not context_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate overlap
        overlap = answer_tokens.intersection(context_tokens)
        
        # Calculate precision, recall, F1
        precision = len(overlap) / len(answer_tokens) if answer_tokens else 0.0
        recall = len(overlap) / len(context_tokens) if context_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_legal_term_overlap(self, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Compute legal terminology overlap between answer and contexts.
        
        Args:
            answer: Generated answer text
            contexts: List of retrieved context texts
            
        Returns:
            Dictionary with legal term precision, recall, and F1 scores
        """
        answer_tokens = set(self._preprocess_text(answer))
        answer_legal_terms = answer_tokens.intersection(self.legal_terms)
        
        if not contexts or not answer_legal_terms:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Combine all contexts
        combined_context = " ".join(contexts)
        context_tokens = set(self._preprocess_text(combined_context))
        context_legal_terms = context_tokens.intersection(self.legal_terms)
        
        if not context_legal_terms:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate legal term overlap
        legal_overlap = answer_legal_terms.intersection(context_legal_terms)
        
        # Calculate precision, recall, F1
        precision = len(legal_overlap) / len(answer_legal_terms) if answer_legal_terms else 0.0
        recall = len(legal_overlap) / len(context_legal_terms) if context_legal_terms else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_semantic_similarity(self, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Compute semantic similarity between answer and contexts using embeddings.
        
        Args:
            answer: Generated answer text
            contexts: List of retrieved context texts
            
        Returns:
            Dictionary with similarity scores
        """
        if not contexts or not answer.strip():
            return {"combined_similarity": 0.0, "max_similarity": 0.0, "avg_similarity": 0.0}
        
        # Get embeddings
        answer_embedding = self.embedding_model.encode([answer])
        context_embeddings = self.embedding_model.encode(contexts)
        
        # Compute similarities with individual contexts
        similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
        
        # Compute similarity with combined context
        combined_context = " ".join(contexts)
        combined_embedding = self.embedding_model.encode([combined_context])
        combined_similarity = cosine_similarity(answer_embedding, combined_embedding)[0][0]
        
        return {
            "combined_similarity": float(combined_similarity),
            "max_similarity": float(np.max(similarities)),
            "avg_similarity": float(np.mean(similarities))
        }
    
    def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair against retrieved contexts.
        
        Args:
            question: Input question
            answer: Generated answer
            contexts: Retrieved context documents
            
        Returns:
            Dictionary containing all ROUGE-C metrics
        """
        # Compute all metrics
        rouge_l = self.compute_rouge_l(answer, contexts)
        token_overlap = self.compute_token_overlap(answer, contexts)
        legal_overlap = self.compute_legal_term_overlap(answer, contexts)
        semantic_sim = self.compute_semantic_similarity(answer, contexts)
        
        return {
            "question": question,
            "answer_length": len(answer.split()),
            "num_contexts": len(contexts),
            "rouge_l": rouge_l,
            "token_overlap": token_overlap,
            "legal_term_overlap": legal_overlap,
            "semantic_similarity": semantic_sim,
            "overall_score": self._compute_overall_score(rouge_l, token_overlap, legal_overlap, semantic_sim)
        }
    
    def _compute_overall_score(self, rouge_l: Dict, token_overlap: Dict, 
                              legal_overlap: Dict, semantic_sim: Dict) -> float:
        """
        Compute overall ROUGE-C score as weighted combination of metrics.
        
        Args:
            rouge_l: ROUGE-L metrics
            token_overlap: Token overlap metrics
            legal_overlap: Legal term overlap metrics
            semantic_sim: Semantic similarity metrics
            
        Returns:
            Overall score (0-1)
        """
        # Weighted combination (adjust weights based on importance)
        weights = {
            "rouge_l_f1": 0.25,
            "token_overlap_f1": 0.25,
            "legal_overlap_f1": 0.25,
            "semantic_similarity": 0.25
        }
        
        score = (
            weights["rouge_l_f1"] * rouge_l["f1"] +
            weights["token_overlap_f1"] * token_overlap["f1"] +
            weights["legal_overlap_f1"] * legal_overlap["f1"] +
            weights["semantic_similarity"] * semantic_sim["combined_similarity"]
        )
        
        return score
    
    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple question-answer pairs and compute aggregate statistics.
        
        Args:
            evaluations: List of evaluation results from evaluate_single
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not evaluations:
            return {}
        
        # Extract metrics for aggregation
        rouge_l_f1 = [eval_result["rouge_l"]["f1"] for eval_result in evaluations]
        token_overlap_f1 = [eval_result["token_overlap"]["f1"] for eval_result in evaluations]
        legal_overlap_f1 = [eval_result["legal_term_overlap"]["f1"] for eval_result in evaluations]
        semantic_sim = [eval_result["semantic_similarity"]["combined_similarity"] for eval_result in evaluations]
        overall_scores = [eval_result["overall_score"] for eval_result in evaluations]
        
        return {
            "num_evaluations": len(evaluations),
            "rouge_l": {
                "mean_f1": np.mean(rouge_l_f1),
                "std_f1": np.std(rouge_l_f1),
                "min_f1": np.min(rouge_l_f1),
                "max_f1": np.max(rouge_l_f1)
            },
            "token_overlap": {
                "mean_f1": np.mean(token_overlap_f1),
                "std_f1": np.std(token_overlap_f1),
                "min_f1": np.min(token_overlap_f1),
                "max_f1": np.max(token_overlap_f1)
            },
            "legal_term_overlap": {
                "mean_f1": np.mean(legal_overlap_f1),
                "std_f1": np.std(legal_overlap_f1),
                "min_f1": np.min(legal_overlap_f1),
                "max_f1": np.max(legal_overlap_f1)
            },
            "semantic_similarity": {
                "mean": np.mean(semantic_sim),
                "std": np.std(semantic_sim),
                "min": np.min(semantic_sim),
                "max": np.max(semantic_sim)
            },
            "overall_score": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "min": np.min(overall_scores),
                "max": np.max(overall_scores)
            }
        }
    
    def display_results(self, results: Dict[str, Any], batch_stats: Dict[str, Any] = None):
        """
        Display ROUGE-C evaluation results in a formatted table.
        
        Args:
            results: Individual evaluation results
            batch_stats: Batch statistics (optional)
        """
        # Individual results table
        table = Table(title="ROUGE-C Evaluation Results - Legal Past Cases")
        table.add_column("Metric", style="cyan")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right") 
        table.add_column("F1-Score", justify="right")
        
        # ROUGE-L results
        rouge_l = results["rouge_l"]
        table.add_row("ROUGE-L", f"{rouge_l['precision']:.3f}", 
                     f"{rouge_l['recall']:.3f}", f"{rouge_l['f1']:.3f}")
        
        # Token overlap results
        token_overlap = results["token_overlap"]
        table.add_row("Token Overlap", f"{token_overlap['precision']:.3f}", 
                     f"{token_overlap['recall']:.3f}", f"{token_overlap['f1']:.3f}")
        
        # Legal term overlap results
        legal_overlap = results["legal_term_overlap"]
        table.add_row("Legal Terms", f"{legal_overlap['precision']:.3f}", 
                     f"{legal_overlap['recall']:.3f}", f"{legal_overlap['f1']:.3f}")
        
        console.print(table)
        
        # Semantic similarity table
        sim_table = Table(title="Semantic Similarity Scores")
        sim_table.add_column("Similarity Type", style="cyan")
        sim_table.add_column("Score", justify="right")
        
        semantic_sim = results["semantic_similarity"]
        sim_table.add_row("Combined Context", f"{semantic_sim['combined_similarity']:.3f}")
        sim_table.add_row("Maximum Individual", f"{semantic_sim['max_similarity']:.3f}")
        sim_table.add_row("Average Individual", f"{semantic_sim['avg_similarity']:.3f}")
        
        console.print(sim_table)
        
        # Overall score
        console.print(f"\n[bold green]Overall ROUGE-C Score: {results['overall_score']:.3f}[/bold green]")
        
        # Batch statistics if provided
        if batch_stats:
            self._display_batch_stats(batch_stats)
    
    def _display_batch_stats(self, batch_stats: Dict[str, Any]):
        """Display batch evaluation statistics."""
        stats_table = Table(title=f"Batch Statistics ({batch_stats['num_evaluations']} evaluations)")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Mean", justify="right")
        stats_table.add_column("Std Dev", justify="right")
        stats_table.add_column("Min", justify="right")
        stats_table.add_column("Max", justify="right")
        
        # Add rows for each metric
        stats_table.add_row("ROUGE-L F1", 
                           f"{batch_stats['rouge_l']['mean_f1']:.3f}",
                           f"{batch_stats['rouge_l']['std_f1']:.3f}",
                           f"{batch_stats['rouge_l']['min_f1']:.3f}",
                           f"{batch_stats['rouge_l']['max_f1']:.3f}")
        
        stats_table.add_row("Token Overlap F1",
                           f"{batch_stats['token_overlap']['mean_f1']:.3f}",
                           f"{batch_stats['token_overlap']['std_f1']:.3f}",
                           f"{batch_stats['token_overlap']['min_f1']:.3f}",
                           f"{batch_stats['token_overlap']['max_f1']:.3f}")
        
        stats_table.add_row("Legal Terms F1",
                           f"{batch_stats['legal_term_overlap']['mean_f1']:.3f}",
                           f"{batch_stats['legal_term_overlap']['std_f1']:.3f}",
                           f"{batch_stats['legal_term_overlap']['min_f1']:.3f}",
                           f"{batch_stats['legal_term_overlap']['max_f1']:.3f}")
        
        stats_table.add_row("Semantic Similarity",
                           f"{batch_stats['semantic_similarity']['mean']:.3f}",
                           f"{batch_stats['semantic_similarity']['std']:.3f}",
                           f"{batch_stats['semantic_similarity']['min']:.3f}",
                           f"{batch_stats['semantic_similarity']['max']:.3f}")
        
        stats_table.add_row("Overall Score",
                           f"{batch_stats['overall_score']['mean']:.3f}",
                           f"{batch_stats['overall_score']['std']:.3f}",
                           f"{batch_stats['overall_score']['min']:.3f}",
                           f"{batch_stats['overall_score']['max']:.3f}")
        
        console.print(stats_table)