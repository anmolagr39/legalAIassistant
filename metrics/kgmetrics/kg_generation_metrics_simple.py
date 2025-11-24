"""
Simplified Knowledge Graph Generation Metrics
Uses only ROUGE-L and LCS (Longest Common Subsequence)
Adapted from legal-rag-system/rouge-c/rouge_c.py
"""

import re
import numpy as np
from typing import List, Dict, Any

class KGGenerationMetrics:
    """
    Computes ROUGE-L and LCS metrics for generated text quality.
    Based on token-level overlap and longest common subsequence.
    """
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Initialize generation metrics evaluator.
        
        Args:
            remove_stopwords: Whether to remove stopwords from text
        """
        self.remove_stopwords = remove_stopwords
        
        # Common stopwords (simplified set)
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        print(f"✅ KG Generation Metrics initialized (stopwords: {remove_stopwords})")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, lowercasing, and optionally removing stopwords.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        # Basic tokenization and normalization
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Compute length of Longest Common Subsequence between two token sequences.
        Uses dynamic programming approach.
        
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
    
    def compute_rouge_l(self, generated_text: str, reference_text: str) -> Dict[str, float]:
        """
        Compute ROUGE-L scores between generated text and reference.
        ROUGE-L measures longest common subsequence overlap.
        
        Args:
            generated_text: Generated answer text
            reference_text: Reference text (e.g., retrieved context or expected answer)
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Preprocess texts
        gen_tokens = self._preprocess_text(generated_text)
        ref_tokens = self._preprocess_text(reference_text)
        
        if not gen_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Compute LCS length
        lcs_len = self._lcs_length(gen_tokens, ref_tokens)
        
        # Compute ROUGE-L metrics
        precision = lcs_len / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def compute_lcs_score(self, generated_text: str, reference_text: str) -> float:
        """
        Compute normalized LCS score between generated and reference text.
        Returns F1-score of ROUGE-L as a single metric.
        
        Args:
            generated_text: Generated answer text
            reference_text: Reference text
            
        Returns:
            LCS F1-score [0, 1]
        """
        rouge_l = self.compute_rouge_l(generated_text, reference_text)
        return rouge_l['f1']
    
    def evaluate(self, generated_text: str, reference_text: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated text.
        
        Args:
            generated_text: Generated answer text
            reference_text: Reference text (context or expected answer)
            
        Returns:
            Dictionary with ROUGE-L and LCS metrics
        """
        rouge_l = self.compute_rouge_l(generated_text, reference_text)
        lcs_score = rouge_l['f1']  # LCS score is the F1 from ROUGE-L
        
        return {
            'rouge_l_precision': rouge_l['precision'],
            'rouge_l_recall': rouge_l['recall'],
            'rouge_l_f1': rouge_l['f1'],
            'lcs_score': lcs_score
        }
    
    def batch_evaluate(self, text_pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate multiple generated-reference text pairs.
        
        Args:
            text_pairs: List of dicts with 'generated' and 'reference' keys
            
        Returns:
            Aggregate metrics and individual results
        """
        individual_results = []
        
        print(f"Evaluating {len(text_pairs)} text pairs...")
        
        for pair in text_pairs:
            generated = pair.get('generated', '')
            reference = pair.get('reference', '')
            
            metrics = self.evaluate(generated, reference)
            
            individual_results.append({
                'generated': generated[:100] + '...' if len(generated) > 100 else generated,
                'reference': reference[:100] + '...' if len(reference) > 100 else reference,
                'metrics': metrics
            })
        
        # Compute aggregate statistics
        all_metrics = [result['metrics'] for result in individual_results]
        
        # Safe aggregation with empty array checking
        rouge_l_precisions = [m['rouge_l_precision'] for m in all_metrics]
        rouge_l_recalls = [m['rouge_l_recall'] for m in all_metrics]
        rouge_l_f1s = [m['rouge_l_f1'] for m in all_metrics]
        lcs_scores = [m['lcs_score'] for m in all_metrics]
        
        aggregate_metrics = {
            'avg_rouge_l_precision': float(np.mean(rouge_l_precisions)) if rouge_l_precisions else 0.0,
            'std_rouge_l_precision': float(np.std(rouge_l_precisions)) if rouge_l_precisions else 0.0,
            'avg_rouge_l_recall': float(np.mean(rouge_l_recalls)) if rouge_l_recalls else 0.0,
            'std_rouge_l_recall': float(np.std(rouge_l_recalls)) if rouge_l_recalls else 0.0,
            'avg_rouge_l_f1': float(np.mean(rouge_l_f1s)) if rouge_l_f1s else 0.0,
            'std_rouge_l_f1': float(np.std(rouge_l_f1s)) if rouge_l_f1s else 0.0,
            'avg_lcs_score': float(np.mean(lcs_scores)) if lcs_scores else 0.0,
            'std_lcs_score': float(np.std(lcs_scores)) if lcs_scores else 0.0
        }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'individual_results': individual_results,
            'total_pairs': len(text_pairs)
        }
