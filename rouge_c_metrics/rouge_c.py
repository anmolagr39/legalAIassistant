"""
ROUGE-C: Reference-free Context-based ROUGE metric for IPC RAG systems.
Measures quality of generated text against retrieved context without requiring reference answers.

Based on: Lin, 'ROUGE: A Package for Automatic Evaluation of Summaries', ACL 2004 
(adapted for reference-free evaluation)
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class RougeC:
    """
    ROUGE-C: Context-based reference-free ROUGE metric.
    Evaluates generated text quality against retrieved context using:
    1. ROUGE-L: Longest Common Subsequence overlap
    2. Semantic similarity: Cosine similarity between embeddings
    3. Token overlap: Precision, recall, and F1 of token matches
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 use_stemming: bool = False, remove_stopwords: bool = True):
        """
        Initialize ROUGE-C evaluator.
        
        Args:
            embedding_model: Model for semantic similarity computation
            use_stemming: Whether to apply stemming (requires nltk)
            remove_stopwords: Whether to remove stopwords
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        
        # Common stopwords (simplified set)
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        print(f"✅ ROUGE-C initialized with embedding model: {embedding_model}")
        
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
                print("🔤 Stemming enabled")
            except ImportError:
                print("⚠️ Warning: NLTK not available, stemming disabled")
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
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
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
    
    def compute_rouge_l(self, generated_text: str, context_text: str) -> Dict[str, float]:
        """
        Compute ROUGE-L scores between generated text and context.
        
        Args:
            generated_text: Generated answer text
            context_text: Retrieved context text
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Preprocess texts
        gen_tokens = self._preprocess_text(generated_text)
        ctx_tokens = self._preprocess_text(context_text)
        
        if not gen_tokens or not ctx_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Compute LCS length
        lcs_len = self._lcs_length(gen_tokens, ctx_tokens)
        
        # Compute ROUGE-L metrics
        precision = lcs_len / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
        recall = lcs_len / len(ctx_tokens) if len(ctx_tokens) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall, 
            "f1": f1
        }
    
    def compute_token_overlap(self, generated_text: str, context_text: str) -> Dict[str, float]:
        """
        Compute token-level overlap metrics between generated text and context.
        
        Args:
            generated_text: Generated answer text
            context_text: Retrieved context text
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Preprocess texts
        gen_tokens = self._preprocess_text(generated_text)
        ctx_tokens = self._preprocess_text(context_text)
        
        if not gen_tokens or not ctx_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Convert to sets for overlap computation
        gen_set = set(gen_tokens)
        ctx_set = set(ctx_tokens)
        
        # Compute overlap
        overlap = len(gen_set & ctx_set)
        
        # Compute metrics
        precision = overlap / len(gen_set) if len(gen_set) > 0 else 0.0
        recall = overlap / len(ctx_set) if len(ctx_set) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def compute_semantic_similarity(self, generated_text: str, context_text: str) -> float:
        """
        Compute semantic similarity between generated text and context using embeddings.
        
        Args:
            generated_text: Generated answer text
            context_text: Retrieved context text
            
        Returns:
            Cosine similarity score [0, 1]
        """
        if not generated_text.strip() or not context_text.strip():
            return 0.0
        
        # Encode texts
        gen_embedding = self.embedding_model.encode([generated_text])
        ctx_embedding = self.embedding_model.encode([context_text])
        
        # Compute cosine similarity
        similarity = cosine_similarity(gen_embedding, ctx_embedding)[0][0]
        
        # Normalize to [0, 1] range
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))
        
        return similarity
    
    def compute_rouge_c_single(self, generated_text: str, retrieved_contexts: List[str]) -> Dict[str, Any]:
        """
        Compute ROUGE-C metrics for a single generated text against retrieved contexts.
        
        Args:
            generated_text: Generated answer text
            retrieved_contexts: List of retrieved context texts
            
        Returns:
            Dictionary with all ROUGE-C metrics
        """
        if not retrieved_contexts or not generated_text.strip():
            return {
                "rouge_l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "token_overlap": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "semantic_similarity": 0.0,
                "num_contexts": 0
            }
        
        # Combine all contexts into single text
        combined_context = " ".join(retrieved_contexts)
        
        # Compute different ROUGE-C variants
        rouge_l = self.compute_rouge_l(generated_text, combined_context)
        token_overlap = self.compute_token_overlap(generated_text, combined_context)
        semantic_sim = self.compute_semantic_similarity(generated_text, combined_context)
        
        # Also compute average semantic similarity with individual contexts
        individual_sims = []
        for context in retrieved_contexts:
            sim = self.compute_semantic_similarity(generated_text, context)
            individual_sims.append(sim)
        
        avg_individual_sim = np.mean(individual_sims) if individual_sims else 0.0
        max_individual_sim = np.max(individual_sims) if individual_sims else 0.0
        
        return {
            "rouge_l": rouge_l,
            "token_overlap": token_overlap,
            "semantic_similarity": semantic_sim,
            "avg_context_similarity": avg_individual_sim,
            "max_context_similarity": max_individual_sim,
            "num_contexts": len(retrieved_contexts)
        }
    
    def compute_rouge_c_batch(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute ROUGE-C metrics for multiple question-answer pairs.
        
        Args:
            qa_pairs: List of dicts with 'question', 'generated_answer', 'retrieved_contexts'
            
        Returns:
            Aggregate ROUGE-C metrics
        """
        if not qa_pairs:
            return {}
        
        all_results = []
        
        print(f"🔍 Computing ROUGE-C metrics for {len(qa_pairs)} Q-A pairs...")
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            generated_text = qa_pair.get('generated_answer', '')
            contexts = qa_pair.get('retrieved_contexts', [])
            
            result = self.compute_rouge_c_single(generated_text, contexts)
            result['question'] = qa_pair.get('question', '')
            
            all_results.append(result)
            
            if i % 2 == 0:  # Progress indicator
                print(f"   Processed {i}/{len(qa_pairs)} pairs...")
        
        # Compute aggregate statistics
        return self._compute_aggregate_stats(all_results)
    
    def _compute_aggregate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics from individual results."""
        if not results:
            return {}
        
        # Extract metrics for aggregation
        rouge_l_precision = [r['rouge_l']['precision'] for r in results]
        rouge_l_recall = [r['rouge_l']['recall'] for r in results]
        rouge_l_f1 = [r['rouge_l']['f1'] for r in results]
        
        token_precision = [r['token_overlap']['precision'] for r in results]
        token_recall = [r['token_overlap']['recall'] for r in results]
        token_f1 = [r['token_overlap']['f1'] for r in results]
        
        semantic_sims = [r['semantic_similarity'] for r in results]
        avg_context_sims = [r['avg_context_similarity'] for r in results]
        max_context_sims = [r['max_context_similarity'] for r in results]
        
        # Compute statistics
        aggregate_metrics = {
            'rouge_l': {
                'avg_precision': np.mean(rouge_l_precision),
                'std_precision': np.std(rouge_l_precision),
                'avg_recall': np.mean(rouge_l_recall),
                'std_recall': np.std(rouge_l_recall),
                'avg_f1': np.mean(rouge_l_f1),
                'std_f1': np.std(rouge_l_f1)
            },
            'token_overlap': {
                'avg_precision': np.mean(token_precision),
                'std_precision': np.std(token_precision),
                'avg_recall': np.mean(token_recall),
                'std_recall': np.std(token_recall),
                'avg_f1': np.mean(token_f1),
                'std_f1': np.std(token_f1)
            },
            'semantic_similarity': {
                'avg_combined': np.mean(semantic_sims),
                'std_combined': np.std(semantic_sims),
                'avg_individual': np.mean(avg_context_sims),
                'std_individual': np.std(avg_context_sims),
                'avg_max': np.mean(max_context_sims),
                'std_max': np.std(max_context_sims)
            },
            'summary': {
                'total_pairs': len(results),
                'avg_contexts_per_query': np.mean([r['num_contexts'] for r in results])
            }
        }
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'individual_results': results
        }
    
    def display_results(self, results: Dict[str, Any]):
        """
        Display ROUGE-C results in formatted output.
        
        Args:
            results: Results from compute_rouge_c_batch or single evaluation
        """
        if 'aggregate_metrics' in results:
            # Display aggregate results
            self._display_aggregate_results(results['aggregate_metrics'])
        else:
            # Display single result
            self._display_single_result(results)
    
    def _display_aggregate_results(self, metrics: Dict[str, Any]):
        """Display aggregate ROUGE-C results."""
        print("\n" + "="*80)
        print("🎯 ROUGE-C AGGREGATE RESULTS")
        print("="*80)
        
        # ROUGE-L metrics
        print("\n📊 ROUGE-L METRICS:")
        rouge_l = metrics['rouge_l']
        print(f"   Precision: {rouge_l['avg_precision']:.3f} ± {rouge_l['std_precision']:.3f}")
        print(f"   Recall:    {rouge_l['avg_recall']:.3f} ± {rouge_l['std_recall']:.3f}")
        print(f"   F1-Score:  {rouge_l['avg_f1']:.3f} ± {rouge_l['std_f1']:.3f}")
        
        # Token overlap metrics
        print("\n🔤 TOKEN OVERLAP METRICS:")
        token = metrics['token_overlap']
        print(f"   Precision: {token['avg_precision']:.3f} ± {token['std_precision']:.3f}")
        print(f"   Recall:    {token['avg_recall']:.3f} ± {token['std_recall']:.3f}")
        print(f"   F1-Score:  {token['avg_f1']:.3f} ± {token['std_f1']:.3f}")
        
        # Semantic similarity metrics
        print("\n🧠 SEMANTIC SIMILARITY METRICS:")
        sem = metrics['semantic_similarity']
        print(f"   Combined Context:   {sem['avg_combined']:.3f} ± {sem['std_combined']:.3f}")
        print(f"   Avg Individual:     {sem['avg_individual']:.3f} ± {sem['std_individual']:.3f}")
        print(f"   Max Individual:     {sem['avg_max']:.3f} ± {sem['std_max']:.3f}")
        
        # Summary
        summary = metrics['summary']
        print(f"\n📈 SUMMARY:")
        print(f"   Total Q-A pairs:          {summary['total_pairs']}")
        print(f"   Avg contexts per query:   {summary['avg_contexts_per_query']:.1f}")
        
        # Performance assessment
        self._display_performance_assessment(metrics)
    
    def _display_single_result(self, result: Dict[str, Any]):
        """Display single ROUGE-C result."""
        print("\n" + "="*60)
        print("🎯 ROUGE-C RESULTS")
        print("="*60)
        
        # ROUGE-L metrics
        rouge_l = result['rouge_l']
        print(f"\n📊 ROUGE-L:")
        print(f"   Precision: {rouge_l['precision']:.3f}")
        print(f"   Recall:    {rouge_l['recall']:.3f}")
        print(f"   F1-Score:  {rouge_l['f1']:.3f}")
        
        # Token overlap metrics
        token = result['token_overlap']
        print(f"\n🔤 TOKEN OVERLAP:")
        print(f"   Precision: {token['precision']:.3f}")
        print(f"   Recall:    {token['recall']:.3f}")
        print(f"   F1-Score:  {token['f1']:.3f}")
        
        # Semantic similarity
        print(f"\n🧠 SEMANTIC SIMILARITY:")
        print(f"   Combined Context: {result['semantic_similarity']:.3f}")
        print(f"   Avg Individual:   {result['avg_context_similarity']:.3f}")
        print(f"   Max Individual:   {result['max_context_similarity']:.3f}")
        print(f"   Contexts Used:    {result['num_contexts']}")
    
    def _display_performance_assessment(self, metrics: Dict[str, Any]):
        """Display performance assessment."""
        rouge_l_f1 = metrics['rouge_l']['avg_f1']
        token_f1 = metrics['token_overlap']['avg_f1']
        sem_combined = metrics['semantic_similarity']['avg_combined']
        overall_score = (rouge_l_f1 + token_f1 + sem_combined) / 3
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        
        # Individual assessments
        print(f"   ROUGE-L F1: {rouge_l_f1:.3f} - {'🟢 High' if rouge_l_f1 > 0.5 else '🟡 Moderate' if rouge_l_f1 > 0.3 else '🔴 Low'} sequential overlap")
        print(f"   Token F1:   {token_f1:.3f} - {'🟢 High' if token_f1 > 0.4 else '🟡 Moderate' if token_f1 > 0.2 else '🔴 Low'} lexical overlap")
        print(f"   Semantic:   {sem_combined:.3f} - {'🟢 High' if sem_combined > 0.7 else '🟡 Moderate' if sem_combined > 0.5 else '🔴 Low'} semantic alignment")
        
        print(f"\n🏆 OVERALL SCORE: {overall_score:.3f}")
        
        if overall_score > 0.6:
            print("   ✅ EXCELLENT: Answers are well-grounded in retrieved contexts")
        elif overall_score > 0.4:
            print("   🟡 GOOD: Answers are reasonably grounded in contexts")  
        elif overall_score > 0.2:
            print("   🟠 FAIR: Answers have some grounding in contexts")
        else:
            print("   🔴 POOR: Answers are poorly grounded in contexts")
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save ROUGE-C results to JSON file.
        
        Args:
            results: ROUGE-C evaluation results
            filepath: Path to save results
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"💾 ROUGE-C results saved to: {filepath}")