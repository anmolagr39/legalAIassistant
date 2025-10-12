"""
SUPERT evaluation metric implementation for Legal RAG system.
Measures semantic similarity between generated answers and retrieved context.
"""

import numpy as np
import pandas as pd
import nltk
import warnings
import re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore')

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_packages = ['punkt', 'punkt_tab']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                pass  # Silent fail for missing packages

# Ensure NLTK data is available
ensure_nltk_data()

class SUPERTEvaluator:
    """
    SUPERT evaluation metric for Legal RAG systems.
    Measures semantic similarity between generated answers and retrieved context using
    optimal bipartite matching of sentence embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/distilroberta-base-msmarco-v2"):
        """Initialize SUPERT evaluator with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        print(f"✅ SUPERT evaluator initialized with model: {model_name}")
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize sentence text."""
        if not sentence or not isinstance(sentence, str):
            return ""
        # Lowercase and strip
        sentence = sentence.strip().lower()
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence)
        # Remove boilerplate or non-informative sentences
        if sentence in ["no context available", "see above", "n/a", "none"]:
            return ""
        # Filter out very short sentences (less than 3 words)
        if len(sentence.split()) < 3:
            return ""
        return sentence
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with fallback."""
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Try using NLTK sentence tokenizer
            sentences = nltk.sent_tokenize(text)
        except (LookupError, AttributeError):
            # Fallback to simple sentence splitting if NLTK is not available
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            clean_sent = self._clean_sentence(sent)
            if clean_sent:
                cleaned_sentences.append(clean_sent)
        
        return cleaned_sentences
    
    def _compute_similarity_matrix(self, answer_sentences: List[str], 
                                 context_sentences: List[str]) -> np.ndarray:
        """Compute cosine similarity matrix between answer and context sentences."""
        if not answer_sentences or not context_sentences:
            return np.array([])
        
        # Encode all sentences
        all_sentences = answer_sentences + context_sentences
        embeddings = self.model.encode(all_sentences, convert_to_tensor=False)
        
        # Split embeddings
        n_answer = len(answer_sentences)
        answer_embeddings = embeddings[:n_answer]
        context_embeddings = embeddings[n_answer:]
        
        # Compute cosine similarity
        similarity_matrix = np.dot(answer_embeddings, context_embeddings.T)
        
        # Normalize by vector norms for cosine similarity
        answer_norms = np.linalg.norm(answer_embeddings, axis=1, keepdims=True)
        context_norms = np.linalg.norm(context_embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        answer_norms = np.maximum(answer_norms, 1e-8)
        context_norms = np.maximum(context_norms, 1e-8)
        
        similarity_matrix = similarity_matrix / (answer_norms @ context_norms.T)
        
        return similarity_matrix
    
    def _optimal_matching(self, similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """Perform optimal bipartite matching using Hungarian algorithm."""
        if similarity_matrix.size == 0:
            return [], 0.0
        
        # Convert similarity to cost matrix (negative for minimization)
        cost_matrix = -similarity_matrix
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_indices, col_indices))
        
        # Calculate total similarity from optimal matching
        total_similarity = similarity_matrix[row_indices, col_indices].sum()
        
        return matched_pairs, total_similarity
    
    def evaluate_single(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate a single answer-context pair using SUPERT metric.
        
        Args:
            answer: Generated answer text
            context: Retrieved context text
            
        Returns:
            Dictionary containing SUPERT F1, precision, recall and other metrics
        """
        # Split into sentences
        answer_sentences = self._split_into_sentences(answer)
        context_sentences = self._split_into_sentences(context)
        
        # Handle empty cases
        if not answer_sentences or not context_sentences:
            return {
                'supert_f1': 0.0,
                'supert_precision': 0.0,
                'supert_recall': 0.0,
                'n_answer_sentences': len(answer_sentences),
                'n_context_sentences': len(context_sentences),
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'matched_pairs': 0
            }
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(answer_sentences, context_sentences)
        
        # Perform optimal matching
        matched_pairs, total_matched_similarity = self._optimal_matching(similarity_matrix)
        
        # Calculate metrics
        n_answer = len(answer_sentences)
        n_context = len(context_sentences)
        
        # SUPERT precision: how well answer sentences are supported by context
        precision = total_matched_similarity / n_answer if n_answer > 0 else 0.0
        
        # SUPERT recall: how much context is covered by answer
        recall = total_matched_similarity / n_context if n_context > 0 else 0.0
        
        # SUPERT F1: harmonic mean of precision and recall
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        # Additional statistics
        mean_similarity = similarity_matrix.mean() if similarity_matrix.size > 0 else 0.0
        max_similarity = similarity_matrix.max() if similarity_matrix.size > 0 else 0.0
        
        return {
            'supert_f1': f1_score,
            'supert_precision': precision,
            'supert_recall': recall,
            'n_answer_sentences': n_answer,
            'n_context_sentences': n_context,
            'mean_similarity': mean_similarity,
            'max_similarity': max_similarity,
            'matched_pairs': len(matched_pairs)
        }
    
    def evaluate_rag_response(self, rag_result: Dict[str, Any], vector_db=None) -> Dict[str, Any]:
        """
        Evaluate a RAG system response using SUPERT.
        Uses the exact context string passed to the LLM if available, otherwise reconstructs from sources.
        """
        answer = rag_result.get('answer', '')
        sources = rag_result.get('sources', [])
        
        # Try to use the exact context string if available (e.g., rag_result['context'])
        context = rag_result.get('context', None)
        
        if not context:
            # Need to reconstruct context from chunk IDs if vector_db is available
            context_parts = []
            if vector_db and sources:
                try:
                    # Extract chunk IDs from sources
                    chunk_ids = [source.get('chunk_id') for source in sources if isinstance(source, dict) and source.get('chunk_id')]
                    if chunk_ids:
                        # Query the vector database to get the actual content
                        documents = vector_db.collection.get(ids=chunk_ids, include=['documents'])
                        if documents and documents.get('documents'):
                            context_parts = documents['documents']
                except Exception as e:
                    print(f"Warning: Could not retrieve content from vector DB: {e}")
            
            # Fallback: try to extract from source content/document fields
            if not context_parts:
                for source in sources:
                    if isinstance(source, dict):
                        content = source.get('content', source.get('document', ''))
                        if content:
                            context_parts.append(content)
            
            context = ' '.join(context_parts) if context_parts else "No context available"
        
        # Evaluate using SUPERT
        evaluation = self.evaluate_single(answer, context)
        evaluation['query'] = rag_result.get('query', 'Unknown')
        evaluation['num_sources'] = len(sources)
        # Save context and answer for debugging
        evaluation['eval_context'] = context
        evaluation['eval_answer'] = answer
        return evaluation


def get_legal_test_queries() -> List[str]:
    """
    Get test queries for legal RAG evaluation.
    Optimized for Supreme Court cases - only queries relevant to the dataset.
    Irrelevant queries filtered out for agentic workflow specialization.
    """
    return [
        "What are the fundamental rights guaranteed under the Indian Constitution?",
        "What cases deal with property rights and land acquisition?",
        "Find cases related to criminal procedure and bail provisions",
        "What are the legal requirements for filing a petition in Supreme Court?",
        "What cases establish precedents for freedom of speech?",
        "Explain the principles of natural justice in administrative law",
        "What are the grounds for judicial review in Indian courts?",
        "Find cases related to constitutional interpretation and Article 14"
    ]


def evaluate_legal_rag_with_supert(rag_system, test_queries: Optional[List[str]] = None, 
                                  num_docs: int = 5) -> pd.DataFrame:
    """
    Evaluate Legal RAG system using SUPERT metric.
    
    Args:
        rag_system: Legal RAG system instance
        test_queries: List of test queries (optional)
        num_docs: Number of documents to retrieve per query
        
    Returns:
        DataFrame with evaluation results
    """
    if test_queries is None:
        test_queries = get_legal_test_queries()
    
    evaluator = SUPERTEvaluator()
    results = []
    print(f"🔍 Evaluating RAG system with {len(test_queries)} queries using SUPERT metric...")
    print("=" * 80)
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"📝 Query {i}/{len(test_queries)}: {query[:60]}...")
            rag_result = rag_system.answer_question(query, num_docs=num_docs)
            supert_result = evaluator.evaluate_rag_response(rag_result, rag_system.vector_db)
            supert_result['query_id'] = i
            supert_result['processing_time'] = rag_result.get('metadata', {}).get('processing_time_ms', 0)
            results.append(supert_result)
            # Print individual result
            print(f"   F1: {supert_result['supert_f1']:.3f} | "
                  f"P: {supert_result['supert_precision']:.3f} | "
                  f"R: {supert_result['supert_recall']:.3f}")
            # Warn if context or answer is too short
            if supert_result['n_context_sentences'] < 2 or supert_result['n_answer_sentences'] < 2:
                print(f"   ⚠️  Warning: Very short context or answer. SUPERT may be unreliable.")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'query_id': i,
                'query': query,
                'supert_f1': 0.0,
                'supert_precision': 0.0,
                'supert_recall': 0.0,
                'n_answer_sentences': 0,
                'n_context_sentences': 0,
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'matched_pairs': 0,
                'num_sources': 0,
                'processing_time': 0,
                'error': str(e),
                'eval_context': '',
                'eval_answer': ''
            })
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("📊 SUPERT EVALUATION SUMMARY")
    print("=" * 80)
    if len(df) > 0:
        print(f"Total Queries Evaluated: {len(df)}")
        print(f"Mean SUPERT F1 Score: {df['supert_f1'].mean():.4f} (±{df['supert_f1'].std():.4f})")
        print(f"Mean SUPERT Precision: {df['supert_precision'].mean():.4f} (±{df['supert_precision'].std():.4f})")
        print(f"Mean SUPERT Recall: {df['supert_recall'].mean():.4f} (±{df['supert_recall'].std():.4f})")
        print(f"Average Answer Length: {df['n_answer_sentences'].mean():.1f} sentences")
        print(f"Average Context Length: {df['n_context_sentences'].mean():.1f} sentences")
        print(f"Average Processing Time: {df['processing_time'].mean():.0f}ms")
        print(f"\n🏆 TOP 3 PERFORMING QUERIES (by F1):")
        top_queries = df.nlargest(3, 'supert_f1')[['query_id', 'supert_f1', 'supert_precision', 'supert_recall']]
        for _, row in top_queries.iterrows():
            print(f"   Query {row['query_id']}: F1={row['supert_f1']:.3f}, P={row['supert_precision']:.3f}, R={row['supert_recall']:.3f}")
        print(f"\n⚠️  LOWEST 3 PERFORMING QUERIES (by F1):")
        low_queries = df.nsmallest(3, 'supert_f1')[['query_id', 'supert_f1', 'supert_precision', 'supert_recall', 'eval_context', 'eval_answer']]
        for _, row in low_queries.iterrows():
            print(f"   Query {row['query_id']}: F1={row['supert_f1']:.3f}, P={row['supert_precision']:.3f}, R={row['supert_recall']:.3f}")
            print(f"      Context: {row['eval_context'][:200]}{'...' if len(row['eval_context']) > 200 else ''}")
            print(f"      Answer:  {row['eval_answer'][:200]}{'...' if len(row['eval_answer']) > 200 else ''}")
    print("=" * 80)
    return df
