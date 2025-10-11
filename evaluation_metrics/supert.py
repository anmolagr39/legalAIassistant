import numpy as np
import pandas as pd
import nltk
import warnings
import re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# Download required NLTK data - handle both old and new versions
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_packages = ['punkt', 'punkt_tab']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"✅ Found NLTK {package}")
        except LookupError:
            try:
                print(f"📥 Downloading NLTK {package}...")
                nltk.download(package, quiet=True)
                print(f"✅ Successfully downloaded NLTK {package}")
            except Exception as e:
                print(f"⚠️ Could not download {package}: {e}")

# Call the function to ensure NLTK data
ensure_nltk_data()

class SUPERTEvaluator:
    """
    SUPERT evaluation metric for RAG systems.
    Measures semantic similarity between generated answers and retrieved context.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize SUPERT evaluator with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        print(f"✅ SUPERT evaluator initialized with model: {model_name}")
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize sentence text."""
        sentence = re.sub(r'\s+', ' ', sentence.strip())
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
        except LookupError:
            # Fallback to simple sentence splitting if NLTK data is not available
            print("⚠️ NLTK punkt tokenizer not available, using simple sentence splitting")
            # Simple sentence splitting based on periods, exclamation marks, question marks
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        cleaned_sentences = []
        for sent in sentences:
            clean_sent = self._clean_sentence(sent)
            if clean_sent:
                cleaned_sentences.append(clean_sent)
        
        return cleaned_sentences
    
    def _compute_similarity_matrix(self, answer_sentences: List[str], 
                                 context_sentences: List[str]) -> np.ndarray:
        """Compute sentence-to-sentence similarity matrix."""
        if not answer_sentences or not context_sentences:
            return np.array([])
        
        all_sentences = answer_sentences + context_sentences
        embeddings = self.model.encode(all_sentences, convert_to_tensor=False)
        
        n_answer = len(answer_sentences)
        answer_embeddings = embeddings[:n_answer]
        context_embeddings = embeddings[n_answer:]
        
        similarity_matrix = np.dot(answer_embeddings, context_embeddings.T)
        
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
        
        cost_matrix = -similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_indices, col_indices))
        total_similarity = similarity_matrix[row_indices, col_indices].sum()
        
        return matched_pairs, total_similarity
    
    def evaluate_single(self, answer: str, context: str) -> Dict[str, Any]:
        """Evaluate a single answer-context pair."""
        answer_sentences = self._split_into_sentences(answer)
        context_sentences = self._split_into_sentences(context)
        
        if not answer_sentences or not context_sentences:
            return {
                'supert_f1': 0.0,
                'supert_precision': 0.0,
                'supert_recall': 0.0,
                'n_answer_sentences': len(answer_sentences),
                'n_context_sentences': len(context_sentences),
                'mean_similarity': 0.0,
                'max_similarity': 0.0,
                'matched_pairs': 0,
                'total_matched_similarity': 0.0
            }
        
        similarity_matrix = self._compute_similarity_matrix(answer_sentences, context_sentences)
        matched_pairs, total_matched_similarity = self._optimal_matching(similarity_matrix)
        
        n_answer = len(answer_sentences)
        n_context = len(context_sentences)
        
        precision_like = total_matched_similarity / n_answer if n_answer > 0 else 0.0
        recall_like = total_matched_similarity / n_context if n_context > 0 else 0.0
        
        if precision_like + recall_like > 0:
            f1_score = 2 * precision_like * recall_like / (precision_like + recall_like)
        else:
            f1_score = 0.0
        
        mean_similarity = similarity_matrix.mean() if similarity_matrix.size > 0 else 0.0
        max_similarity = similarity_matrix.max() if similarity_matrix.size > 0 else 0.0
        
        return {
            'supert_f1': f1_score,
            'supert_precision': precision_like,
            'supert_recall': recall_like,
            'n_answer_sentences': n_answer,
            'n_context_sentences': n_context,
            'mean_similarity': mean_similarity,
            'max_similarity': max_similarity,
            'matched_pairs': len(matched_pairs),
            'total_matched_similarity': total_matched_similarity
        }
    
    def evaluate_batch(self, examples: List[Dict[str, str]], 
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """Evaluate a batch of answer-context pairs."""
        results = []
        
        print(f"🔍 Evaluating {len(examples)} examples with SUPERT metric...")
        
        for i, example in enumerate(examples):
            answer = example.get('answer', '')
            context = example.get('context', '')
            query = example.get('query', f'Example_{i}')
            
            try:
                result = self.evaluate_single(answer, context)
                result['query'] = query
                result['example_id'] = i
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error evaluating example {i}: {e}")
                # Add a default result for failed examples
                results.append({
                    'supert_f1': 0.0,
                    'supert_precision': 0.0,
                    'supert_recall': 0.0,
                    'query': query,
                    'example_id': i,
                    'n_answer_sentences': 0,
                    'n_context_sentences': 0,
                    'mean_similarity': 0.0,
                    'max_similarity': 0.0,
                    'matched_pairs': 0,
                    'total_matched_similarity': 0.0
                })
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {i + 1}/{len(examples)} examples...")
        
        df = pd.DataFrame(results)
        
        column_order = [
            'example_id', 'query', 'supert_f1', 'supert_precision', 'supert_recall',
            'n_answer_sentences', 'n_context_sentences', 'mean_similarity', 
            'max_similarity', 'matched_pairs', 'total_matched_similarity'
        ]
        df = df[column_order]
        
        if output_path:
            try:
                df.to_csv(output_path, index=False)
                print(f"✅ Results saved to: {output_path}")
            except Exception as e:
                print(f"⚠️ Could not save results to CSV: {e}")
        
        self._print_summary_stats(df)
        return df
    
    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("📊 SUPERT EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Examples: {len(df)}")
        
        if len(df) > 0:
            print(f"Mean SUPERT F1: {df['supert_f1'].mean():.4f} (±{df['supert_f1'].std():.4f})")
            print(f"Mean Precision: {df['supert_precision'].mean():.4f} (±{df['supert_precision'].std():.4f})")
            print(f"Mean Recall: {df['supert_recall'].mean():.4f} (±{df['supert_recall'].std():.4f})")
            print(f"Mean Answer Length: {df['n_answer_sentences'].mean():.1f} sentences")
            print(f"Mean Context Length: {df['n_context_sentences'].mean():.1f} sentences")
            print(f"Mean Similarity: {df['mean_similarity'].mean():.4f}")
        else:
            print("No results to display")
        
        print("="*60)


def create_evaluation_examples_from_rag(rag_engine, queries: List[str]) -> List[Dict[str, str]]:
    """Create evaluation examples by running queries through RAG system."""
    examples = []
    
    print(f"🔄 Processing {len(queries)} queries through RAG system...")
    
    for i, query in enumerate(queries):
        try:
            print(f"   Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            # Run query through RAG system
            result = rag_engine.query(query)
            answer = result.get('answer', '')
            
            # Get context from retrieved documents
            retrieval_result = rag_engine.retrieve_relevant_cases(query)
            context_docs = retrieval_result.get('documents', [[]])[0]
            context = ' '.join(context_docs[:3]) if context_docs else "No context retrieved"
            
            examples.append({
                'query': query,
                'answer': answer,
                'context': context
            })
            
        except Exception as e:
            print(f"❌ Error processing query '{query[:30]}...': {e}")
            examples.append({
                'query': query,
                'answer': f"Error: {str(e)}",
                'context': "No context due to error"
            })
    
    print(f"✅ Successfully processed {len(examples)} examples")
    return examples


def get_test_queries() -> List[str]:
    """Get test queries for FIR legal analysis."""
    return [
        "What are the IPC sections for cheating and fraud?",
        "Explain section 420 of Indian Penal Code",
        "Cases related to cybercrime and online fraud", 
        "IPC sections for theft and burglary",
        "What is criminal breach of trust under IPC?",
        "Forgery related sections in Indian Penal Code",
        "Cases involving domestic violence",
        "IPC sections for assault and battery",
        "What are the penalties for dowry harassment?",
        "Criminal conspiracy cases under IPC"
    ]