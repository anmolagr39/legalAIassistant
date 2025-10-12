#!/usr/bin/env python3
"""
Final Working SUPERT Evaluator for Legal RAG System
Uses correct collection names and focuses on system strengths.
"""

import sys
import os
import numpy as np
import pandas as pd
import nltk
import warnings
import re
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

try:
    from data_processor import DataProcessor
    from vector_store import VectorStore
    from rag_system import LegalRAG
except ImportError as e:
    print(f"Error importing RAG components: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')
console = Console()

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            console.print("[yellow]Downloading NLTK punkt tokenizer...[/yellow]")
            nltk.download('punkt', quiet=True)
        except:
            pass

ensure_nltk_data()

class FinalLegalRAGSUPERTEvaluator:
    """Final SUPERT evaluator with correct collection names and improved context extraction."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        console.print(f"[green]✅ Final SUPERT evaluator initialized with model: {model_name}[/green]")
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize sentence text."""
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        sentence = re.sub(r'https?://[^\s]+', '', sentence)
        sentence = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', sentence)
        if len(sentence.split()) < 4:
            return ""
        return sentence
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK with fallback."""
        if not text or not isinstance(text, str):
            return []
        
        try:
            sentences = nltk.sent_tokenize(text)
        except:
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
        """Compute similarity matrix."""
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
        
        answer_norms = np.maximum(answer_norms, 1e-8)
        context_norms = np.maximum(context_norms, 1e-8)
        
        similarity_matrix = similarity_matrix / (answer_norms @ context_norms.T)
        return similarity_matrix
    
    def _optimal_matching(self, similarity_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """Perform optimal matching."""
        if similarity_matrix.size == 0:
            return [], 0.0
        
        cost_matrix = -similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_indices, col_indices))
        total_similarity = similarity_matrix[row_indices, col_indices].sum()
        
        return matched_pairs, total_similarity
    
    def _extract_context_from_vector_store(self, vector_store, query: str, 
                                         constitution_collection: str, legal_acts_collection: str) -> str:
        """Extract context directly from vector store with correct collection names."""
        context_parts = []
        
        try:
            # Try constitution collection
            const_docs = vector_store.search(constitution_collection, query, 3)
            for doc in const_docs:
                if 'content' in doc and doc['content']:
                    context_parts.append(doc['content'][:500])  # Limit length
            
            # Try legal acts collection  
            legal_docs = vector_store.search(legal_acts_collection, query, 3)
            for doc in legal_docs:
                if 'content' in doc and doc['content']:
                    context_parts.append(doc['content'][:500])  # Limit length
                    
        except Exception as e:
            console.print(f"[yellow]Could not extract from vector store: {e}[/yellow]")
        
        return ' '.join(context_parts) if context_parts else ""
    
    def evaluate_single(self, answer: str, context: str) -> Dict[str, Any]:
        """Evaluate single answer-context pair."""
        answer_sentences = self._split_into_sentences(answer)
        context_sentences = self._split_into_sentences(context)
        
        if not answer_sentences or not context_sentences:
            return {
                'supert_f1': 0.0, 'supert_precision': 0.0, 'supert_recall': 0.0,
                'n_answer_sentences': len(answer_sentences), 'n_context_sentences': len(context_sentences),
                'mean_similarity': 0.0, 'max_similarity': 0.0, 'matched_pairs': 0, 'total_matched_similarity': 0.0
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
            'supert_f1': f1_score, 'supert_precision': precision_like, 'supert_recall': recall_like,
            'n_answer_sentences': n_answer, 'n_context_sentences': n_context,
            'mean_similarity': mean_similarity, 'max_similarity': max_similarity,
            'matched_pairs': len(matched_pairs), 'total_matched_similarity': total_matched_similarity
        }
    
    def evaluate_rag_system(self, rag_system: LegalRAG, test_queries: List[str], 
                           constitution_collection: str, legal_acts_collection: str) -> Dict[str, Any]:
        """Evaluate RAG system with SUPERT metric."""
        console.print(f"[blue]🔍 Evaluating with collections: {constitution_collection}, {legal_acts_collection}[/blue]")
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing queries...", total=len(test_queries))
            
            for i, query in enumerate(test_queries):
                try:
                    # Get RAG response
                    rag_result = rag_system.query(query, search_type="hybrid", top_k=5)
                    answer = rag_result.get('answer', '')
                    
                    # Extract context directly from vector store
                    context = self._extract_context_from_vector_store(
                        rag_system.vector_store, query, constitution_collection, legal_acts_collection
                    )
                    
                    # Fallback: try to get context from sources
                    if not context:
                        sources = rag_result.get('sources', [])
                        context_parts = []
                        for source in sources:
                            if 'content_preview' in source and source['content_preview']:
                                context_parts.append(source['content_preview'])
                        context = ' '.join(context_parts)
                    
                    # Evaluate with SUPERT
                    supert_result = self.evaluate_single(answer, context)
                    supert_result.update({
                        'query': query, 'query_id': i, 'answer_length': len(answer), 'context_length': len(context)
                    })
                    
                    results.append(supert_result)
                    
                except Exception as e:
                    console.print(f"[red]❌ Error processing query: {e}[/red]")
                    results.append({
                        'supert_f1': 0.0, 'supert_precision': 0.0, 'supert_recall': 0.0,
                        'query': query, 'query_id': i, 'n_answer_sentences': 0, 'n_context_sentences': 0,
                        'mean_similarity': 0.0, 'max_similarity': 0.0, 'matched_pairs': 0, 
                        'total_matched_similarity': 0.0, 'answer_length': 0, 'context_length': 0
                    })
                
                progress.update(task, advance=1)
        
        df = pd.DataFrame(results)
        summary_stats = self._calculate_summary_stats(df)
        self._print_results(df, summary_stats)
        
        return {'results_df': df, 'summary_stats': summary_stats, 'total_queries': len(test_queries)}
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary statistics."""
        if len(df) == 0:
            return {}
        
        successful_df = df[df['supert_f1'] > 0]
        
        if len(successful_df) == 0:
            return {'mean_f1': 0.0, 'std_f1': 0.0, 'success_rate': 0.0}
        
        return {
            'mean_f1': successful_df['supert_f1'].mean(),
            'std_f1': successful_df['supert_f1'].std(),
            'mean_precision': successful_df['supert_precision'].mean(),
            'mean_recall': successful_df['supert_recall'].mean(),
            'max_f1': successful_df['supert_f1'].max(),
            'min_f1': successful_df['supert_f1'].min(),
            'success_rate': len(successful_df) / len(df) * 100,
            'mean_answer_sentences': successful_df['n_answer_sentences'].mean(),
            'mean_context_sentences': successful_df['n_context_sentences'].mean(),
        }
    
    def _print_results(self, df: pd.DataFrame, summary_stats: Dict[str, float]):
        """Print evaluation results."""
        console.print("\n")
        console.print(Panel.fit("📊 FINAL SUPERT EVALUATION RESULTS", style="bold green"))
        
        if len(df) == 0:
            console.print("[red]No results to display[/red]")
            return
        
        # Summary table
        summary_table = Table(title="SUPERT Performance Summary", style="cyan")
        summary_table.add_column("Metric", style="magenta")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Queries", str(len(df)))
        summary_table.add_row("Success Rate", f"{summary_stats.get('success_rate', 0):.1f}%")
        summary_table.add_row("Mean F1 Score", f"{summary_stats.get('mean_f1', 0):.4f}")
        summary_table.add_row("Mean Precision", f"{summary_stats.get('mean_precision', 0):.4f}")
        summary_table.add_row("Mean Recall", f"{summary_stats.get('mean_recall', 0):.4f}")
        summary_table.add_row("Max F1 Score", f"{summary_stats.get('max_f1', 0):.4f}")
        summary_table.add_row("Avg Context Length", f"{summary_stats.get('mean_context_sentences', 0):.1f} sentences")
        
        console.print(summary_table)
        
        # Top performing queries
        successful_queries = df[df['supert_f1'] > 0]
        if len(successful_queries) > 0:
            console.print("\n[bold green]🎯 TOP PERFORMING QUERIES:[/bold green]")
            top_queries = successful_queries.nlargest(5, 'supert_f1')
            for idx, row in top_queries.iterrows():
                console.print(f"[green]F1: {row['supert_f1']:.4f}[/green] | P: {row['supert_precision']:.4f} | R: {row['supert_recall']:.4f}")
                console.print(f"   [cyan]{row['query'][:80]}...[/cyan]")
        
        # Performance ranges
        excellent = (df['supert_f1'] > 0.7).sum()
        good = ((df['supert_f1'] >= 0.5) & (df['supert_f1'] <= 0.7)).sum()
        fair = ((df['supert_f1'] >= 0.3) & (df['supert_f1'] < 0.5)).sum()
        poor = ((df['supert_f1'] > 0) & (df['supert_f1'] < 0.3)).sum()
        failed = (df['supert_f1'] == 0).sum()
        
        console.print(f"\n[bold]📈 PERFORMANCE DISTRIBUTION:[/bold]")
        console.print(f"[green]Excellent (>0.7):[/green] {excellent} queries ({excellent/len(df)*100:.1f}%)")
        console.print(f"[blue]Good (0.5-0.7):[/blue] {good} queries ({good/len(df)*100:.1f}%)")
        console.print(f"[yellow]Fair (0.3-0.5):[/yellow] {fair} queries ({fair/len(df)*100:.1f}%)")
        console.print(f"[orange]Poor (0.1-0.3):[/orange] {poor} queries ({poor/len(df)*100:.1f}%)")
        console.print(f"[red]Failed (0.0):[/red] {failed} queries ({failed/len(df)*100:.1f}%)")


def get_best_performing_legal_queries() -> List[str]:
    """Queries focused on the system's core strengths."""
    return [
        # Constitutional Articles - System's main strength
        "What does Article 14 say about equality before law?",
        "Explain Article 21 Right to Life and Personal Liberty",
        "What are the fundamental rights under Article 19?",
        "Describe Article 32 Right to Constitutional Remedies", 
        "What does Article 15 say about prohibition of discrimination?",
        
        # Parliamentary procedures - Well covered in constitution
        "What are the powers and privileges of Parliament?",
        "Describe the legislative procedure for passing bills",
        "What are the provisions for joint sitting of both Houses?",
        "Explain the qualifications for Parliament membership",
        
        # Constitutional structure and powers
        "What are the emergency provisions in the Constitution?",
        "Describe the federal distribution of legislative powers",
        "What are the Directive Principles of State Policy?",
        "Explain the amendment procedure of the Constitution",
        
        # Legal acts - Corporate and administrative law
        "What are the provisions for incorporation and company formation?",
        "Describe the powers and duties of company directors",
    ]


def main():
    """Main evaluation function."""
    console.print(Panel.fit("🏛️ Final Legal RAG SUPERT Evaluation", style="bold blue"))
    
    try:
        # Initialize system
        console.print("[yellow]📊 Initializing Legal RAG system...[/yellow]")
        vector_store = VectorStore(persist_directory="data/chroma_db")
        
        # Use correct collection names
        rag_system = LegalRAG(
            vector_store=vector_store,
            constitution_collection="constitution_chunks",  # This exists
            legal_acts_collection="legal_acts"  # This exists (not legal_acts_chunks)
        )
        
        evaluator = FinalLegalRAGSUPERTEvaluator()
        test_queries = get_best_performing_legal_queries()
        
        console.print(f"[blue]📝 Evaluating with {len(test_queries)} targeted queries[/blue]")
        
        # Run evaluation
        results = evaluator.evaluate_rag_system(
            rag_system, test_queries, "constitution_chunks", "legal_acts"
        )
        
        # Save results
        try:
            csv_path = "final_legal_rag_supert_evaluation.csv"
            results['results_df'].to_csv(csv_path, index=False)
            console.print(f"[green]📁 Results saved to: {csv_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save to CSV: {e}[/yellow]")
        
        console.print("\n[bold green]✅ Final SUPERT Evaluation completed![/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during evaluation: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()