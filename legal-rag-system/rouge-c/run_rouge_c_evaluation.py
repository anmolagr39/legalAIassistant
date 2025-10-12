"""
Simple ROUGE-C evaluation script for Legal RAG system.
Automatically runs ROUGE-C evaluation on 10 legal queries.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import LegalRAG
from src.vector_store import VectorStore
from rouge_c import RougeC
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def get_legal_queries() -> List[str]:
    """Return 10 legal queries for evaluation."""
    return [
        "What are the fundamental rights in the Constitution?",
        "How can constitutional amendments be made?",
        "What is the procedure for impeachment of the President?",
        "What are the powers of the Supreme Court?",
        "What is the role of the Election Commission?",
        "How are states reorganized under the Constitution?",
        "What are the directive principles of state policy?",
        "What is the procedure for declaring emergency?",
        "What are the qualifications for becoming a Member of Parliament?",
        "How is the Chief Justice of India appointed?"
    ]

def initialize_system():
    """Initialize RAG system and ROUGE-C evaluator."""
    try:
        # Initialize vector store
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
        vector_store = VectorStore(persist_directory=data_path)
        
        # Initialize RAG system
        rag_system = LegalRAG(
            vector_store=vector_store,
            constitution_collection="constitution_articles",
            legal_acts_collection="legal_acts"
        )
        
        # Initialize ROUGE-C evaluator
        rouge_c_evaluator = RougeC(
            embedding_model="all-MiniLM-L6-v2",
            use_stemming=False,
            remove_stopwords=True
        )
        
        return rag_system, rouge_c_evaluator
        
    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        return None, None

def generate_qa_pairs(rag_system):
    """Generate question-answer pairs with contexts."""
    queries = get_legal_queries()
    qa_pairs = []
    
    console.print(f"[blue]Generating answers for {len(queries)} legal queries...[/blue]\n")
    
    for i, question in enumerate(queries, 1):
        console.print(f"[yellow]Processing query {i}/{len(queries)}: {question[:60]}...[/yellow]")
        
        try:
            # Get full RAG response
            response = rag_system.query(
                question=question,
                search_type="hybrid",
                top_k=5,
                include_sources=True
            )
            
            # Extract answer and contexts
            generated_answer = response.get('answer', '')
            sources = response.get('sources', [])
            
            # Extract context texts
            contexts = []
            for source in sources:
                if 'content' in source:
                    contexts.append(source['content'])
                elif 'content_preview' in source:
                    contexts.append(source['content_preview'])
            
            # Only add if we have both answer and contexts
            if generated_answer.strip() and contexts:
                qa_pairs.append({
                    'question': question,
                    'generated_answer': generated_answer,
                    'retrieved_contexts': contexts
                })
                
                console.print(f"  ✓ Answer: {len(generated_answer)} chars, Contexts: {len(contexts)}")
            else:
                console.print(f"  ✗ Missing answer or contexts")
            
        except Exception as e:
            console.print(f"  ✗ Error: {e}")
    
    return qa_pairs

def display_final_results(results):
    """Display comprehensive ROUGE-C results."""
    console.print("\n" + "="*80)
    console.print(Panel.fit("[bold green]ROUGE-C Evaluation Results[/bold green]"))
    
    if 'aggregate_metrics' not in results:
        console.print("[red]No aggregate metrics available[/red]")
        return
    
    metrics = results['aggregate_metrics']
    
    # Main results table
    main_table = Table(show_header=True, header_style="bold magenta", title="ROUGE-C Metrics Summary")
    main_table.add_column("Metric Category", style="cyan", no_wrap=True)
    main_table.add_column("Precision", justify="center", style="green")
    main_table.add_column("Recall", justify="center", style="blue")
    main_table.add_column("F1-Score", justify="center", style="yellow")
    
    # Add ROUGE-L metrics
    rouge_l = metrics['rouge_l']
    main_table.add_row(
        "ROUGE-L",
        f"{rouge_l['avg_precision']:.3f} ±{rouge_l['std_precision']:.3f}",
        f"{rouge_l['avg_recall']:.3f} ±{rouge_l['std_recall']:.3f}",
        f"{rouge_l['avg_f1']:.3f} ±{rouge_l['std_f1']:.3f}"
    )
    
    # Add Token Overlap metrics
    token = metrics['token_overlap']
    main_table.add_row(
        "Token Overlap",
        f"{token['avg_precision']:.3f} ±{token['std_precision']:.3f}",
        f"{token['avg_recall']:.3f} ±{token['std_recall']:.3f}",
        f"{token['avg_f1']:.3f} ±{token['std_f1']:.3f}"
    )
    
    console.print(main_table)
    
    # Semantic similarity table
    sem_table = Table(show_header=True, header_style="bold magenta", title="Semantic Similarity Metrics")
    sem_table.add_column("Similarity Type", style="cyan")
    sem_table.add_column("Average", justify="center", style="green")
    sem_table.add_column("Std Dev", justify="center", style="blue")
    
    sem = metrics['semantic_similarity']
    sem_table.add_row("Combined Context", f"{sem['avg_combined']:.3f}", f"{sem['std_combined']:.3f}")
    sem_table.add_row("Average Individual", f"{sem['avg_individual']:.3f}", f"{sem['std_individual']:.3f}")
    sem_table.add_row("Max Individual", f"{sem['avg_max']:.3f}", f"{sem['std_max']:.3f}")
    
    console.print(sem_table)
    
    # Summary statistics
    summary = metrics['summary']
    console.print(f"\n[bold]Evaluation Summary:[/bold]")
    console.print(f"• Total Q-A pairs evaluated: {summary['total_pairs']}")
    console.print(f"• Average contexts per query: {summary['avg_contexts_per_query']:.1f}")
    
    # Performance assessment
    rouge_l_f1 = rouge_l['avg_f1']
    token_f1 = token['avg_f1']
    sem_combined = sem['avg_combined']
    overall_score = (rouge_l_f1 + token_f1 + sem_combined) / 3
    
    console.print(f"\n[bold yellow]Performance Assessment:[/bold yellow]")
    console.print(f"• ROUGE-L F1: {rouge_l_f1:.3f} - {'High' if rouge_l_f1 > 0.5 else 'Moderate' if rouge_l_f1 > 0.3 else 'Low'} sequential overlap")
    console.print(f"• Token F1: {token_f1:.3f} - {'High' if token_f1 > 0.4 else 'Moderate' if token_f1 > 0.2 else 'Low'} lexical overlap")
    console.print(f"• Semantic Sim: {sem_combined:.3f} - {'High' if sem_combined > 0.7 else 'Moderate' if sem_combined > 0.5 else 'Low'} semantic alignment")
    console.print(f"• Overall Score: {overall_score:.3f}")
    
    if overall_score > 0.6:
        console.print("[green]✓ Excellent: Answers are well-grounded in retrieved contexts[/green]")
    elif overall_score > 0.4:
        console.print("[yellow]◐ Good: Answers are reasonably grounded in contexts[/yellow]")  
    elif overall_score > 0.2:
        console.print("[orange3]◑ Fair: Answers have some grounding in contexts[/orange3]")
    else:
        console.print("[red]✗ Poor: Answers are poorly grounded in contexts[/red]")

def main():
    """Main evaluation function."""
    console.print(Panel.fit(
        "[bold blue]ROUGE-C Evaluation for Legal RAG System[/bold blue]\n"
        "Evaluating answer quality against retrieved contexts using ROUGE-C metrics\n"
        "• ROUGE-L: Sequential overlap (longest common subsequence)\n"
        "• Token Overlap: Lexical overlap between answer and context\n" 
        "• Semantic Similarity: Embedding-based semantic alignment",
        title="ROUGE-C Evaluation"
    ))
    
    # Initialize system
    console.print("\n[blue]Initializing system...[/blue]")
    rag_system, rouge_c_evaluator = initialize_system()
    
    if not rag_system:
        console.print("[red]Failed to initialize system. Exiting.[/red]")
        return
    
    console.print("[green]✓ System initialized successfully![/green]")
    
    # Generate Q-A pairs
    qa_pairs = generate_qa_pairs(rag_system)
    
    if not qa_pairs:
        console.print("[red]No valid Q-A pairs generated. Cannot evaluate.[/red]")
        return
    
    console.print(f"\n[green]Generated {len(qa_pairs)} valid Q-A pairs[/green]")
    
    # Run ROUGE-C evaluation
    console.print("\n[blue]Computing ROUGE-C metrics...[/blue]")
    results = rouge_c_evaluator.compute_rouge_c_batch(qa_pairs)
    
    # Display results
    display_final_results(results)
    
    # Save results
    save_path = os.path.join(os.path.dirname(__file__), "rouge_c_evaluation_results.json")
    rouge_c_evaluator.save_results(results, save_path)

if __name__ == "__main__":
    main()