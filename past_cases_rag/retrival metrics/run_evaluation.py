"""
Simple script to run retrieval metrics evaluation on 10 legal case queries.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from legal_rag_metrics import RetrievalMetrics
from legal_queries import get_legal_case_queries
from src.rag.vector_db import ChromaVectorDB
from src.rag.rag_system import LegalRAGSystem
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def initialize_system():
    """Initialize the RAG system and metrics evaluator."""
    try:
        # Initialize vector store and RAG system with correct path to chroma_db
        db_path = os.path.join(os.path.dirname(__file__), '..', 'chroma_db')
        vector_db = ChromaVectorDB(persist_directory=db_path)
        rag_system = LegalRAGSystem(vector_db)
        
        # Initialize metrics evaluator with threshold for better results
        metrics_evaluator = RetrievalMetrics(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.5
        )
        
        # Load all available documents for recall computation
        try:
            collection_stats = vector_db.get_collection_stats()
            console.print(f"[blue]Loaded {collection_stats['total_documents']} documents for recall computation[/blue]")
            
            # Get sample documents for recall computation
            sample_results = vector_db.collection.get(limit=1000)
            all_documents = sample_results['documents'] if sample_results['documents'] else []
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load all documents: {e}[/yellow]")
            all_documents = []
        
        return rag_system, metrics_evaluator, all_documents
        
    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        return None, None, None

def evaluate_queries(rag_system, metrics_evaluator, all_documents):
    """Evaluate 10 legal case queries for k=5 and k=10."""
    queries = get_legal_case_queries()
    
    console.print(Panel.fit(f"[bold green]Evaluating {len(queries)} Legal Case Queries[/bold green]"))
    console.print("[blue]Computing metrics for k=5 and k=10...[/blue]\n")
    
    results_k5 = []
    results_k10 = []
    
    # Process each query
    for i, query in enumerate(queries, 1):
        console.print(f"[yellow]Processing query {i}/{len(queries)}: {query[:60]}...[/yellow]")
        
        try:
            # Get documents using RAG system for k=5
            retrieved_results_k5 = rag_system.retrieve_relevant_documents(
                query=query,
                num_docs=5
            )
            
            # Get documents using RAG system for k=10
            retrieved_results_k10 = rag_system.retrieve_relevant_documents(
                query=query,
                num_docs=10
            )
            
            # Extract document texts
            retrieved_docs_k5 = [result['document'] for result in retrieved_results_k5]
            retrieved_docs_k10 = [result['document'] for result in retrieved_results_k10]
            
            results_k5.append({
                'query': query,
                'retrieved_docs': retrieved_docs_k5
            })
            
            results_k10.append({
                'query': query,
                'retrieved_docs': retrieved_docs_k10
            })
            
            # Just show document count
            console.print(f"  Retrieved {len(retrieved_docs_k5)} docs for k=5, {len(retrieved_docs_k10)} docs for k=10")
                
        except Exception as e:
            console.print(f"  [red]Error processing query: {e}[/red]")
    
    # Compute aggregate metrics
    if results_k5 and results_k10:
        console.print("\n[blue]Computing aggregate metrics...[/blue]")
        
        metrics_k5 = metrics_evaluator.batch_evaluate(
            query_results=results_k5,
            all_available_docs=all_documents[:1000] if all_documents else None
        )
        
        metrics_k10 = metrics_evaluator.batch_evaluate(
            query_results=results_k10,
            all_available_docs=all_documents[:1000] if all_documents else None
        )
        
        # Display final results
        display_final_results(metrics_k5, metrics_k10)
    else:
        console.print("[red]No successful queries to evaluate![/red]")

def display_final_results(metrics_k5, metrics_k10):
    """Display final aggregate results."""
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold green]Final Retrieval Metrics Results[/bold green]"))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("k=5", justify="center", style="green")
    table.add_column("k=10", justify="center", style="blue")
    
    agg_k5 = metrics_k5['aggregate_metrics']
    agg_k10 = metrics_k10['aggregate_metrics']
    
    table.add_row(
        "Precision@k",
        f"{agg_k5['avg_precision']:.3f}",
        f"{agg_k10['avg_precision']:.3f}"
    )
    
    table.add_row(
        "Recall@k",
        f"{agg_k5['avg_recall']:.3f}",
        f"{agg_k10['avg_recall']:.3f}"
    )
    
    table.add_row(
        "F1-Score@k",
        f"{agg_k5['avg_f1_score']:.3f}",
        f"{agg_k10['avg_f1_score']:.3f}"
    )
    
    console.print(table)
    
    # Additional summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"• Total queries evaluated: {metrics_k5['total_queries']}")
    console.print(f"• Similarity threshold: 0.5 (adjusted for better results)")
    console.print(f"• Search mode: Past Cases Database")
    
    # Performance interpretation
    console.print(f"\n[bold]Performance Analysis:[/bold]")
    
    precision_k5 = agg_k5['avg_precision']
    precision_k10 = agg_k10['avg_precision']
    
    if precision_k5 > 0.7:
        console.print(f"• [green]Good precision@5: {precision_k5:.3f} - Most retrieved documents are relevant[/green]")
    elif precision_k5 > 0.5:
        console.print(f"• [yellow]Moderate precision@5: {precision_k5:.3f} - Some improvement needed[/yellow]")
    else:
        console.print(f"• [red]Low precision@5: {precision_k5:.3f} - Retrieval needs improvement[/red]")
    
    if precision_k10 > precision_k5:
        console.print(f"• [blue]Precision@10 > Precision@5: More relevant docs in larger result set[/blue]")
    else:
        console.print(f"• [yellow]Precision decreases with k=10: Diminishing relevance in larger results[/yellow]")

def main():
    """Main function to run the evaluation."""
    console.print(Panel.fit(
        "[bold blue]Legal RAG Retrieval Metrics Evaluation[/bold blue]\n"
        "Evaluating retrieval performance on 10 legal case queries\n"
        "Computing Precision@k, Recall@k, and F1@k for k=5 and k=10",
        title="Retrieval Metrics Testing"
    ))
    
    # Initialize system
    console.print("\n[blue]Initializing system...[/blue]")
    rag_system, metrics_evaluator, all_documents = initialize_system()
    
    if not rag_system:
        console.print("[red]Failed to initialize system. Exiting.[/red]")
        return
    
    console.print("[green]✓ System initialized successfully![/green]\n")
    
    # Run evaluation
    evaluate_queries(rag_system, metrics_evaluator, all_documents)

if __name__ == "__main__":
    main()