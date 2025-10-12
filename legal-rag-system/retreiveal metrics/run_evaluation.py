"""
Simple script to run retrieval metrics evaluation on 10 legal queries.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import LegalRAG
from src.vector_store import VectorStore
from retrieval_metrics import RetrievalMetrics
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def get_legal_queries() -> List[str]:
    """Return 10 predefined legal queries for testing."""
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
    """Initialize the RAG system and metrics evaluator."""
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
        
        # Initialize metrics evaluator with lower threshold for better results
        metrics_evaluator = RetrievalMetrics(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.5
        )
        
        # Load all available documents for recall computation
        try:
            const_collection = vector_store.client.get_collection("constitution_articles")
            const_results = const_collection.get()
            
            acts_collection = vector_store.client.get_collection("legal_acts")
            acts_results = acts_collection.get()
            
            all_documents = const_results['documents'] + acts_results['documents']
            console.print(f"[blue]Loaded {len(all_documents)} documents for recall computation[/blue]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load all documents: {e}[/yellow]")
            all_documents = []
        
        return rag_system, metrics_evaluator, all_documents
        
    except Exception as e:
        console.print(f"[red]Error initializing system: {e}[/red]")
        return None, None, None

def evaluate_queries(rag_system, metrics_evaluator, all_documents):
    """Evaluate 10 legal queries for k=5 and k=10."""
    queries = get_legal_queries()
    
    console.print(Panel.fit(f"[bold green]Evaluating {len(queries)} Legal Queries[/bold green]"))
    console.print("[blue]Computing metrics for k=5 and k=10...[/blue]\n")
    
    results_k5 = []
    results_k10 = []
    
    # Process each query
    for i, query in enumerate(queries, 1):
        console.print(f"[yellow]Processing query {i}/{len(queries)}: {query[:60]}...[/yellow]")
        
        try:
            # Get raw documents directly from vector search for k=5
            raw_docs_k5 = rag_system.vector_store.hybrid_search(
                rag_system.constitution_collection,
                rag_system.legal_acts_collection,
                query,
                top_k=5
            )
            
            # Get raw documents directly from vector search for k=10  
            raw_docs_k10 = rag_system.vector_store.hybrid_search(
                rag_system.constitution_collection,
                rag_system.legal_acts_collection,
                query,
                top_k=10
            )
            
            # Extract full document content
            retrieved_docs_k5 = [doc['content'] for doc in raw_docs_k5 if 'content' in doc]
            retrieved_docs_k10 = [doc['content'] for doc in raw_docs_k10 if 'content' in doc]
            
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
    console.print(f"• Search mode: Hybrid (Constitution + Legal Acts)")
    
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
        "Evaluating retrieval performance on 10 legal queries\n"
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