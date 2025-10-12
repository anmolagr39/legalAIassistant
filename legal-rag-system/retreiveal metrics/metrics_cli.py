"""
Command Line Interface for testing retrieval metrics on Legal RAG system.
Focuses on Constitution and Legal Acts queries only.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import LegalRAG
from src.vector_store import VectorStore
from retrieval_metrics import RetrievalMetrics
from typing import List, Dict, Any
import json
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
import pandas as pd

console = Console()

class RetrievalMetricsCLI:
    """CLI for testing retrieval metrics on the Legal RAG system."""
    
    def __init__(self):
        self.rag_system = None
        self.vector_store = None
        self.metrics_evaluator = None
        self.all_documents = []
        
    def initialize_system(self):
        """Initialize the RAG system and metrics evaluator."""
        console.print(Panel.fit("[bold blue]Initializing Legal RAG System for Metrics Testing[/bold blue]"))
        
        try:
            # Initialize vector store
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
            self.vector_store = VectorStore(persist_directory=data_path)
            
            # Initialize RAG system
            self.rag_system = LegalRAG(
                vector_store=self.vector_store,
                constitution_collection="constitution_articles",
                legal_acts_collection="legal_acts"
            )
            
            # Initialize metrics evaluator (hardcoded settings)
            similarity_threshold = 0.7
            
            self.metrics_evaluator = RetrievalMetrics(
                embedding_model="all-MiniLM-L6-v2",
                similarity_threshold=similarity_threshold
            )
            
            # Load all available documents for recall computation
            self._load_all_documents()
            
            console.print("[green]✓ System initialized successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            return False
        
        return True
    
    def _load_all_documents(self):
        """Load all documents from both collections for recall computation."""
        try:
            # Get all documents from constitution collection
            const_collection = self.vector_store.client.get_collection("constitution_articles")
            const_results = const_collection.get()
            
            # Get all documents from legal acts collection
            acts_collection = self.vector_store.client.get_collection("legal_acts")
            acts_results = acts_collection.get()
            
            # Combine all documents
            self.all_documents = const_results['documents'] + acts_results['documents']
            
            console.print(f"[blue]Loaded {len(self.all_documents)} total documents for recall computation[/blue]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load all documents for recall computation: {e}[/yellow]")
            self.all_documents = []
    
    def get_predefined_legal_queries(self) -> List[str]:
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
    
    def single_query_evaluation(self):
        """Evaluate a single query interactively."""
        console.print(Panel.fit("[bold cyan]Single Query Evaluation[/bold cyan]"))
        
        # Get query from user
        query = Prompt.ask("[green]Enter your legal query[/green]")
        
        if not query.strip():
            console.print("[red]Query cannot be empty![/red]")
            return
        
        # Hardcoded search parameters
        search_type = "hybrid"
        top_k = 5
        
        # Perform retrieval for k=5 and k=10
        console.print(f"[blue]Retrieving documents for: '{query}'[/blue]")
        
        try:
            # Get results for k=5
            response_k5 = self.rag_system.query(
                question=query,
                search_type=search_type,
                top_k=5,
                include_sources=True
            )
            retrieved_docs_k5 = [doc['content'] for doc in response_k5.get('sources', [])]
            
            # Get results for k=10
            response_k10 = self.rag_system.query(
                question=query,
                search_type=search_type,
                top_k=10,
                include_sources=True
            )
            retrieved_docs_k10 = [doc['content'] for doc in response_k10.get('sources', [])]
            
            if not retrieved_docs_k5 and not retrieved_docs_k10:
                console.print("[red]No documents retrieved![/red]")
                return
            
            console.print(f"[green]Retrieved {len(retrieved_docs_k5)} documents for k=5, {len(retrieved_docs_k10)} for k=10[/green]")
            
            # Evaluate retrieval metrics for both k values
            metrics_k5 = self.metrics_evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs_k5,
                all_available_docs=self.all_documents if self.all_documents else None
            )
            
            metrics_k10 = self.metrics_evaluator.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs_k10,
                all_available_docs=self.all_documents if self.all_documents else None
            )
            
            # Display simple metrics comparison
            self._display_single_query_metrics(query, metrics_k5, metrics_k10)
            
            # Show retrieved documents if requested
            if Confirm.ask("[cyan]Show retrieved documents?[/cyan]"):
                console.print("\n[bold cyan]Documents for k=5:[/bold cyan]")
                self._display_retrieved_docs(retrieved_docs_k5, response_k5.get('sources', []))
                console.print("\n[bold cyan]Documents for k=10:[/bold cyan]")
                self._display_retrieved_docs(retrieved_docs_k10, response_k10.get('sources', []))
            
        except Exception as e:
            console.print(f"[red]Error during evaluation: {e}[/red]")
    
    def batch_evaluation(self):
        """Evaluate 10 predefined legal queries."""
        console.print(Panel.fit("[bold cyan]Evaluating 10 Legal Queries[/bold cyan]"))
        
        # Always use predefined queries (10 queries)
        queries = self.get_predefined_legal_queries()
        console.print(f"[blue]Using {len(queries)} predefined legal queries[/blue]")
        
        # Hardcoded search parameters
        search_type = "hybrid"
        
        # Process all queries for k=5 and k=10
        console.print(f"[blue]Processing {len(queries)} queries for k=5 and k=10...[/blue]")
        
        results_k5 = []
        results_k10 = []
        
        for i, query in enumerate(queries, 1):
            console.print(f"[yellow]Processing query {i}/{len(queries)}: {query[:50]}...[/yellow]")
            
            try:
                # Get results for k=5
                response_k5 = self.rag_system.query(
                    question=query,
                    search_type=search_type,
                    top_k=5,
                    include_sources=True
                )
                retrieved_docs_k5 = [doc['content'] for doc in response_k5.get('sources', [])]
                
                # Get results for k=10  
                response_k10 = self.rag_system.query(
                    question=query,
                    search_type=search_type,
                    top_k=10,
                    include_sources=True
                )
                retrieved_docs_k10 = [doc['content'] for doc in response_k10.get('sources', [])]
                
                results_k5.append({
                    'query': query,
                    'retrieved_docs': retrieved_docs_k5
                })
                
                results_k10.append({
                    'query': query,
                    'retrieved_docs': retrieved_docs_k10
                })
                
            except Exception as e:
                console.print(f"[red]Error processing query {i}: {e}[/red]")
        
        if not results_k5:
            console.print("[red]No successful queries to evaluate![/red]")
            return
        
        # Evaluate all queries for both k values
        console.print("[blue]Computing retrieval metrics for k=5...[/blue]")
        metrics_k5 = self.metrics_evaluator.batch_evaluate(
            query_results=results_k5,
            all_available_docs=self.all_documents if self.all_documents else None
        )
        
        console.print("[blue]Computing retrieval metrics for k=10...[/blue]")
        metrics_k10 = self.metrics_evaluator.batch_evaluate(
            query_results=results_k10,
            all_available_docs=self.all_documents if self.all_documents else None
        )
        
        # Display simplified results
        self._display_simple_metrics(metrics_k5, metrics_k10)
    
    def _display_simple_metrics(self, metrics_k5: Dict[str, Any], metrics_k10: Dict[str, Any]):
        """Display simplified metrics for k=5 and k=10."""
        console.print("\n[bold green]Retrieval Metrics Comparison[/bold green]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("k=5", justify="center")
        table.add_column("k=10", justify="center")
        
        # Extract key metrics
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
        console.print(f"\n[blue]Total queries evaluated: {metrics_k5['total_queries']}[/blue]")
    
    def _display_single_query_metrics(self, query: str, metrics_k5: Dict[str, float], metrics_k10: Dict[str, float]):
        """Display metrics for a single query."""
        console.print(f"\n[bold green]Metrics for Query: {query[:60]}...[/bold green]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("k=5", justify="center")
        table.add_column("k=10", justify="center")
        
        table.add_row(
            "Precision@k",
            f"{metrics_k5['precision']:.3f}",
            f"{metrics_k10['precision']:.3f}"
        )
        
        table.add_row(
            "Recall@k", 
            f"{metrics_k5['recall']:.3f}",
            f"{metrics_k10['recall']:.3f}"
        )
        
        table.add_row(
            "F1-Score@k",
            f"{metrics_k5['f1_score']:.3f}",
            f"{metrics_k10['f1_score']:.3f}"
        )
        
        console.print(table)
    
    def _display_retrieved_docs(self, docs: List[str], sources: List[Dict]):
        """Display retrieved documents with metadata."""
        console.print("\n[bold green]Retrieved Documents:[/bold green]")
        
        for i, (doc, source) in enumerate(zip(docs, sources), 1):
            console.print(f"\n[bold cyan]Document {i}:[/bold cyan]")
            console.print(f"[yellow]Similarity: {source.get('similarity', 'N/A'):.3f}[/yellow]")
            
            if 'metadata' in source:
                metadata = source['metadata']
                console.print(f"[yellow]Source: {metadata.get('metadata_source', 'N/A')}[/yellow]")
                console.print(f"[yellow]Article: {metadata.get('metadata_article', 'N/A')}[/yellow]")
            
            # Truncate long documents
            content = doc[:300] + "..." if len(doc) > 300 else doc
            console.print(f"[white]{content}[/white]")
    
    def main_menu(self):
        """Display main menu and handle user choices."""
        while True:
            console.print("\n" + "="*60)
            console.print(Panel.fit(
                "[bold blue]Legal RAG Retrieval Metrics Testing[/bold blue]\n\n"
                "1. Run Evaluation (10 Legal Queries)\n"
                "2. Single Query Evaluation\n"
                "3. View System Status\n"
                "4. Exit",
                title="Main Menu"
            ))
            
            choice = Prompt.ask(
                "[green]Select an option[/green]",
                choices=["1", "2", "3", "4"],
                default="1"
            )
            
            if choice == "1":
                self.batch_evaluation()
            elif choice == "2":
                self.single_query_evaluation()
            elif choice == "3":
                self._show_system_status()
            elif choice == "4":
                console.print("[green]Goodbye![/green]")
                break
    
    def _show_system_status(self):
        """Show current system status and statistics."""
        console.print(Panel.fit("[bold cyan]System Status[/bold cyan]"))
        
        try:
            # Check collections
            collections = self.vector_store.list_collections()
            console.print(f"[green]Available collections: {collections}[/green]")
            
            for collection_name in collections:
                stats = self.vector_store.get_collection_stats(collection_name)
                console.print(f"[blue]  {collection_name}: {stats['document_count']} documents[/blue]")
            
            console.print(f"[blue]Total documents for recall: {len(self.all_documents)}[/blue]")
            console.print(f"[blue]Similarity threshold: {self.metrics_evaluator.similarity_threshold}[/blue]")
            
        except Exception as e:
            console.print(f"[red]Error getting system status: {e}[/red]")
    
    def run(self):
        """Run the CLI application."""
        console.print(Panel.fit(
            "[bold green]Legal RAG Retrieval Metrics Testing[/bold green]\n\n"
            "This tool evaluates retrieval performance using semantic similarity-based metrics\n"
            "without requiring ground truth annotations.",
            title="Welcome"
        ))
        
        # Initialize system
        if not self.initialize_system():
            console.print("[red]Failed to initialize system. Exiting.[/red]")
            return
        
        # Run main menu
        self.main_menu()

def main():
    """Entry point for the CLI application."""
    cli = RetrievalMetricsCLI()
    cli.run()

if __name__ == "__main__":
    main()