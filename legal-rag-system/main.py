#!/usr/bin/env python3
"""
Legal RAG System - Terminal Interface
A comprehensive RAG system for Indian Constitution and Legal Acts
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_processor import DataProcessor
    from vector_store import VectorStore
    from rag_system import LegalRAG
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress
except ImportError as e:
    print(f"Missing dependencies. Please install requirements: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Configuration
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from settings import *

console = Console()

class LegalRAGCLI:
    """Command-line interface for the Legal RAG system."""
    
    def __init__(self):
        self.data_processor = None
        self.vector_store = None
        self.rag_system = None
        self.setup_complete = False
    
    def setup_system(self, reset_db: bool = False, max_acts: int = None):
        """Setup the RAG system by processing data and creating vector store."""
        console.print(Panel.fit("🏛️ Legal RAG System Setup", style="bold blue"))
        
        # Initialize components
        self.data_processor = DataProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
        self.vector_store = VectorStore(CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL)
        
        # Check if setup is needed
        collections = self.vector_store.list_collections()
        if not reset_db and CONSTITUTION_COLLECTION_NAME in collections and LEGAL_ACTS_COLLECTION_NAME in collections:
            console.print("[green]Vector store already exists. Skipping data processing.[/green]")
            self.rag_system = LegalRAG(
                self.vector_store, 
                CONSTITUTION_COLLECTION_NAME, 
                LEGAL_ACTS_COLLECTION_NAME
            )
            self.setup_complete = True
            return
        
        try:
            # Process Constitution
            console.print("\n[bold]Step 1: Processing Constitution of India[/bold]")
            constitution_path = os.path.join(os.path.dirname(__file__), CONSTITUTION_PDF_PATH)
            
            if os.path.exists(constitution_path):
                const_text = self.data_processor.extract_pdf_text(constitution_path)
                const_chunks = self.data_processor.process_constitution_text(const_text)
                
                # Add to vector store
                self.vector_store.add_documents(
                    CONSTITUTION_COLLECTION_NAME, 
                    const_chunks, 
                    reset=reset_db
                )
                
                # Save processed data
                self.data_processor.save_processed_data(const_chunks, "constitution_chunks.csv")
            else:
                console.print(f"[red]Constitution PDF not found at: {constitution_path}[/red]")
                return False
            
            # Process Legal Acts
            console.print("\n[bold]Step 2: Processing Legal Acts[/bold]")
            legal_acts_path = os.path.join(os.path.dirname(__file__), LEGAL_ACTS_CSV_PATH)
            
            if os.path.exists(legal_acts_path):
                if max_acts:
                    console.print(f"[yellow]Processing only first {max_acts} legal acts for testing[/yellow]")
                
                acts_chunks = self.data_processor.process_legal_acts_csv(legal_acts_path, max_acts)
                
                # Add to vector store
                self.vector_store.add_documents(
                    LEGAL_ACTS_COLLECTION_NAME, 
                    acts_chunks, 
                    reset=reset_db
                )
                
                # Save processed data
                self.data_processor.save_processed_data(acts_chunks, "legal_acts_chunks.csv")
            else:
                console.print(f"[red]Legal Acts CSV not found at: {legal_acts_path}[/red]")
                return False
            
            # Initialize RAG system
            self.rag_system = LegalRAG(
                self.vector_store, 
                CONSTITUTION_COLLECTION_NAME, 
                LEGAL_ACTS_COLLECTION_NAME
            )
            
            self.setup_complete = True
            console.print("\n[bold green]✅ System setup completed successfully![/bold green]")
            
            # Show stats
            self.show_stats()
            
            return True
            
        except Exception as e:
            console.print(f"[red]Setup failed: {e}[/red]")
            return False
    
    def interactive_mode(self):
        """Run the system in interactive query mode."""
        if not self.setup_complete:
            console.print("[red]System not set up. Run setup first.[/red]")
            return
        
        console.print(Panel.fit("🤖 Legal RAG System - Interactive Mode", style="bold green"))
        console.print("[cyan]Ask questions about Indian Constitution and Legal Acts[/cyan]")
        console.print("[yellow]Type 'quit' to exit, 'help' for commands, 'stats' for system info[/yellow]\n")
        
        while True:
            try:
                query = Prompt.ask("\n[bold blue]Legal Query[/bold blue]")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif query.lower() == 'help':
                    self.show_help()
                    continue
                
                elif query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                elif query.lower().startswith('search:'):
                    # Advanced search mode
                    self.advanced_search_mode(query[7:].strip())
                    continue
                
                # Process query
                self.process_query(query)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    def process_query(self, query: str, search_type: str = "hybrid"):
        """Process a single query."""
        console.print(f"\n[blue]Searching for: '{query}'[/blue]")
        
        with console.status("[bold green]Processing query..."):
            result = self.rag_system.query(query, search_type=search_type, top_k=TOP_K)
        
        # Display answer
        console.print("\n" + "="*80)
        console.print(Panel(result['answer'], title="📋 Legal Analysis", border_style="green"))
        
        # Display sources
        if result['sources']:
            console.print(f"\n[bold]📚 Sources ({len(result['sources'])} documents):[/bold]")
            for i, source in enumerate(result['sources'], 1):
                self.display_source(i, source)
    
    def display_source(self, index: int, source: Dict):
        """Display a single source."""
        source_type = source['source_type']
        similarity = source['similarity_score']
        
        if source_type == 'constitution':
            title = f"Constitution - {source.get('article', 'Unknown Article')}"
        elif source_type == 'legal_acts':
            title = f"{source.get('act_title', 'Unknown Act')} ({source.get('enactment_date', 'Unknown Date')})"
        else:
            title = f"Document {index}"
        
        console.print(f"\n[bold cyan]{index}. {title}[/bold cyan] [dim](Similarity: {similarity})[/dim]")
        console.print(f"[dim]{source['content_preview']}[/dim]")
    
    def advanced_search_mode(self, query: str):
        """Advanced search with options."""
        console.print(f"\n[bold]Advanced Search Mode[/bold]")
        
        # Search type selection
        search_type = Prompt.ask(
            "Search in", 
            choices=["hybrid", "constitution", "legal_acts"], 
            default="hybrid"
        )
        
        # Number of results
        top_k = int(Prompt.ask("Number of results", default="5"))
        
        self.process_query(query, search_type)
    
    def show_stats(self):
        """Display system statistics."""
        if not self.rag_system:
            console.print("[red]System not initialized[/red]")
            return
        
        stats = self.rag_system.get_system_stats()
        
        table = Table(title="📊 System Statistics")
        table.add_column("Collection", style="cyan")
        table.add_column("Documents", justify="right", style="green")
        table.add_column("Status", style="yellow")
        
        table.add_row(
            "Constitution",
            str(stats['constitution_collection']['document_count']),
            stats['constitution_collection']['status']
        )
        
        table.add_row(
            "Legal Acts",
            str(stats['legal_acts_collection']['document_count']),
            stats['legal_acts_collection']['status']
        )
        
        table.add_row(
            "Total",
            str(stats['total_documents']),
            "Active" if stats['total_documents'] > 0 else "Empty"
        )
        
        console.print(table)
    
    def show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]Legal RAG System Commands:[/bold cyan]

[yellow]Basic Commands:[/yellow]
• Type any legal question to get an answer
• [cyan]quit/exit/q[/cyan] - Exit the system
• [cyan]help[/cyan] - Show this help
• [cyan]stats[/cyan] - Show system statistics

[yellow]Advanced Search:[/yellow]
• [cyan]search: <query>[/cyan] - Advanced search with options

[yellow]Example Questions:[/yellow]
• "What are fundamental rights in Indian Constitution?"
• "Explain Article 21 of the Constitution"
• "What is the right to privacy?"
• "Show me acts related to consumer protection"

[yellow]Search Types:[/yellow]
• [green]hybrid[/green] - Search both Constitution and Legal Acts
• [green]constitution[/green] - Search only Constitution
• [green]legal_acts[/green] - Search only Legal Acts
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def batch_query(self, queries_file: str):
        """Process queries from a file."""
        if not os.path.exists(queries_file):
            console.print(f"[red]Queries file not found: {queries_file}[/red]")
            return
        
        with open(queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("[green]Processing queries...", total=len(queries))
            
            for query in queries:
                result = self.rag_system.query(query)
                results.append({
                    'query': query,
                    'answer': result['answer'],
                    'sources_count': len(result['sources'])
                })
                progress.update(task, advance=1)
        
        # Save results
        output_file = queries_file.replace('.txt', '_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]Results saved to: {output_file}[/green]")

def main():
    parser = argparse.ArgumentParser(description="Legal RAG System")
    parser.add_argument('--setup', action='store_true', help='Setup the system')
    parser.add_argument('--reset', action='store_true', help='Reset vector database')
    parser.add_argument('--max-acts', type=int, help='Limit number of acts for testing')
    parser.add_argument('--query', type=str, help='Single query mode')
    parser.add_argument('--batch', type=str, help='Batch query from file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    cli = LegalRAGCLI()
    
    # Setup if requested
    if args.setup or not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        if not cli.setup_system(reset_db=args.reset, max_acts=args.max_acts):
            console.print("[red]Setup failed. Exiting.[/red]")
            return
    else:
        # Initialize existing system
        cli.vector_store = VectorStore(CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL)
        cli.rag_system = LegalRAG(
            cli.vector_store, 
            CONSTITUTION_COLLECTION_NAME, 
            LEGAL_ACTS_COLLECTION_NAME
        )
        cli.setup_complete = True
    
    # Handle different modes
    if args.query:
        cli.process_query(args.query)
    elif args.batch:
        cli.batch_query(args.batch)
    elif args.interactive or len(sys.argv) == 1:
        cli.interactive_mode()
    else:
        console.print("[yellow]Use --help for usage information[/yellow]")

if __name__ == "__main__":
    main()