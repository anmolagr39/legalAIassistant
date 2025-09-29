#!/usr/bin/env python3
"""
Legal RAG System - Terminal Interface
A comprehensive RAG system for Indian Constitution and Legal Acts
"""

import sys
import os
import argparse
from typing import Dict, Any
import json

# Configuration constants
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
CONSTITUTION_COLLECTION_NAME = "constitution_articles"
LEGAL_ACTS_COLLECTION_NAME = "legal_acts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 8192
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
SIMILARITY_THRESHOLD = 0.7
CONSTITUTION_PDF_PATH = "../the_constitution_of_india (1).pdf"
LEGAL_ACTS_CSV_PATH = "../legal-acts.csv"

def check_dependencies():
    """Check if all required packages are installed."""
    missing_packages = []
    
    try:
        import chromadb
    except ImportError:
        missing_packages.append("chromadb")
    
    try:
        import google.generativeai
    except ImportError:
        missing_packages.append("google-generativeai")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_packages.append("sentence-transformers")
    
    try:
        import fitz
    except ImportError:
        missing_packages.append("PyMuPDF")
    
    try:
        import pandas
    except ImportError:
        missing_packages.append("pandas")
    
    try:
        from rich.console import Console
    except ImportError:
        missing_packages.append("rich")
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nPlease install them with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main entry point."""
    if not check_dependencies():
        sys.exit(1)
    
    # Now import the modules after checking dependencies
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data_processor import DataProcessor
        from vector_store import VectorStore
        from rag_system import LegalRAG
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.progress import Progress
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)
    
    console = Console()
    
    parser = argparse.ArgumentParser(description="Legal RAG System")
    parser.add_argument('--setup', action='store_true', help='Setup the system')
    parser.add_argument('--reset', action='store_true', help='Reset vector database')
    parser.add_argument('--max-acts', type=int, help='Limit number of acts for testing')
    parser.add_argument('--query', type=str, help='Single query mode')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--test', action='store_true', help='Run with limited data for testing')
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        args.max_acts = 100  # Limit to 100 acts for testing
        console.print("[yellow]Running in test mode with limited data[/yellow]")
    
    # Initialize components
    data_processor = DataProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
    vector_store = VectorStore(CHROMA_PERSIST_DIRECTORY, EMBEDDING_MODEL)
    
    # Setup if requested or if database doesn't exist
    if args.setup or not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        console.print(Panel.fit("🏛️ Legal RAG System Setup", style="bold blue"))
        
        # Check if collections exist
        collections = vector_store.list_collections()
        if not args.reset and CONSTITUTION_COLLECTION_NAME in collections and LEGAL_ACTS_COLLECTION_NAME in collections:
            console.print("[green]Vector store already exists. Skipping data processing.[/green]")
        else:
            # Process Constitution
            console.print("\n[bold]Step 1: Processing Constitution of India[/bold]")
            constitution_path = os.path.join(os.path.dirname(__file__), CONSTITUTION_PDF_PATH)
            
            if os.path.exists(constitution_path):
                const_text = data_processor.extract_pdf_text(constitution_path)
                const_chunks = data_processor.process_constitution_text(const_text)
                
                # Add to vector store
                vector_store.add_documents(
                    CONSTITUTION_COLLECTION_NAME, 
                    const_chunks, 
                    reset=args.reset
                )
                
                # Save processed data
                data_processor.save_processed_data(const_chunks, "constitution_chunks.csv")
            else:
                console.print(f"[red]Constitution PDF not found at: {constitution_path}[/red]")
                sys.exit(1)
            
            # Process Legal Acts
            console.print("\n[bold]Step 2: Processing Legal Acts[/bold]")
            legal_acts_path = os.path.join(os.path.dirname(__file__), LEGAL_ACTS_CSV_PATH)
            
            if os.path.exists(legal_acts_path):
                if args.max_acts:
                    console.print(f"[yellow]Processing only first {args.max_acts} legal acts[/yellow]")
                
                acts_chunks = data_processor.process_legal_acts_csv(legal_acts_path, args.max_acts)
                
                # Add to vector store
                vector_store.add_documents(
                    LEGAL_ACTS_COLLECTION_NAME, 
                    acts_chunks, 
                    reset=args.reset
                )
                
                # Save processed data
                data_processor.save_processed_data(acts_chunks, "legal_acts_chunks.csv")
            else:
                console.print(f"[red]Legal Acts CSV not found at: {legal_acts_path}[/red]")
                sys.exit(1)
            
            console.print("\n[bold green]✅ System setup completed successfully![/bold green]")
    
    # Initialize RAG system
    rag_system = LegalRAG(
        vector_store, 
        CONSTITUTION_COLLECTION_NAME, 
        LEGAL_ACTS_COLLECTION_NAME
    )
    
    # Show stats
    stats = rag_system.get_system_stats()
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
    
    # Handle different modes
    if args.query:
        # Single query mode
        console.print(f"\n[blue]Processing query: '{args.query}'[/blue]")
        
        with console.status("[bold green]Searching..."):
            result = rag_system.query(args.query, top_k=TOP_K)
        
        # Display answer
        console.print("\n" + "="*80)
        console.print(Panel(result['answer'], title="📋 Legal Analysis", border_style="green"))
        
        # Display sources
        if result['sources']:
            console.print(f"\n[bold]📚 Sources ({len(result['sources'])} documents):[/bold]")
            for i, source in enumerate(result['sources'], 1):
                source_type = source['source_type']
                similarity = source['similarity_score']
                
                if source_type == 'constitution':
                    title = f"Constitution - {source.get('article', 'Unknown Article')}"
                elif source_type == 'legal_acts':
                    title = f"{source.get('act_title', 'Unknown Act')} ({source.get('enactment_date', 'Unknown Date')})"
                else:
                    title = f"Document {i}"
                
                console.print(f"\n[bold cyan]{i}. {title}[/bold cyan] [dim](Similarity: {similarity})[/dim]")
                console.print(f"[dim]{source['content_preview']}[/dim]")
        
    elif args.interactive or len(sys.argv) == 1:
        # Interactive mode
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
                    help_text = """
[bold cyan]Legal RAG System Commands:[/bold cyan]

[yellow]Basic Commands:[/yellow]
• Type any legal question to get an answer
• [cyan]quit/exit/q[/cyan] - Exit the system
• [cyan]help[/cyan] - Show this help
• [cyan]stats[/cyan] - Show system statistics

[yellow]Example Questions:[/yellow]
• "What are fundamental rights in Indian Constitution?"
• "Explain Article 21 of the Constitution"
• "What is the right to privacy?"
• "Show me acts related to consumer protection"
                    """
                    console.print(Panel(help_text, title="Help", border_style="blue"))
                    continue
                
                elif query.lower() == 'stats':
                    console.print(table)
                    continue
                
                # Process query
                console.print(f"[blue]Searching for: '{query}'[/blue]")
                
                with console.status("[bold green]Processing query..."):
                    result = rag_system.query(query, top_k=TOP_K)
                
                # Display answer
                console.print("\n" + "="*80)
                console.print(Panel(result['answer'], title="📋 Legal Analysis", border_style="green"))
                
                # Display sources
                if result['sources']:
                    console.print(f"\n[bold]📚 Sources ({len(result['sources'])} documents):[/bold]")
                    for i, source in enumerate(result['sources'], 1):
                        source_type = source['source_type']
                        similarity = source['similarity_score']
                        
                        if source_type == 'constitution':
                            title = f"Constitution - {source.get('article', 'Unknown Article')}"
                        elif source_type == 'legal_acts':
                            title = f"{source.get('act_title', 'Unknown Act')} ({source.get('enactment_date', 'Unknown Date')})"
                        else:
                            title = f"Document {i}"
                        
                        console.print(f"\n[bold cyan]{i}. {title}[/bold cyan] [dim](Similarity: {similarity})[/dim]")
                        console.print(f"[dim]{source['content_preview']}[/dim]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    else:
        console.print("[yellow]Use --help for usage information[/yellow]")

if __name__ == "__main__":
    main()