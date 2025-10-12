"""
Command-line interface for the Legal Assistant RAG system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from pathlib import Path
import json

from src.utils import setup_logging, get_logger, GOOGLE_API_KEY
from src.rag import DocumentIngestionPipeline, LegalRAGSystem, ChromaVectorDB

console = Console()
logger = get_logger(__name__)

class LegalAssistantCLI:
    """Command-line interface for the Legal Assistant RAG system."""
    
    def __init__(self):
        self.rag_system = None
        self.vector_db = None
        
    def initialize_system(self):
        """Initialize the RAG system components."""
        try:
            self.vector_db = ChromaVectorDB()
            self.rag_system = LegalRAGSystem(self.vector_db)
            return True
        except Exception as e:
            console.print(f"[red]Error initializing system: {str(e)}[/red]")
            return False

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Legal Assistant RAG System - Command Line Interface"""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    ctx.obj['verbose'] = verbose

@cli.command()
@click.option('--limit', type=int, help='Limit number of documents to process (for testing)')
@click.pass_context
def ingest(ctx, limit):
    """Ingest legal documents into the vector database."""
    console.print("[bold blue]Starting Document Ingestion Process[/bold blue]")
    
    # Check if GOOGLE_API_KEY is set
    if not GOOGLE_API_KEY:
        console.print("[red]Error: GOOGLE_API_KEY not found in environment variables.[/red]")
        console.print("Please set your Google API key in the .env file or environment.")
        return
    
    try:
        # Initialize ingestion pipeline
        pipeline = DocumentIngestionPipeline()
        
        # Run the pipeline
        with console.status("[bold green]Processing documents..."):
            result = pipeline.run_full_pipeline(limit=limit)
        
        if result['status'] == 'success':
            stats = result['statistics']
            
            # Display results in a table
            table = Table(title="Ingestion Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Files Found", str(stats['total_files']))
            table.add_row("Documents Processed", str(stats['processed_docs']))
            table.add_row("Failed Documents", str(stats['failed_docs']))
            table.add_row("Total Chunks Created", str(stats['total_chunks']))
            table.add_row("Chunks Ingested", str(stats['ingested_chunks']))
            
            console.print(table)
            console.print(f"[green]✓ Ingestion completed successfully![/green]")
            console.print(f"Summary saved to: {result.get('summary_file', 'N/A')}")
        
        else:
            console.print(f"[red]✗ Ingestion failed: {result.get('error', 'Unknown error')}[/red]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Ingestion failed: {str(e)}[/red]")

@cli.command()
@click.pass_context
def status(ctx):
    """Check the status of the vector database and system."""
    console.print("[bold blue]System Status[/bold blue]")
    
    try:
        # Initialize vector database
        vector_db = ChromaVectorDB()
        stats = vector_db.get_collection_stats()
        
        # Display status
        panel_content = f"""
[green]✓ Vector Database[/green]: Connected
[green]✓ Collection[/green]: {stats.get('collection_name', 'Unknown')}
[green]✓ Total Documents[/green]: {stats.get('total_documents', 'Unknown')}
[green]✓ Persist Directory[/green]: {stats.get('persist_directory', 'Unknown')}

[cyan]Sample Metadata Keys:[/cyan]
{', '.join(stats.get('sample_metadata_keys', []))}
        """
        
        console.print(Panel(panel_content, title="System Status", border_style="green"))
        
        # Check API key
        if GOOGLE_API_KEY:
            console.print("[green]✓ Google API Key: Configured[/green]")
        else:
            console.print("[red]✗ Google API Key: Not configured[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ Error checking status: {str(e)}[/red]")

@cli.command()
@click.argument('query', required=False)
@click.option('--num-docs', default=5, help='Number of documents to retrieve')
@click.option('--interactive', is_flag=True, help='Start interactive query mode')
@click.pass_context
def query(ctx, query, num_docs, interactive):
    """Query the legal database with natural language questions."""
    
    # Check if API key is configured
    if not GOOGLE_API_KEY:
        console.print("[red]Error: GOOGLE_API_KEY not found in environment variables.[/red]")
        return
    
    try:
        # Initialize CLI helper
        cli_helper = LegalAssistantCLI()
        
        with console.status("[bold green]Initializing RAG system..."):
            if not cli_helper.initialize_system():
                return
        
        console.print("[green]✓ RAG system initialized successfully![/green]\n")
        
        if interactive:
            # Interactive mode
            console.print("[bold cyan]Interactive Legal Assistant[/bold cyan]")
            console.print("Ask questions about Indian Supreme Court cases. Type 'quit' to exit.\n")
            
            while True:
                try:
                    user_query = Prompt.ask("[bold yellow]Your Question[/bold yellow]")
                    
                    if user_query.lower() in ['quit', 'exit', 'q']:
                        console.print("[yellow]Goodbye![/yellow]")
                        break
                    
                    if not user_query.strip():
                        continue
                    
                    _process_query(cli_helper.rag_system, user_query, num_docs)
                    console.print("\n" + "─" * 80 + "\n")
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
        
        else:
            # Single query mode
            if not query:
                console.print("[red]Error: Please provide a query or use --interactive mode[/red]")
                return
            
            _process_query(cli_helper.rag_system, query, num_docs)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

def _process_query(rag_system, user_query, num_docs):
    """Process a single query and display results."""
    
    with console.status(f"[bold green]Searching for: '{user_query}'..."):
        result = rag_system.answer_question(user_query, num_docs=num_docs)
    
    # Display the answer
    console.print(Panel(
        result['answer'], 
        title=f"Answer to: {user_query}", 
        border_style="blue"
    ))
    
    # Display sources
    if result['sources']:
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        
        sources_table = Table()
        sources_table.add_column("Case Name", style="green", width=30)
        sources_table.add_column("Court", style="cyan", width=20)
        sources_table.add_column("Date", style="yellow", width=15)
        sources_table.add_column("Relevance", style="magenta", width=10)
        
        for source in result['sources']:
            relevance = f"{source['relevance']:.1%}"
            sources_table.add_row(
                source['case_name'][:27] + "..." if len(source['case_name']) > 30 else source['case_name'],
                source['court'],
                source['date'],
                relevance
            )
        
        console.print(sources_table)
    
    # Display metadata
    metadata = result['metadata']
    console.print(f"\n[dim]Processing Time: {metadata['processing_time_ms']}ms | "
                 f"Sources: {metadata['num_sources']} | "
                 f"Model: {metadata.get('model_used', 'Unknown')}[/dim]")

@cli.command()
@click.argument('topic')
@click.option('--num-cases', default=10, help='Number of cases to return')
@click.pass_context
def search_topic(ctx, topic, num_cases):
    """Search for cases related to a specific legal topic."""
    
    try:
        cli_helper = LegalAssistantCLI()
        
        with console.status("[bold green]Initializing system..."):
            if not cli_helper.initialize_system():
                return
        
        with console.status(f"[bold green]Searching for cases related to '{topic}'..."):
            cases = cli_helper.rag_system.search_cases_by_topic(topic, num_cases)
        
        if cases:
            console.print(f"[bold green]Found {len(cases)} cases related to '{topic}':[/bold green]\n")
            
            cases_table = Table()
            cases_table.add_column("Case Name", style="green", width=40)
            cases_table.add_column("Court", style="cyan", width=20)
            cases_table.add_column("Date", style="yellow", width=15)
            cases_table.add_column("Relevance", style="magenta", width=10)
            
            for case in cases:
                relevance = f"{case['relevance']:.1%}"
                cases_table.add_row(
                    case['case_name'][:37] + "..." if len(case['case_name']) > 40 else case['case_name'],
                    case['court'],
                    case['date'],
                    relevance
                )
            
            console.print(cases_table)
        else:
            console.print(f"[yellow]No cases found related to '{topic}'.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.argument('case_name')
@click.pass_context
def summarize(ctx, case_name):
    """Get a summary of a specific case."""
    
    try:
        cli_helper = LegalAssistantCLI()
        
        with console.status("[bold green]Initializing system..."):
            if not cli_helper.initialize_system():
                return
        
        with console.status(f"[bold green]Generating summary for '{case_name}'..."):
            summary_result = cli_helper.rag_system.get_case_summary(case_name)
        
        if summary_result['status'] == 'success':
            # Display case details
            details = f"""
[bold]Case:[/bold] {summary_result['case_name']}
[bold]Court:[/bold] {summary_result['court']}
[bold]Date:[/bold] {summary_result['date']}
[bold]Judge:[/bold] {summary_result['judge']}
            """
            console.print(Panel(details.strip(), title="Case Details", border_style="cyan"))
            
            # Display summary
            console.print(Panel(
                summary_result['summary'], 
                title="Case Summary", 
                border_style="green"
            ))
        else:
            console.print(f"[yellow]{summary_result['summary']}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.pass_context
def reset(ctx):
    """Reset the vector database (WARNING: This will delete all data)."""
    
    confirm = Prompt.ask(
        "[red]Are you sure you want to reset the database? This will delete all ingested data.[/red]",
        choices=["yes", "no"],
        default="no"
    )
    
    if confirm == "yes":
        try:
            vector_db = ChromaVectorDB()
            
            with console.status("[bold red]Resetting database..."):
                success = vector_db.delete_collection()
            
            if success:
                console.print("[green]✓ Database reset successfully![/green]")
            else:
                console.print("[red]✗ Failed to reset database.[/red]")
                
        except Exception as e:
            console.print(f"[red]✗ Error resetting database: {str(e)}[/red]")
    else:
        console.print("[yellow]Reset cancelled.[/yellow]")

if __name__ == '__main__':
    cli()