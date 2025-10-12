"""
CLI for testing ROUGE-C metrics on Legal RAG system.
Evaluates generated answers against retrieved contexts using ROUGE-C.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import LegalRAG
from src.vector_store import VectorStore
from rouge_c import RougeC
from typing import List, Dict, Any
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

class RougeCCLI:
    """CLI for testing ROUGE-C metrics on the Legal RAG system."""
    
    def __init__(self):
        self.rag_system = None
        self.rouge_c_evaluator = None
        
    def initialize_system(self):
        """Initialize the RAG system and ROUGE-C evaluator."""
        console.print(Panel.fit("[bold blue]Initializing Legal RAG System for ROUGE-C Testing[/bold blue]"))
        
        try:
            # Initialize vector store
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
            vector_store = VectorStore(persist_directory=data_path)
            
            # Initialize RAG system
            self.rag_system = LegalRAG(
                vector_store=vector_store,
                constitution_collection="constitution_articles",
                legal_acts_collection="legal_acts"
            )
            
            # Initialize ROUGE-C evaluator
            self.rouge_c_evaluator = RougeC(
                embedding_model="all-MiniLM-L6-v2",
                use_stemming=False,
                remove_stopwords=True
            )
            
            console.print("[green]✓ System initialized successfully![/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing system: {e}[/red]")
            return False
        
        return True
    
    def get_legal_queries(self) -> List[str]:
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
    
    def generate_qa_pairs(self) -> List[Dict[str, Any]]:
        """Generate question-answer pairs with retrieved contexts."""
        queries = self.get_legal_queries()
        qa_pairs = []
        
        console.print(f"[blue]Generating Q-A pairs for {len(queries)} queries...[/blue]")
        
        for i, question in enumerate(queries, 1):
            console.print(f"[yellow]Processing query {i}/{len(queries)}: {question[:60]}...[/yellow]")
            
            try:
                # Get RAG response with full pipeline
                response = self.rag_system.query(
                    question=question,
                    search_type="hybrid",
                    top_k=5,
                    include_sources=True
                )
                
                # Extract generated answer and contexts
                generated_answer = response.get('answer', '')
                sources = response.get('sources', [])
                
                # Get full context texts from sources
                contexts = []
                for source in sources:
                    if 'content' in source:
                        contexts.append(source['content'])
                    elif 'content_preview' in source:
                        contexts.append(source['content_preview'])
                
                qa_pairs.append({
                    'question': question,
                    'generated_answer': generated_answer,
                    'retrieved_contexts': contexts
                })
                
                console.print(f"  Generated answer length: {len(generated_answer)} chars, "
                             f"Retrieved {len(contexts)} contexts")
                
            except Exception as e:
                console.print(f"  [red]Error processing query: {e}[/red]")
        
        console.print(f"[green]Generated {len(qa_pairs)} Q-A pairs successfully[/green]")
        return qa_pairs
    
    def run_rouge_c_evaluation(self):
        """Run ROUGE-C evaluation on legal Q-A pairs."""
        console.print(Panel.fit("[bold green]ROUGE-C Evaluation on Legal RAG System[/bold green]"))
        
        # Generate Q-A pairs
        qa_pairs = self.generate_qa_pairs()
        
        if not qa_pairs:
            console.print("[red]No Q-A pairs generated. Cannot evaluate.[/red]")
            return
        
        # Filter out pairs with empty answers or contexts
        valid_pairs = []
        for pair in qa_pairs:
            if (pair['generated_answer'].strip() and 
                pair['retrieved_contexts'] and 
                any(ctx.strip() for ctx in pair['retrieved_contexts'])):
                valid_pairs.append(pair)
        
        console.print(f"[blue]Evaluating {len(valid_pairs)} valid Q-A pairs...[/blue]")
        
        if not valid_pairs:
            console.print("[red]No valid Q-A pairs found for evaluation.[/red]")
            return
        
        # Compute ROUGE-C metrics
        results = self.rouge_c_evaluator.compute_rouge_c_batch(valid_pairs)
        
        # Display results
        self.rouge_c_evaluator.display_results(results)
        
        # Show interpretation
        self._display_interpretation(results)
        
        # Optionally save results
        save_path = os.path.join(os.path.dirname(__file__), "rouge_c_results.json")
        self.rouge_c_evaluator.save_results(results, save_path)
    
    def _display_interpretation(self, results: Dict[str, Any]):
        """Display interpretation of ROUGE-C results."""
        if 'aggregate_metrics' not in results:
            return
        
        metrics = results['aggregate_metrics']
        
        console.print("\n[bold yellow]Performance Interpretation:[/bold yellow]")
        
        # ROUGE-L interpretation
        rouge_l_f1 = metrics['rouge_l']['avg_f1']
        if rouge_l_f1 > 0.5:
            console.print(f"• [green]Good ROUGE-L F1: {rouge_l_f1:.3f} - Strong sequential overlap with context[/green]")
        elif rouge_l_f1 > 0.3:
            console.print(f"• [yellow]Moderate ROUGE-L F1: {rouge_l_f1:.3f} - Some sequential alignment[/yellow]")
        else:
            console.print(f"• [red]Low ROUGE-L F1: {rouge_l_f1:.3f} - Limited sequential overlap[/red]")
        
        # Token overlap interpretation
        token_f1 = metrics['token_overlap']['avg_f1']
        if token_f1 > 0.4:
            console.print(f"• [green]Good token overlap F1: {token_f1:.3f} - Strong lexical alignment[/green]")
        elif token_f1 > 0.2:
            console.print(f"• [yellow]Moderate token overlap F1: {token_f1:.3f} - Some lexical alignment[/yellow]")
        else:
            console.print(f"• [red]Low token overlap F1: {token_f1:.3f} - Limited lexical overlap[/red]")
        
        # Semantic similarity interpretation
        sem_combined = metrics['semantic_similarity']['avg_combined']
        if sem_combined > 0.7:
            console.print(f"• [green]High semantic similarity: {sem_combined:.3f} - Strong contextual alignment[/green]")
        elif sem_combined > 0.5:
            console.print(f"• [yellow]Moderate semantic similarity: {sem_combined:.3f} - Good contextual alignment[/yellow]")
        else:
            console.print(f"• [red]Low semantic similarity: {sem_combined:.3f} - Weak contextual alignment[/red]")
        
        # Overall assessment
        overall_score = (rouge_l_f1 + token_f1 + sem_combined) / 3
        console.print(f"\n[bold]Overall ROUGE-C Score: {overall_score:.3f}[/bold]")
        
        if overall_score > 0.6:
            console.print("[green]✓ Excellent: Generated answers are well-grounded in retrieved contexts[/green]")
        elif overall_score > 0.4:
            console.print("[yellow]◐ Good: Generated answers are reasonably grounded in contexts[/yellow]")
        elif overall_score > 0.2:
            console.print("[orange3]◑ Fair: Generated answers have some grounding in contexts[/orange3]")
        else:
            console.print("[red]✗ Poor: Generated answers are poorly grounded in retrieved contexts[/red]")
    
    def run_single_example(self):
        """Run ROUGE-C on a single example for demonstration."""
        console.print(Panel.fit("[bold cyan]Single Example ROUGE-C Evaluation[/bold cyan]"))
        
        # Use a sample question
        question = "What are the fundamental rights in the Constitution?"
        
        console.print(f"[blue]Question: {question}[/blue]")
        
        # Get RAG response
        response = self.rag_system.query(
            question=question,
            search_type="hybrid",
            top_k=3,
            include_sources=True
        )
        
        generated_answer = response.get('answer', '')
        sources = response.get('sources', [])
        contexts = [source.get('content', source.get('content_preview', '')) for source in sources]
        
        console.print(f"\n[green]Generated Answer:[/green]")
        console.print(f"[white]{generated_answer}[/white]")
        
        console.print(f"\n[cyan]Retrieved Contexts ({len(contexts)}):[/cyan]")
        for i, context in enumerate(contexts, 1):
            preview = context[:200] + "..." if len(context) > 200 else context
            console.print(f"  {i}. {preview}")
        
        # Compute ROUGE-C
        result = self.rouge_c_evaluator.compute_rouge_c_single(generated_answer, contexts)
        
        # Display result
        self.rouge_c_evaluator.display_results(result)
    
    def run(self):
        """Run the CLI application."""
        console.print(Panel.fit(
            "[bold green]ROUGE-C Evaluation for Legal RAG[/bold green]\n\n"
            "This tool evaluates generated answers against retrieved contexts using ROUGE-C metrics\n"
            "without requiring reference answers.",
            title="Welcome"
        ))
        
        # Initialize system
        if not self.initialize_system():
            console.print("[red]Failed to initialize system. Exiting.[/red]")
            return
        
        # Main menu
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold blue]ROUGE-C Evaluation Options[/bold blue]\n\n"
            "1. Run Full Evaluation (10 Legal Queries)\n"
            "2. Single Example Evaluation\n"
            "3. Exit",
            title="Main Menu"
        ))
        
        choice = input("Select option [1-3]: ").strip()
        
        if choice == "1":
            self.run_rouge_c_evaluation()
        elif choice == "2":
            self.run_single_example()
        elif choice == "3":
            console.print("[green]Goodbye![/green]")
        else:
            console.print("[red]Invalid choice. Please run again.[/red]")

def main():
    """Entry point for the CLI application."""
    cli = RougeCCLI()
    cli.run()

if __name__ == "__main__":
    main()