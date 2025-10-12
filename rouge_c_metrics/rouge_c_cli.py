"""
ROUGE-C CLI: Interactive command-line interface for evaluating IPC RAG system with ROUGE-C metrics.
"""

import json
import argparse
from typing import Dict, List, Any
from rouge_c import RougeC
import sys
import os

class RougeCCLI:
    """Command-line interface for ROUGE-C evaluation of IPC RAG systems."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.rouge_c = None
        
    def setup_rouge_c(self, embedding_model: str = "all-MiniLM-L6-v2", 
                     use_stemming: bool = False, remove_stopwords: bool = True):
        """
        Setup ROUGE-C evaluator with specified configuration.
        
        Args:
            embedding_model: Sentence transformer model for semantic similarity
            use_stemming: Whether to apply stemming
            remove_stopwords: Whether to remove stopwords
        """
        print(f"🔧 Setting up ROUGE-C with model: {embedding_model}")
        self.rouge_c = RougeC(
            embedding_model=embedding_model,
            use_stemming=use_stemming,
            remove_stopwords=remove_stopwords
        )
        print("✅ ROUGE-C setup complete!")
    
    def load_qa_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load Q-A evaluation data from JSON file.
        
        Expected format:
        [
            {
                "question": "What is IPC section 302?",
                "generated_answer": "IPC section 302 deals with...",
                "retrieved_contexts": ["Context 1", "Context 2", ...]
            },
            ...
        ]
        
        Args:
            filepath: Path to JSON file with Q-A data
            
        Returns:
            List of Q-A dictionaries
        """
        print(f"📂 Loading Q-A data from: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded {len(data)} Q-A pairs")
            
            # Validate data format
            for i, item in enumerate(data):
                if not all(key in item for key in ['question', 'generated_answer', 'retrieved_contexts']):
                    print(f"⚠️ Warning: Q-A pair {i+1} missing required fields")
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format: {e}")
            return []
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []
    
    def evaluate_single(self, question: str, generated_answer: str, contexts: List[str]):
        """
        Evaluate a single Q-A pair.
        
        Args:
            question: The question text
            generated_answer: Generated answer text
            contexts: List of retrieved context texts
        """
        if not self.rouge_c:
            print("❌ Error: ROUGE-C not initialized. Run setup first.")
            return
        
        print(f"\n🔍 Evaluating single Q-A pair...")
        print(f"Question: {question[:100]}...")
        print(f"Answer: {generated_answer[:100]}...")
        print(f"Contexts: {len(contexts)} retrieved")
        
        result = self.rouge_c.compute_rouge_c_single(generated_answer, contexts)
        self.rouge_c.display_results(result)
        
        return result
    
    def evaluate_batch(self, qa_data: List[Dict[str, Any]], output_file: str = None):
        """
        Evaluate multiple Q-A pairs.
        
        Args:
            qa_data: List of Q-A dictionaries
            output_file: Optional file to save results
        """
        if not self.rouge_c:
            print("❌ Error: ROUGE-C not initialized. Run setup first.")
            return
        
        if not qa_data:
            print("❌ Error: No Q-A data provided")
            return
        
        print(f"\n🔍 Starting batch evaluation of {len(qa_data)} Q-A pairs...")
        
        results = self.rouge_c.compute_rouge_c_batch(qa_data)
        
        if results:
            self.rouge_c.display_results(results)
            
            if output_file:
                self.rouge_c.save_results(results, output_file)
        
        return results
    
    def interactive_evaluation(self):
        """Run interactive evaluation mode."""
        print("\n🎯 ROUGE-C Interactive Evaluation Mode")
        print("="*50)
        
        if not self.rouge_c:
            print("Setting up ROUGE-C with default configuration...")
            self.setup_rouge_c()
        
        while True:
            print("\nChoose an option:")
            print("1. Evaluate single Q-A pair")
            print("2. Load and evaluate batch from file")
            print("3. Reconfigure ROUGE-C")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self._interactive_single_evaluation()
            elif choice == '2':
                self._interactive_batch_evaluation()
            elif choice == '3':
                self._interactive_reconfigure()
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
    
    def _interactive_single_evaluation(self):
        """Interactive single evaluation."""
        print("\n📝 Single Q-A Evaluation")
        print("-" * 30)
        
        question = input("Enter question: ").strip()
        if not question:
            print("❌ Question cannot be empty")
            return
        
        generated_answer = input("Enter generated answer: ").strip()
        if not generated_answer:
            print("❌ Generated answer cannot be empty")
            return
        
        print("Enter retrieved contexts (enter empty line to finish):")
        contexts = []
        while True:
            context = input(f"Context {len(contexts)+1}: ").strip()
            if not context:
                break
            contexts.append(context)
        
        if not contexts:
            print("❌ At least one context is required")
            return
        
        self.evaluate_single(question, generated_answer, contexts)
    
    def _interactive_batch_evaluation(self):
        """Interactive batch evaluation."""
        print("\n📂 Batch Evaluation from File")
        print("-" * 30)
        
        filepath = input("Enter path to Q-A data file: ").strip()
        if not filepath:
            print("❌ File path cannot be empty")
            return
        
        qa_data = self.load_qa_data(filepath)
        if not qa_data:
            return
        
        output_file = input("Enter output file path (optional): ").strip()
        if not output_file:
            output_file = None
        
        self.evaluate_batch(qa_data, output_file)
    
    def _interactive_reconfigure(self):
        """Interactive reconfiguration."""
        print("\n⚙️ Reconfigure ROUGE-C")
        print("-" * 25)
        
        print("Available embedding models:")
        print("1. all-MiniLM-L6-v2 (default, fast)")
        print("2. all-mpnet-base-v2 (more accurate)")
        print("3. custom (enter your own)")
        
        model_choice = input("Choose embedding model (1-3): ").strip()
        
        if model_choice == '1':
            embedding_model = "all-MiniLM-L6-v2"
        elif model_choice == '2':
            embedding_model = "all-mpnet-base-v2"
        elif model_choice == '3':
            embedding_model = input("Enter custom model name: ").strip()
            if not embedding_model:
                print("❌ Invalid model name")
                return
        else:
            print("❌ Invalid choice, using default")
            embedding_model = "all-MiniLM-L6-v2"
        
        use_stemming = input("Use stemming? (y/n): ").strip().lower() == 'y'
        remove_stopwords = input("Remove stopwords? (y/n): ").strip().lower() == 'y'
        
        self.setup_rouge_c(embedding_model, use_stemming, remove_stopwords)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ROUGE-C: Reference-free evaluation for IPC RAG systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python rouge_c_cli.py
  
  # Batch evaluation
  python rouge_c_cli.py --batch qa_data.json --output results.json
  
  # Single evaluation
  python rouge_c_cli.py --question "What is IPC 302?" --answer "Murder" --contexts "Section 302..."
  
  # Custom configuration
  python rouge_c_cli.py --batch qa_data.json --model all-mpnet-base-v2 --stemming
        """
    )
    
    # Evaluation mode
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode (default)')
    parser.add_argument('--batch', '-b', type=str,
                        help='Path to JSON file with Q-A pairs for batch evaluation')
    
    # Single evaluation
    parser.add_argument('--question', '-q', type=str,
                        help='Question for single evaluation')
    parser.add_argument('--answer', '-a', type=str,
                        help='Generated answer for single evaluation')
    parser.add_argument('--contexts', '-c', nargs='+',
                        help='Retrieved contexts for single evaluation')
    
    # Configuration
    parser.add_argument('--model', '-m', type=str, default='all-MiniLM-L6-v2',
                        help='Embedding model for semantic similarity')
    parser.add_argument('--stemming', action='store_true',
                        help='Enable stemming (requires nltk)')
    parser.add_argument('--keep-stopwords', action='store_true',
                        help='Keep stopwords (default: remove)')
    
    # Output
    parser.add_argument('--output', '-o', type=str,
                        help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = RougeCCLI()
    
    # Setup ROUGE-C
    cli.setup_rouge_c(
        embedding_model=args.model,
        use_stemming=args.stemming,
        remove_stopwords=not args.keep_stopwords
    )
    
    # Run evaluation based on arguments
    if args.batch:
        # Batch evaluation
        qa_data = cli.load_qa_data(args.batch)
        if qa_data:
            cli.evaluate_batch(qa_data, args.output)
    
    elif args.question and args.answer and args.contexts:
        # Single evaluation
        cli.evaluate_single(args.question, args.answer, args.contexts)
    
    else:
        # Interactive mode (default)
        cli.interactive_evaluation()

if __name__ == "__main__":
    main()