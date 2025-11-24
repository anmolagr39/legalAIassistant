"""
Master control script for LLM-as-a-Judge evaluation
Provides easy interface for all evaluation tasks
"""
import sys
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("=" * 80)
    print(" " * 20 + "LLM-as-a-Judge Evaluation System")
    print(" " * 25 + "Legal Assistant Evaluator")
    print("=" * 80)
    print()


def print_menu():
    """Print main menu"""
    print("Available Commands:")
    print()
    print("  Setup & Testing:")
    print("    1. test         - Test system setup and configuration")
    print()
    print("  Run Evaluations:")
    print("    2. quick <N>    - Quick evaluation with N questions (e.g., quick 5)")
    print("    3. full         - Full evaluation (all 30 questions)")
    print("    4. category <C> - Evaluate specific category (e.g., category IPC_Section)")
    print("    5. list-cat     - List available categories")
    print()
    print("  Analysis:")
    print("    6. analyze <F>  - Analyze results file (e.g., analyze results.json)")
    print("    7. visualize <F>- Create visualizations (e.g., visualize results.json)")
    print()
    print("  Help:")
    print("    8. help         - Show this menu")
    print("    9. exit         - Exit")
    print()


def run_test():
    """Run setup test"""
    import test_setup
    test_setup.main()


def run_quick_eval(n: int = 5):
    """Run quick evaluation"""
    from run_evaluation import run_quick_eval
    run_quick_eval(n)


def run_full_eval():
    """Run full evaluation"""
    from llm_judge_evaluator import main
    main()


def run_category_eval(category: str):
    """Run category evaluation"""
    from run_evaluation import run_category_eval
    run_category_eval(category)


def list_categories():
    """List available categories"""
    import json
    script_dir = Path(__file__).parent
    with open(script_dir / "ground_truth_dataset.json", 'r') as f:
        data = json.load(f)
    
    categories = {}
    for item in data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nAvailable Categories:")
    print("-" * 80)
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<30} ({count} questions)")
    print()


def analyze_results(filename: str):
    """Analyze results file"""
    from analyze_results import EvaluationAnalyzer
    
    filepath = Path(filename)
    if not filepath.exists():
        # Try looking in evaluation_results folder
        filepath = Path(__file__).parent / "evaluation_results" / filename
        if not filepath.exists():
            print(f"Error: File not found: {filename}")
            return
    
    print(f"\nAnalyzing: {filepath}")
    analyzer = EvaluationAnalyzer(str(filepath))
    
    output_file = filepath.parent / "detailed_analysis_report.txt"
    analyzer.generate_detailed_report(str(output_file))
    
    print(f"\n✓ Analysis complete! Report saved to: {output_file}")


def visualize_results(filename: str):
    """Create visualizations"""
    try:
        from visualize_results import EvaluationVisualizer, VISUALIZATION_AVAILABLE
        
        if not VISUALIZATION_AVAILABLE:
            print("\nVisualization packages not installed!")
            print("Install with: pip install matplotlib seaborn pandas")
            return
        
        filepath = Path(filename)
        if not filepath.exists():
            # Try looking in evaluation_results folder
            filepath = Path(__file__).parent / "evaluation_results" / filename
            if not filepath.exists():
                print(f"Error: File not found: {filename}")
                return
        
        print(f"\nCreating visualizations from: {filepath}")
        visualizer = EvaluationVisualizer(str(filepath))
        
        output_dir = filepath.parent / "visualizations"
        visualizer.generate_all_plots(str(output_dir))
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install visualization packages with: pip install matplotlib seaborn pandas")


def find_latest_results():
    """Find the most recent results file"""
    results_dir = Path(__file__).parent / "evaluation_results"
    if not results_dir.exists():
        return None
    
    json_files = list(results_dir.glob("*results*.json"))
    if not json_files:
        return None
    
    # Sort by modification time
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest


def interactive_mode():
    """Run in interactive mode"""
    print_banner()
    print("Running in interactive mode. Type 'help' for commands.\n")
    
    while True:
        try:
            command = input("eval> ").strip().lower()
            
            if not command:
                continue
            
            parts = command.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else None
            
            print()  # Blank line for readability
            
            if cmd in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            elif cmd in ['help', 'h', '?']:
                print_menu()
            
            elif cmd == 'test':
                run_test()
            
            elif cmd == 'quick':
                n = int(arg) if arg else 5
                run_quick_eval(n)
            
            elif cmd == 'full':
                print("Starting full evaluation (30 questions)...")
                print("This may take 10-15 minutes.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    run_full_eval()
            
            elif cmd == 'category':
                if not arg:
                    print("Usage: category <category_name>")
                    print("Use 'list-cat' to see available categories")
                else:
                    run_category_eval(arg)
            
            elif cmd == 'list-cat':
                list_categories()
            
            elif cmd == 'analyze':
                if not arg:
                    # Try to find latest results
                    latest = find_latest_results()
                    if latest:
                        print(f"No file specified. Using latest: {latest.name}")
                        analyze_results(str(latest))
                    else:
                        print("Usage: analyze <results_file.json>")
                else:
                    analyze_results(arg)
            
            elif cmd == 'visualize':
                if not arg:
                    # Try to find latest results
                    latest = find_latest_results()
                    if latest:
                        print(f"No file specified. Using latest: {latest.name}")
                        visualize_results(str(latest))
                    else:
                        print("Usage: visualize <results_file.json>")
                else:
                    visualize_results(arg)
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
            
            print()  # Blank line after command
            
        except KeyboardInterrupt:
            print("\n\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {e}")
            print()


def command_line_mode():
    """Run in command-line mode"""
    if len(sys.argv) < 2:
        print_banner()
        print("Usage: python master_eval.py <command> [args]")
        print()
        print_menu()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        run_test()
    
    elif command == 'quick':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_quick_eval(n)
    
    elif command == 'full':
        run_full_eval()
    
    elif command == 'category':
        if len(sys.argv) < 3:
            print("Usage: python master_eval.py category <category_name>")
        else:
            run_category_eval(sys.argv[2])
    
    elif command == 'list-cat':
        list_categories()
    
    elif command == 'analyze':
        if len(sys.argv) < 3:
            print("Usage: python master_eval.py analyze <results_file.json>")
        else:
            analyze_results(sys.argv[2])
    
    elif command == 'visualize':
        if len(sys.argv) < 3:
            print("Usage: python master_eval.py visualize <results_file.json>")
        else:
            visualize_results(sys.argv[2])
    
    elif command in ['interactive', 'i']:
        interactive_mode()
    
    else:
        print(f"Unknown command: {command}")
        print()
        print_menu()


def main():
    """Main entry point"""
    if len(sys.argv) == 1 or sys.argv[1].lower() in ['interactive', 'i']:
        interactive_mode()
    else:
        command_line_mode()


if __name__ == "__main__":
    main()
