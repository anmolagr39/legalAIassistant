"""
Quick evaluation script with configurable options
Run subset evaluations for testing
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from llmasjudgeeval.llm_judge_evaluator import AgenticOrchestrator, LLMJudge, EvaluationRunner

load_dotenv()


def run_quick_eval(num_questions: int = 5):
    """Run quick evaluation on a subset of questions"""
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    script_dir = Path(__file__).parent
    ground_truth_file = script_dir / "ground_truth_dataset.json"
    output_dir = script_dir / "evaluation_results"
    
    print("=" * 80)
    print(f"Quick Evaluation - Testing with {num_questions} questions")
    print("=" * 80)
    
    # Initialize
    orchestrator = AgenticOrchestrator()
    judge = LLMJudge(gemini_api_key=GEMINI_API_KEY, judge_model="gemini-2.0-flash-lite")
    runner = EvaluationRunner(orchestrator, judge, str(output_dir))
    
    # Run
    runner.run_evaluation(
        ground_truth_file=str(ground_truth_file),
        max_questions=num_questions,
        delay_seconds=1.5
    )


def run_category_eval(category: str):
    """Run evaluation on specific category only"""
    import json
    
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    script_dir = Path(__file__).parent
    ground_truth_file = script_dir / "ground_truth_dataset.json"
    
    # Load and filter by category
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = [item for item in data if item['category'] == category]
    
    if not filtered_data:
        print(f"No questions found for category: {category}")
        return
    
    # Save filtered data temporarily
    temp_file = script_dir / f"temp_{category}.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Evaluating {len(filtered_data)} questions from category: {category}")
    
    output_dir = script_dir / "evaluation_results" / category
    
    orchestrator = AgenticOrchestrator()
    judge = LLMJudge(gemini_api_key=GEMINI_API_KEY, judge_model="gemini-2.0-flash-lite")
    runner = EvaluationRunner(orchestrator, judge, str(output_dir))
    
    runner.run_evaluation(
        ground_truth_file=str(temp_file),
        max_questions=None,
        delay_seconds=2.0
    )
    
    # Clean up temp file
    temp_file.unlink()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Legal Assistant Evaluation")
    parser.add_argument('--quick', type=int, help='Run quick eval with N questions', metavar='N')
    parser.add_argument('--category', type=str, help='Run eval for specific category')
    parser.add_argument('--list-categories', action='store_true', help='List available categories')
    
    args = parser.parse_args()
    
    if args.list_categories:
        import json
        script_dir = Path(__file__).parent
        with open(script_dir / "ground_truth_dataset.json", 'r') as f:
            data = json.load(f)
        categories = sorted(set(item['category'] for item in data))
        print("\nAvailable categories:")
        for cat in categories:
            count = sum(1 for item in data if item['category'] == cat)
            print(f"  - {cat} ({count} questions)")
        print()
    elif args.quick:
        run_quick_eval(args.quick)
    elif args.category:
        run_category_eval(args.category)
    else:
        print("Usage:")
        print("  python run_evaluation.py --quick 5          # Test with 5 questions")
        print("  python run_evaluation.py --category IPC_Section  # Eval specific category")
        print("  python run_evaluation.py --list-categories  # List all categories")
        print("\nFor full evaluation, use: python llm_judge_evaluator.py")
