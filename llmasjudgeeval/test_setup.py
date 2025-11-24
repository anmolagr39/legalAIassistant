"""
Test script to verify evaluation system setup
Run this before full evaluation to check configuration
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_environment():
    """Check environment variables"""
    print("Checking environment variables...")
    
    errors = []
    
    if not os.getenv('GEMINI_API_KEY'):
        errors.append("❌ GEMINI_API_KEY not set in .env")
    else:
        print("  ✓ GEMINI_API_KEY found")
    
    if not os.getenv('NEO4J_PASSWORD'):
        errors.append("❌ NEO4J_PASSWORD not set in .env")
    else:
        print("  ✓ NEO4J_PASSWORD found")
    
    print("  ✓ NEO4J_URI:", os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    print("  ✓ NEO4J_USER:", os.getenv('NEO4J_USER', 'neo4j'))
    
    return errors


def check_files():
    """Check required files exist"""
    print("\nChecking required files...")
    
    errors = []
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    # Check ground truth
    ground_truth = script_dir / "ground_truth_dataset.json"
    if ground_truth.exists():
        print(f"  ✓ Ground truth dataset found")
        import json
        with open(ground_truth, 'r') as f:
            data = json.load(f)
        print(f"    - {len(data)} questions loaded")
    else:
        errors.append(f"❌ Ground truth dataset not found: {ground_truth}")
    
    # Check orchestrator
    orchestrator_file = parent_dir / "agentic_orchestrator_v2.py"
    if orchestrator_file.exists():
        print(f"  ✓ Agentic orchestrator found")
    else:
        errors.append(f"❌ Orchestrator not found: {orchestrator_file}")
    
    return errors


def check_imports():
    """Check required packages can be imported"""
    print("\nChecking required packages...")
    
    errors = []
    
    packages = [
        ('google.generativeai', 'google-generativeai'),
        ('dotenv', 'python-dotenv'),
        ('chromadb', 'chromadb'),
        ('neo4j', 'neo4j')
    ]
    
    for module, package in packages:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            errors.append(f"❌ {package} not installed")
    
    return errors


def test_orchestrator_import():
    """Test importing the orchestrator"""
    print("\nTesting orchestrator import...")
    
    try:
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from agentic_orchestrator_v2 import AgenticOrchestrator
        print("  ✓ AgenticOrchestrator imported successfully")
        return []
    except Exception as e:
        return [f"❌ Failed to import orchestrator: {e}"]


def test_neo4j_connection():
    """Test Neo4j connection"""
    print("\nTesting Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD')
        
        if not password:
            return ["❌ NEO4J_PASSWORD not set, skipping connection test"]
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        driver.close()
        
        print("  ✓ Neo4j connection successful")
        return []
    except Exception as e:
        return [f"❌ Neo4j connection failed: {e}"]


def test_gemini_api():
    """Test Gemini API"""
    print("\nTesting Gemini API...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return ["❌ GEMINI_API_KEY not set"]
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        response = model.generate_content("Say 'Hello' in one word")
        result_text = response.text.strip()
        
        print(f"  ✓ Gemini API working (response: '{result_text}')")
        return []
    except Exception as e:
        return [f"❌ Gemini API test failed: {e}"]


def main():
    """Run all checks"""
    print("=" * 80)
    print("LLM-as-a-Judge Evaluation System - Setup Verification")
    print("=" * 80)
    
    all_errors = []
    
    # Run checks
    all_errors.extend(check_environment())
    all_errors.extend(check_files())
    all_errors.extend(check_imports())
    all_errors.extend(test_orchestrator_import())
    all_errors.extend(test_neo4j_connection())
    all_errors.extend(test_gemini_api())
    
    print("\n" + "=" * 80)
    
    if all_errors:
        print("❌ SETUP INCOMPLETE - Issues found:")
        print("=" * 80)
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the issues above before running evaluation.")
        return False
    else:
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print("\nYour evaluation system is ready!")
        print("\nNext steps:")
        print("  1. Run quick test: python run_evaluation.py --quick 3")
        print("  2. Run full evaluation: python llm_judge_evaluator.py")
        print("  3. Analyze results: python analyze_results.py <results_file>")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
