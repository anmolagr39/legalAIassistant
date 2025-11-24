"""
LLM-as-a-Judge Evaluation System for Orchestrator WITHOUT Knowledge Graph
Evaluates orchestrator responses (3 RAGs only, no KG) against ground truth using LLM judgment
"""
import os
import sys
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Import necessary components from orchestrator
from config.neo4j_config import Neo4jConnection
from extraction.groq_extractor import GroqExtractor


class OrchestratorNoKG:
    """Orchestrator with only RAG systems (no Knowledge Graph)"""
    
    def __init__(self):
        """Initialize orchestrator with 3 RAG systems only"""
        # Import from actual orchestrator to reuse existing RAG systems
        from agentic_orchestrator_v2 import IPCRagSystem, PastCasesRagSystem, LegalActsRagSystem
        
        # Initialize RAG systems (same as full orchestrator)
        print("Initializing IPC RAG...")
        self.ipc_rag = IPCRagSystem()
        
        print("Initializing Past Cases RAG...")
        self.past_cases_rag = PastCasesRagSystem()
        
        print("Initializing Legal Acts RAG...")
        self.legal_acts_rag = LegalActsRagSystem()
        
        # Initialize Gemini for answer synthesis
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        
    def process_query(self, query: str) -> Dict:
        """
        Process query using only RAG systems (no KG)
        
        Args:
            query: User query
            
        Returns:
            Dict with final_answer and routing info
        """
        try:
            # Query all 3 RAG systems
            print(f"  → Querying IPC RAG...")
            ipc_result = self.ipc_rag.query(query)
            
            print(f"  → Querying Past Cases RAG...")
            cases_result = self.past_cases_rag.query(query)
            
            print(f"  → Querying Legal Acts RAG...")
            legal_result = self.legal_acts_rag.query(query)
            
            # Combine context from all RAG systems
            combined_context = ""
            systems_used = []
            
            if ipc_result.get('answer'):
                combined_context += "\n\n=== IPC RAG ===\n"
                combined_context += ipc_result['answer']
                systems_used.append('ipc_rag')
            
            if cases_result.get('answer'):
                combined_context += "\n\n=== Past Cases RAG ===\n"
                combined_context += cases_result['answer']
                systems_used.append('past_cases_rag')
            
            if legal_result.get('answer'):
                combined_context += "\n\n=== Legal Acts RAG ===\n"
                combined_context += legal_result['answer']
                systems_used.append('legal_acts_rag')
            
            # Generate final answer using Gemini
            synthesis_prompt = f"""You are a legal expert assistant. Based on the following information from different legal databases, synthesize a comprehensive answer to the user's question.

Information from Legal Databases:
{combined_context}

User Question: {query}

Provide a detailed, accurate answer that synthesizes the information above. If the databases provide relevant information, use it. If not, provide the best answer you can based on what's available."""

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=synthesis_prompt
            )
            
            final_answer = response.text.strip()
            
            return {
                'final_answer': final_answer,
                'routing': {
                    'systems': systems_used if systems_used else ['no_results']
                }
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'final_answer': f"Error processing query: {str(e)}",
                'routing': {
                    'systems': ['error']
                }
            }


class LLMJudge:
    """LLM-as-a-Judge for evaluating legal assistant responses"""
    
    def __init__(self, gemini_api_key: str, judge_model: str = "gemini-2.5-flash"):
        """Initialize LLM Judge"""
        self.client = genai.Client(api_key=gemini_api_key)
        self.judge_model = judge_model
        
    def evaluate_response(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        category: str
    ) -> Dict:
        """Evaluate a generated answer against ground truth"""
        
        evaluation_prompt = f"""You are an expert legal evaluator. Your task is to evaluate the quality of a legal assistant's answer against the ground truth.

QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

CATEGORY: {category}

Please evaluate the generated answer on the following criteria (score each 0-10):

1. CORRECTNESS (0-10):
   - Are the legal facts, case names, section numbers, and dates accurate?
   - Are there any factual errors or misstatements of law?
   - 10 = Perfectly accurate, 0 = Completely wrong

2. COMPLETENESS (0-10):
   - Does the answer cover all key points from the ground truth?
   - Are important details missing?
   - 10 = All key points covered, 0 = Missing all information

3. RELEVANCE (0-10):
   - Does the answer directly address the question?
   - Is there unnecessary or off-topic information?
   - 10 = Perfectly relevant, 0 = Off-topic

4. LEGAL REASONING (0-10):
   - Is the legal reasoning sound and well-explained?
   - Are citations and precedents used appropriately?
   - 10 = Excellent reasoning, 0 = Poor or no reasoning

Provide your evaluation in the following JSON format:
{{
    "correctness_score": <0-10>,
    "completeness_score": <0-10>,
    "relevance_score": <0-10>,
    "legal_reasoning_score": <0-10>,
    "overall_score": <0-10>,
    "verdict": "<PASS|FAIL>",
    "reasoning": "<detailed explanation>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...]
}}

IMPORTANT:
- Be fair in your evaluation
- PASS if overall_score >= 5.0, FAIL otherwise
- For legal questions, accuracy of case names, sections, and legal principles is critical
- Consider that the generated answer might phrase things differently but still be correct
- Focus on substance over style
"""

        try:
            response = self.client.models.generate_content(
                model=self.judge_model,
                contents=evaluation_prompt
            )
            response_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            evaluation = json.loads(json_text)
            
            # Validate required fields
            required_fields = ["correctness_score", "completeness_score", "relevance_score", 
                             "legal_reasoning_score", "overall_score", "verdict", "reasoning"]
            for field in required_fields:
                if field not in evaluation:
                    raise ValueError(f"Missing required field: {field}")
            
            return evaluation
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "correctness_score": 0,
                "completeness_score": 0,
                "relevance_score": 0,
                "legal_reasoning_score": 0,
                "overall_score": 0,
                "verdict": "FAIL",
                "reasoning": f"Evaluation failed due to error: {str(e)}",
                "strengths": [],
                "weaknesses": ["Evaluation error occurred"]
            }


class EvaluationRunner:
    """Runs evaluation on ground truth dataset"""
    
    def __init__(
        self,
        orchestrator: OrchestratorNoKG,
        judge: LLMJudge,
        output_dir: str = "orchestrator_no_kg_evaluation_results"
    ):
        self.orchestrator = orchestrator
        self.judge = judge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_ground_truth(self, filepath: str) -> List[Dict]:
        """Load ground truth dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_evaluation(
        self,
        ground_truth_file: str,
        max_questions: Optional[int] = None,
        delay_seconds: float = 2.0
    ) -> Dict:
        """Run evaluation on ground truth dataset"""
        
        print("=" * 80)
        print("LLM-as-a-Judge Evaluation for Orchestrator (No KG - 3 RAGs Only)")
        print("=" * 80)
        
        ground_truth = self.load_ground_truth(ground_truth_file)
        if max_questions:
            ground_truth = ground_truth[:max_questions]
        
        print(f"\nLoaded {len(ground_truth)} questions from ground truth dataset")
        print(f"Output directory: {self.output_dir}")
        print(f"Delay between questions: {delay_seconds}s")
        print("\n" + "=" * 80 + "\n")
        
        results = []
        start_time = time.time()
        
        for idx, item in enumerate(ground_truth, 1):
            question_id = item['id']
            question = item['question']
            ground_truth_answer = item['ground_truth_answer']
            category = item['category']
            source = item['source']
            
            print(f"[{idx}/{len(ground_truth)}] Evaluating Question {question_id}")
            print(f"Category: {category}")
            print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
            
            try:
                # Get answer from orchestrator (no KG)
                print("  → Querying orchestrator (3 RAGs only)...")
                query_start = time.time()
                orchestrator_response = self.orchestrator.process_query(question)
                query_time = time.time() - query_start
                
                generated_answer = orchestrator_response.get('final_answer', '')
                routing = orchestrator_response.get('routing', {})
                systems_used = routing.get('systems', [])
                
                print(f"  → Got response in {query_time:.2f}s")
                print(f"  → Systems used: {', '.join(systems_used)}")
                
                # Evaluate with LLM judge
                print("  → Evaluating with LLM judge...")
                eval_start = time.time()
                evaluation = self.judge.evaluate_response(
                    question=question,
                    ground_truth=ground_truth_answer,
                    generated_answer=generated_answer,
                    category=category
                )
                eval_time = time.time() - eval_start
                
                print(f"  → Evaluation complete in {eval_time:.2f}s")
                print(f"  → Verdict: {evaluation['verdict']} (Overall Score: {evaluation['overall_score']}/10)")
                
                # Compile result
                result = {
                    "question_id": question_id,
                    "question": question,
                    "category": category,
                    "source": source,
                    "ground_truth": ground_truth_answer,
                    "generated_answer": generated_answer,
                    "systems_used": systems_used,
                    "query_time": query_time,
                    "evaluation_time": eval_time,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Save intermediate results
                self._save_results(results, prefix="intermediate")
                
                print(f"  ✓ Question {question_id} complete\n")
                
            except Exception as e:
                print(f"  ✗ Error processing question {question_id}: {e}\n")
                result = {
                    "question_id": question_id,
                    "question": question,
                    "category": category,
                    "source": source,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
            
            # Delay to avoid rate limits
            if idx < len(ground_truth):
                time.sleep(delay_seconds)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        statistics = self._calculate_statistics(results, total_time)
        
        # Save final results
        final_output = {
            "metadata": {
                "total_questions": len(ground_truth),
                "completed": len([r for r in results if 'evaluation' in r]),
                "failed": len([r for r in results if 'error' in r]),
                "total_time_seconds": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "statistics": statistics,
            "results": results
        }
        
        self._save_results([final_output], prefix="final", filename="evaluation_report.json")
        self._generate_summary_report(final_output)
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        print(f"\nTotal Questions: {statistics['total_questions']}")
        print(f"Passed: {statistics['passed']} ({statistics['pass_rate']:.1f}%)")
        print(f"Failed: {statistics['failed']} ({statistics['fail_rate']:.1f}%)")
        print(f"Average Overall Score: {statistics['average_overall_score']:.2f}/10")
        print(f"\nResults saved to: {self.output_dir}")
        
        return final_output
    
    def _calculate_statistics(self, results: List[Dict], total_time: float) -> Dict:
        """Calculate evaluation statistics"""
        
        completed_results = [r for r in results if 'evaluation' in r]
        
        if not completed_results:
            return {
                "total_questions": len(results),
                "completed": 0,
                "failed": len(results),
                "pass_rate": 0.0,
                "fail_rate": 100.0
            }
        
        passed = sum(1 for r in completed_results if r['evaluation']['verdict'] == 'PASS')
        failed = len(completed_results) - passed
        
        # Average scores
        avg_correctness = sum(r['evaluation']['correctness_score'] for r in completed_results) / len(completed_results)
        avg_completeness = sum(r['evaluation']['completeness_score'] for r in completed_results) / len(completed_results)
        avg_relevance = sum(r['evaluation']['relevance_score'] for r in completed_results) / len(completed_results)
        avg_legal_reasoning = sum(r['evaluation']['legal_reasoning_score'] for r in completed_results) / len(completed_results)
        avg_overall = sum(r['evaluation']['overall_score'] for r in completed_results) / len(completed_results)
        
        # Category breakdown
        category_stats = {}
        for result in completed_results:
            cat = result['category']
            if cat not in category_stats:
                category_stats[cat] = {
                    "total": 0,
                    "passed": 0,
                    "scores": []
                }
            category_stats[cat]["total"] += 1
            category_stats[cat]["scores"].append(result['evaluation']['overall_score'])
            if result['evaluation']['verdict'] == 'PASS':
                category_stats[cat]["passed"] += 1
        
        for cat in category_stats:
            stats = category_stats[cat]
            stats["pass_rate"] = (stats["passed"] / stats["total"]) * 100
            stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
            del stats["scores"]
        
        return {
            "total_questions": len(results),
            "completed": len(completed_results),
            "failed": len(results) - len(completed_results),
            "passed": passed,
            "not_passed": failed,
            "pass_rate": (passed / len(completed_results)) * 100,
            "fail_rate": (failed / len(completed_results)) * 100,
            "average_correctness_score": round(avg_correctness, 2),
            "average_completeness_score": round(avg_completeness, 2),
            "average_relevance_score": round(avg_relevance, 2),
            "average_legal_reasoning_score": round(avg_legal_reasoning, 2),
            "average_overall_score": round(avg_overall, 2),
            "total_time_seconds": round(total_time, 2),
            "average_query_time": round(sum(r.get('query_time', 0) for r in completed_results) / len(completed_results), 2),
            "average_evaluation_time": round(sum(r.get('evaluation_time', 0) for r in completed_results) / len(completed_results), 2),
            "category_breakdown": category_stats
        }
    
    def _save_results(self, results: List[Dict], prefix: str = "", filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_results_{timestamp}.json" if prefix else f"results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _generate_summary_report(self, evaluation_output: Dict):
        """Generate human-readable summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"summary_report_{timestamp}.txt"
        
        stats = evaluation_output['statistics']
        metadata = evaluation_output['metadata']
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM-as-a-Judge Evaluation Summary Report\n")
            f.write("Orchestrator WITHOUT Knowledge Graph (3 RAGs Only)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {metadata['timestamp']}\n")
            f.write(f"Total Time: {stats['total_time_seconds']:.2f} seconds\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Questions: {stats['total_questions']}\n")
            f.write(f"Completed: {stats['completed']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            f.write(f"Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)\n")
            f.write(f"Not Passed: {stats['not_passed']} ({stats['fail_rate']:.1f}%)\n\n")
            
            f.write("AVERAGE SCORES (0-10)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correctness:     {stats['average_correctness_score']:.2f}/10\n")
            f.write(f"Completeness:    {stats['average_completeness_score']:.2f}/10\n")
            f.write(f"Relevance:       {stats['average_relevance_score']:.2f}/10\n")
            f.write(f"Legal Reasoning: {stats['average_legal_reasoning_score']:.2f}/10\n")
            f.write(f"Overall:         {stats['average_overall_score']:.2f}/10\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Query Time: {stats['average_query_time']:.2f}s\n")
            f.write(f"Average Evaluation Time: {stats['average_evaluation_time']:.2f}s\n\n")
            
            if 'category_breakdown' in stats:
                f.write("CATEGORY BREAKDOWN\n")
                f.write("-" * 80 + "\n")
                for category, cat_stats in stats['category_breakdown'].items():
                    f.write(f"\n{category}:\n")
                    f.write(f"  Total: {cat_stats['total']}\n")
                    f.write(f"  Passed: {cat_stats['passed']} ({cat_stats['pass_rate']:.1f}%)\n")
                    f.write(f"  Average Score: {cat_stats['average_score']:.2f}/10\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nSummary report saved to: {filepath}")


def main():
    """Main execution function"""
    
    # Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    # Paths
    script_dir = Path(__file__).parent
    ground_truth_file = script_dir / "ground_truth_dataset.json"
    output_dir = script_dir / "orchestrator_no_kg_evaluation_results"
    
    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")
    
    print("Initializing evaluation system...")
    print(f"Ground truth file: {ground_truth_file}")
    print(f"Output directory: {output_dir}")
    
    # Initialize components
    print("\nInitializing Orchestrator (No KG - 3 RAGs Only)...")
    orchestrator = OrchestratorNoKG()
    
    print("Initializing LLM Judge...")
    judge = LLMJudge(gemini_api_key=GEMINI_API_KEY, judge_model="gemini-2.5-flash")
    
    print("Initializing Evaluation Runner...")
    runner = EvaluationRunner(
        orchestrator=orchestrator,
        judge=judge,
        output_dir=str(output_dir)
    )
    
    # Run evaluation
    print("\nStarting evaluation...\n")
    results = runner.run_evaluation(
        ground_truth_file=str(ground_truth_file),
        max_questions=None,
        delay_seconds=2.0
    )
    
    print("\n✓ Evaluation complete!")
    print(f"Check {output_dir} for detailed results")


if __name__ == "__main__":
    main()
