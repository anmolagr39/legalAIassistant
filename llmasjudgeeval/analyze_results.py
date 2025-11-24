"""
Analysis tools for evaluation results
Analyze and visualize evaluation metrics
"""
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class EvaluationAnalyzer:
    """Analyze evaluation results and generate insights"""
    
    def __init__(self, results_file: str):
        """Load evaluation results from JSON file"""
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                # Handle intermediate results format
                self.data = data[0] if 'metadata' in data[0] else {'results': data}
            else:
                self.data = data
        
        self.results = self.data.get('results', [])
        self.statistics = self.data.get('statistics', {})
        self.metadata = self.data.get('metadata', {})
    
    def get_failed_questions(self) -> List[Dict]:
        """Get all questions that failed evaluation"""
        failed = []
        for result in self.results:
            if 'evaluation' in result and result['evaluation']['verdict'] == 'FAIL':
                failed.append({
                    'question_id': result['question_id'],
                    'question': result['question'],
                    'category': result['category'],
                    'score': result['evaluation']['overall_score'],
                    'reasoning': result['evaluation']['reasoning'],
                    'weaknesses': result['evaluation'].get('weaknesses', [])
                })
        return failed
    
    def get_top_performers(self, n: int = 5) -> List[Dict]:
        """Get top N performing questions"""
        scored = [r for r in self.results if 'evaluation' in r]
        sorted_results = sorted(
            scored,
            key=lambda x: x['evaluation']['overall_score'],
            reverse=True
        )
        
        return [{
            'question_id': r['question_id'],
            'question': r['question'],
            'category': r['category'],
            'score': r['evaluation']['overall_score'],
            'verdict': r['evaluation']['verdict']
        } for r in sorted_results[:n]]
    
    def get_worst_performers(self, n: int = 5) -> List[Dict]:
        """Get worst N performing questions"""
        scored = [r for r in self.results if 'evaluation' in r]
        sorted_results = sorted(
            scored,
            key=lambda x: x['evaluation']['overall_score']
        )
        
        return [{
            'question_id': r['question_id'],
            'question': r['question'],
            'category': r['category'],
            'score': r['evaluation']['overall_score'],
            'verdict': r['evaluation']['verdict'],
            'weaknesses': r['evaluation'].get('weaknesses', [])
        } for r in sorted_results[:n]]
    
    def analyze_by_category(self) -> Dict:
        """Detailed analysis by category"""
        category_data = defaultdict(lambda: {
            'questions': [],
            'scores': [],
            'passed': 0,
            'failed': 0
        })
        
        for result in self.results:
            if 'evaluation' not in result:
                continue
            
            cat = result['category']
            category_data[cat]['questions'].append(result['question_id'])
            category_data[cat]['scores'].append(result['evaluation']['overall_score'])
            
            if result['evaluation']['verdict'] == 'PASS':
                category_data[cat]['passed'] += 1
            else:
                category_data[cat]['failed'] += 1
        
        # Calculate stats
        analysis = {}
        for cat, data in category_data.items():
            analysis[cat] = {
                'total_questions': len(data['questions']),
                'passed': data['passed'],
                'failed': data['failed'],
                'pass_rate': (data['passed'] / len(data['questions'])) * 100,
                'average_score': sum(data['scores']) / len(data['scores']),
                'min_score': min(data['scores']),
                'max_score': max(data['scores']),
                'question_ids': data['questions']
            }
        
        return analysis
    
    def analyze_by_source(self) -> Dict:
        """Analyze performance by data source"""
        source_data = defaultdict(lambda: {
            'questions': [],
            'scores': [],
            'passed': 0,
            'failed': 0
        })
        
        for result in self.results:
            if 'evaluation' not in result:
                continue
            
            source = result.get('source', 'Unknown')
            source_data[source]['questions'].append(result['question_id'])
            source_data[source]['scores'].append(result['evaluation']['overall_score'])
            
            if result['evaluation']['verdict'] == 'PASS':
                source_data[source]['passed'] += 1
            else:
                source_data[source]['failed'] += 1
        
        # Calculate stats
        analysis = {}
        for source, data in source_data.items():
            analysis[source] = {
                'total_questions': len(data['questions']),
                'passed': data['passed'],
                'failed': data['failed'],
                'pass_rate': (data['passed'] / len(data['questions'])) * 100,
                'average_score': sum(data['scores']) / len(data['scores'])
            }
        
        return analysis
    
    def get_common_weaknesses(self) -> List[tuple]:
        """Identify most common weaknesses across all evaluations"""
        weakness_counts = defaultdict(int)
        
        for result in self.results:
            if 'evaluation' in result:
                for weakness in result['evaluation'].get('weaknesses', []):
                    weakness_counts[weakness] += 1
        
        return sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)
    
    def get_score_distribution(self) -> Dict:
        """Get distribution of scores"""
        distribution = {
            '9-10': 0,
            '7-8.9': 0,
            '5-6.9': 0,
            '3-4.9': 0,
            '0-2.9': 0
        }
        
        for result in self.results:
            if 'evaluation' not in result:
                continue
            
            score = result['evaluation']['overall_score']
            if score >= 9:
                distribution['9-10'] += 1
            elif score >= 7:
                distribution['7-8.9'] += 1
            elif score >= 5:
                distribution['5-6.9'] += 1
            elif score >= 3:
                distribution['3-4.9'] += 1
            else:
                distribution['0-2.9'] += 1
        
        return distribution
    
    def generate_detailed_report(self, output_file: str):
        """Generate comprehensive analysis report"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED EVALUATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in self.statistics.items():
                if key != 'category_breakdown':
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Score distribution
            f.write("SCORE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            dist = self.get_score_distribution()
            for range_label, count in dist.items():
                f.write(f"{range_label}: {count} questions\n")
            f.write("\n")
            
            # Category analysis
            f.write("CATEGORY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            cat_analysis = self.analyze_by_category()
            for cat, stats in cat_analysis.items():
                f.write(f"\n{cat}:\n")
                f.write(f"  Total: {stats['total_questions']}\n")
                f.write(f"  Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)\n")
                f.write(f"  Average Score: {stats['average_score']:.2f}\n")
                f.write(f"  Score Range: {stats['min_score']:.1f} - {stats['max_score']:.1f}\n")
            f.write("\n")
            
            # Source analysis
            f.write("SOURCE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            source_analysis = self.analyze_by_source()
            for source, stats in source_analysis.items():
                f.write(f"\n{source}:\n")
                f.write(f"  Total: {stats['total_questions']}\n")
                f.write(f"  Passed: {stats['passed']} ({stats['pass_rate']:.1f}%)\n")
                f.write(f"  Average Score: {stats['average_score']:.2f}\n")
            f.write("\n")
            
            # Top performers
            f.write("TOP 5 PERFORMING QUESTIONS\n")
            f.write("-" * 80 + "\n")
            for i, q in enumerate(self.get_top_performers(5), 1):
                f.write(f"\n{i}. Question {q['question_id']} (Score: {q['score']}/10)\n")
                f.write(f"   Category: {q['category']}\n")
                f.write(f"   Question: {q['question'][:100]}...\n")
            f.write("\n")
            
            # Worst performers
            f.write("BOTTOM 5 PERFORMING QUESTIONS\n")
            f.write("-" * 80 + "\n")
            for i, q in enumerate(self.get_worst_performers(5), 1):
                f.write(f"\n{i}. Question {q['question_id']} (Score: {q['score']}/10)\n")
                f.write(f"   Category: {q['category']}\n")
                f.write(f"   Question: {q['question'][:100]}...\n")
                if q['weaknesses']:
                    f.write(f"   Weaknesses: {', '.join(q['weaknesses'][:3])}\n")
            f.write("\n")
            
            # Common weaknesses
            f.write("COMMON WEAKNESSES (Top 10)\n")
            f.write("-" * 80 + "\n")
            for weakness, count in self.get_common_weaknesses()[:10]:
                f.write(f"{count}x: {weakness}\n")
            f.write("\n")
            
            # Failed questions details
            failed = self.get_failed_questions()
            f.write(f"FAILED QUESTIONS DETAILS ({len(failed)} total)\n")
            f.write("-" * 80 + "\n")
            for q in failed:
                f.write(f"\nQuestion {q['question_id']} (Score: {q['score']}/10)\n")
                f.write(f"Category: {q['category']}\n")
                f.write(f"Question: {q['question'][:150]}...\n")
                f.write(f"Reasoning: {q['reasoning'][:200]}...\n")
                if q['weaknesses']:
                    f.write(f"Weaknesses:\n")
                    for w in q['weaknesses']:
                        f.write(f"  - {w}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Analysis Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"Detailed analysis saved to: {output_file}")


def main():
    """Main analysis function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("\nExample: python analyze_results.py evaluation_results/final_results_20231122_143000.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        return
    
    print(f"Analyzing results from: {results_file}")
    
    analyzer = EvaluationAnalyzer(results_file)
    
    # Generate detailed report
    output_file = Path(results_file).parent / "detailed_analysis_report.txt"
    analyzer.generate_detailed_report(str(output_file))
    
    # Print quick summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    
    if analyzer.statistics:
        stats = analyzer.statistics
        print(f"\nTotal Questions: {stats.get('total_questions', 0)}")
        print(f"Pass Rate: {stats.get('pass_rate', 0):.1f}%")
        print(f"Average Score: {stats.get('average_overall_score', 0):.2f}/10")
        
        print("\nScore Breakdown:")
        print(f"  Correctness:     {stats.get('average_correctness_score', 0):.2f}/10")
        print(f"  Completeness:    {stats.get('average_completeness_score', 0):.2f}/10")
        print(f"  Relevance:       {stats.get('average_relevance_score', 0):.2f}/10")
        print(f"  Legal Reasoning: {stats.get('average_legal_reasoning_score', 0):.2f}/10")
    
    print("\n" + "=" * 80)
    print(f"Full analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
