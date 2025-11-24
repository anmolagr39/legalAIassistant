"""
Optional: Visualization tools for evaluation results
Requires: pip install matplotlib seaborn pandas
"""
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib, seaborn, or pandas not installed")
    print("Install with: pip install matplotlib seaborn pandas")


class EvaluationVisualizer:
    """Create visualizations for evaluation results"""
    
    def __init__(self, results_file: str):
        if not VISUALIZATION_AVAILABLE:
            raise ImportError("Visualization packages not available")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                self.data = data[0] if 'metadata' in data[0] else {'results': data}
            else:
                self.data = data
        
        self.results = self.data.get('results', [])
        self.statistics = self.data.get('statistics', {})
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_score_distribution(self, output_file: str = None):
        """Plot distribution of overall scores"""
        scores = [r['evaluation']['overall_score'] 
                 for r in self.results if 'evaluation' in r]
        
        fig, ax = plt.subplots()
        ax.hist(scores, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(6.0, color='red', linestyle='--', label='Pass Threshold')
        ax.set_xlabel('Overall Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Overall Scores')
        ax.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_category_performance(self, output_file: str = None):
        """Plot performance by category"""
        category_data = {}
        for result in self.results:
            if 'evaluation' not in result:
                continue
            cat = result['category']
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(result['evaluation']['overall_score'])
        
        # Create dataframe
        data_list = []
        for cat, scores in category_data.items():
            for score in scores:
                data_list.append({'Category': cat, 'Score': score})
        df = pd.DataFrame(data_list)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(data=df, x='Category', y='Score', ax=ax)
        ax.axhline(6.0, color='red', linestyle='--', label='Pass Threshold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Overall Score')
        ax.set_title('Performance by Category')
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metric_comparison(self, output_file: str = None):
        """Compare different evaluation metrics"""
        metrics = ['correctness_score', 'completeness_score', 
                  'relevance_score', 'legal_reasoning_score']
        metric_names = ['Correctness', 'Completeness', 'Relevance', 'Legal Reasoning']
        
        avg_scores = []
        for metric in metrics:
            scores = [r['evaluation'][metric] 
                     for r in self.results if 'evaluation' in r]
            avg_scores.append(sum(scores) / len(scores))
        
        fig, ax = plt.subplots()
        bars = ax.bar(metric_names, avg_scores, edgecolor='black', alpha=0.7)
        ax.axhline(6.0, color='red', linestyle='--', label='Pass Threshold')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Metric')
        ax.set_ylim(0, 10)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=15, ha='right')
        ax.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pass_fail_by_category(self, output_file: str = None):
        """Stacked bar chart of pass/fail by category"""
        category_data = {}
        for result in self.results:
            if 'evaluation' not in result:
                continue
            cat = result['category']
            if cat not in category_data:
                category_data[cat] = {'PASS': 0, 'FAIL': 0}
            verdict = result['evaluation']['verdict']
            category_data[cat][verdict] += 1
        
        categories = list(category_data.keys())
        pass_counts = [category_data[cat]['PASS'] for cat in categories]
        fail_counts = [category_data[cat]['FAIL'] for cat in categories]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = range(len(categories))
        ax.bar(x, pass_counts, label='Pass', color='green', alpha=0.7)
        ax.bar(x, fail_counts, bottom=pass_counts, label='Fail', 
               color='red', alpha=0.7)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Pass/Fail Distribution by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_heatmap(self, output_file: str = None):
        """Heatmap of scores across categories and metrics"""
        metrics = ['correctness_score', 'completeness_score', 
                  'relevance_score', 'legal_reasoning_score']
        metric_names = ['Correctness', 'Completeness', 'Relevance', 'Legal Reasoning']
        
        # Collect data
        category_metrics = {}
        for result in self.results:
            if 'evaluation' not in result:
                continue
            cat = result['category']
            if cat not in category_metrics:
                category_metrics[cat] = {m: [] for m in metrics}
            
            for metric in metrics:
                category_metrics[cat][metric].append(result['evaluation'][metric])
        
        # Calculate averages
        categories = list(category_metrics.keys())
        data = []
        for cat in categories:
            row = []
            for metric in metrics:
                avg = sum(category_metrics[cat][metric]) / len(category_metrics[cat][metric])
                row.append(avg)
            data.append(row)
        
        df = pd.DataFrame(data, index=categories, columns=metric_names)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=0, vmax=10, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Average Scores Heatmap by Category and Metric')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Category')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_plots(self, output_dir: str):
        """Generate all visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Generating visualizations...")
        
        print("  1. Score distribution...")
        self.plot_score_distribution(str(output_path / "score_distribution.png"))
        
        print("  2. Category performance...")
        self.plot_category_performance(str(output_path / "category_performance.png"))
        
        print("  3. Metric comparison...")
        self.plot_metric_comparison(str(output_path / "metric_comparison.png"))
        
        print("  4. Pass/Fail by category...")
        self.plot_pass_fail_by_category(str(output_path / "pass_fail_distribution.png"))
        
        print("  5. Score heatmap...")
        self.plot_score_heatmap(str(output_path / "score_heatmap.png"))
        
        print(f"\n✓ All plots saved to: {output_path}")


def main():
    """Main visualization function"""
    import sys
    
    if not VISUALIZATION_AVAILABLE:
        print("Error: Visualization packages not installed")
        print("Install with: pip install matplotlib seaborn pandas")
        return
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results_file.json>")
        print("\nExample: python visualize_results.py evaluation_results/final_results_20231122_143000.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        return
    
    print(f"Creating visualizations from: {results_file}")
    
    visualizer = EvaluationVisualizer(results_file)
    
    # Generate all plots
    output_dir = Path(results_file).parent / "visualizations"
    visualizer.generate_all_plots(str(output_dir))


if __name__ == "__main__":
    main()
