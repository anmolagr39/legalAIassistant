"""
Comprehensive Results Storage for IPC RAG System Evaluation

This module provides functionality to store, retrieve, and analyze results from:
- ROUGE-C metrics (text generation quality)
- Retrieval metrics (document retrieval quality) 
- Generation metrics (text generation evaluation)

Features:
- Unified result storage format
- Experiment tracking with metadata
- Performance comparison across runs
- Export capabilities for analysis
- Historical trend analysis
"""

import json
import os
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ExperimentMetadata:
    """Metadata for an evaluation experiment."""
    experiment_id: str
    timestamp: str
    experiment_name: str
    description: str
    rag_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    dataset_info: Dict[str, Any]

@dataclass
class RougeResults:
    """ROUGE-C evaluation results."""
    rouge_l_f1: float
    rouge_l_precision: float
    rouge_l_recall: float
    token_overlap_f1: float
    token_overlap_precision: float
    token_overlap_recall: float
    semantic_similarity: float
    avg_context_similarity: float
    max_context_similarity: float
    overall_score: float
    num_queries: int

@dataclass
class RetrievalResults:
    """Retrieval evaluation results."""
    f1_at_5: float
    precision_at_5: float
    recall_at_5: float
    f1_at_10: float
    precision_at_10: float
    recall_at_10: float
    num_queries: int
    avg_retrieved_docs: float

@dataclass
class GenerationResults:
    """Text generation evaluation results."""
    bleu_score: Optional[float] = None
    rouge_1_f1: Optional[float] = None
    rouge_2_f1: Optional[float] = None
    rouge_l_f1: Optional[float] = None
    bertscore_f1: Optional[float] = None
    perplexity: Optional[float] = None
    coherence_score: Optional[float] = None
    num_queries: int = 0

@dataclass
class ComprehensiveResults:
    """Complete evaluation results for an experiment."""
    metadata: ExperimentMetadata
    rouge_c_results: Optional[RougeResults] = None
    retrieval_results: Optional[RetrievalResults] = None
    generation_results: Optional[GenerationResults] = None
    overall_performance_score: Optional[float] = None
    notes: str = ""

class ResultsManager:
    """Manager for storing and retrieving evaluation results."""
    
    def __init__(self, results_file: str = "../comprehensive_evaluation_results.json"):
        """
        Initialize results manager.
        
        Args:
            results_file: Path to JSON file for storing results (relative to project root)
        """
        # Convert relative path to absolute path from project root
        if not os.path.isabs(results_file):
            # Get the project root directory (parent of rouge_c_metrics)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            results_file = os.path.join(project_root, results_file.lstrip('../'))
        
        self.results_file = results_file
        self.results_history: List[ComprehensiveResults] = []
        self.load_results()
    
    def create_experiment_metadata(self, 
                                 experiment_name: str,
                                 description: str,
                                 rag_config: Dict[str, Any] = None,
                                 evaluation_config: Dict[str, Any] = None,
                                 dataset_info: Dict[str, Any] = None) -> ExperimentMetadata:
        """
        Create experiment metadata.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            rag_config: RAG system configuration
            evaluation_config: Evaluation configuration
            dataset_info: Dataset information
            
        Returns:
            ExperimentMetadata object
        """
        return ExperimentMetadata(
            experiment_id=str(uuid.uuid4())[:8],
            timestamp=datetime.datetime.now().isoformat(),
            experiment_name=experiment_name,
            description=description,
            rag_config=rag_config or {},
            evaluation_config=evaluation_config or {},
            dataset_info=dataset_info or {}
        )
    
    def store_rouge_c_results(self, 
                             rouge_c_data: Dict[str, Any]) -> RougeResults:
        """
        Convert ROUGE-C results to structured format.
        
        Args:
            rouge_c_data: Raw ROUGE-C results from evaluation
            
        Returns:
            RougeResults object
        """
        if 'aggregate_metrics' in rouge_c_data:
            metrics = rouge_c_data['aggregate_metrics']
            
            return RougeResults(
                rouge_l_f1=metrics['rouge_l'].get('avg_f1', 0.0),
                rouge_l_precision=metrics['rouge_l'].get('avg_precision', 0.0),
                rouge_l_recall=metrics['rouge_l'].get('avg_recall', 0.0),
                token_overlap_f1=metrics['token_overlap'].get('avg_f1', 0.0),
                token_overlap_precision=metrics['token_overlap'].get('avg_precision', 0.0),
                token_overlap_recall=metrics['token_overlap'].get('avg_recall', 0.0),
                semantic_similarity=metrics['semantic_similarity'].get('avg_combined', 0.0),
                avg_context_similarity=metrics['semantic_similarity'].get('avg_individual', 0.0),
                max_context_similarity=metrics['semantic_similarity'].get('avg_max', 0.0),
                overall_score=(
                    metrics['rouge_l'].get('avg_f1', 0.0) + 
                    metrics['token_overlap'].get('avg_f1', 0.0) + 
                    metrics['semantic_similarity'].get('avg_combined', 0.0)
                ) / 3,
                num_queries=metrics.get('summary', {}).get('total_pairs', 0)
            )
        else:
            # Single result format
            return RougeResults(
                rouge_l_f1=rouge_c_data.get('rouge_l', {}).get('f1', 0.0),
                rouge_l_precision=rouge_c_data.get('rouge_l', {}).get('precision', 0.0),
                rouge_l_recall=rouge_c_data.get('rouge_l', {}).get('recall', 0.0),
                token_overlap_f1=rouge_c_data.get('token_overlap', {}).get('f1', 0.0),
                token_overlap_precision=rouge_c_data.get('token_overlap', {}).get('precision', 0.0),
                token_overlap_recall=rouge_c_data.get('token_overlap', {}).get('recall', 0.0),
                semantic_similarity=rouge_c_data.get('semantic_similarity', 0.0),
                avg_context_similarity=rouge_c_data.get('avg_context_similarity', 0.0),
                max_context_similarity=rouge_c_data.get('max_context_similarity', 0.0),
                overall_score=0.0,  # Calculate if needed
                num_queries=1
            )
    
    def store_retrieval_results(self, 
                              retrieval_data: Dict[str, Any]) -> RetrievalResults:
        """
        Convert retrieval results to structured format.
        
        Args:
            retrieval_data: Raw retrieval results from evaluation
            
        Returns:
            RetrievalResults object
        """
        return RetrievalResults(
            f1_at_5=retrieval_data.get('f1_at_5', 0.0),
            precision_at_5=retrieval_data.get('precision_at_5', 0.0),
            recall_at_5=retrieval_data.get('recall_at_5', 0.0),
            f1_at_10=retrieval_data.get('f1_at_10', 0.0),
            precision_at_10=retrieval_data.get('precision_at_10', 0.0),
            recall_at_10=retrieval_data.get('recall_at_10', 0.0),
            num_queries=retrieval_data.get('num_queries', 0),
            avg_retrieved_docs=retrieval_data.get('avg_retrieved_docs', 0.0)
        )
    
    def store_generation_results(self, 
                               generation_data: Dict[str, Any]) -> GenerationResults:
        """
        Convert generation results to structured format.
        
        Args:
            generation_data: Raw generation results from evaluation
            
        Returns:
            GenerationResults object
        """
        return GenerationResults(
            bleu_score=generation_data.get('bleu_score'),
            rouge_1_f1=generation_data.get('rouge_1_f1'),
            rouge_2_f1=generation_data.get('rouge_2_f1'),
            rouge_l_f1=generation_data.get('rouge_l_f1'),
            bertscore_f1=generation_data.get('bertscore_f1'),
            perplexity=generation_data.get('perplexity'),
            coherence_score=generation_data.get('coherence_score'),
            num_queries=generation_data.get('num_queries', 0)
        )
    
    def add_experiment_results(self, 
                             metadata: ExperimentMetadata,
                             rouge_c_data: Dict[str, Any] = None,
                             retrieval_data: Dict[str, Any] = None,
                             generation_data: Dict[str, Any] = None,
                             notes: str = "") -> str:
        """
        Add complete experiment results.
        
        Args:
            metadata: Experiment metadata
            rouge_c_data: ROUGE-C evaluation results
            retrieval_data: Retrieval evaluation results
            generation_data: Generation evaluation results
            notes: Additional notes
            
        Returns:
            Experiment ID
        """
        # Process results
        rouge_results = None
        if rouge_c_data:
            rouge_results = self.store_rouge_c_results(rouge_c_data)
        
        retrieval_results = None
        if retrieval_data:
            retrieval_results = self.store_retrieval_results(retrieval_data)
        
        generation_results = None
        if generation_data:
            generation_results = self.store_generation_results(generation_data)
        
        # Calculate overall performance score
        overall_score = self._calculate_overall_score(
            rouge_results, retrieval_results, generation_results
        )
        
        # Create comprehensive results
        comprehensive_results = ComprehensiveResults(
            metadata=metadata,
            rouge_c_results=rouge_results,
            retrieval_results=retrieval_results,
            generation_results=generation_results,
            overall_performance_score=overall_score,
            notes=notes
        )
        
        # Add to history
        self.results_history.append(comprehensive_results)
        
        # Save to file
        self.save_results()
        
        print(f"✅ Experiment results stored: {metadata.experiment_id}")
        return metadata.experiment_id
    
    def _calculate_overall_score(self, 
                               rouge_results: Optional[RougeResults],
                               retrieval_results: Optional[RetrievalResults],
                               generation_results: Optional[GenerationResults]) -> float:
        """Calculate overall performance score."""
        scores = []
        
        if rouge_results:
            scores.append(rouge_results.overall_score)
        
        if retrieval_results:
            # Use F1@5 as primary retrieval metric
            scores.append(retrieval_results.f1_at_5)
        
        if generation_results:
            # Use ROUGE-L F1 if available, otherwise BLEU
            if generation_results.rouge_l_f1 is not None:
                scores.append(generation_results.rouge_l_f1)
            elif generation_results.bleu_score is not None:
                scores.append(generation_results.bleu_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[ComprehensiveResults]:
        """Get experiment results by ID."""
        for result in self.results_history:
            if result.metadata.experiment_id == experiment_id:
                return result
        return None
    
    def get_latest_experiment(self) -> Optional[ComprehensiveResults]:
        """Get the most recent experiment."""
        if self.results_history:
            return max(self.results_history, 
                      key=lambda x: x.metadata.timestamp)
        return None
    
    def get_experiments_by_name(self, experiment_name: str) -> List[ComprehensiveResults]:
        """Get all experiments with a specific name."""
        return [result for result in self.results_history 
                if result.metadata.experiment_name == experiment_name]
    
    def compare_experiments(self, 
                          experiment_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            
        Returns:
            Comparison results
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment_by_id(exp_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            return {}
        
        comparison = {
            'experiments': [],
            'best_rouge_c': None,
            'best_retrieval': None,
            'best_generation': None,
            'best_overall': None
        }
        
        best_overall_score = -1
        best_rouge_score = -1
        best_retrieval_score = -1
        best_generation_score = -1
        
        for exp in experiments:
            exp_summary = {
                'experiment_id': exp.metadata.experiment_id,
                'experiment_name': exp.metadata.experiment_name,
                'timestamp': exp.metadata.timestamp,
                'overall_score': exp.overall_performance_score or 0,
                'rouge_c_score': exp.rouge_c_results.overall_score if exp.rouge_c_results else 0,
                'retrieval_score': exp.retrieval_results.f1_at_5 if exp.retrieval_results else 0,
                'generation_score': (
                    exp.generation_results.rouge_l_f1 or exp.generation_results.bleu_score or 0
                    if exp.generation_results else 0
                )
            }
            
            comparison['experiments'].append(exp_summary)
            
            # Track best performers
            if exp_summary['overall_score'] > best_overall_score:
                best_overall_score = exp_summary['overall_score']
                comparison['best_overall'] = exp_summary
            
            if exp_summary['rouge_c_score'] > best_rouge_score:
                best_rouge_score = exp_summary['rouge_c_score']
                comparison['best_rouge_c'] = exp_summary
            
            if exp_summary['retrieval_score'] > best_retrieval_score:
                best_retrieval_score = exp_summary['retrieval_score']
                comparison['best_retrieval'] = exp_summary
            
            if exp_summary['generation_score'] > best_generation_score:
                best_generation_score = exp_summary['generation_score']
                comparison['best_generation'] = exp_summary
        
        return comparison
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        trends = {
            'timestamps': [],
            'overall_scores': [],
            'rouge_c_scores': [],
            'retrieval_scores': [],
            'generation_scores': []
        }
        
        # Sort by timestamp
        sorted_results = sorted(self.results_history, 
                              key=lambda x: x.metadata.timestamp)
        
        for result in sorted_results:
            trends['timestamps'].append(result.metadata.timestamp)
            trends['overall_scores'].append(result.overall_performance_score or 0)
            trends['rouge_c_scores'].append(
                result.rouge_c_results.overall_score if result.rouge_c_results else 0
            )
            trends['retrieval_scores'].append(
                result.retrieval_results.f1_at_5 if result.retrieval_results else 0
            )
            trends['generation_scores'].append(
                (result.generation_results.rouge_l_f1 or 
                 result.generation_results.bleu_score or 0)
                if result.generation_results else 0
            )
        
        return trends
    
    def export_results_summary(self, filepath: str = None):
        """Export results summary to file."""
        if filepath is None:
            # Save to project root by default
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            filename = f"results_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(project_root, filename)
        
        summary = {
            'export_timestamp': datetime.datetime.now().isoformat(),
            'total_experiments': len(self.results_history),
            'performance_trends': self.get_performance_trends(),
            'experiments': []
        }
        
        for result in self.results_history:
            exp_data = {
                'metadata': asdict(result.metadata),
                'results_summary': {
                    'overall_score': result.overall_performance_score,
                    'rouge_c': asdict(result.rouge_c_results) if result.rouge_c_results else None,
                    'retrieval': asdict(result.retrieval_results) if result.retrieval_results else None,
                    'generation': asdict(result.generation_results) if result.generation_results else None
                },
                'notes': result.notes
            }
            summary['experiments'].append(exp_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Results summary exported to: {filepath}")
        return filepath
    
    def display_latest_results(self):
        """Display the latest experiment results."""
        latest = self.get_latest_experiment()
        if not latest:
            print("❌ No experiment results found")
            return
        
        print("\n" + "="*80)
        print("📊 LATEST EXPERIMENT RESULTS")
        print("="*80)
        
        # Metadata
        meta = latest.metadata
        print(f"\n🏷️  EXPERIMENT INFO:")
        print(f"   ID: {meta.experiment_id}")
        print(f"   Name: {meta.experiment_name}")
        print(f"   Description: {meta.description}")
        print(f"   Timestamp: {meta.timestamp}")
        
        # Overall score
        if latest.overall_performance_score:
            print(f"\n🏆 OVERALL PERFORMANCE: {latest.overall_performance_score:.3f}")
        
        # ROUGE-C results
        if latest.rouge_c_results:
            rouge = latest.rouge_c_results
            print(f"\n🎯 ROUGE-C RESULTS:")
            print(f"   Overall Score: {rouge.overall_score:.3f}")
            print(f"   ROUGE-L F1: {rouge.rouge_l_f1:.3f}")
            print(f"   Token F1: {rouge.token_overlap_f1:.3f}")
            print(f"   Semantic Sim: {rouge.semantic_similarity:.3f}")
            print(f"   Queries: {rouge.num_queries}")
        
        # Retrieval results
        if latest.retrieval_results:
            retr = latest.retrieval_results
            print(f"\n🔍 RETRIEVAL RESULTS:")
            print(f"   F1@5: {retr.f1_at_5:.3f}")
            print(f"   Precision@5: {retr.precision_at_5:.3f}")
            print(f"   Recall@5: {retr.recall_at_5:.3f}")
            print(f"   Queries: {retr.num_queries}")
        
        # Generation results
        if latest.generation_results:
            gen = latest.generation_results
            print(f"\n📝 GENERATION RESULTS:")
            if gen.bleu_score:
                print(f"   BLEU Score: {gen.bleu_score:.3f}")
            if gen.rouge_l_f1:
                print(f"   ROUGE-L F1: {gen.rouge_l_f1:.3f}")
            if gen.bertscore_f1:
                print(f"   BERTScore F1: {gen.bertscore_f1:.3f}")
            print(f"   Queries: {gen.num_queries}")
        
        # Notes
        if latest.notes:
            print(f"\n📋 NOTES: {latest.notes}")
    
    def save_results(self):
        """Save results to JSON file."""
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = []
        for result in self.results_history:
            serializable_result = {
                'metadata': asdict(result.metadata),
                'rouge_c_results': asdict(result.rouge_c_results) if result.rouge_c_results else None,
                'retrieval_results': asdict(result.retrieval_results) if result.retrieval_results else None,
                'generation_results': asdict(result.generation_results) if result.generation_results else None,
                'overall_performance_score': result.overall_performance_score,
                'notes': result.notes
            }
            serializable_results.append(serializable_result)
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def load_results(self):
        """Load results from JSON file."""
        if not os.path.exists(self.results_file):
            return
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.results_history = []
            for item in data:
                # Reconstruct dataclasses
                metadata = ExperimentMetadata(**item['metadata'])
                
                rouge_c_results = None
                if item['rouge_c_results']:
                    rouge_c_results = RougeResults(**item['rouge_c_results'])
                
                retrieval_results = None
                if item['retrieval_results']:
                    retrieval_results = RetrievalResults(**item['retrieval_results'])
                
                generation_results = None
                if item['generation_results']:
                    generation_results = GenerationResults(**item['generation_results'])
                
                comprehensive_result = ComprehensiveResults(
                    metadata=metadata,
                    rouge_c_results=rouge_c_results,
                    retrieval_results=retrieval_results,
                    generation_results=generation_results,
                    overall_performance_score=item['overall_performance_score'],
                    notes=item['notes']
                )
                
                self.results_history.append(comprehensive_result)
            
            print(f"📂 Loaded {len(self.results_history)} experiment results from {self.results_file}")
            
        except Exception as e:
            print(f"⚠️ Error loading results: {e}")
            self.results_history = []


# Convenience functions for easy usage
def create_results_manager(results_file: str = "../comprehensive_evaluation_results.json") -> ResultsManager:
    """Create a new results manager."""
    return ResultsManager(results_file)

def quick_store_results(experiment_name: str,
                       description: str,
                       rouge_c_data: Dict[str, Any] = None,
                       retrieval_data: Dict[str, Any] = None,
                       generation_data: Dict[str, Any] = None,
                       notes: str = "",
                       results_file: str = "../comprehensive_evaluation_results.json") -> str:
    """
    Quick function to store evaluation results.
    
    Args:
        experiment_name: Name of the experiment
        description: Description of the experiment
        rouge_c_data: ROUGE-C evaluation results
        retrieval_data: Retrieval evaluation results
        generation_data: Generation evaluation results
        notes: Additional notes
        results_file: Path to results file
        
    Returns:
        Experiment ID
    """
    manager = ResultsManager(results_file)
    metadata = manager.create_experiment_metadata(experiment_name, description)
    
    return manager.add_experiment_results(
        metadata=metadata,
        rouge_c_data=rouge_c_data,
        retrieval_data=retrieval_data,
        generation_data=generation_data,
        notes=notes
    )