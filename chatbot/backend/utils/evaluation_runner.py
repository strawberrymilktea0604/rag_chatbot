"""
Comprehensive evaluation runner for RAG system retrieval performance.
Provides detailed analysis and visualization of retrieval metrics.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict

from .retrieval_metrics import RetrievalEvaluator, RetrievalMetrics

logger = logging.getLogger(__name__)


class ComprehensiveRetrievalEvaluator:
    """
    Enhanced retrieval evaluator with comprehensive analysis and reporting.
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values to evaluate (default: [1, 3, 5, 10, 20])
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.metrics = [
            'recall', 'precision', 'mrr', 'ndcg', 'hit_rate', 'ap'
        ]
    
    def evaluate_retrieval(self, qa_gold_path: str, retrieval_run_path: str) -> Dict[str, Any]:
        """
        Comprehensive retrieval evaluation with detailed analysis.
        
        Args:
            qa_gold_path: Path to gold standard QA CSV file
            retrieval_run_path: Path to retrieval run CSV file
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Loading gold standard from: {qa_gold_path}")
        qa_gold = RetrievalEvaluator.load_qa_gold(qa_gold_path)
        
        logger.info(f"Loading retrieval runs from: {retrieval_run_path}")
        retrieval_runs = RetrievalEvaluator.load_retrieval_runs(retrieval_run_path)
        
        # Track metrics for all questions
        all_metrics = defaultdict(list)
        question_results = []
        
        # Document-level aggregation
        doc_metrics = defaultdict(lambda: defaultdict(list))
        
        logger.info(f"Evaluating {len(qa_gold)} questions...")
        
        for question_id, gold_data in qa_gold.items():
            if question_id not in retrieval_runs:
                logger.warning(f"No retrieval results for question: {question_id}")
                continue
            
            run_data = retrieval_runs[question_id]
            gold_chunks = set(gold_data['gold_support_chunk_ids'])
            retrieved_chunks = run_data['retrieved_chunk_ids']
            document_id = gold_data.get('document_id', 'unknown')
            
            question_metrics = {'question_id': question_id, 'document_id': document_id}
            
            # Compute metrics for all K values
            for k in self.k_values:
                metrics_at_k = RetrievalMetrics.compute_all_metrics(
                    relevant_items=gold_chunks,
                    retrieved_items=retrieved_chunks,
                    k=k
                )
                
                # Store in question-level results
                question_metrics.update(metrics_at_k)
                
                # Aggregate for overall statistics
                for metric_name, score in metrics_at_k.items():
                    all_metrics[metric_name].append(score)
                    # Document-level aggregation
                    doc_metrics[document_id][metric_name].append(score)
            
            question_results.append(question_metrics)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(all_metrics)
        
        # Compute document-level statistics
        document_stats = self._compute_document_statistics(doc_metrics)
        
        # Performance analysis
        performance_analysis = self._analyze_performance(question_results)
        
        return {
            'summary': summary_stats,
            'per_question': question_results,
            'per_document': document_stats,
            'analysis': performance_analysis,
            'metadata': {
                'total_questions': len(question_results),
                'k_values': self.k_values,
                'metrics_computed': self.metrics
            }
        }
    
    def _compute_summary_statistics(self, all_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute comprehensive summary statistics for all metrics."""
        summary = {}
        
        for metric_name, values in all_metrics.items():
            if not values:
                continue
                
            summary[metric_name] = {
                'mean': float(pd.Series(values).mean()),
                'median': float(pd.Series(values).median()),
                'std': float(pd.Series(values).std()),
                'min': float(pd.Series(values).min()),
                'max': float(pd.Series(values).max()),
                'q25': float(pd.Series(values).quantile(0.25)),
                'q75': float(pd.Series(values).quantile(0.75)),
                'count': len(values),
                'zero_count': sum(1 for v in values if v == 0.0),
                'perfect_count': sum(1 for v in values if v == 1.0)
            }
        
        return summary
    
    def _compute_document_statistics(self, doc_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Compute per-document performance statistics."""
        document_stats = {}
        
        for doc_id, metrics in doc_metrics.items():
            doc_summary = {}
            for metric_name, values in metrics.items():
                if values:
                    doc_summary[metric_name] = {
                        'mean': float(pd.Series(values).mean()),
                        'std': float(pd.Series(values).std()),
                        'count': len(values)
                    }
            document_stats[doc_id] = doc_summary
        
        return document_stats
    
    def _analyze_performance(self, question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance patterns and identify issues."""
        analysis = {}
        
        # Find questions with poor performance
        poor_performance = []
        excellent_performance = []
        
        for result in question_results:
            recall_5 = result.get('recall@5', 0.0)
            mrr_5 = result.get('mrr@5', 0.0)
            
            if recall_5 == 0.0:  # No relevant chunks retrieved
                poor_performance.append({
                    'question_id': result['question_id'],
                    'document_id': result['document_id'],
                    'recall@5': recall_5,
                    'mrr@5': mrr_5
                })
            elif recall_5 >= 0.8 and mrr_5 >= 0.8:  # Excellent performance
                excellent_performance.append({
                    'question_id': result['question_id'],
                    'document_id': result['document_id'],
                    'recall@5': recall_5,
                    'mrr@5': mrr_5
                })
        
        analysis['poor_performance'] = poor_performance[:10]  # Top 10 worst
        analysis['excellent_performance'] = excellent_performance[:10]  # Top 10 best
        analysis['performance_distribution'] = {
            'zero_recall_count': len([r for r in question_results if r.get('recall@5', 0) == 0]),
            'perfect_recall_count': len([r for r in question_results if r.get('recall@5', 0) == 1.0]),
            'high_mrr_count': len([r for r in question_results if r.get('mrr@5', 0) >= 0.8])
        }
        
        return analysis
    
    def generate_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate a comprehensive evaluation report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save full results as JSON
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        summary_path = report_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RAG RETRIEVAL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for k in self.k_values:
                f.write(f"\nMetrics @ K={k}:\n")
                for metric in self.metrics:
                    key = f"{metric}@{k}"
                    if key in results['summary']:
                        stats = results['summary'][key]
                        f.write(f"  {metric:10s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                               f"(median: {stats['median']:.4f})\n")
            
            f.write(f"\nPERFORMANCE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            analysis = results['analysis']
            dist = analysis['performance_distribution']
            total = results['metadata']['total_questions']
            
            f.write(f"Total questions evaluated: {total}\n")
            f.write(f"Questions with zero recall@5: {dist['zero_recall_count']} ({dist['zero_recall_count']/total*100:.1f}%)\n")
            f.write(f"Questions with perfect recall@5: {dist['perfect_recall_count']} ({dist['perfect_recall_count']/total*100:.1f}%)\n")
            f.write(f"Questions with high MRR@5 (≥0.8): {dist['high_mrr_count']} ({dist['high_mrr_count']/total*100:.1f}%)\n")
            
            if analysis['poor_performance']:
                f.write(f"\nWORST PERFORMING QUESTIONS (Top 10):\n")
                for i, question in enumerate(analysis['poor_performance'][:10], 1):
                    f.write(f"{i:2d}. {question['question_id']} (doc: {question['document_id']}) - "
                           f"Recall@5: {question['recall@5']:.3f}\n")
        
        logger.info(f"Report saved to: {json_path} and {summary_path}")
    
    def save_detailed_csv(self, results: Dict[str, Any], output_path: str) -> None:
        """Save detailed per-question results to CSV for further analysis."""
        csv_path = Path(output_path).with_suffix('.csv')
        
        if results['per_question']:
            df = pd.DataFrame(results['per_question'])
            df.to_csv(csv_path, index=False)
            logger.info(f"Detailed results saved to: {csv_path}")


def run_comprehensive_evaluation(qa_gold_path: str, retrieval_run_path: str, 
                               output_dir: str = None) -> Dict[str, Any]:
    """
    Run comprehensive retrieval evaluation with detailed reporting.
    
    Args:
        qa_gold_path: Path to gold standard QA file
        retrieval_run_path: Path to retrieval run results
        output_dir: Directory to save results (default: same as qa_gold_path)
        
    Returns:
        Complete evaluation results
    """
    if output_dir is None:
        output_dir = Path(qa_gold_path).parent
    
    evaluator = ComprehensiveRetrievalEvaluator()
    results = evaluator.evaluate_retrieval(qa_gold_path, retrieval_run_path)
    
    # Generate reports
    output_base = Path(output_dir) / "retrieval_evaluation_detailed"
    evaluator.generate_report(results, str(output_base))
    evaluator.save_detailed_csv(results, str(output_base))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive retrieval evaluation")
    parser.add_argument("--qa-gold", required=True, help="Path to gold standard QA CSV")
    parser.add_argument("--retrieval-run", required=True, help="Path to retrieval run CSV")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    results = run_comprehensive_evaluation(
        args.qa_gold, 
        args.retrieval_run, 
        args.output_dir
    )
    
    print("Evaluation completed successfully!")
    print(f"Summary statistics for key metrics:")
    for metric in ['recall@5', 'precision@5', 'mrr@5', 'ndcg@5']:
        if metric in results['summary']:
            stats = results['summary'][metric]
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")