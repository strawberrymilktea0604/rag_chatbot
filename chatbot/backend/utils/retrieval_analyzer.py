"""
Visualization and analysis tools for retrieval evaluation metrics.
Provides charts, statistical analysis, and diagnostic tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RetrievalAnalyzer:
    """
    Advanced analysis tools for retrieval evaluation results.
    Provides statistical analysis, correlation studies, and diagnostic insights.
    """
    
    def __init__(self):
        self.standard_metrics = ['recall', 'precision', 'mrr', 'ndcg', 'hit_rate', 'ap']
        self.standard_k_values = [1, 3, 5, 10, 20]
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_metric_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations between different metrics.
        Helps understand which metrics provide complementary information.
        """
        df = pd.DataFrame(results['per_question'])
        
        # Extract metric columns for correlation analysis
        metric_cols = [col for col in df.columns if any(metric in col for metric in self.standard_metrics)]
        metric_df = df[metric_cols].select_dtypes(include=[np.number])
        
        if metric_df.empty:
            return {'error': 'No numeric metric columns found'}
        
        # Compute correlation matrix
        correlation_matrix = metric_df.corr()
        
        # Find highly correlated metric pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'summary': {
                'metrics_analyzed': len(metric_cols),
                'high_correlation_pairs': len(high_correlations)
            }
        }
    
    def analyze_k_value_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how performance changes with different K values.
        Helps determine optimal K for your use case.
        """
        df = pd.DataFrame(results['per_question'])
        
        k_analysis = {}
        
        for metric in self.standard_metrics:
            metric_data = {}
            k_values_found = []
            
            for k in self.standard_k_values:
                col_name = f"{metric}@{k}"
                if col_name in df.columns:
                    k_values_found.append(k)
                    metric_data[k] = {
                        'mean': float(df[col_name].mean()),
                        'std': float(df[col_name].std()),
                        'median': float(df[col_name].median())
                    }
            
            if metric_data:
                # Calculate improvement rates
                improvements = {}
                for i in range(1, len(k_values_found)):
                    k_prev, k_curr = k_values_found[i-1], k_values_found[i]
                    prev_mean = metric_data[k_prev]['mean']
                    curr_mean = metric_data[k_curr]['mean']
                    
                    if prev_mean > 0:
                        improvement = (curr_mean - prev_mean) / prev_mean * 100
                        improvements[f"k{k_prev}_to_k{k_curr}"] = float(improvement)
                
                k_analysis[metric] = {
                    'performance_by_k': metric_data,
                    'improvements': improvements,
                    'diminishing_returns_threshold': self._find_diminishing_returns(metric_data)
                }
        
        return k_analysis
    
    def _find_diminishing_returns(self, metric_data: Dict[int, Dict[str, float]]) -> Optional[int]:
        """Find the K value where diminishing returns start (improvement < 5%)."""
        k_values = sorted(metric_data.keys())
        
        for i in range(1, len(k_values)):
            k_prev, k_curr = k_values[i-1], k_values[i]
            prev_mean = metric_data[k_prev]['mean']
            curr_mean = metric_data[k_curr]['mean']
            
            if prev_mean > 0:
                improvement = (curr_mean - prev_mean) / prev_mean * 100
                if improvement < 5.0:  # Less than 5% improvement
                    return k_prev
        
        return None
    
    def analyze_document_performance_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance patterns across different documents.
        Identifies which documents are hardest/easiest for retrieval.
        """
        df = pd.DataFrame(results['per_question'])
        
        if 'document_id' not in df.columns:
            return {'error': 'No document_id column found'}
        
        doc_analysis = {}
        
        # Group by document
        for doc_id, doc_df in df.groupby('document_id'):
            doc_metrics = {}
            
            # Calculate performance for each metric
            for metric in self.standard_metrics:
                for k in [5, 10]:  # Focus on common K values
                    col_name = f"{metric}@{k}"
                    if col_name in doc_df.columns:
                        doc_metrics[col_name] = {
                            'mean': float(doc_df[col_name].mean()),
                            'std': float(doc_df[col_name].std()),
                            'questions': len(doc_df),
                            'zero_performance': int((doc_df[col_name] == 0).sum())
                        }
            
            doc_analysis[doc_id] = doc_metrics
        
        # Rank documents by performance
        doc_rankings = self._rank_documents_by_performance(doc_analysis)
        
        return {
            'per_document': doc_analysis,
            'rankings': doc_rankings,
            'summary': {
                'total_documents': len(doc_analysis),
                'documents_analyzed': list(doc_analysis.keys())
            }
        }
    
    def _rank_documents_by_performance(self, doc_analysis: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Rank documents by different performance metrics."""
        rankings = {}
        
        metric_cols = ['recall@5', 'precision@5', 'mrr@5', 'ndcg@5']
        
        for metric_col in metric_cols:
            doc_scores = []
            for doc_id, metrics in doc_analysis.items():
                if metric_col in metrics:
                    doc_scores.append({
                        'document_id': doc_id,
                        'score': metrics[metric_col]['mean'],
                        'questions': metrics[metric_col]['questions']
                    })
            
            # Sort by score descending
            doc_scores.sort(key=lambda x: x['score'], reverse=True)
            rankings[metric_col] = doc_scores
        
        return rankings
    
    def identify_failure_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify patterns in retrieval failures to guide system improvements.
        """
        df = pd.DataFrame(results['per_question'])
        
        # Define failure criteria
        failures = {
            'zero_recall': df[df.get('recall@5', pd.Series()) == 0.0],
            'low_precision': df[df.get('precision@5', pd.Series()) < 0.2],
            'poor_ranking': df[df.get('mrr@5', pd.Series()) < 0.3]
        }
        
        failure_analysis = {}
        
        for failure_type, failure_df in failures.items():
            if failure_df.empty:
                continue
            
            analysis = {
                'count': len(failure_df),
                'percentage': float(len(failure_df) / len(df) * 100),
                'document_distribution': failure_df['document_id'].value_counts().to_dict() if 'document_id' in failure_df.columns else {}
            }
            
            # Sample failed questions for manual review
            sample_size = min(5, len(failure_df))
            if sample_size > 0:
                sample_questions = failure_df.sample(n=sample_size)['question_id'].tolist()
                analysis['sample_questions'] = sample_questions
            
            failure_analysis[failure_type] = analysis
        
        return failure_analysis
    
    def generate_performance_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable insights for improving retrieval performance.
        """
        insights = []
        
        # Analyze overall performance
        summary = results.get('summary', {})
        
        # Check recall performance
        recall_5_mean = summary.get('recall@5', {}).get('mean', 0)
        if recall_5_mean < 0.5:
            insights.append({
                'type': 'performance_issue',
                'metric': 'recall@5',
                'issue': f'Low recall@5 ({recall_5_mean:.3f}). Consider improving embedding quality or chunk strategy.',
                'priority': 'high'
            })
        
        # Check precision vs recall trade-off
        precision_5_mean = summary.get('precision@5', {}).get('mean', 0)
        if precision_5_mean > 0.8 and recall_5_mean < 0.6:
            insights.append({
                'type': 'optimization',
                'issue': 'High precision but low recall suggests conservative retrieval. Consider increasing top_k.',
                'priority': 'medium'
            })
        
        # Check MRR for ranking quality
        mrr_5_mean = summary.get('mrr@5', {}).get('mean', 0)
        if mrr_5_mean < 0.4:
            insights.append({
                'type': 'ranking_issue',
                'metric': 'mrr@5',
                'issue': f'Poor ranking quality (MRR@5: {mrr_5_mean:.3f}). Consider reranking or better embeddings.',
                'priority': 'high'
            })
        
        # Analyze failure patterns
        if 'analysis' in results:
            dist = results['analysis'].get('performance_distribution', {})
            zero_recall_pct = dist.get('zero_recall_count', 0) / results['metadata']['total_questions'] * 100
            
            if zero_recall_pct > 20:
                insights.append({
                    'type': 'critical_issue',
                    'issue': f'{zero_recall_pct:.1f}% of questions have zero recall. Check embedding coverage and chunking strategy.',
                    'priority': 'critical'
                })
        
        return {
            'insights': insights,
            'recommendations': self._generate_recommendations(insights),
            'overall_health': self._assess_overall_health(summary)
        }
    
    def _generate_recommendations(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        for insight in insights:
            if insight['type'] == 'performance_issue' and 'recall' in insight.get('metric', ''):
                recommendations.extend([
                    "Experiment with different embedding models (e.g., larger models or domain-specific ones)",
                    "Optimize chunk size and overlap parameters",
                    "Consider hybrid search (sparse + dense)",
                    "Increase top_k parameter in initial retrieval"
                ])
            
            elif insight['type'] == 'ranking_issue':
                recommendations.extend([
                    "Implement reranking with CrossEncoder models",
                    "Fine-tune embedding models on your domain data",
                    "Experiment with different similarity metrics (cosine, dot product)",
                    "Consider query expansion techniques"
                ])
            
            elif insight['type'] == 'critical_issue':
                recommendations.extend([
                    "Review document preprocessing pipeline",
                    "Check for embedding model compatibility",
                    "Validate chunk creation process",
                    "Ensure proper text cleaning and normalization"
                ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_overall_health(self, summary: Dict[str, Any]) -> str:
        """Assess overall retrieval system health."""
        recall_5 = summary.get('recall@5', {}).get('mean', 0)
        mrr_5 = summary.get('mrr@5', {}).get('mean', 0)
        
        if recall_5 >= 0.7 and mrr_5 >= 0.6:
            return "excellent"
        elif recall_5 >= 0.5 and mrr_5 >= 0.4:
            return "good"
        elif recall_5 >= 0.3 and mrr_5 >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def export_analysis_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Export comprehensive analysis report."""
        output_path = Path(output_path)
        
        # Run all analyses
        correlation_analysis = self.analyze_metric_correlations(results)
        k_impact_analysis = self.analyze_k_value_impact(results)
        document_patterns = self.analyze_document_performance_patterns(results)
        failure_patterns = self.identify_failure_patterns(results)
        insights = self.generate_performance_insights(results)
        
        # Combine all analyses
        full_analysis = {
            'correlation_analysis': correlation_analysis,
            'k_impact_analysis': k_impact_analysis,
            'document_patterns': document_patterns,
            'failure_patterns': failure_patterns,
            'insights': insights,
            'metadata': {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'source_results': str(output_path.parent / "retrieval_evaluation_detailed.json")
            }
        }
        
        # Save detailed analysis
        analysis_path = output_path.with_name(f"{output_path.stem}_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(full_analysis, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable summary
        summary_path = output_path.with_name(f"{output_path.stem}_analysis_summary.txt")
        self._write_analysis_summary(full_analysis, summary_path)
        
        logger.info(f"Analysis report saved to: {analysis_path}")
        logger.info(f"Analysis summary saved to: {summary_path}")
    
    def _write_analysis_summary(self, analysis: Dict[str, Any], output_path: Path) -> None:
        """Write human-readable analysis summary."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RAG RETRIEVAL SYSTEM ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall health assessment
            insights = analysis.get('insights', {})
            health = insights.get('overall_health', 'unknown')
            f.write(f"OVERALL SYSTEM HEALTH: {health.upper()}\n\n")
            
            # Key insights
            if 'insights' in insights:
                f.write("KEY INSIGHTS:\n")
                f.write("-" * 20 + "\n")
                for i, insight in enumerate(insights['insights'], 1):
                    f.write(f"{i}. [{insight.get('priority', 'medium').upper()}] {insight.get('issue', 'N/A')}\n")
                f.write("\n")
            
            # Recommendations
            if 'recommendations' in insights:
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(insights['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            # K-value optimization
            k_analysis = analysis.get('k_impact_analysis', {})
            if k_analysis:
                f.write("K-VALUE OPTIMIZATION:\n")
                f.write("-" * 25 + "\n")
                for metric, data in k_analysis.items():
                    diminishing_k = data.get('diminishing_returns_threshold')
                    if diminishing_k:
                        f.write(f"{metric}: Consider K={diminishing_k} (diminishing returns beyond this)\n")
                f.write("\n")
            
            # Document performance patterns
            doc_patterns = analysis.get('document_patterns', {})
            if 'rankings' in doc_patterns:
                f.write("DOCUMENT PERFORMANCE (by recall@5):\n")
                f.write("-" * 35 + "\n")
                recall_ranking = doc_patterns['rankings'].get('recall@5', [])
                for i, doc_info in enumerate(recall_ranking[:5], 1):  # Top 5
                    f.write(f"{i}. {doc_info['document_id']}: {doc_info['score']:.3f} "
                           f"({doc_info['questions']} questions)\n")
                f.write("\n")
            
            # Failure analysis
            failure_patterns = analysis.get('failure_patterns', {})
            if failure_patterns:
                f.write("FAILURE ANALYSIS:\n")
                f.write("-" * 18 + "\n")
                for failure_type, data in failure_patterns.items():
                    f.write(f"{failure_type}: {data['count']} questions ({data['percentage']:.1f}%)\n")


def analyze_retrieval_results(results_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to run complete analysis on evaluation results.
    
    Args:
        results_path: Path to evaluation results JSON file
        output_dir: Directory to save analysis results
        
    Returns:
        Complete analysis results
    """
    if output_dir is None:
        output_dir = Path(results_path).parent
    
    analyzer = RetrievalAnalyzer()
    results = analyzer.load_results(results_path)
    
    # Export comprehensive analysis
    output_base = Path(output_dir) / "retrieval_analysis"
    analyzer.export_analysis_report(results, str(output_base))
    
    # Return analysis for programmatic use
    return {
        'correlation_analysis': analyzer.analyze_metric_correlations(results),
        'k_impact_analysis': analyzer.analyze_k_value_impact(results),
        'document_patterns': analyzer.analyze_document_performance_patterns(results),
        'failure_patterns': analyzer.identify_failure_patterns(results),
        'insights': analyzer.generate_performance_insights(results)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze retrieval evaluation results")
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output-dir", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    analysis = analyze_retrieval_results(args.results, args.output_dir)
    
    print("Analysis completed successfully!")
    insights = analysis.get('insights', {})
    if 'overall_health' in insights:
        print(f"Overall system health: {insights['overall_health']}")