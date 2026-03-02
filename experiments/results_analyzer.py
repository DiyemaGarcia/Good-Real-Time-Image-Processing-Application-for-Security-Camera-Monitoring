"""
Results analyzer for statistical analysis and reporting
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and generate statistical reports from experiment results"""
    
    def __init__(self, results_dir: str = 'results'):
        """Initialize ResultsAnalyzer"""
        self.results_dir = Path(results_dir)
        
    def load_results(self, dataset_name: str) -> Dict:
        """Load results from JSON file"""
        results_file = self.results_dir / f"{dataset_name}_results.json"
        
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return {}
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded results from {results_file}")
        return results
    
    def compute_statistics(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Compute statistical summary of results"""
        stats_data = []
        
        for method, method_results in results.items():
            if not method_results:
                continue
            
            metrics = {
                'num_features1': [],
                'num_features2': [],
                'num_matches': [],
                'matching_efficiency': [],
                'avg_matching_distance': [],
                'spatial_entropy1': [],
                'spatial_entropy2': [],
                'detection_time': [],
                'matching_time': [],
                'total_time': []
            }
            
            for result in method_results:
                for metric in metrics.keys():
                    if metric in result:
                        metrics[metric].append(result[metric])
            
            method_stats = {'Method': method}
            
            for metric, values in metrics.items():
                if values:
                    method_stats[f'{metric}_mean'] = np.mean(values)
                    method_stats[f'{metric}_std'] = np.std(values)
                    method_stats[f'{metric}_min'] = np.min(values)
                    method_stats[f'{metric}_max'] = np.max(values)
                    method_stats[f'{metric}_median'] = np.median(values)
            
            stats_data.append(method_stats)
        
        df = pd.DataFrame(stats_data)
        return df
    
    def compare_methods(self, results: Dict[str, List[Dict]], 
                       metric: str = 'matching_efficiency') -> Dict:
        """Statistical comparison between methods"""
        method_values = {}
        for method, method_results in results.items():
            if not method_results:
                continue
            
            values = [r[metric] for r in method_results if metric in r]
            if values:
                method_values[method] = values
        
        if len(method_values) < 2:
            logger.warning("Need at least 2 methods for comparison")
            return {}
        
        comparison_results = {}
        methods = list(method_values.keys())
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                t_stat, p_value = stats.ttest_ind(
                    method_values[method1],
                    method_values[method2]
                )
                
                comparison_key = f"{method1}_vs_{method2}"
                comparison_results[comparison_key] = {
                    'metric': metric,
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'mean_diff': np.mean(method_values[method1]) - np.mean(method_values[method2])
                }
        
        return comparison_results
    
    def generate_latex_table(self, df: pd.DataFrame, 
                            metrics: List[str] = None) -> str:
        """Generate LaTeX table from results"""
        if metrics is None:
            metrics = ['num_matches', 'matching_efficiency', 'avg_matching_distance',
                      'spatial_entropy1', 'spatial_entropy2']
        
        columns = ['Method']
        for metric in metrics:
            columns.append(f'{metric}_mean')
        
        table_df = df[columns].copy()
        
        display_names = {
            'num_matches_mean': 'Matches',
            'matching_efficiency_mean': 'Efficiency (%)',
            'avg_matching_distance_mean': 'Avg Distance',
            'spatial_entropy1_mean': 'Entropy 1',
            'spatial_entropy2_mean': 'Entropy 2'
        }
        
        table_df.rename(columns=display_names, inplace=True)
        
        latex_str = table_df.to_latex(index=False, float_format='%.2f')
        
        return latex_str
    
    def generate_report(self, dataset_name: str) -> str:
        """Generate comprehensive text report"""
        results = self.load_results(dataset_name)
        
        if not results:
            return "No results found"
        
        stats_df = self.compute_statistics(results)
        
        report = []
        report.append("=" * 80)
        report.append(f"FEATURE DETECTION COMPARISON REPORT - {dataset_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append("")
        
        for _, row in stats_df.iterrows():
            method = row['Method']
            report.append(f"{method}:")
            report.append(f"  Number of Matches: {row.get('num_matches_mean', 0):.2f} ± {row.get('num_matches_std', 0):.2f}")
            report.append(f"  Matching Efficiency: {row.get('matching_efficiency_mean', 0):.2f}% ± {row.get('matching_efficiency_std', 0):.2f}%")
            report.append(f"  Avg Match Distance: {row.get('avg_matching_distance_mean', 0):.4f} ± {row.get('avg_matching_distance_std', 0):.4f}")
            report.append(f"  Spatial Entropy (Img1): {row.get('spatial_entropy1_mean', 0):.4f} ± {row.get('spatial_entropy1_std', 0):.4f}")
            report.append(f"  Spatial Entropy (Img2): {row.get('spatial_entropy2_mean', 0):.4f} ± {row.get('spatial_entropy2_std', 0):.4f}")
            report.append(f"  Detection Time: {row.get('detection_time_mean', 0):.4f}s ± {row.get('detection_time_std', 0):.4f}s")
            report.append(f"  Matching Time: {row.get('matching_time_mean', 0):.4f}s ± {row.get('matching_time_std', 0):.4f}s")
            report.append("")
        
        report.append("STATISTICAL COMPARISONS")
        report.append("-" * 80)
        report.append("")
        
        for metric in ['matching_efficiency', 'avg_matching_distance', 'num_matches']:
            report.append(f"Metric: {metric}")
            comparisons = self.compare_methods(results, metric)
            
            for comparison_key, comparison_result in comparisons.items():
                report.append(f"  {comparison_key}:")
                report.append(f"    p-value: {comparison_result['p_value']:.4f}")
                report.append(f"    Significant: {comparison_result['significant']}")
                report.append(f"    Mean difference: {comparison_result['mean_diff']:.4f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, dataset_name: str, output_file: str = None) -> None:
        """Generate and save report to file"""
        report = self.generate_report(dataset_name)
        
        if output_file is None:
            output_file = self.results_dir / f"{dataset_name}_report.txt"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_file}")
    
    def export_to_csv(self, dataset_name: str) -> None:
        """Export results to CSV file"""
        results = self.load_results(dataset_name)
        
        if not results:
            logger.error("No results to export")
            return
        
        all_data = []
        for method, method_results in results.items():
            for result in method_results:
                result_copy = result.copy()
                result_copy['method'] = method
                all_data.append(result_copy)
        
        df = pd.DataFrame(all_data)
        
        csv_file = self.results_dir / f"{dataset_name}_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results exported to {csv_file}")
        
        stats_df = self.compute_statistics(results)
        stats_file = self.results_dir / f"{dataset_name}_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        
        logger.info(f"Statistics exported to {stats_file}")