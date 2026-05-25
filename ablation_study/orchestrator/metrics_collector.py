"""
Ablation Metrics Collector
==========================

Collects, stores, and analyzes metrics from ablation study experiments.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import statistics


@dataclass
class RunMetrics:
    """Metrics collected from a single experiment run."""
    ablation_id: str
    application: str
    run_id: int
    
    # Timing
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Coverage metrics
    total_features: int
    detected_features: int
    feature_coverage: float
    
    # Exploration metrics
    states_explored: int
    actions_executed: int
    unique_pages: int
    
    # Quality metrics
    precision: float
    recall: float
    f1_score: float
    
    # Component-specific
    component_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    success: bool = True


@dataclass
class AblationSummary:
    """Summary statistics for an ablation across all runs."""
    ablation_id: str
    ablation_description: str
    total_runs: int
    successful_runs: int
    
    # Coverage stats (mean ± std)
    coverage_mean: float
    coverage_std: float
    coverage_min: float
    coverage_max: float
    
    # Precision/Recall/F1 stats
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    
    # Exploration stats
    states_mean: float
    states_std: float
    duration_mean: float
    duration_std: float
    
    # Per-application breakdown
    by_application: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AblationMetricsCollector:
    """
    Collects and analyzes metrics from ablation study experiments.
    
    Features:
    - Store run metrics
    - Compute summary statistics
    - Compare ablations
    - Export for analysis
    
    Usage:
        collector = AblationMetricsCollector()
        
        # Record a run
        collector.record_run(metrics)
        
        # Get summary for an ablation
        summary = collector.get_ablation_summary("A1.1")
        
        # Compare two ablations
        comparison = collector.compare_ablations("baseline", "A1.1")
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize metrics collector.
        
        Args:
            results_dir: Directory for storing metrics.
                        Defaults to ablation_study/results.
        """
        if results_dir is None:
            current = Path(__file__).resolve()
            results_dir = current.parent.parent / "results"
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.results_dir / "all_metrics.json"
        self._metrics: List[RunMetrics] = []
        self._load()
    
    def _load(self):
        """Load existing metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self._metrics = [RunMetrics(**m) for m in data.get('runs', [])]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load metrics file: {e}")
                self._metrics = []
    
    def _save(self):
        """Save metrics to file."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_runs': len(self._metrics),
            'runs': [asdict(m) for m in self._metrics]
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_run(self, metrics: RunMetrics):
        """Record metrics from a completed run."""
        self._metrics.append(metrics)
        self._save()
        
        # Also save individual run file
        run_file = self.results_dir / f"{metrics.ablation_id}_{metrics.application}_run{metrics.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
    
    def get_runs(
        self,
        ablation_id: Optional[str] = None,
        application: Optional[str] = None,
        successful_only: bool = True
    ) -> List[RunMetrics]:
        """Get runs, optionally filtered."""
        runs = self._metrics
        
        if ablation_id:
            runs = [r for r in runs if r.ablation_id == ablation_id]
        if application:
            runs = [r for r in runs if r.application == application]
        if successful_only:
            runs = [r for r in runs if r.success]
        
        return runs
    
    def _compute_stats(self, values: List[float]) -> Tuple[float, float, float, float]:
        """Compute mean, std, min, max for a list of values."""
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std, min(values), max(values)
    
    def get_ablation_summary(
        self,
        ablation_id: str,
        description: str = ""
    ) -> AblationSummary:
        """
        Compute summary statistics for an ablation.
        
        Args:
            ablation_id: The ablation ID
            description: Description of the ablation
            
        Returns:
            AblationSummary with statistics
        """
        runs = self.get_runs(ablation_id=ablation_id, successful_only=True)
        all_runs = self.get_runs(ablation_id=ablation_id, successful_only=False)
        
        if not runs:
            return AblationSummary(
                ablation_id=ablation_id,
                ablation_description=description,
                total_runs=len(all_runs),
                successful_runs=0,
                coverage_mean=0.0, coverage_std=0.0, coverage_min=0.0, coverage_max=0.0,
                precision_mean=0.0, precision_std=0.0,
                recall_mean=0.0, recall_std=0.0,
                f1_mean=0.0, f1_std=0.0,
                states_mean=0.0, states_std=0.0,
                duration_mean=0.0, duration_std=0.0
            )
        
        coverages = [r.feature_coverage for r in runs]
        precisions = [r.precision for r in runs]
        recalls = [r.recall for r in runs]
        f1s = [r.f1_score for r in runs]
        states = [float(r.states_explored) for r in runs]
        durations = [r.duration_seconds for r in runs]
        
        cov_mean, cov_std, cov_min, cov_max = self._compute_stats(coverages)
        prec_mean, prec_std, _, _ = self._compute_stats(precisions)
        rec_mean, rec_std, _, _ = self._compute_stats(recalls)
        f1_mean, f1_std, _, _ = self._compute_stats(f1s)
        states_mean, states_std, _, _ = self._compute_stats(states)
        dur_mean, dur_std, _, _ = self._compute_stats(durations)
        
        # Per-application breakdown
        by_app = {}
        applications = set(r.application for r in runs)
        for app in applications:
            app_runs = [r for r in runs if r.application == app]
            app_coverages = [r.feature_coverage for r in app_runs]
            app_f1s = [r.f1_score for r in app_runs]
            by_app[app] = {
                'runs': len(app_runs),
                'coverage_mean': statistics.mean(app_coverages) if app_coverages else 0.0,
                'f1_mean': statistics.mean(app_f1s) if app_f1s else 0.0
            }
        
        return AblationSummary(
            ablation_id=ablation_id,
            ablation_description=description,
            total_runs=len(all_runs),
            successful_runs=len(runs),
            coverage_mean=cov_mean, coverage_std=cov_std,
            coverage_min=cov_min, coverage_max=cov_max,
            precision_mean=prec_mean, precision_std=prec_std,
            recall_mean=rec_mean, recall_std=rec_std,
            f1_mean=f1_mean, f1_std=f1_std,
            states_mean=states_mean, states_std=states_std,
            duration_mean=dur_mean, duration_std=dur_std,
            by_application=by_app
        )
    
    def compare_ablations(
        self,
        baseline_id: str,
        ablation_id: str
    ) -> Dict[str, Any]:
        """
        Compare an ablation against baseline.
        
        Args:
            baseline_id: Baseline ablation ID
            ablation_id: Comparison ablation ID
            
        Returns:
            Comparison results with differences
        """
        baseline = self.get_ablation_summary(baseline_id)
        ablation = self.get_ablation_summary(ablation_id)
        
        def diff(a: float, b: float) -> float:
            return a - b
        
        def pct_diff(a: float, b: float) -> float:
            if b == 0:
                return 0.0
            return ((a - b) / b) * 100
        
        return {
            'baseline_id': baseline_id,
            'ablation_id': ablation_id,
            'coverage': {
                'baseline_mean': baseline.coverage_mean,
                'ablation_mean': ablation.coverage_mean,
                'difference': diff(ablation.coverage_mean, baseline.coverage_mean),
                'pct_change': pct_diff(ablation.coverage_mean, baseline.coverage_mean)
            },
            'f1_score': {
                'baseline_mean': baseline.f1_mean,
                'ablation_mean': ablation.f1_mean,
                'difference': diff(ablation.f1_mean, baseline.f1_mean),
                'pct_change': pct_diff(ablation.f1_mean, baseline.f1_mean)
            },
            'precision': {
                'baseline_mean': baseline.precision_mean,
                'ablation_mean': ablation.precision_mean,
                'difference': diff(ablation.precision_mean, baseline.precision_mean),
                'pct_change': pct_diff(ablation.precision_mean, baseline.precision_mean)
            },
            'recall': {
                'baseline_mean': baseline.recall_mean,
                'ablation_mean': ablation.recall_mean,
                'difference': diff(ablation.recall_mean, baseline.recall_mean),
                'pct_change': pct_diff(ablation.recall_mean, baseline.recall_mean)
            },
            'efficiency': {
                'baseline_states': baseline.states_mean,
                'ablation_states': ablation.states_mean,
                'states_difference': diff(ablation.states_mean, baseline.states_mean),
                'baseline_duration': baseline.duration_mean,
                'ablation_duration': ablation.duration_mean,
                'duration_difference': diff(ablation.duration_mean, baseline.duration_mean)
            }
        }
    
    def export_for_latex(self, output_file: Optional[str] = None) -> str:
        """
        Export summary table in LaTeX format.
        
        Args:
            output_file: Optional file path to write table
            
        Returns:
            LaTeX table string
        """
        ablation_ids = sorted(set(r.ablation_id for r in self._metrics))
        
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ablation Study Results}",
            "\\label{tab:ablation_results}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Ablation & Coverage & Precision & Recall & F1 & States & Duration \\\\",
            "\\midrule"
        ]
        
        for ablation_id in ablation_ids:
            summary = self.get_ablation_summary(ablation_id)
            line = (
                f"{ablation_id} & "
                f"{summary.coverage_mean:.2f}±{summary.coverage_std:.2f} & "
                f"{summary.precision_mean:.2f}±{summary.precision_std:.2f} & "
                f"{summary.recall_mean:.2f}±{summary.recall_std:.2f} & "
                f"{summary.f1_mean:.2f}±{summary.f1_std:.2f} & "
                f"{summary.states_mean:.0f}±{summary.states_std:.0f} & "
                f"{summary.duration_mean:.0f}s \\\\"
            )
            lines.append(line)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        table = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(table)
        
        return table
    
    def export_for_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export all runs to CSV format.
        
        Args:
            output_file: Optional file path
            
        Returns:
            CSV string
        """
        headers = [
            "ablation_id", "application", "run_id",
            "feature_coverage", "precision", "recall", "f1_score",
            "states_explored", "duration_seconds", "success"
        ]
        
        lines = [",".join(headers)]
        
        for r in self._metrics:
            line = ",".join([
                r.ablation_id,
                r.application,
                str(r.run_id),
                f"{r.feature_coverage:.4f}",
                f"{r.precision:.4f}",
                f"{r.recall:.4f}",
                f"{r.f1_score:.4f}",
                str(r.states_explored),
                f"{r.duration_seconds:.2f}",
                str(r.success)
            ])
            lines.append(line)
        
        csv_content = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(csv_content)
        
        return csv_content
    
    def get_statistics(self) -> Dict:
        """Get overall collection statistics."""
        ablation_ids = set(r.ablation_id for r in self._metrics)
        applications = set(r.application for r in self._metrics)
        
        return {
            'total_runs': len(self._metrics),
            'successful_runs': sum(1 for r in self._metrics if r.success),
            'failed_runs': sum(1 for r in self._metrics if not r.success),
            'unique_ablations': len(ablation_ids),
            'unique_applications': len(applications),
            'ablations': sorted(ablation_ids),
            'applications': sorted(applications)
        }
    
    def print_summary(self):
        """Print summary of collected metrics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 70)
        print("ABLATION STUDY METRICS SUMMARY")
        print("=" * 70)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Successful: {stats['successful_runs']}")
        print(f"Failed: {stats['failed_runs']}")
        print(f"Ablations: {stats['unique_ablations']}")
        print(f"Applications: {stats['unique_applications']}")
        
        if stats['total_runs'] > 0:
            print("\nPer-Ablation Summary:")
            print("-" * 70)
            print(f"{'Ablation':<12} {'Runs':>6} {'Coverage':>12} {'F1':>12} {'States':>10}")
            print("-" * 70)
            
            for ablation_id in sorted(stats['ablations']):
                summary = self.get_ablation_summary(ablation_id)
                print(
                    f"{ablation_id:<12} "
                    f"{summary.successful_runs:>6} "
                    f"{summary.coverage_mean:>5.1f}±{summary.coverage_std:<5.1f} "
                    f"{summary.f1_mean:>5.2f}±{summary.f1_std:<5.2f} "
                    f"{summary.states_mean:>10.0f}"
                )
        
        print("=" * 70 + "\n")
