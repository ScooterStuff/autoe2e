#!/usr/bin/env python3
"""
Analyze Ablation Study Results
==============================

Analyze and visualize results from ablation study experiments.

Usage:
    # Generate summary report
    python -m ablation_study.scripts.analyze_ablations
    
    # Compare specific ablations against baseline
    python -m ablation_study.scripts.analyze_ablations --compare A1.1 A1.2
    
    # Export to LaTeX table
    python -m ablation_study.scripts.analyze_ablations --latex
    
    # Export to CSV
    python -m ablation_study.scripts.analyze_ablations --csv results.csv
    
    # Statistical significance tests
    python -m ablation_study.scripts.analyze_ablations --significance
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_study.orchestrator import (
    AblationConfigLoader,
    AblationMetricsCollector
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ablation study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Analysis options
    parser.add_argument(
        '--compare',
        nargs='+',
        metavar='ID',
        help='Compare specific ablations against baseline'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default='baseline',
        help='Baseline ablation ID for comparison (default: baseline)'
    )
    
    parser.add_argument(
        '--by-app',
        action='store_true',
        help='Show breakdown by application'
    )
    
    parser.add_argument(
        '--by-component',
        action='store_true',
        help='Show breakdown by component'
    )
    
    # Export options
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Generate LaTeX table'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        metavar='FILE',
        help='Export to CSV file'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        metavar='FILE',
        help='Export full analysis to JSON'
    )
    
    # Statistical options
    parser.add_argument(
        '--significance',
        action='store_true',
        help='Perform statistical significance tests'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    
    # Output
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for generated files'
    )
    
    return parser.parse_args()


def print_summary_table(collector: AblationMetricsCollector, loader: AblationConfigLoader):
    """Print summary table of all ablations."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 100)
    
    stats = collector.get_statistics()
    
    if stats['total_runs'] == 0:
        print("\nNo results found. Run some experiments first.")
        return
    
    print(f"\nTotal runs: {stats['total_runs']} "
          f"(successful: {stats['successful_runs']}, failed: {stats['failed_runs']})")
    print(f"Ablations tested: {stats['unique_ablations']}")
    print(f"Applications tested: {stats['unique_applications']}")
    
    print("\n" + "-" * 100)
    print(f"{'Ablation':<12} {'Runs':>6} {'Coverage':>14} {'Precision':>14} "
          f"{'Recall':>14} {'F1':>14} {'States':>10}")
    print("-" * 100)
    
    for ablation_id in sorted(stats['ablations']):
        ablation = loader.get_ablation(ablation_id)
        desc = ablation.description[:30] if ablation else ''
        summary = collector.get_ablation_summary(ablation_id, desc)
        
        print(f"{ablation_id:<12} "
              f"{summary.successful_runs:>6} "
              f"{summary.coverage_mean:>6.1f}±{summary.coverage_std:<6.1f} "
              f"{summary.precision_mean:>6.2f}±{summary.precision_std:<6.2f} "
              f"{summary.recall_mean:>6.2f}±{summary.recall_std:<6.2f} "
              f"{summary.f1_mean:>6.2f}±{summary.f1_std:<6.2f} "
              f"{summary.states_mean:>10.0f}")
    
    print("-" * 100)


def compare_ablations(
    collector: AblationMetricsCollector,
    loader: AblationConfigLoader,
    ablation_ids: List[str],
    baseline_id: str = 'baseline'
):
    """Compare ablations against baseline."""
    print("\n" + "=" * 90)
    print(f"ABLATION COMPARISON (vs {baseline_id})")
    print("=" * 90)
    
    baseline = collector.get_ablation_summary(baseline_id)
    
    print(f"\nBaseline ({baseline_id}):")
    print(f"  Coverage: {baseline.coverage_mean:.2f}%")
    print(f"  F1 Score: {baseline.f1_mean:.3f}")
    print(f"  States:   {baseline.states_mean:.0f}")
    
    print("\n" + "-" * 90)
    print(f"{'Ablation':<12} {'Description':<30} {'Δ Coverage':>12} {'Δ F1':>10} {'Δ States':>10}")
    print("-" * 90)
    
    for ablation_id in ablation_ids:
        if ablation_id == baseline_id:
            continue
        
        comparison = collector.compare_ablations(baseline_id, ablation_id)
        ablation = loader.get_ablation(ablation_id)
        desc = ablation.description[:28] if ablation else ''
        
        cov_diff = comparison['coverage']['difference']
        f1_diff = comparison['f1_score']['difference']
        states_diff = comparison['efficiency']['states_difference']
        
        # Color indicators (for terminal)
        cov_sign = "+" if cov_diff > 0 else ""
        f1_sign = "+" if f1_diff > 0 else ""
        states_sign = "+" if states_diff > 0 else ""
        
        print(f"{ablation_id:<12} {desc:<30} "
              f"{cov_sign}{cov_diff:>10.2f}% "
              f"{f1_sign}{f1_diff:>9.3f} "
              f"{states_sign}{states_diff:>9.0f}")
    
    print("-" * 90)


def analyze_by_component(
    collector: AblationMetricsCollector,
    loader: AblationConfigLoader
):
    """Analyze results grouped by component."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY COMPONENT")
    print("=" * 80)
    
    components = {
        'C1 - Context': ['A1.1', 'A1.2', 'A1.3', 'A1.4'],
        'C2 - Prompting': ['A2.1', 'A2.2', 'A2.3'],
        'C3 - Scoring': ['A3.1', 'A3.2', 'A3.3', 'A3.4', 'A3.5'],
        'C4 - Accumulation': ['A5.1', 'A5.2', 'A5.3'],
        'C5 - Threshold': ['A7.1', 'A7.2', 'A7.3']
    }
    
    baseline = collector.get_ablation_summary('baseline')
    
    for component_name, ablation_ids in components.items():
        print(f"\n{component_name}:")
        print("-" * 60)
        
        has_data = False
        for ablation_id in ablation_ids:
            summary = collector.get_ablation_summary(ablation_id)
            if summary.successful_runs > 0:
                has_data = True
                ablation = loader.get_ablation(ablation_id)
                desc = ablation.description[:40] if ablation else ''
                
                cov_diff = summary.coverage_mean - baseline.coverage_mean
                cov_sign = "+" if cov_diff > 0 else ""
                
                print(f"  {ablation_id}: {desc}")
                print(f"         Coverage: {summary.coverage_mean:.1f}% ({cov_sign}{cov_diff:.1f}%)")
        
        if not has_data:
            print("  No data available")


def analyze_by_application(
    collector: AblationMetricsCollector,
    loader: AblationConfigLoader
):
    """Analyze results grouped by application."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY APPLICATION")
    print("=" * 80)
    
    stats = collector.get_statistics()
    
    for app in sorted(stats['applications']):
        print(f"\n{app.upper()}:")
        print("-" * 60)
        
        app_runs = collector.get_runs(application=app)
        ablation_ids = set(r.ablation_id for r in app_runs)
        
        for ablation_id in sorted(ablation_ids):
            runs = [r for r in app_runs if r.ablation_id == ablation_id]
            if runs:
                coverages = [r.feature_coverage for r in runs]
                import statistics
                mean_cov = statistics.mean(coverages) * 100
                std_cov = statistics.stdev(coverages) * 100 if len(coverages) > 1 else 0
                
                print(f"  {ablation_id:<12} Coverage: {mean_cov:>5.1f}±{std_cov:<5.1f}% "
                      f"(n={len(runs)})")


def perform_significance_tests(
    collector: AblationMetricsCollector,
    loader: AblationConfigLoader,
    baseline_id: str = 'baseline',
    alpha: float = 0.05
):
    """Perform statistical significance tests."""
    print("\n" + "=" * 80)
    print(f"STATISTICAL SIGNIFICANCE TESTS (α = {alpha})")
    print("=" * 80)
    
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        print("\nError: scipy is required for significance tests.")
        print("Install with: pip install scipy")
        return
    
    baseline_runs = collector.get_runs(ablation_id=baseline_id)
    if not baseline_runs:
        print(f"\nNo baseline data found for '{baseline_id}'")
        return
    
    baseline_coverages = [r.feature_coverage for r in baseline_runs]
    
    all_stats = collector.get_statistics()
    
    print(f"\nBaseline ({baseline_id}): n={len(baseline_runs)}, "
          f"mean coverage={sum(baseline_coverages)/len(baseline_coverages)*100:.1f}%")
    
    print("\n" + "-" * 80)
    print(f"{'Ablation':<12} {'n':>4} {'Mean':>8} {'t-stat':>10} {'p-value':>10} {'Sig?':>8}")
    print("-" * 80)
    
    for ablation_id in sorted(all_stats['ablations']):
        if ablation_id == baseline_id:
            continue
        
        ablation_runs = collector.get_runs(ablation_id=ablation_id)
        if len(ablation_runs) < 2:
            continue
        
        ablation_coverages = [r.feature_coverage for r in ablation_runs]
        
        # Perform Welch's t-test (doesn't assume equal variances)
        t_stat, p_value = scipy_stats.ttest_ind(
            baseline_coverages,
            ablation_coverages,
            equal_var=False
        )
        
        is_significant = p_value < alpha
        sig_marker = "YES **" if is_significant and ablation_coverages else "no"
        if is_significant:
            direction = "↑" if sum(ablation_coverages)/len(ablation_coverages) > sum(baseline_coverages)/len(baseline_coverages) else "↓"
            sig_marker = f"YES {direction}"
        
        mean_cov = sum(ablation_coverages)/len(ablation_coverages)*100
        
        print(f"{ablation_id:<12} {len(ablation_runs):>4} {mean_cov:>7.1f}% "
              f"{t_stat:>10.3f} {p_value:>10.4f} {sig_marker:>8}")
    
    print("-" * 80)
    print("\nNote: Using Welch's t-test (unequal variances assumed)")
    print("↑ = significantly better, ↓ = significantly worse")


def export_latex(collector: AblationMetricsCollector, output_file: Optional[str] = None):
    """Export results as LaTeX table."""
    table = collector.export_for_latex(output_file)
    
    if output_file:
        print(f"\nLaTeX table exported to: {output_file}")
    else:
        print("\n" + "=" * 60)
        print("LATEX TABLE")
        print("=" * 60)
        print(table)


def export_json_analysis(
    collector: AblationMetricsCollector,
    loader: AblationConfigLoader,
    output_file: str
):
    """Export complete analysis to JSON."""
    stats = collector.get_statistics()
    
    analysis = {
        'generated_at': datetime.now().isoformat(),
        'statistics': stats,
        'ablations': {},
        'comparisons': []
    }
    
    # Add ablation summaries
    for ablation_id in stats['ablations']:
        ablation = loader.get_ablation(ablation_id)
        summary = collector.get_ablation_summary(ablation_id)
        
        analysis['ablations'][ablation_id] = {
            'description': ablation.description if ablation else '',
            'runs': summary.successful_runs,
            'coverage_mean': summary.coverage_mean,
            'coverage_std': summary.coverage_std,
            'f1_mean': summary.f1_mean,
            'f1_std': summary.f1_std,
            'by_application': summary.by_application
        }
    
    # Add comparisons against baseline
    if 'baseline' in stats['ablations']:
        for ablation_id in stats['ablations']:
            if ablation_id != 'baseline':
                comparison = collector.compare_ablations('baseline', ablation_id)
                analysis['comparisons'].append(comparison)
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nFull analysis exported to: {output_file}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize
    loader = AblationConfigLoader()
    collector = AblationMetricsCollector()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle exports first
    if args.latex:
        latex_file = output_dir / "ablation_results.tex" if args.output_dir else None
        export_latex(collector, str(latex_file) if latex_file else None)
    
    if args.csv:
        csv_file = output_dir / args.csv if args.output_dir else args.csv
        collector.export_for_csv(str(csv_file))
        print(f"\nCSV exported to: {csv_file}")
    
    if args.json:
        json_file = output_dir / args.json if args.output_dir else args.json
        export_json_analysis(collector, loader, str(json_file))
    
    # Print summary (unless only doing exports)
    if not (args.latex or args.csv or args.json):
        print_summary_table(collector, loader)
    
    # Handle comparison
    if args.compare:
        compare_ablations(collector, loader, args.compare, args.baseline)
    
    # Handle by-component analysis
    if args.by_component:
        analyze_by_component(collector, loader)
    
    # Handle by-app analysis
    if args.by_app:
        analyze_by_application(collector, loader)
    
    # Handle significance tests
    if args.significance:
        perform_significance_tests(collector, loader, args.baseline, args.alpha)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
