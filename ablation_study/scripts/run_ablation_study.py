#!/usr/bin/env python3
"""
Run Ablation Study
==================

Main entry point for running the complete ablation study.

Usage:
    # Run all ablations on all recommended apps (3 repetitions each)
    python -m ablation_study.scripts.run_ablation_study
    
    # Run specific ablations
    python -m ablation_study.scripts.run_ablation_study -A A1.1 A1.2 A1.3
    
    # Run on specific apps
    python -m ablation_study.scripts.run_ablation_study -a petclinic realworld
    
    # Custom repetitions
    python -m ablation_study.scripts.run_ablation_study -R 5
    
    # Dry run (show plan only)
    python -m ablation_study.scripts.run_ablation_study --dry-run
    
    # Force re-run (ignore checkpoints)
    python -m ablation_study.scripts.run_ablation_study --force
    
    # Show current status
    python -m ablation_study.scripts.run_ablation_study --status
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_study.orchestrator import (
    AblationConfigLoader,
    AblationRunner,
    AblationCheckpointManager,
    AblationMetricsCollector
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AutoE2E Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Selection arguments
    parser.add_argument(
        '-A', '--ablations',
        nargs='+',
        metavar='ID',
        help='Ablation IDs to run (e.g., A1.1 A1.2). Default: all ablations'
    )
    
    parser.add_argument(
        '-a', '--apps', '--applications',
        nargs='+',
        metavar='APP',
        help='Applications to test (e.g., petclinic realworld). Default: recommended subset'
    )
    
    parser.add_argument(
        '-R', '--runs', '--repetitions',
        type=int,
        default=None,
        metavar='N',
        help='Number of repetitions per ablation-app pair. Default: 3'
    )
    
    # Execution control
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if already completed'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint (default: True)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignoring checkpoint'
    )
    
    # Output control
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory for results'
    )
    
    # Info commands
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current checkpoint status and exit'
    )
    
    parser.add_argument(
        '--list-ablations',
        action='store_true',
        help='List all available ablations and exit'
    )
    
    parser.add_argument(
        '--list-apps',
        action='store_true',
        help='List all available applications and exit'
    )
    
    parser.add_argument(
        '--show-plan',
        action='store_true',
        help='Show execution plan and exit'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration summary and exit'
    )
    
    return parser.parse_args()


def show_ablations():
    """List all available ablations."""
    loader = AblationConfigLoader()
    
    print("\n" + "=" * 70)
    print("AVAILABLE ABLATIONS")
    print("=" * 70)
    
    ablations = loader.get_ablations()
    
    # Group by component
    groups = {
        'Baseline': [],
        'C1 - Context Extraction': [],
        'C2 - Prompting Strategy': [],
        'C3 - Scoring Function': [],
        'C4 - Feature Aggregation': [],
        'C5 - Score Accumulation': [],
        'C6 - Finality Detection': [],
        'C7 - Probability Filtering': []
    }
    
    for a in ablations:
        if a.id == 'baseline':
            groups['Baseline'].append(a)
        elif a.id.startswith('A1'):
            groups['C1 - Context Extraction'].append(a)
        elif a.id.startswith('A2'):
            groups['C2 - Prompting Strategy'].append(a)
        elif a.id.startswith('A3'):
            groups['C3 - Scoring Function'].append(a)
        elif a.id.startswith('A4'):
            groups['C4 - Feature Aggregation'].append(a)
        elif a.id.startswith('A5'):
            groups['C5 - Score Accumulation'].append(a)
        elif a.id.startswith('A6'):
            groups['C6 - Finality Detection'].append(a)
        elif a.id.startswith('A7'):
            groups['C7 - Probability Filtering'].append(a)
    
    for group_name, items in groups.items():
        if items:
            print(f"\n{group_name}:")
            for a in items:
                print(f"  {a.id:<12} - {a.description}")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(ablations)} ablations")
    print("=" * 70 + "\n")


def show_apps():
    """List all available applications."""
    loader = AblationConfigLoader()
    
    print("\n" + "=" * 70)
    print("AVAILABLE APPLICATIONS")
    print("=" * 70)
    
    apps = loader.get_applications()
    recommended = loader.get_recommended_apps()
    
    for app in apps:
        rec_marker = " [recommended]" if app.name in recommended else ""
        print(f"\n  {app.name}{rec_marker}")
        print(f"    Config: {app.config_name}")
        print(f"    Features: {app.feature_count}")
        print(f"    URL: {app.url}")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(apps)} applications")
    print(f"Recommended subset: {', '.join(recommended)}")
    print("=" * 70 + "\n")


def show_status():
    """Show current checkpoint status."""
    checkpoint = AblationCheckpointManager()
    metrics = AblationMetricsCollector()
    
    checkpoint.print_status()
    metrics.print_summary()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle info commands
    if args.list_ablations:
        show_ablations()
        return 0
    
    if args.list_apps:
        show_apps()
        return 0
    
    if args.status:
        show_status()
        return 0
    
    if args.show_config:
        loader = AblationConfigLoader()
        loader.print_summary()
        return 0
    
    # Setup runner
    verbose = not args.quiet and args.verbose
    
    runner = AblationRunner(
        results_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=verbose
    )
    
    # Handle show plan
    if args.show_plan or args.dry_run:
        runner.show_plan(
            ablation_ids=args.ablations,
            applications=args.apps,
            repetitions=args.runs
        )
        if not args.dry_run:
            return 0
    
    # Determine resume behavior
    resume = args.resume and not args.no_resume
    
    # Run the study
    try:
        results = runner.run_all(
            ablation_ids=args.ablations,
            applications=args.apps,
            repetitions=args.runs,
            force=args.force,
            resume=resume
        )
        
        if results['failed'] > 0:
            print(f"\nWarning: {results['failed']} runs failed")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print("Run with --resume to continue.")
        return 130
    
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
