#!/usr/bin/env python3
"""
Run Single Ablation
===================

Run a single ablation experiment for quick testing or debugging.

Usage:
    # Run baseline on petclinic
    python -m ablation_study.scripts.run_single_ablation baseline petclinic
    
    # Run A1.1 on realworld with custom run ID
    python -m ablation_study.scripts.run_single_ablation A1.1 realworld --run-id 2
    
    # Force re-run
    python -m ablation_study.scripts.run_single_ablation A1.1 petclinic --force
    
    # Dry run
    python -m ablation_study.scripts.run_single_ablation A1.1 petclinic --dry-run
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
    ComponentFactory
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single ablation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'ablation_id',
        type=str,
        help='Ablation ID (e.g., baseline, A1.1, A2.3)'
    )
    
    parser.add_argument(
        'application',
        type=str,
        help='Application name (e.g., petclinic, realworld)'
    )
    
    parser.add_argument(
        '--run-id',
        type=int,
        default=1,
        help='Run ID number (default: 1)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if already completed'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )
    
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
        '--show-config',
        action='store_true',
        help='Show ablation configuration and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Ollama model to use (e.g., qwen2.5-coder:7b, llava-llama3:8b)'
    )
    
    return parser.parse_args()


def show_ablation_config(ablation_id: str):
    """Show detailed configuration for an ablation."""
    loader = AblationConfigLoader()
    ablation = loader.get_ablation(ablation_id)
    
    if ablation is None:
        print(f"Error: Unknown ablation '{ablation_id}'")
        print("\nAvailable ablations:")
        for aid in loader.get_ablation_ids():
            print(f"  - {aid}")
        return False
    
    print("\n" + "=" * 70)
    print(f"ABLATION CONFIGURATION: {ablation.id}")
    print("=" * 70)
    print(f"Name: {ablation.name}")
    print(f"Description: {ablation.description}")
    
    # Create components to show their configuration
    components = ComponentFactory.create_all(ablation)
    
    print("\nComponent Configuration:")
    print("-" * 50)
    
    print(f"\n1. Context Extraction:")
    ctx = components['context_extractor']
    print(f"   - Include screenshot: {ctx.include_screenshot}")
    print(f"   - Include previous state: {ctx.include_previous_state}")
    print(f"   - Include previous action: {ctx.include_previous_action}")
    
    print(f"\n2. Prompting Strategy:")
    pm = components['prompt_manager']
    print(f"   - Strategy: {pm.strategy}")
    
    print(f"\n3. Scoring Function:")
    sf = components['scoring_function']
    print(f"   - Method: {sf.method}")
    print(f"   - Parameters: p={sf.p}, R={sf.R}")
    
    print(f"\n4. Score Accumulation:")
    sa = components['score_accumulator']
    print(f"   - Method: {sa.method}")
    
    print(f"\n5. Score Threshold:")
    st = components['score_threshold']
    print(f"   - Enabled: {st.enabled}")
    print(f"   - Min score: {st.min_score}")
    
    print("\n" + "=" * 70 + "\n")
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate ablation
    loader = AblationConfigLoader()
    ablation = loader.get_ablation(args.ablation_id)
    
    if ablation is None:
        print(f"Error: Unknown ablation '{args.ablation_id}'")
        print("\nAvailable ablations:")
        for aid in loader.get_ablation_ids():
            a = loader.get_ablation(aid)
            print(f"  {aid:<12} - {a.description if a else ''}")
        return 1
    
    # Validate application
    app = loader.get_application(args.application)
    if app is None:
        print(f"Error: Unknown application '{args.application}'")
        print("\nAvailable applications:")
        for app_name in loader.get_application_names():
            print(f"  - {app_name}")
        return 1
    
    # Handle show config
    if args.show_config:
        show_ablation_config(args.ablation_id)
        return 0
    
    # Setup runner
    verbose = not args.quiet and args.verbose
    
    runner = AblationRunner(
        results_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=verbose,
        model=args.model
    )
    
    # Get model info for display
    model_name = args.model if args.model else runner.config_loader.get_model().name
    
    print("\n" + "=" * 70)
    print("SINGLE ABLATION RUN")
    print("=" * 70)
    print(f"Ablation: {args.ablation_id}")
    print(f"Application: {args.application}")
    print(f"Model: {model_name}")
    print(f"Run ID: {args.run_id}")
    print(f"Description: {ablation.description}")
    print("=" * 70 + "\n")
    
    try:
        success = runner.run_single(
            ablation_id=args.ablation_id,
            app_name=args.application,
            run_id=args.run_id,
            force=args.force
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
