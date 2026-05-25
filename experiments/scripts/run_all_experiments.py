#!/usr/bin/env python3
"""
Run All Experiments Script
==========================

Entry point for running AutoE2E replication experiments.
Orchestrates experiments across all models and applications.

Usage:
    # Run all experiments
    python scripts/run_all_experiments.py
    
    # Run specific model only
    python scripts/run_all_experiments.py --model qwen3:8b
    
    # Run specific application only
    python scripts/run_all_experiments.py --app petclinic
    
    # Run specific combination
    python scripts/run_all_experiments.py --model devstral:24b --app conduit
    
    # Resume from checkpoint
    python scripts/run_all_experiments.py --resume
    
    # Dry run (show what would be executed)
    python scripts/run_all_experiments.py --dry-run
    
    # Skip Docker container management
    python scripts/run_all_experiments.py --skip-docker
"""

import os
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.orchestrator import (
    ConfigLoader,
    CheckpointManager,
    ResultCollector,
    ExperimentRunner
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AutoE2E replication experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run all experiments
  %(prog)s --model qwen3:8b         Run only qwen3:8b model
  %(prog)s --app petclinic          Run only petclinic application
  %(prog)s --max-states 30          Limit to 30 states explored
  %(prog)s --timeout 60             Limit to 60 minutes runtime
  %(prog)s --dry-run                Show what would run
  %(prog)s --resume                 Resume from checkpoint
  %(prog)s --reset-checkpoint       Clear checkpoint and start fresh
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        nargs='+',
        help="Model(s) to run (e.g., qwen3:8b devstral:24b)"
    )
    
    parser.add_argument(
        '--app', '-a',
        type=str,
        nargs='+',
        help="Application(s) to run (e.g., petclinic realworld)"
    )
    
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        default=True,
        help="Resume from checkpoint (skip completed runs)"
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help="Don't resume - run all experiments including completed"
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        '--skip-docker',
        action='store_true',
        help="Skip Docker container management (assume running)"
    )
    
    parser.add_argument(
        '--reset-checkpoint',
        action='store_true',
        help="Clear checkpoint data before running"
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help="List available models and exit"
    )
    
    parser.add_argument(
        '--list-apps',
        action='store_true',
        help="List available applications and exit"
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help="Show checkpoint status and exit"
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        help="Path to configuration directory"
    )
    
    parser.add_argument(
        '--runs', '-R',
        type=int,
        default=1,
        help="Number of repetitions (default: 1)"
    )

    parser.add_argument(
        '--max-states', '-S',
        type=int,
        help="Override maximum number of states to explore (default: 1000)"
    )

    parser.add_argument(
        '--timeout', '-T',
        type=int,
        help="Override maximum runtime in minutes (default: 720)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show real-time output from main.py (can be verbose)"
    )


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\n" + "=" * 50)
    print("INTERRUPT RECEIVED")
    print("Saving progress and cleaning up...")
    print("=" * 50)
    sys.exit(1)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    # Initialize components
    config_dir = args.config_dir
    if config_dir:
        config_dir = Path(config_dir)
    
    config = ConfigLoader(config_dir)
    checkpoint = CheckpointManager()
    results = ResultCollector()
    
    # Handle info commands
    if args.list_models:
        print("\nAvailable Models:")
        print("-" * 50)
        for m in config.get_models():
            print(f"  {m.name}")
            print(f"    Context: {m.context_length}, {m.description}")
        return 0
    
    if args.list_apps:
        print("\nAvailable Applications:")
        print("-" * 50)
        for a in config.get_applications():
            print(f"  {a.name}")
            print(f"    Features: {a.feature_count}, {a.description}")
        return 0
    
    if args.status:
        checkpoint.print_status()
        return 0
    
    # Reset checkpoint if requested
    if args.reset_checkpoint:
        print("Resetting checkpoint...")
        checkpoint.reset()
    
    # Print header
    print("\n" + "=" * 70)
    print("AutoE2E REPLICATION EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Validate configuration
    issues = config.validate_config()
    if issues['errors']:
        print("\nConfiguration Errors:")
        for e in issues['errors']:
            print(f"  ✗ {e}")
        return 1
    
    if issues['warnings']:
        print("\nConfiguration Warnings:")
        for w in issues['warnings']:
            print(f"  ⚠ {w}")
    
    # Print configuration summary
    config.print_summary()
    
    # Set repetitions (default 1)
    config.get_experiment_params().execution.repetitions = args.runs
    print(f"Repetitions: {args.runs}")
    
    # Initialize runner
    runner = ExperimentRunner(
        config_loader=config,
        checkpoint_manager=checkpoint,
        result_collector=results
    )
    
    # Override max states if specified (must be after runner creation to update runner.params)
    if args.max_states:
        runner.params.exploration.max_states = args.max_states
        print(f"Overriding max_states to: {args.max_states}")
    
    # Override timeout if specified (must be after runner creation to update runner.params)
    if args.timeout:
        runner.params.exploration.timeout_minutes = args.timeout
        print(f"Overriding timeout to: {args.timeout} minutes")
    
    # Determine which models/apps to run
    models = args.model if args.model else None
    apps = args.app if args.app else None
    
    # Get the actual model configs that will be used
    model_configs = config.get_models()
    if models:
        model_configs = [m for m in model_configs if m.name in models]
    
    # Check Ollama status - only for models that will be used
    print("\nChecking Ollama status...")
    models_to_check = [m.ollama_model for m in model_configs]
    ready, missing = runner.check_ollama_status(models_to_check)
    
    if not ready:
        print("\n" + "=" * 50)
        print("ERROR: Ollama is not ready")
        print("=" * 50)
        print("\nIssues found:")
        for issue in missing:
            print(f"  - {issue}")
        
        print("\nTo fix:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull required models:")
        
        for m in model_configs:
            print(f"   ollama pull {m.ollama_model}")
        print(f"   ollama pull {config.get_embedding_model()}")
        
        return 1
    
    print("✓ Ollama is ready")
    
    resume = args.resume and not args.no_resume
    
    # Run experiments
    print("\n" + "=" * 70)
    print("STARTING EXPERIMENTS")
    print("=" * 70)
    
    try:
        results_dict = runner.run_all_experiments(
            models=models,
            apps=apps,
            resume=resume,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        print("=" * 70)
        
        # Count results
        total_success = sum(
            sum(1 for r in runs if r.success)
            for runs in results_dict.values()
        )
        total_failed = sum(
            sum(1 for r in runs if not r.success)
            for runs in results_dict.values()
        )
        
        print(f"\nResults:")
        print(f"  Successful: {total_success}")
        print(f"  Failed: {total_failed}")
        
        # Show checkpoint status
        checkpoint.print_status()
        
        return 0 if total_failed == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        checkpoint.print_status()
        return 130
    
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
