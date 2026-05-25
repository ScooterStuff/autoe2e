#!/usr/bin/env python3
"""
Run Single Experiment Script
============================

Run a single model-application-run combination.
Useful for testing or re-running specific experiments.

Usage:
    python scripts/run_single_experiment.py --model qwen3:8b --app petclinic --run 1
    python scripts/run_single_experiment.py -m devstral:24b -a realworld -r 2 --skip-docker
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
        description="Run a single AutoE2E experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help="Model to use (e.g., qwen3:8b)"
    )
    
    parser.add_argument(
        '--app', '-a',
        type=str,
        required=True,
        help="Application to test (e.g., petclinic)"
    )
    
    parser.add_argument(
        '--run', '-r',
        type=int,
        default=1,
        help="Run ID (default: 1)"
    )
    
    parser.add_argument(
        '--skip-docker',
        action='store_true',
        help="Skip Docker container management"
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help="Run even if already completed"
    )
    
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help="Don't save to checkpoint"
    )
    
    parser.add_argument(
        '--max-states', '-S',
        type=int,
        help="Override maximum number of states to explore"
    )
    
    parser.add_argument(
        '--timeout', '-T',
        type=int,
        help="Override maximum runtime in minutes"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Show real-time output from main.py (can be verbose)"
    )

    return parser.parse_args()


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\n\nInterrupt received, cleaning up...")
    sys.exit(1)


def main():
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    # Initialize components
    config = ConfigLoader()
    checkpoint = CheckpointManager() if not args.no_checkpoint else None
    results = ResultCollector()
    
    # Get configurations
    model_config = config.get_model(args.model)
    if model_config is None:
        print(f"Error: Model '{args.model}' not found")
        print(f"Available models: {config.get_model_names()}")
        return 1
    
    app_config = config.get_application(args.app)
    if app_config is None:
        print(f"Error: Application '{args.app}' not found")
        print(f"Available applications: {config.get_application_names()}")
        return 1
    
    # Check if already completed
    if checkpoint and not args.force:
        if checkpoint.is_completed(model_config.name, app_config.name, args.run):
            print(f"Run already completed: {model_config.name}/{app_config.name}/run_{args.run}")
            print("Use --force to re-run")
            return 0
    
    # Print header
    print("\n" + "=" * 70)
    print("SINGLE EXPERIMENT RUN")
    print("=" * 70)
    print(f"Model: {model_config.name} ({model_config.provider})")
    print(f"Application: {app_config.name}")
    print(f"Run ID: {args.run}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70 + "\n")
    
    # Create runner with optional checkpoint
    runner = ExperimentRunner(
        config_loader=config,
        checkpoint_manager=checkpoint,
        result_collector=results
    )
    
    # Override max states if specified (must be after runner creation to update runner.params)
    if args.max_states:
        runner.params.exploration.max_states = args.max_states
        print(f"Max states: {args.max_states}")
    
    # Override timeout if specified (must be after runner creation to update runner.params)
    if args.timeout:
        runner.params.exploration.timeout_minutes = args.timeout
        print(f"Timeout: {args.timeout} minutes")
    
    # Check Ollama - only needed for ollama provider models
    if model_config.provider != 'openai':
        ready, missing = runner.check_ollama_status([model_config.ollama_model])
        if not ready:
            print(f"Error: Ollama not ready - {missing}")
            return 1
    
    # Run experiment
    result = runner.run_experiment(
        model=model_config,
        app=app_config,
        run_id=args.run,
        skip_docker=args.skip_docker,
        verbose=args.verbose
    )
    
    # Print result
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    
    if result.success:
        print(f"✓ Success in {result.duration_seconds:.1f}s")
        if result.metrics:
            print(f"\nMetrics:")
            print(f"  States explored: {result.metrics.states_explored}")
            print(f"  LLM queries: {result.metrics.llm_queries}")
            print(f"  Total time: {result.metrics.total_time_seconds:.1f}s")
        return 0
    else:
        print(f"✗ Failed: {result.error_message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
