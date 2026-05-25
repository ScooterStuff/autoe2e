"""
Ablation Study Module for AutoE2E
=================================

This module provides a systematic ablation study framework to measure
the contribution of each architectural component in AutoE2E.

Components being ablated:
- C1: Context Extraction (screenshot, previous state, previous action)
- C2: Dual-Prompt Feature Inference (single vs pair vs merged)
- C3: Probabilistic Scoring Function (geometric, uniform, linear, binary, exponential)
- C4: Evidence Accumulation (differential vs sum vs max vs final vs weighted)
- C5: Feature Probability Filtering (thresholds: 0, 0.3, 0.5, 0.7)

Usage:
    from ablation_study import AblationRunner, AblationConfigLoader
    
    # Load configuration
    config = AblationConfigLoader()
    config.print_summary()
    
    # Run experiments
    runner = AblationRunner()
    
    # Show plan
    runner.show_plan()
    
    # Run all ablations
    runner.run_all()
    
    # Run specific ablation
    runner.run_ablation("A1.1", applications=["petclinic"])
    
    # Run single experiment
    runner.run_single("A1.1", "petclinic", run_id=1)

CLI Usage:
    # List available ablations
    python -m ablation_study.scripts.run_ablation_study --list-ablations
    
    # Run all ablations
    python -m ablation_study.scripts.run_ablation_study
    
    # Run specific ablation
    python -m ablation_study.scripts.run_single_ablation A1.1 petclinic
    
    # Analyze results
    python -m ablation_study.scripts.analyze_ablations
"""

# Components
from .components import (
    ContextExtractor,
    PromptManager,
    ScoringFunction,
    ScoreAccumulator,
    ScoreThreshold
)

# Orchestrator
from .orchestrator import (
    AblationConfigLoader,
    AblationRunner,
    ComponentFactory,
    AblationMetricsCollector,
    AblationCheckpointManager
)

__version__ = "1.0.0"

__all__ = [
    # Components
    'ContextExtractor',
    'PromptManager',
    'ScoringFunction',
    'ScoreAccumulator',
    'ScoreThreshold',
    # Orchestrator
    'AblationConfigLoader',
    'AblationRunner', 
    'ComponentFactory',
    'AblationMetricsCollector',
    'AblationCheckpointManager'
]
