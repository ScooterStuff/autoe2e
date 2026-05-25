"""
Ablation Integration Module
===========================

This module provides a global registry for ablation study components.
When running in ablation mode, main.py will load components from an
ablation_config.json and register them here.

When components are registered, the inference functions will use them
instead of the default (baseline) behavior.

This design ensures:
1. Original behavior is preserved when no ablation config is provided
2. Ablation components can modify behavior without changing function signatures
3. Multiple modules can access the same component configuration

Usage in main.py:
    from autoe2e.ablation_integration import init_ablation_components, is_ablation_mode
    
    if args.ablation_config:
        init_ablation_components(args.ablation_config)
    
Usage in infer_utils.py:
    from autoe2e.ablation_integration import get_component, is_ablation_mode
    
    if is_ablation_mode():
        context_extractor = get_component('context_extractor')
        if context_extractor and not context_extractor.should_include_screenshot():
            # Skip screenshot
            ...
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# Global State
# =============================================================================

# Flag indicating if ablation mode is active (use is_ablation_mode() for reliable access)
ABLATION_MODE = False

# Ablation configuration (loaded from JSON)
_ablation_config: Optional[Dict[str, Any]] = None

# Component instances (created from config)
_components: Dict[str, Any] = {}

# Ablation metadata
_ablation_id: Optional[str] = None
_ablation_description: Optional[str] = None


def is_ablation_mode() -> bool:
    """
    Check if ablation mode is currently active.
    
    Use this function instead of directly accessing ABLATION_MODE
    to ensure you get the current value (not a stale import).
    
    Returns:
        True if ablation mode is active, False otherwise
    """
    return ABLATION_MODE


def get_ablation_id() -> Optional[str]:
    """Get the current ablation ID."""
    return _ablation_id


def get_ablation_description() -> Optional[str]:
    """Get the current ablation description."""
    return _ablation_description


# =============================================================================
# Component Factory Import (deferred to avoid circular imports)
# =============================================================================

def _get_component_factory():
    """Lazily import ComponentFactory to avoid circular imports."""
    try:
        # Add ablation_study to path if needed
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from ablation_study.orchestrator.component_factory import ComponentFactory
        return ComponentFactory
    except ImportError as e:
        print(f"Warning: Could not import ComponentFactory: {e}")
        return None


# =============================================================================
# Initialization
# =============================================================================

def init_ablation_components(config_path: str) -> bool:
    """
    Initialize ablation components from a configuration file.
    
    Args:
        config_path: Path to ablation_config.json file
        
    Returns:
        True if initialization successful, False otherwise
    """
    global ABLATION_MODE, _ablation_config, _components, _ablation_id, _ablation_description
    
    try:
        with open(config_path, 'r') as f:
            _ablation_config = json.load(f)
        
        # Store metadata
        _ablation_id = _ablation_config.get('ablation_id', 'unknown')
        _ablation_description = _ablation_config.get('ablation_description', '')
        
        # Get component configurations
        components_config = _ablation_config.get('components', {})
        
        # Create component instances using factory
        ComponentFactory = _get_component_factory()
        if ComponentFactory is None:
            print("Warning: ComponentFactory not available, ablation mode disabled")
            return False
        
        _components = ComponentFactory.create_from_dict(components_config)
        
        # Mark ablation mode as active
        ABLATION_MODE = True
        
        print("=" * 70)
        print(f"ABLATION MODE ENABLED: {_ablation_id}")
        print(f"Description: {_ablation_description}")
        print("-" * 70)
        print("Component Configuration:")
        print(f"  Context:     {_components['context_extractor'].get_description()}")
        print(f"  Prompting:   {_components['prompt_manager'].get_description()}")
        print(f"  Scoring:     {_components['scoring_function'].get_description()}")
        print(f"  Accumulation:{_components['score_accumulator'].get_description()}")
        print(f"  Threshold:   {_components['score_threshold'].get_description()}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"Error initializing ablation components: {e}")
        ABLATION_MODE = False
        return False


def reset_ablation_components():
    """Reset ablation state (useful for testing)."""
    global ABLATION_MODE, _ablation_config, _components, _ablation_id, _ablation_description
    
    ABLATION_MODE = False
    _ablation_config = None
    _components = {}
    _ablation_id = None
    _ablation_description = None


# =============================================================================
# Component Accessors
# =============================================================================

def get_component(name: str) -> Optional[Any]:
    """
    Get a component by name.
    
    Args:
        name: Component name (context_extractor, prompt_manager, scoring_function,
              score_accumulator, score_threshold)
              
    Returns:
        Component instance or None if not in ablation mode
    """
    if not ABLATION_MODE:
        return None
    return _components.get(name)


def get_context_extractor():
    """Get the context extractor component."""
    return get_component('context_extractor')


def get_prompt_manager():
    """Get the prompt manager component."""
    return get_component('prompt_manager')


def get_scoring_function():
    """Get the scoring function component."""
    return get_component('scoring_function')


def get_score_accumulator():
    """Get the score accumulator component."""
    return get_component('score_accumulator')


def get_score_threshold():
    """Get the score threshold component."""
    return get_component('score_threshold')


def get_ablation_id() -> Optional[str]:
    """Get the current ablation ID."""
    return _ablation_id if ABLATION_MODE else None


def get_ablation_config() -> Optional[Dict[str, Any]]:
    """Get the full ablation configuration."""
    return _ablation_config if ABLATION_MODE else None


# =============================================================================
# Helper Functions for Common Operations
# =============================================================================

def should_include_screenshot() -> bool:
    """Check if screenshot should be included (respects ablation config)."""
    ctx = get_context_extractor()
    if ctx:
        return ctx.should_include_screenshot()
    return True  # Default: include screenshot


def should_include_previous_state() -> bool:
    """Check if previous state context should be included."""
    ctx = get_context_extractor()
    if ctx:
        return ctx.include_previous_state
    return True  # Default: include


def should_include_previous_action() -> bool:
    """Check if previous action context should be included."""
    ctx = get_context_extractor()
    if ctx:
        return ctx.include_previous_action
    return True  # Default: include


def get_prompt_types(has_previous_action: bool = True) -> list:
    """Get which prompt types to execute based on strategy."""
    pm = get_prompt_manager()
    if pm:
        return pm.get_prompt_types(has_previous_action)
    # Default: dual strategy (single + double)
    return ["single", "double"] if has_previous_action else ["single"]


def compute_score(rank: int) -> float:
    """
    Compute score for a rank using configured scoring function.
    
    Args:
        rank: Feature rank (1-indexed, 1 is best). None for penalty.
        
    Returns:
        Score value
    """
    sf = get_scoring_function()
    if sf:
        return sf.score(rank)
    # Default: geometric scoring
    import numpy as np
    p, R = 0.5, 10
    if rank is not None and rank >= 1:
        return (rank - 1) * np.log(1 - p) + np.log(p)
    return R * np.log(1 - p) + np.log(p)


def should_keep_feature(score: float) -> bool:
    """Check if a feature should be kept based on score threshold."""
    st = get_score_threshold()
    if st:
        return st.should_keep(score)
    return True  # Default: keep all
