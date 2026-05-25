"""
Ablation Study Components Module
================================

Configurable components for ablation experiments.
Each component can be configured via YAML to test different variants.
"""

from .context_extractor import ContextExtractor
from .prompt_manager import PromptManager
from .scoring_function import ScoringFunction
from .score_accumulator import ScoreAccumulator
from .score_threshold import ScoreThreshold

__all__ = [
    'ContextExtractor',
    'PromptManager',
    'ScoringFunction',
    'ScoreAccumulator',
    'ScoreThreshold'
]
