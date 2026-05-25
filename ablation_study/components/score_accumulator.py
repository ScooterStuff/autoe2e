"""
Score Accumulator Component
===========================

Configurable score accumulation across action chains for ablation study.
Controls how evidence is combined across multiple actions.

Ablations:
- A5.1: Final only (method="final_only")
- A5.2: Simple sum (method="simple_sum")
- A5.3: Maximum (method="maximum")
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class AccumulationMethod(Enum):
    """Available accumulation methods."""
    DIFFERENTIAL = "differential"  # Δscore = score(pair) - score(single) (baseline)
    SIMPLE_SUM = "simple_sum"      # Sum of pair scores
    MAXIMUM = "maximum"            # Max score across chain
    FINAL_ONLY = "final_only"      # Only last action's score


@dataclass
class AccumulationConfig:
    """Configuration for score accumulation."""
    method: AccumulationMethod = AccumulationMethod.DIFFERENTIAL


class ScoreAccumulator:
    """
    Handles score accumulation across action chains.
    
    AutoE2E uses differential accumulation by default:
        Δscore(F) = score(F|Ai, Ai-1) - score(F|Ai-1)
    
    This captures how much the action-pair query adds beyond the single-action
    query. This component allows testing alternative accumulation strategies:
    
    - differential: Sum of (pair_score - single_score) (baseline)
    - simple_sum: Sum of pair scores without differential
    - maximum: Maximum score observed across chain
    - final_only: Only use the last action's score
    
    Usage:
        accumulator = ScoreAccumulator(method="differential")
        
        # Accumulate scores for a feature across actions
        final_score = accumulator.accumulate(feature_scores)
    """
    
    METHODS = ["differential", "simple_sum", "maximum", "final_only"]
    
    def __init__(self, method: str = "differential"):
        """
        Initialize score accumulator.
        
        Args:
            method: Accumulation method
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Must be one of {self.METHODS}")
        
        self.method = AccumulationMethod(method)
        self.config = AccumulationConfig(method=self.method)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ScoreAccumulator':
        """Create accumulator from configuration dictionary."""
        return cls(method=config.get('method', 'differential'))
    
    def get_description(self) -> str:
        """Get human-readable description of current method."""
        descriptions = {
            AccumulationMethod.DIFFERENTIAL: "differential (pair - single)",
            AccumulationMethod.SIMPLE_SUM: "simple sum of pair scores",
            AccumulationMethod.MAXIMUM: "maximum score",
            AccumulationMethod.FINAL_ONLY: "final action only"
        }
        return descriptions.get(self.method, "unknown")
    
    def accumulate(
        self,
        feature_scores: List[Tuple[float, float]]
    ) -> float:
        """
        Accumulate scores across action chain.
        
        Args:
            feature_scores: List of (score_pair, score_single) tuples for each action
                           score_pair = p(F|Ai, Ai-1) or score from action-pair query
                           score_single = p(F|Ai-1) or score from single-action query
                           
        Returns:
            Accumulated score
        """
        if not feature_scores:
            return 0.0
        
        if self.method == AccumulationMethod.DIFFERENTIAL:
            return self._differential_accumulate(feature_scores)
        elif self.method == AccumulationMethod.SIMPLE_SUM:
            return self._simple_sum_accumulate(feature_scores)
        elif self.method == AccumulationMethod.MAXIMUM:
            return self._maximum_accumulate(feature_scores)
        elif self.method == AccumulationMethod.FINAL_ONLY:
            return self._final_only_accumulate(feature_scores)
        
        return 0.0
    
    def _differential_accumulate(self, feature_scores: List[Tuple[float, float]]) -> float:
        """
        Differential accumulation: sum of (pair_score - single_score).
        
        This captures the additional evidence from action-pair queries
        beyond what single-action queries provide.
        """
        return sum(pair - single for pair, single in feature_scores)
    
    def _simple_sum_accumulate(self, feature_scores: List[Tuple[float, float]]) -> float:
        """
        Simple sum: sum all pair scores without differential.
        """
        return sum(pair for pair, single in feature_scores)
    
    def _maximum_accumulate(self, feature_scores: List[Tuple[float, float]]) -> float:
        """
        Maximum: take the highest pair score observed.
        """
        return max(pair for pair, single in feature_scores)
    
    def _final_only_accumulate(self, feature_scores: List[Tuple[float, float]]) -> float:
        """
        Final only: use only the last action's pair score.
        """
        return feature_scores[-1][0]
    
    def compute_update(
        self,
        current_pair_score: float,
        previous_single_score: float
    ) -> float:
        """
        Compute the score update for a single action step.
        
        This is called during the main loop to update feature scores.
        
        Args:
            current_pair_score: Score from p(F|Ai, Ai-1)
            previous_single_score: Score from p(F|Ai-1)
            
        Returns:
            Score delta to add
        """
        if self.method == AccumulationMethod.DIFFERENTIAL:
            return current_pair_score - previous_single_score
        elif self.method == AccumulationMethod.SIMPLE_SUM:
            return current_pair_score
        elif self.method == AccumulationMethod.MAXIMUM:
            # For maximum, we'll handle this differently in the main loop
            # Just return the current score
            return current_pair_score
        elif self.method == AccumulationMethod.FINAL_ONLY:
            # For final_only, don't accumulate - just use current
            return current_pair_score
        
        return 0.0
    
    def is_incremental(self) -> bool:
        """
        Check if this method supports incremental updates.
        
        Some methods (like maximum) need to track all scores, not just
        add incremental updates.
        """
        return self.method in [AccumulationMethod.DIFFERENTIAL, AccumulationMethod.SIMPLE_SUM]
    
    def needs_full_history(self) -> bool:
        """
        Check if this method needs full score history.
        
        Maximum and final_only need different handling than incremental methods.
        """
        return self.method in [AccumulationMethod.MAXIMUM, AccumulationMethod.FINAL_ONLY]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'method': self.method.value
        }
    
    def __repr__(self) -> str:
        return f"ScoreAccumulator(method={self.method.value})"
