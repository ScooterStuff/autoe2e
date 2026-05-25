"""
Scoring Function Component
==========================

Pluggable scoring functions for rank-to-probability conversion.
Tests different ways to convert LLM feature rankings to scores.

Ablations:
- A3.1: Uniform scoring (function="uniform")
- A3.2: Linear scoring (function="linear")
- A3.3: Binary scoring (function="binary")
- A3.4a-d: Geometric with different p values
- A3.5a-c: Different R (max candidates) values
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ScoringMethod(Enum):
    """Available scoring methods."""
    GEOMETRIC = "geometric"
    UNIFORM = "uniform"
    LINEAR = "linear"
    BINARY = "binary"


@dataclass
class ScoringConfig:
    """Configuration for scoring function."""
    function: ScoringMethod = ScoringMethod.GEOMETRIC
    p: float = 0.5
    R: int = 10


class ScoringFunction:
    """
    Pluggable scoring function for rank-to-probability conversion.
    
    AutoE2E uses geometric distribution by default:
        score(r) = (r-1) * log(1-p) + log(p)
    
    This component allows testing alternative scoring methods:
    - geometric: Original AutoE2E scoring
    - uniform: Equal weight for all top-R features
    - linear: Linear decay by rank
    - binary: 1 if in top-R, 0 otherwise
    
    Usage:
        scorer = ScoringFunction(method="geometric", p=0.5, R=10)
        score = scorer.score(rank=3)
    """
    
    METHODS = ["geometric", "uniform", "linear", "binary"]
    
    def __init__(
        self,
        method: str = "geometric",
        p: float = 0.5,
        R: int = 10
    ):
        """
        Initialize scoring function.
        
        Args:
            method: Scoring method ("geometric", "uniform", "linear", "binary")
            p: Parameter for geometric distribution (probability of first rank)
            R: Maximum number of candidate features to consider
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Must be one of {self.METHODS}")
        
        if not 0 < p < 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        
        if R < 1:
            raise ValueError(f"R must be at least 1, got {R}")
        
        self.method = ScoringMethod(method)
        self.p = p
        self.R = R
        
        self.config = ScoringConfig(
            function=self.method,
            p=p,
            R=R
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ScoringFunction':
        """Create scoring function from configuration dictionary."""
        return cls(
            method=config.get('function', 'geometric'),
            p=config.get('p', 0.5),
            R=config.get('R', 10)
        )
    
    def get_description(self) -> str:
        """Get human-readable description of current configuration."""
        if self.method == ScoringMethod.GEOMETRIC:
            return f"geometric(p={self.p}, R={self.R})"
        return f"{self.method.value}(R={self.R})"
    
    def score(self, rank: int) -> float:
        """
        Convert rank to score.
        
        Args:
            rank: Feature rank (1-indexed, 1 is best)
            
        Returns:
            Score value
        """
        if rank is None or rank < 1:
            return self._default_score()
        
        if self.method == ScoringMethod.GEOMETRIC:
            return self._geometric_score(rank)
        elif self.method == ScoringMethod.UNIFORM:
            return self._uniform_score(rank)
        elif self.method == ScoringMethod.LINEAR:
            return self._linear_score(rank)
        elif self.method == ScoringMethod.BINARY:
            return self._binary_score(rank)
        
        return self._default_score()
    
    def _geometric_score(self, rank: int) -> float:
        """
        Original AutoE2E geometric scoring.
        
        Formula: score(r) = (r-1) * log(1-p) + log(p)
        
        This corresponds to the log probability of rank r under a geometric
        distribution with parameter p.
        """
        if rank <= self.R:
            return (rank - 1) * math.log(1 - self.p) + math.log(self.p)
        # For ranks beyond R, use penalty score
        return self.R * math.log(1 - self.p) + math.log(self.p)
    
    def _uniform_score(self, rank: int) -> float:
        """
        Uniform scoring: equal weight for all in top-R.
        
        Formula: score(r) = 1/R if r <= R else 0
        """
        if rank <= self.R:
            return 1.0 / self.R
        return 0.0
    
    def _linear_score(self, rank: int) -> float:
        """
        Linear scoring: linear decay by rank.
        
        Formula: score(r) = max(0, R - r + 1)
        """
        return max(0.0, float(self.R - rank + 1))
    
    def _binary_score(self, rank: int) -> float:
        """
        Binary scoring: 1 if in top-R, 0 otherwise.
        
        Formula: score(r) = 1 if r <= R else 0
        """
        return 1.0 if rank <= self.R else 0.0
    
    def _default_score(self) -> float:
        """Get default/penalty score for invalid ranks."""
        if self.method == ScoringMethod.GEOMETRIC:
            # Penalty for features not in top-R
            return self.R * math.log(1 - self.p) + math.log(self.p) - math.log(self.p)
        return 0.0
    
    def score_features(self, features: list) -> list:
        """
        Score a list of features by their rank order.
        
        Args:
            features: List of features (assumed to be in rank order)
            
        Returns:
            List of (feature, score) tuples
        """
        scored = []
        for i, feature in enumerate(features):
            rank = i + 1  # 1-indexed
            scored.append((feature, self.score(rank)))
        return scored
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'function': self.method.value,
            'p': self.p,
            'R': self.R
        }
    
    def __repr__(self) -> str:
        return f"ScoringFunction(method={self.method.value}, p={self.p}, R={self.R})"
