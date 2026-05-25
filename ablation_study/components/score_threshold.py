"""
Score Threshold Component
=========================

Configurable score threshold filtering for ablation study.
Filters out features below an accumulated score threshold.

Ablations:
- A7.1: Score threshold >= 0 (non-negative scores)
- A7.2: Score threshold >= 1.0 (moderate confidence)
- A7.3: Score threshold >= 2.0 (high confidence)

Note: The geometric scoring function produces negative values that accumulate
via differential scoring. Higher accumulated scores indicate stronger evidence.

Example scores with p=0.5:
- Rank 1: -0.693
- Rank 2: -1.386
- Rank 3: -2.079

After differential accumulation across observations, positive scores indicate
the feature appeared at better ranks in the current action than previous.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    """Configuration for score threshold filtering."""
    enabled: bool = False
    min_score: Optional[float] = None


class ScoreThreshold:
    """
    Filters features based on accumulated score threshold.
    
    AutoE2E accumulates scores through differential scoring:
    score += (current_rank_score - previous_rank_score)
    
    This component filters final features by their accumulated score,
    keeping only those above a minimum threshold.
    
    Usage:
        threshold = ScoreThreshold(enabled=True, min_score=1.0)
        
        # Check if feature should be kept
        if threshold.should_keep(feature_score):
            # Include in final results
            ...
    """
    
    def __init__(
        self,
        enabled: bool = False,
        min_score: Optional[float] = None
    ):
        """
        Initialize score threshold filter.
        
        Args:
            enabled: Whether filtering is enabled
            min_score: Minimum accumulated score to keep feature (None = no threshold)
        """
        self.enabled = enabled
        self.min_score = min_score
        self.config = ThresholdConfig(enabled=enabled, min_score=min_score)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ScoreThreshold':
        """Create filter from configuration dictionary."""
        return cls(
            enabled=config.get('enabled', False),
            min_score=config.get('min_score', None)
        )
    
    def get_description(self) -> str:
        """Get human-readable description of current configuration."""
        if not self.enabled or self.min_score is None:
            return "disabled (keep all)"
        return f"score >= {self.min_score}"
    
    def is_enabled(self) -> bool:
        """Check if filtering is enabled."""
        return self.enabled and self.min_score is not None
    
    def get_min_score(self) -> Optional[float]:
        """Get the minimum score threshold."""
        return self.min_score
    
    def should_keep(self, score: float) -> bool:
        """
        Check if a feature should be kept based on accumulated score.
        
        Args:
            score: Feature's accumulated score
            
        Returns:
            True if feature should be kept
        """
        if not self.enabled or self.min_score is None:
            return True
        return score >= self.min_score
    
    def filter_features(
        self,
        features: List[Dict[str, Any]],
        score_key: str = 'score'
    ) -> List[Dict[str, Any]]:
        """
        Filter features based on accumulated score.
        
        Args:
            features: List of feature dicts with score
            score_key: Key for the score field in each dict
            
        Returns:
            Filtered list of features
        """
        if not self.enabled or self.min_score is None:
            return features
        
        return [
            f for f in features
            if f.get(score_key, float('-inf')) >= self.min_score
        ]
    
    def filter_by_scores(
        self,
        features: List[str],
        scores: List[float]
    ) -> tuple[List[str], List[float]]:
        """
        Filter parallel lists of features and scores.
        
        Args:
            features: List of feature strings
            scores: List of accumulated scores
            
        Returns:
            Tuple of (filtered_features, filtered_scores)
        """
        if not self.enabled or self.min_score is None:
            return features, scores
        
        filtered_features = []
        filtered_scores = []
        
        for feature, score in zip(features, scores):
            if score >= self.min_score:
                filtered_features.append(feature)
                filtered_scores.append(score)
        
        return filtered_features, filtered_scores
    
    def get_filter_stats(
        self,
        original_count: int,
        filtered_count: int
    ) -> Dict[str, Any]:
        """
        Get statistics about filtering.
        
        Args:
            original_count: Number of features before filtering
            filtered_count: Number of features after filtering
            
        Returns:
            Dictionary with filter statistics
        """
        removed = original_count - filtered_count
        removal_rate = removed / original_count if original_count > 0 else 0.0
        
        return {
            'enabled': self.enabled,
            'min_score': self.min_score,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed,
            'removal_rate': removal_rate
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'min_score': self.min_score
        }
    
    def __repr__(self) -> str:
        return f"ScoreThreshold(enabled={self.enabled}, min_score={self.min_score})"
