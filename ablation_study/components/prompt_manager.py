"""
Prompt Manager Component
========================

Configurable prompting strategy for feature inference ablation study.
Controls whether to use single-action, action-pair, or merged prompts.

Ablations:
- A2.1: Single-action only (strategy="single_action")
- A2.2: Action-pair only (strategy="action_pair")  
- A2.3: Single merged query (strategy="merged")
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptStrategy(Enum):
    """Available prompting strategies."""
    DUAL = "dual"              # Both single-action and action-pair (baseline)
    SINGLE_ACTION = "single_action"  # Only p(F|Ai)
    ACTION_PAIR = "action_pair"      # Only p(F|Ai, Ai-1)
    MERGED = "merged"          # Single prompt with both contexts


@dataclass
class PromptConfig:
    """Configuration for prompt strategy."""
    strategy: PromptStrategy = PromptStrategy.DUAL


class PromptManager:
    """
    Manages prompting strategy for feature inference.
    
    AutoE2E normally makes two LLM calls:
    1. Single-action: p(F|Ai) - infer features from current action alone
    2. Action-pair: p(F|Ai, Ai-1) - infer features from action sequence
    
    This component allows testing different strategies:
    - dual: Both queries (baseline)
    - single_action: Only single-action query
    - action_pair: Only action-pair query  
    - merged: Single prompt combining both contexts
    
    Usage:
        manager = PromptManager(strategy="dual")
        
        # Get which prompts to execute
        prompts = manager.get_prompt_types(has_previous_action=True)
        # Returns: ["single", "double"] for dual strategy
    """
    
    STRATEGIES = ["dual", "single_action", "action_pair", "merged"]
    
    def __init__(self, strategy: str = "dual"):
        """
        Initialize prompt manager.
        
        Args:
            strategy: Prompting strategy ("dual", "single_action", "action_pair", "merged")
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {self.STRATEGIES}")
        
        self.strategy = PromptStrategy(strategy)
        self.config = PromptConfig(strategy=self.strategy)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PromptManager':
        """Create manager from configuration dictionary."""
        return cls(strategy=config.get('strategy', 'dual'))
    
    def get_description(self) -> str:
        """Get human-readable description of current strategy."""
        descriptions = {
            PromptStrategy.DUAL: "dual prompts (single + pair)",
            PromptStrategy.SINGLE_ACTION: "single-action only",
            PromptStrategy.ACTION_PAIR: "action-pair only",
            PromptStrategy.MERGED: "merged single prompt"
        }
        return descriptions.get(self.strategy, "unknown")
    
    def get_prompt_types(self, has_previous_action: bool = True) -> List[str]:
        """
        Get list of prompt types to execute based on strategy.
        
        Args:
            has_previous_action: Whether there is a previous action available
            
        Returns:
            List of prompt types: "single" for p(F|Ai), "double" for p(F|Ai, Ai-1)
        """
        if self.strategy == PromptStrategy.DUAL:
            prompts = ["single"]
            if has_previous_action:
                prompts.append("double")
            return prompts
        
        elif self.strategy == PromptStrategy.SINGLE_ACTION:
            return ["single"]
        
        elif self.strategy == PromptStrategy.ACTION_PAIR:
            if has_previous_action:
                return ["double"]
            # Fall back to single if no previous action
            return ["single"]
        
        elif self.strategy == PromptStrategy.MERGED:
            return ["merged"]
        
        return ["single"]  # Default fallback
    
    def should_extract_single(self) -> bool:
        """Check if single-action extraction should be performed."""
        return self.strategy in [PromptStrategy.DUAL, PromptStrategy.SINGLE_ACTION]
    
    def should_extract_double(self, has_previous_action: bool = True) -> bool:
        """Check if action-pair extraction should be performed."""
        if not has_previous_action:
            return False
        return self.strategy in [PromptStrategy.DUAL, PromptStrategy.ACTION_PAIR]
    
    def should_extract_merged(self) -> bool:
        """Check if merged extraction should be performed."""
        return self.strategy == PromptStrategy.MERGED
    
    def get_action_type_for_db(self, prompt_type: str) -> str:
        """
        Get action type string for database storage.
        
        Args:
            prompt_type: "single", "double", or "merged"
            
        Returns:
            Action type string for MongoDB
        """
        type_map = {
            "single": "SINGLE",
            "double": "DOUBLE",
            "merged": "MERGED"
        }
        return type_map.get(prompt_type, "SINGLE")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'strategy': self.strategy.value
        }
    
    def __repr__(self) -> str:
        return f"PromptManager(strategy={self.strategy.value})"
