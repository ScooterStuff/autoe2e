"""
Component Factory
=================

Creates configured component instances from ablation configuration.
Acts as the central factory for all configurable components.
"""

from typing import Dict, Any

from ..components import (
    ContextExtractor,
    PromptManager,
    ScoringFunction,
    ScoreAccumulator,
    ScoreThreshold
)
from .config_loader import AblationConfig


class ComponentFactory:
    """
    Creates configured component instances from ablation configuration.
    
    This factory ensures all components are created consistently
    and provides a single point of configuration injection.
    
    Usage:
        factory = ComponentFactory()
        
        # Create all components from ablation config
        components = factory.create_all(ablation_config)
        
        # Or create individual components
        scorer = factory.create_scoring_function(config.scoring)
    """
    
    @staticmethod
    def create_context_extractor(config: Dict[str, Any]) -> ContextExtractor:
        """
        Create a context extractor from configuration.
        
        Args:
            config: Context configuration dictionary
            
        Returns:
            Configured ContextExtractor
        """
        return ContextExtractor(
            include_screenshot=config.get('include_screenshot', True),
            include_previous_state=config.get('include_previous_state', True),
            include_previous_action=config.get('include_previous_action', True)
        )
    
    @staticmethod
    def create_prompt_manager(config: Dict[str, Any]) -> PromptManager:
        """
        Create a prompt manager from configuration.
        
        Args:
            config: Prompting configuration dictionary
            
        Returns:
            Configured PromptManager
        """
        return PromptManager(
            strategy=config.get('strategy', 'dual')
        )
    
    @staticmethod
    def create_scoring_function(config: Dict[str, Any]) -> ScoringFunction:
        """
        Create a scoring function from configuration.
        
        Args:
            config: Scoring configuration dictionary
            
        Returns:
            Configured ScoringFunction
        """
        return ScoringFunction(
            method=config.get('function', 'geometric'),
            p=config.get('p', 0.5),
            R=config.get('R', 10)
        )
    
    @staticmethod
    def create_score_accumulator(config: Dict[str, Any]) -> ScoreAccumulator:
        """
        Create a score accumulator from configuration.
        
        Args:
            config: Accumulation configuration dictionary
            
        Returns:
            Configured ScoreAccumulator
        """
        return ScoreAccumulator(
            method=config.get('method', 'differential')
        )
    
    @staticmethod
    def create_score_threshold(config: Dict[str, Any]) -> ScoreThreshold:
        """
        Create a score threshold filter from configuration.
        
        Args:
            config: Score threshold configuration dictionary
            
        Returns:
            Configured ScoreThreshold
        """
        return ScoreThreshold(
            enabled=config.get('enabled', False),
            min_score=config.get('min_score', None)
        )
    
    @classmethod
    def create_all(cls, ablation_config: AblationConfig) -> Dict[str, Any]:
        """
        Create all components from an ablation configuration.
        
        Args:
            ablation_config: Complete AblationConfig object
            
        Returns:
            Dictionary with all component instances
        """
        return {
            'context_extractor': cls.create_context_extractor(ablation_config.context),
            'prompt_manager': cls.create_prompt_manager(ablation_config.prompting),
            'scoring_function': cls.create_scoring_function(ablation_config.scoring),
            'score_accumulator': cls.create_score_accumulator(ablation_config.accumulation),
            'score_threshold': cls.create_score_threshold(ablation_config.score_threshold)
        }
    
    @classmethod
    def create_from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create all components from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with component configs
            
        Returns:
            Dictionary with all component instances
        """
        return {
            'context_extractor': cls.create_context_extractor(config.get('context', {})),
            'prompt_manager': cls.create_prompt_manager(config.get('prompting', {})),
            'scoring_function': cls.create_scoring_function(config.get('scoring', {})),
            'score_accumulator': cls.create_score_accumulator(config.get('accumulation', {})),
            'score_threshold': cls.create_score_threshold(config.get('score_threshold', {}))
        }
    
    @staticmethod
    def get_component_summary(components: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of component configurations.
        
        Args:
            components: Dictionary of component instances
            
        Returns:
            Summary string
        """
        lines = ["Component Configuration:"]
        
        if 'context_extractor' in components:
            lines.append(f"  Context: {components['context_extractor'].get_description()}")
        if 'prompt_manager' in components:
            lines.append(f"  Prompting: {components['prompt_manager'].get_description()}")
        if 'scoring_function' in components:
            lines.append(f"  Scoring: {components['scoring_function'].get_description()}")
        if 'score_accumulator' in components:
            lines.append(f"  Accumulation: {components['score_accumulator'].get_description()}")
        if 'score_threshold' in components:
            lines.append(f"  Score Threshold: {components['score_threshold'].get_description()}")
        
        return "\n".join(lines)
