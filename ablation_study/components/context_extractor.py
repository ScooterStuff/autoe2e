"""
Context Extractor Component
===========================

Configurable context extraction for ablation study.
Controls what context is passed to the LLM for state understanding.

Ablations:
- A1.1: No screenshot (include_screenshot=false)
- A1.2: No previous state (include_previous_state=false)  
- A1.3: No previous action (include_previous_action=false)
- A1.4: Minimal context (all false)
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ContextConfig:
    """Configuration for context extraction."""
    include_screenshot: bool = True
    include_previous_state: bool = True
    include_previous_action: bool = True


class ContextExtractor:
    """
    Configurable context extractor for state understanding.
    
    This component controls what information is passed to the LLM
    when extracting state context (webpage understanding).
    
    Usage:
        extractor = ContextExtractor(
            include_screenshot=True,
            include_previous_state=True,
            include_previous_action=True
        )
        
        context = extractor.extract_context(
            driver=driver,
            state=current_state,
            prev_state=previous_state,
            prev_action=previous_action
        )
    """
    
    def __init__(
        self,
        include_screenshot: bool = True,
        include_previous_state: bool = True,
        include_previous_action: bool = True
    ):
        """
        Initialize context extractor.
        
        Args:
            include_screenshot: Whether to include screenshot in context
            include_previous_state: Whether to include previous state context
            include_previous_action: Whether to include previous action HTML
        """
        self.include_screenshot = include_screenshot
        self.include_previous_state = include_previous_state
        self.include_previous_action = include_previous_action
        
        self.config = ContextConfig(
            include_screenshot=include_screenshot,
            include_previous_state=include_previous_state,
            include_previous_action=include_previous_action
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ContextExtractor':
        """Create extractor from configuration dictionary."""
        return cls(
            include_screenshot=config.get('include_screenshot', True),
            include_previous_state=config.get('include_previous_state', True),
            include_previous_action=config.get('include_previous_action', True)
        )
    
    def get_description(self) -> str:
        """Get human-readable description of current configuration."""
        parts = []
        if not self.include_screenshot:
            parts.append("no screenshot")
        if not self.include_previous_state:
            parts.append("no prev state")
        if not self.include_previous_action:
            parts.append("no prev action")
        
        if not parts:
            return "full context"
        return ", ".join(parts)
    
    def prepare_context_inputs(
        self,
        description: str = "None",
        previous_state_context: Optional[str] = None,
        previous_action_html: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Prepare context inputs for the LLM prompt.
        
        Args:
            description: Application description
            previous_state_context: Context string from previous state
            previous_action_html: outerHTML of previous action element
            
        Returns:
            Dictionary with prepared context inputs
        """
        context_inputs = {
            "description": description
        }
        
        # Previous state context
        if self.include_previous_state and previous_state_context:
            context_inputs["previous_state"] = previous_state_context
        else:
            context_inputs["previous_state"] = "None. This is the first state."
        
        # Previous action context
        if self.include_previous_action and previous_action_html:
            context_inputs["previous_action"] = previous_action_html
        else:
            context_inputs["previous_action"] = "None. This is the first state."
        
        return context_inputs
    
    def should_include_screenshot(self) -> bool:
        """Check if screenshot should be included."""
        return self.include_screenshot
    
    def get_placeholder_image(self) -> str:
        """
        Get a placeholder image for when screenshots are disabled.
        Returns a minimal valid base64 PNG (1x1 transparent pixel).
        """
        # 1x1 transparent PNG in base64
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'include_screenshot': self.include_screenshot,
            'include_previous_state': self.include_previous_state,
            'include_previous_action': self.include_previous_action
        }
    
    def __repr__(self) -> str:
        return (
            f"ContextExtractor("
            f"screenshot={self.include_screenshot}, "
            f"prev_state={self.include_previous_state}, "
            f"prev_action={self.include_previous_action})"
        )
