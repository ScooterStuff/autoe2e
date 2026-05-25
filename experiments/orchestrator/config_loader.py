"""
Configuration Loader Module
===========================

Loads and manages experiment configurations from YAML files.
Provides a unified interface for accessing model, application,
and experiment parameter configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    ollama_model: str
    context_length: int
    description: str
    expected_performance: str = "baseline"
    seed: Optional[int] = None  # Optional seed for reproducibility
    provider: str = "ollama"  # "ollama" or "openai"


@dataclass
class ApplicationConfig:
    """Configuration for a benchmark application."""
    name: str
    config_name: str
    docker_compose: str
    url: str
    feature_count: int
    health_check_endpoint: str
    health_check_timeout_seconds: int
    description: str
    ground_truth_file: str


@dataclass
class ExplorationParams:
    """Exploration parameters."""
    max_depth: int = 7
    timeout_minutes: int = 720
    max_states: int = 1000
    search_strategy: str = "bfs"


@dataclass 
class InferenceParams:
    """Inference parameters."""
    temperature: float = 0
    max_candidate_features: int = 10
    geometric_p: float = 0.5
    seed: Optional[int] = None  # Optional global seed (can be overridden per model)


@dataclass
class ExecutionParams:
    """Execution parameters."""
    repetitions: int = 3
    retry_on_failure: int = 3
    query_timeout_seconds: int = 120
    health_check_timeout_seconds: int = 60
    inter_run_delay_seconds: int = 30


@dataclass
class BrowserParams:
    """Browser configuration parameters."""
    headless: bool = True
    window_width: int = 1920
    window_height: int = 1080
    page_load_timeout: int = 30


@dataclass
class ExperimentParams:
    """Complete experiment parameters."""
    exploration: ExplorationParams = field(default_factory=ExplorationParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    execution: ExecutionParams = field(default_factory=ExecutionParams)
    browser: BrowserParams = field(default_factory=BrowserParams)


class ConfigLoader:
    """
    Loads and manages experiment configurations.
    
    Usage:
        config = ConfigLoader()
        models = config.get_models()
        apps = config.get_applications()
        params = config.get_experiment_params()
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Path to configuration directory. 
                       Defaults to experiments/config relative to project root.
        """
        if config_dir is None:
            # Find project root (directory containing main.py)
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent
            config_dir = project_root / "experiments" / "config"
        
        self.config_dir = Path(config_dir)
        self._models_config: Optional[Dict] = None
        self._applications_config: Optional[Dict] = None
        self._params_config: Optional[Dict] = None
        self._paths_config: Optional[Dict] = None
        
    def _load_yaml(self, filename: str) -> Dict:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_models_config(self) -> Dict:
        """Load models configuration with caching."""
        if self._models_config is None:
            self._models_config = self._load_yaml("models.yaml")
        return self._models_config
    
    def _get_applications_config(self) -> Dict:
        """Load applications configuration with caching."""
        if self._applications_config is None:
            self._applications_config = self._load_yaml("applications.yaml")
        return self._applications_config
    
    def _get_params_config(self) -> Dict:
        """Load experiment parameters configuration with caching."""
        if self._params_config is None:
            self._params_config = self._load_yaml("experiment_params.yaml")
        return self._params_config
    
    def _get_paths_config(self) -> Dict:
        """Load paths configuration with caching."""
        if self._paths_config is None:
            self._paths_config = self._load_yaml("paths.yaml")
        return self._paths_config
    
    def get_models(self) -> List[ModelConfig]:
        """Get list of all configured models."""
        config = self._get_models_config()
        models = []
        for m in config.get('models', []):
            models.append(ModelConfig(
                name=m['name'],
                ollama_model=m['ollama_model'],
                context_length=m['context_length'],
                description=m['description'],
                expected_performance=m.get('expected_performance', 'baseline'),
                provider=m.get('provider', 'ollama')
            ))
        return models
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a specific model configuration by name."""
        for model in self.get_models():
            if model.name == name or model.ollama_model == name:
                return model
        return None
    
    def get_default_model(self) -> ModelConfig:
        """Get the default model configuration."""
        config = self._get_models_config()
        default_name = config.get('default_model', 'qwen3:8b')
        model = self.get_model(default_name)
        if model is None:
            models = self.get_models()
            if models:
                return models[0]
            raise ValueError("No models configured")
        return model
    
    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        config = self._get_models_config()
        return config.get('embedding_model', 'nomic-embed-text')
    
    def get_applications(self) -> List[ApplicationConfig]:
        """Get list of all configured applications."""
        config = self._get_applications_config()
        apps = []
        for a in config.get('applications', []):
            apps.append(ApplicationConfig(
                name=a['name'],
                config_name=a['config_name'],
                docker_compose=a['docker_compose'],
                url=a['url'],
                feature_count=a['feature_count'],
                health_check_endpoint=a['health_check_endpoint'],
                health_check_timeout_seconds=a['health_check_timeout_seconds'],
                description=a['description'],
                ground_truth_file=a['ground_truth_file']
            ))
        return apps
    
    def get_application(self, name: str) -> Optional[ApplicationConfig]:
        """Get a specific application configuration by name."""
        for app in self.get_applications():
            if app.name == name or app.config_name == name:
                return app
        return None
    
    def get_default_application(self) -> ApplicationConfig:
        """Get the default application configuration."""
        config = self._get_applications_config()
        default_name = config.get('default_application', 'petclinic')
        app = self.get_application(default_name)
        if app is None:
            apps = self.get_applications()
            if apps:
                return apps[0]
            raise ValueError("No applications configured")
        return app
    
    def get_experiment_params(self) -> ExperimentParams:
        """Get experiment parameters."""
        config = self._get_params_config()
        
        exploration = ExplorationParams(
            max_depth=config.get('exploration', {}).get('max_depth', 12),
            timeout_minutes=config.get('exploration', {}).get('timeout_minutes', 720),
            max_states=config.get('exploration', {}).get('max_states', 1000),
            search_strategy=config.get('exploration', {}).get('search_strategy', 'bfs')
        )
        
        inference = InferenceParams(
            temperature=config.get('inference', {}).get('temperature', 0),
            max_candidate_features=config.get('inference', {}).get('max_candidate_features', 10),
            geometric_p=config.get('inference', {}).get('geometric_p', 0.5)
        )
        
        execution = ExecutionParams(
            repetitions=config.get('execution', {}).get('repetitions', 3),
            retry_on_failure=config.get('execution', {}).get('retry_on_failure', 3),
            query_timeout_seconds=config.get('execution', {}).get('query_timeout_seconds', 120),
            health_check_timeout_seconds=config.get('execution', {}).get('health_check_timeout_seconds', 60),
            inter_run_delay_seconds=config.get('execution', {}).get('inter_run_delay_seconds', 30)
        )
        
        browser = BrowserParams(
            headless=config.get('browser', {}).get('headless', True),
            window_width=config.get('browser', {}).get('window_width', 1920),
            window_height=config.get('browser', {}).get('window_height', 1080),
            page_load_timeout=config.get('browser', {}).get('page_load_timeout', 30)
        )
        
        return ExperimentParams(
            exploration=exploration,
            inference=inference,
            execution=execution,
            browser=browser
        )
    
    def get_paths(self) -> Dict[str, Any]:
        """Get all path configurations."""
        return self._get_paths_config()
    
    def get_path(self, *keys: str, default: str = "") -> str:
        """
        Get a specific path from configuration.
        
        Args:
            *keys: Nested keys to access (e.g., 'output', 'results_base')
            default: Default value if path not found
            
        Returns:
            Path string
        """
        config = self._get_paths_config()
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return str(value) if value else default
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names."""
        return [m.name for m in self.get_models()]
    
    def get_application_names(self) -> List[str]:
        """Get list of all application names."""
        return [a.name for a in self.get_applications()]
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate all configurations and return any issues found.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        issues = {'errors': [], 'warnings': []}
        
        # Validate models
        try:
            models = self.get_models()
            if not models:
                issues['errors'].append("No models configured")
        except Exception as e:
            issues['errors'].append(f"Error loading models config: {e}")
        
        # Validate applications
        try:
            apps = self.get_applications()
            if not apps:
                issues['errors'].append("No applications configured")
        except Exception as e:
            issues['errors'].append(f"Error loading applications config: {e}")
        
        # Validate parameters
        try:
            params = self.get_experiment_params()
            if params.execution.repetitions < 1:
                issues['warnings'].append("Repetitions should be at least 1")
            if params.exploration.timeout_minutes < 1:
                issues['warnings'].append("Timeout should be at least 1 minute")
        except Exception as e:
            issues['errors'].append(f"Error loading params config: {e}")
        
        return issues
    
    def print_summary(self):
        """Print a summary of loaded configurations."""
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"\nModels ({len(self.get_models())}):")
        for m in self.get_models():
            print(f"  - {m.name}: {m.description}")
        
        print(f"\nApplications ({len(self.get_applications())}):")
        for a in self.get_applications():
            print(f"  - {a.name}: {a.feature_count} features")
        
        params = self.get_experiment_params()
        print(f"\nExperiment Parameters:")
        print(f"  - Max states: {params.exploration.max_states}")
        print(f"  - Timeout: {params.exploration.timeout_minutes} minutes")
        print(f"  - Repetitions: {params.execution.repetitions}")
        print(f"  - Temperature: {params.inference.temperature}")
        
        total_runs = (
            len(self.get_models()) * 
            len(self.get_applications()) * 
            params.execution.repetitions
        )
        print(f"\nTotal planned runs: {total_runs}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test the configuration loader
    loader = ConfigLoader()
    
    # Validate
    issues = loader.validate_config()
    if issues['errors']:
        print("Errors found:")
        for e in issues['errors']:
            print(f"  - {e}")
    if issues['warnings']:
        print("Warnings:")
        for w in issues['warnings']:
            print(f"  - {w}")
    
    # Print summary
    loader.print_summary()
