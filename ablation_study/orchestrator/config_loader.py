"""
Ablation Configuration Loader
=============================

Loads and manages ablation study configurations from YAML files.
Handles configuration inheritance (parent/override pattern).
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class AblationConfig:
    """Complete configuration for a single ablation."""
    id: str
    name: str
    description: str
    
    # Component configurations
    context: Dict[str, Any] = field(default_factory=dict)
    prompting: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    aggregation: Dict[str, Any] = field(default_factory=dict)
    accumulation: Dict[str, Any] = field(default_factory=dict)
    finality: Dict[str, Any] = field(default_factory=dict)
    score_threshold: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'context': self.context,
            'prompting': self.prompting,
            'scoring': self.scoring,
            'aggregation': self.aggregation,
            'accumulation': self.accumulation,
            'finality': self.finality,
            'score_threshold': self.score_threshold
        }


@dataclass
class ModelConfig:
    """Configuration for the LLM model."""
    name: str
    ollama_model: str
    context_length: int
    description: str


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
class ExperimentParams:
    """Experiment parameters."""
    timeout_minutes: int = 720
    max_states: int = 1000
    max_depth: int = 12
    repetitions: int = 3
    retry_on_failure: int = 3
    inter_run_delay_seconds: int = 30
    headless: bool = True


class AblationConfigLoader:
    """
    Loads and manages ablation study configurations.
    
    Handles:
    - Loading ablation definitions with inheritance
    - Loading model configuration
    - Loading application configurations
    - Loading experiment parameters
    
    Usage:
        loader = AblationConfigLoader()
        
        # Get all ablations
        ablations = loader.get_ablations()
        
        # Get specific ablation
        ablation = loader.get_ablation("A1.1")
        
        # Get baseline
        baseline = loader.get_baseline()
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Path to configuration directory.
                       Defaults to ablation_study/config.
        """
        if config_dir is None:
            current = Path(__file__).resolve()
            config_dir = current.parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        
        # Cache loaded configurations
        self._ablations_config: Optional[Dict] = None
        self._model_config: Optional[Dict] = None
        self._applications_config: Optional[Dict] = None
        self._params_config: Optional[Dict] = None
        
        # Resolved ablation configs (with inheritance applied)
        self._resolved_ablations: Optional[Dict[str, AblationConfig]] = None
    
    def _load_yaml(self, filename: str) -> Dict:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _get_ablations_raw(self) -> Dict:
        """Load raw ablations configuration."""
        if self._ablations_config is None:
            self._ablations_config = self._load_yaml("ablations.yaml")
        return self._ablations_config
    
    def _get_model_raw(self) -> Dict:
        """Load model configuration."""
        if self._model_config is None:
            self._model_config = self._load_yaml("model.yaml")
        return self._model_config
    
    def _get_applications_raw(self) -> Dict:
        """Load applications configuration."""
        if self._applications_config is None:
            self._applications_config = self._load_yaml("applications.yaml")
        return self._applications_config
    
    def _get_params_raw(self) -> Dict:
        """Load experiment parameters."""
        if self._params_config is None:
            self._params_config = self._load_yaml("experiment_params.yaml")
        return self._params_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        Override values take precedence.
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _resolve_ablation(self, name: str, raw_config: Dict, all_raw: Dict) -> AblationConfig:
        """
        Resolve a single ablation configuration with inheritance.
        
        Args:
            name: Ablation name (key in yaml)
            raw_config: Raw configuration for this ablation
            all_raw: All raw ablation configurations (for parent lookup)
            
        Returns:
            Resolved AblationConfig
        """
        # Start with defaults
        resolved = {
            'context': {
                'include_screenshot': True,
                'include_previous_state': True,
                'include_previous_action': True
            },
            'prompting': {
                'strategy': 'dual'
            },
            'scoring': {
                'function': 'geometric',
                'p': 0.5,
                'R': 10
            },
            'aggregation': {
                'method': 'semantic',
                'similarity_threshold': 0.85,
                'llm_validation': True
            },
            'accumulation': {
                'method': 'differential'
            },
            'finality': {
                'method': 'llm'
            },
            'score_threshold': {
                'enabled': False,
                'min_score': None
            }
        }
        
        # If has parent, resolve parent first and merge
        if 'parent' in raw_config:
            parent_name = raw_config['parent']
            if parent_name in all_raw:
                parent_resolved = self._resolve_ablation(parent_name, all_raw[parent_name], all_raw)
                resolved = parent_resolved.to_dict()
                # Remove non-config fields
                for key in ['id', 'name', 'description']:
                    resolved.pop(key, None)
        
        # Apply this config's values (non-override fields)
        for key in ['context', 'prompting', 'scoring', 'aggregation', 'accumulation', 'finality', 'score_threshold']:
            if key in raw_config:
                resolved[key] = self._deep_merge(resolved.get(key, {}), raw_config[key])
        
        # Apply overrides
        if 'overrides' in raw_config:
            for key, value in raw_config['overrides'].items():
                if key in resolved and isinstance(resolved[key], dict) and isinstance(value, dict):
                    resolved[key] = self._deep_merge(resolved[key], value)
                else:
                    resolved[key] = value
        
        return AblationConfig(
            id=raw_config.get('id', name),
            name=name,
            description=raw_config.get('description', ''),
            context=resolved.get('context', {}),
            prompting=resolved.get('prompting', {}),
            scoring=resolved.get('scoring', {}),
            aggregation=resolved.get('aggregation', {}),
            accumulation=resolved.get('accumulation', {}),
            finality=resolved.get('finality', {}),
            score_threshold=resolved.get('score_threshold', {})
        )
    
    def _resolve_all_ablations(self) -> Dict[str, AblationConfig]:
        """Resolve all ablation configurations."""
        if self._resolved_ablations is not None:
            return self._resolved_ablations
        
        raw = self._get_ablations_raw()
        self._resolved_ablations = {}
        
        for name, config in raw.items():
            self._resolved_ablations[name] = self._resolve_ablation(name, config, raw)
        
        return self._resolved_ablations
    
    def get_ablations(self) -> List[AblationConfig]:
        """Get list of all resolved ablation configurations."""
        resolved = self._resolve_all_ablations()
        return list(resolved.values())
    
    def get_ablation(self, ablation_id: str) -> Optional[AblationConfig]:
        """
        Get a specific ablation configuration.
        
        Args:
            ablation_id: Ablation ID (e.g., "A1.1") or name (e.g., "A1.1_no_screenshot")
            
        Returns:
            AblationConfig or None if not found
        """
        resolved = self._resolve_all_ablations()
        
        # Try by name first
        if ablation_id in resolved:
            return resolved[ablation_id]
        
        # Try by ID
        for ablation in resolved.values():
            if ablation.id == ablation_id:
                return ablation
        
        return None
    
    def get_baseline(self) -> AblationConfig:
        """Get the baseline configuration."""
        ablation = self.get_ablation("baseline")
        if ablation is None:
            raise ValueError("Baseline configuration not found")
        return ablation
    
    def get_ablation_names(self) -> List[str]:
        """Get list of all ablation names."""
        resolved = self._resolve_all_ablations()
        return list(resolved.keys())
    
    def get_ablation_ids(self) -> List[str]:
        """Get list of all ablation IDs."""
        resolved = self._resolve_all_ablations()
        return [a.id for a in resolved.values()]
    
    def get_model(self) -> ModelConfig:
        """Get the model configuration."""
        config = self._get_model_raw()
        model = config.get('model', {})
        return ModelConfig(
            name=model.get('name', 'qwen2.5vl:32b'),
            ollama_model=model.get('ollama_model', 'qwen2.5vl:32b'),
            context_length=model.get('context_length', 8192),
            description=model.get('description', '')
        )
    
    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        config = self._get_model_raw()
        return config.get('embedding_model', 'nomic-embed-text')
    
    def get_ollama_settings(self) -> Dict[str, Any]:
        """Get Ollama settings."""
        config = self._get_model_raw()
        return config.get('ollama', {
            'base_url': 'http://localhost:11434',
            'temperature': 0,
            'num_predict': 4096,
            'timeout': 120
        })
    
    def get_applications(self) -> List[ApplicationConfig]:
        """Get list of all application configurations."""
        config = self._get_applications_raw()
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
        """Get a specific application by name."""
        for app in self.get_applications():
            if app.name == name or app.config_name == name:
                return app
        return None
    
    def get_application_names(self) -> List[str]:
        """Get list of all application names."""
        return [a.name for a in self.get_applications()]
    
    def get_recommended_apps(self) -> List[str]:
        """Get list of recommended applications for ablation study."""
        config = self._get_applications_raw()
        return config.get('recommended_subset', ['petclinic', 'realworld', 'dimeshift', 'mantisbt'])
    
    def get_experiment_params(self) -> ExperimentParams:
        """Get experiment parameters."""
        config = self._get_params_raw()
        
        exploration = config.get('exploration', {})
        execution = config.get('execution', {})
        browser = config.get('browser', {})
        
        return ExperimentParams(
            timeout_minutes=exploration.get('timeout_minutes', 720),
            max_states=exploration.get('max_states', 1000),
            max_depth=exploration.get('max_depth', 12),
            repetitions=execution.get('repetitions', 3),
            retry_on_failure=execution.get('retry_on_failure', 3),
            inter_run_delay_seconds=execution.get('inter_run_delay_seconds', 30),
            headless=browser.get('headless', True)
        )
    
    def print_summary(self):
        """Print a summary of loaded configurations."""
        print("\n" + "=" * 70)
        print("ABLATION STUDY CONFIGURATION SUMMARY")
        print("=" * 70)
        
        ablations = self.get_ablations()
        print(f"\nAblations ({len(ablations)}):")
        
        # Group by component
        components = {
            'C1 Context': [],
            'C2 Prompting': [],
            'C3 Scoring': [],
            'C4 Aggregation': [],
            'C5 Accumulation': [],
            'C6 Finality': [],
            'C7 Probability': [],
            'Baseline': []
        }
        
        for a in ablations:
            if a.id == 'baseline':
                components['Baseline'].append(a)
            elif a.id.startswith('A1'):
                components['C1 Context'].append(a)
            elif a.id.startswith('A2'):
                components['C2 Prompting'].append(a)
            elif a.id.startswith('A3'):
                components['C3 Scoring'].append(a)
            elif a.id.startswith('A4'):
                components['C4 Aggregation'].append(a)
            elif a.id.startswith('A5'):
                components['C5 Accumulation'].append(a)
            elif a.id.startswith('A6'):
                components['C6 Finality'].append(a)
            elif a.id.startswith('A7'):
                components['C7 Probability'].append(a)
        
        for component, items in components.items():
            if items:
                print(f"\n  {component}:")
                for a in items:
                    print(f"    - {a.id}: {a.description}")
        
        model = self.get_model()
        print(f"\nModel: {model.name}")
        
        apps = self.get_applications()
        print(f"\nApplications ({len(apps)}):")
        for a in apps:
            print(f"  - {a.name}: {a.feature_count} features")
        
        params = self.get_experiment_params()
        print(f"\nExperiment Parameters:")
        print(f"  - Max states: {params.max_states}")
        print(f"  - Timeout: {params.timeout_minutes} minutes")
        print(f"  - Repetitions: {params.repetitions}")
        
        total_runs = len(ablations) * len(apps) * params.repetitions
        print(f"\nTotal planned runs: {total_runs}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test the configuration loader
    loader = AblationConfigLoader()
    loader.print_summary()
