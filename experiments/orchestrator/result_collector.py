"""
Result Collector Module
=======================

Collects, stores, and aggregates experiment metrics.
Provides unified interface for saving individual run results
and generating aggregated reports.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
import statistics


@dataclass
class ExperimentMetrics:
    """
    Complete metrics for a single experiment run.
    Follows the AutoE2E evaluation methodology.
    """
    # Primary metrics - Feature Coverage
    feature_coverage: float = 0.0
    total_features_covered: int = 0
    total_features: int = 0
    
    # Inference quality
    inferred_features: int = 0
    correct_inferences: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Test complexity - Action chains
    action_chain_lengths: List[int] = field(default_factory=list)
    min_chain_length: int = 0
    max_chain_length: int = 0
    mean_chain_length: float = 0.0
    
    # Ranking quality
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    
    # Efficiency metrics
    total_time_seconds: float = 0.0
    llm_queries: int = 0
    total_tokens: int = 0
    states_explored: int = 0
    actions_processed: int = 0
    
    # Metadata
    model_name: str = ""
    application_name: str = ""
    run_id: int = 0
    timestamp: str = ""
    
    # Additional run info
    completed: bool = False
    stopped_by_time_limit: bool = False
    stopped_by_state_limit: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentMetrics':
        """Create metrics from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def compute_derived_metrics(self):
        """Compute derived metrics from raw data."""
        if self.action_chain_lengths:
            self.min_chain_length = min(self.action_chain_lengths)
            self.max_chain_length = max(self.action_chain_lengths)
            self.mean_chain_length = statistics.mean(self.action_chain_lengths)
        
        if self.total_features > 0:
            self.feature_coverage = self.total_features_covered / self.total_features
        
        if self.inferred_features > 0:
            self.precision = self.correct_inferences / self.inferred_features
        
        if self.total_features > 0:
            self.recall = self.correct_inferences / self.total_features
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    model_name: str
    application_name: str
    num_runs: int
    
    # Means
    mean_feature_coverage: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_time_seconds: float = 0.0
    mean_llm_queries: float = 0.0
    mean_states_explored: float = 0.0
    
    # Standard deviations
    std_feature_coverage: float = 0.0
    std_precision: float = 0.0
    std_recall: float = 0.0
    std_f1: float = 0.0
    
    # Best run
    best_f1: float = 0.0
    best_coverage: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ResultCollector:
    """
    Collects and manages experiment results.
    
    Features:
    - Save individual run metrics
    - Load metrics from previous runs
    - Aggregate metrics across runs
    - Generate summary reports
    
    Directory Structure:
        results/
        ├── replication/
        │   ├── qwen3-8b/
        │   │   ├── petclinic/
        │   │   │   ├── run_1/
        │   │   │   │   ├── metrics.json
        │   │   │   │   └── ...
    """
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize result collector.
        
        Args:
            results_dir: Base directory for storing results
        """
        if results_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            results_dir = project_root / "experiments" / "results"
        
        self.results_dir = Path(results_dir)
        self.replication_dir = self.results_dir / "replication"
        self.aggregated_dir = self.results_dir / "aggregated"
        
        # Create directories
        self.replication_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as directory/file name."""
        return name.replace(':', '-').replace('/', '-').replace('\\', '-')
    
    def _get_run_dir(self, model_name: str, app_name: str, run_id: int) -> Path:
        """Get directory for a specific run."""
        model_dir = self._sanitize_name(model_name)
        run_dir = self.replication_dir / model_dir / app_name / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def save_metrics(
        self, 
        metrics: ExperimentMetrics,
        additional_data: Optional[Dict] = None
    ) -> str:
        """
        Save metrics for a run.
        
        Args:
            metrics: ExperimentMetrics object
            additional_data: Optional additional data to save
            
        Returns:
            Path to saved metrics file
        """
        run_dir = self._get_run_dir(
            metrics.model_name, 
            metrics.application_name, 
            metrics.run_id
        )
        
        # Combine metrics with additional data
        data = metrics.to_dict()
        if additional_data:
            data['additional'] = additional_data
        
        data['saved_at'] = datetime.now().isoformat()
        
        # Save to JSON
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Metrics saved to: {metrics_file}")
        return str(metrics_file)
    
    def load_metrics(
        self, 
        model_name: str, 
        app_name: str, 
        run_id: int
    ) -> Optional[ExperimentMetrics]:
        """Load metrics for a specific run."""
        run_dir = self._get_run_dir(model_name, app_name, run_id)
        metrics_file = run_dir / "metrics.json"
        
        if not metrics_file.exists():
            return None
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ExperimentMetrics.from_dict(data)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None
    
    def load_all_metrics(
        self, 
        model_name: Optional[str] = None,
        app_name: Optional[str] = None
    ) -> List[ExperimentMetrics]:
        """
        Load all metrics, optionally filtered by model or application.
        
        Args:
            model_name: Filter by model (optional)
            app_name: Filter by application (optional)
            
        Returns:
            List of ExperimentMetrics
        """
        metrics_list = []
        
        # Determine which directories to scan
        if model_name:
            model_dirs = [self.replication_dir / self._sanitize_name(model_name)]
        else:
            model_dirs = list(self.replication_dir.iterdir()) if self.replication_dir.exists() else []
        
        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue
            
            # Determine app directories
            if app_name:
                app_dirs = [model_dir / app_name]
            else:
                app_dirs = list(model_dir.iterdir())
            
            for app_dir in app_dirs:
                if not app_dir.is_dir():
                    continue
                
                # Load all runs
                for run_dir in app_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    
                    metrics_file = run_dir / "metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            metrics_list.append(ExperimentMetrics.from_dict(data))
                        except Exception as e:
                            print(f"Error loading {metrics_file}: {e}")
        
        return metrics_list
    
    def aggregate_by_model(
        self, 
        model_name: str
    ) -> Dict[str, AggregatedMetrics]:
        """
        Aggregate metrics for a model across all applications.
        
        Returns:
            Dictionary mapping app_name to AggregatedMetrics
        """
        all_metrics = self.load_all_metrics(model_name=model_name)
        
        # Group by application
        by_app: Dict[str, List[ExperimentMetrics]] = {}
        for m in all_metrics:
            if m.application_name not in by_app:
                by_app[m.application_name] = []
            by_app[m.application_name].append(m)
        
        # Aggregate
        aggregated = {}
        for app_name, metrics_list in by_app.items():
            aggregated[app_name] = self._aggregate_metrics(
                model_name, app_name, metrics_list
            )
        
        return aggregated
    
    def aggregate_by_application(
        self, 
        app_name: str
    ) -> Dict[str, AggregatedMetrics]:
        """
        Aggregate metrics for an application across all models.
        
        Returns:
            Dictionary mapping model_name to AggregatedMetrics
        """
        all_metrics = self.load_all_metrics(app_name=app_name)
        
        # Group by model
        by_model: Dict[str, List[ExperimentMetrics]] = {}
        for m in all_metrics:
            if m.model_name not in by_model:
                by_model[m.model_name] = []
            by_model[m.model_name].append(m)
        
        # Aggregate
        aggregated = {}
        for model_name, metrics_list in by_model.items():
            aggregated[model_name] = self._aggregate_metrics(
                model_name, app_name, metrics_list
            )
        
        return aggregated
    
    def _aggregate_metrics(
        self, 
        model_name: str, 
        app_name: str,
        metrics_list: List[ExperimentMetrics]
    ) -> AggregatedMetrics:
        """Compute aggregated metrics from a list of runs."""
        if not metrics_list:
            return AggregatedMetrics(model_name, app_name, 0)
        
        n = len(metrics_list)
        
        coverages = [m.feature_coverage for m in metrics_list]
        precisions = [m.precision for m in metrics_list]
        recalls = [m.recall for m in metrics_list]
        f1s = [m.f1_score for m in metrics_list]
        times = [m.total_time_seconds for m in metrics_list]
        queries = [m.llm_queries for m in metrics_list]
        states = [m.states_explored for m in metrics_list]
        
        def safe_stdev(values):
            if len(values) < 2:
                return 0.0
            return statistics.stdev(values)
        
        return AggregatedMetrics(
            model_name=model_name,
            application_name=app_name,
            num_runs=n,
            mean_feature_coverage=statistics.mean(coverages) if coverages else 0,
            mean_precision=statistics.mean(precisions) if precisions else 0,
            mean_recall=statistics.mean(recalls) if recalls else 0,
            mean_f1=statistics.mean(f1s) if f1s else 0,
            mean_time_seconds=statistics.mean(times) if times else 0,
            mean_llm_queries=statistics.mean(queries) if queries else 0,
            mean_states_explored=statistics.mean(states) if states else 0,
            std_feature_coverage=safe_stdev(coverages),
            std_precision=safe_stdev(precisions),
            std_recall=safe_stdev(recalls),
            std_f1=safe_stdev(f1s),
            best_f1=max(f1s) if f1s else 0,
            best_coverage=max(coverages) if coverages else 0
        )
    
    def generate_summary_csv(self) -> str:
        """
        Generate summary CSV with all results.
        
        Returns:
            Path to generated CSV file
        """
        all_metrics = self.load_all_metrics()
        
        if not all_metrics:
            print("No metrics found to generate summary")
            return ""
        
        csv_file = self.aggregated_dir / "summary_all_runs.csv"
        
        fieldnames = [
            'model_name', 'application_name', 'run_id',
            'feature_coverage', 'precision', 'recall', 'f1_score',
            'total_features_covered', 'total_features',
            'inferred_features', 'correct_inferences',
            'total_time_seconds', 'llm_queries', 'states_explored',
            'completed', 'error', 'timestamp'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in all_metrics:
                writer.writerow({
                    'model_name': m.model_name,
                    'application_name': m.application_name,
                    'run_id': m.run_id,
                    'feature_coverage': f"{m.feature_coverage:.4f}",
                    'precision': f"{m.precision:.4f}",
                    'recall': f"{m.recall:.4f}",
                    'f1_score': f"{m.f1_score:.4f}",
                    'total_features_covered': m.total_features_covered,
                    'total_features': m.total_features,
                    'inferred_features': m.inferred_features,
                    'correct_inferences': m.correct_inferences,
                    'total_time_seconds': f"{m.total_time_seconds:.2f}",
                    'llm_queries': m.llm_queries,
                    'states_explored': m.states_explored,
                    'completed': m.completed,
                    'error': m.error or '',
                    'timestamp': m.timestamp
                })
        
        print(f"Summary CSV saved to: {csv_file}")
        return str(csv_file)
    
    def generate_aggregated_json(self) -> str:
        """
        Generate aggregated JSON with means and standard deviations.
        
        Returns:
            Path to generated JSON file
        """
        all_metrics = self.load_all_metrics()
        
        if not all_metrics:
            print("No metrics found to generate aggregated report")
            return ""
        
        # Group by model and app
        aggregated = {}
        
        # Get unique models and apps
        models = set(m.model_name for m in all_metrics)
        apps = set(m.application_name for m in all_metrics)
        
        for model in models:
            model_key = self._sanitize_name(model)
            aggregated[model_key] = {}
            
            for app in apps:
                metrics_for_pair = [
                    m for m in all_metrics 
                    if m.model_name == model and m.application_name == app
                ]
                
                if metrics_for_pair:
                    agg = self._aggregate_metrics(model, app, metrics_for_pair)
                    aggregated[model_key][app] = agg.to_dict()
        
        json_file = self.aggregated_dir / "detailed_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_runs': len(all_metrics),
                'models': list(models),
                'applications': list(apps),
                'results': aggregated
            }, f, indent=2)
        
        print(f"Aggregated JSON saved to: {json_file}")
        return str(json_file)
    
    def save_run_artifact(
        self,
        model_name: str,
        app_name: str,
        run_id: int,
        artifact_name: str,
        data: Any
    ) -> str:
        """
        Save an artifact (JSON serializable) for a run.
        
        Args:
            model_name: Model name
            app_name: Application name
            run_id: Run ID
            artifact_name: Name of the artifact (e.g., 'feature_db', 'action_chains')
            data: Data to save (must be JSON serializable)
            
        Returns:
            Path to saved artifact
        """
        run_dir = self._get_run_dir(model_name, app_name, run_id)
        artifact_file = run_dir / f"{artifact_name}.json"
        
        with open(artifact_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return str(artifact_file)
    
    def get_run_artifacts(
        self,
        model_name: str,
        app_name: str,
        run_id: int
    ) -> Dict[str, Path]:
        """
        Get all artifact files for a run.
        
        Returns:
            Dictionary mapping artifact name to file path
        """
        run_dir = self._get_run_dir(model_name, app_name, run_id)
        
        artifacts = {}
        if run_dir.exists():
            for f in run_dir.iterdir():
                if f.suffix == '.json':
                    artifacts[f.stem] = f
        
        return artifacts


if __name__ == "__main__":
    # Test the result collector
    collector = ResultCollector()
    
    # Create test metrics
    metrics = ExperimentMetrics(
        model_name="qwen3:8b",
        application_name="petclinic",
        run_id=1,
        feature_coverage=0.65,
        total_features_covered=15,
        total_features=23,
        precision=0.75,
        recall=0.65,
        f1_score=0.70,
        total_time_seconds=3600,
        llm_queries=150,
        states_explored=45,
        completed=True,
        timestamp=datetime.now().isoformat()
    )
    
    # Save metrics
    collector.save_metrics(metrics)
    
    # Load and verify
    loaded = collector.load_metrics("qwen3:8b", "petclinic", 1)
    print(f"Loaded metrics: {loaded.feature_coverage if loaded else 'Not found'}")
    
    # Generate summaries
    collector.generate_summary_csv()
    collector.generate_aggregated_json()
