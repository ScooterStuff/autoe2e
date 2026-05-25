"""
Ablation Runner
===============

Main orchestrator for running ablation study experiments.
Coordinates component configuration, experiment execution, and metrics collection.
"""

import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .config_loader import AblationConfigLoader, AblationConfig, ApplicationConfig
from .component_factory import ComponentFactory
from .checkpoint_manager import AblationCheckpointManager
from .metrics_collector import AblationMetricsCollector, RunMetrics


@dataclass
class RunConfig:
    """Configuration for a single experiment run."""
    ablation_config: AblationConfig
    application: ApplicationConfig
    run_id: int
    output_dir: Path
    timeout_minutes: int = 720
    headless: bool = True


class AblationRunner:
    """
    Main orchestrator for ablation study experiments.
    
    Responsibilities:
    - Load and validate configurations
    - Coordinate experiment execution
    - Manage checkpoints for resumption
    - Collect and store metrics
    
    Usage:
        runner = AblationRunner()
        
        # Run all ablations
        runner.run_all()
        
        # Run specific ablation
        runner.run_ablation("A1.1")
        
        # Run with specific apps
        runner.run_ablation("A1.1", applications=["petclinic"])
    """
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = True,
        model: Optional[str] = None
    ):
        """
        Initialize ablation runner.
        
        Args:
            config_dir: Configuration directory path
            results_dir: Results directory path
            dry_run: If True, only show what would be run
            verbose: Print detailed progress
            model: Ollama model name override (e.g., "qwen2.5-coder:7b")
        """
        self.config_loader = AblationConfigLoader(config_dir)
        self.checkpoint = AblationCheckpointManager()
        self.metrics = AblationMetricsCollector(results_dir)
        
        self.dry_run = dry_run
        self.verbose = verbose
        self.model = model  # Model override
        
        # Setup results directory
        if results_dir is None:
            current = Path(__file__).resolve()
            results_dir = current.parent.parent / "results"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def _create_run_dir(self, ablation_id: str, app_name: str, run_id: int) -> Path:
        """Create directory for run outputs."""
        run_dir = self.results_dir / ablation_id / app_name / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _generate_run_config_file(
        self,
        run_config: RunConfig,
        components: Dict[str, Any]
    ) -> Path:
        """
        Generate a configuration file for the run.
        
        This file will be read by the main.py script to configure
        the ablation-specific components.
        """
        config_file = run_config.output_dir / "ablation_config.json"
        
        # Get model config
        model_obj = self.config_loader.get_model()
        model_name = self.model if self.model else model_obj.ollama_model
        
        config = {
            'ablation_id': run_config.ablation_config.id,
            'ablation_name': run_config.ablation_config.name,
            'ablation_description': run_config.ablation_config.description,
            'application': run_config.application.name,
            'run_id': run_config.run_id,
            'model': model_name,
            'components': {
                'context': components['context_extractor'].to_dict(),
                'prompting': components['prompt_manager'].to_dict(),
                'scoring': components['scoring_function'].to_dict(),
                'accumulation': components['score_accumulator'].to_dict(),
                'score_threshold': components['score_threshold'].to_dict()
            },
            'settings': {
                'timeout_minutes': run_config.timeout_minutes,
                'headless': run_config.headless
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def _load_app_config(self, app: ApplicationConfig) -> Path:
        """Load the application's main config file."""
        config_path = PROJECT_ROOT / "configs" / f"{app.config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Application config not found: {config_path}")
        return config_path
    
    def _start_docker(self, app: ApplicationConfig) -> bool:
        """
        Start Docker containers for the application.
        
        Returns:
            True if started successfully
        """
        # Handle PetClinic separately - uses simple docker run, not docker-compose
        if app.name == 'petclinic':
            return self._start_petclinic_container(app)
        
        docker_compose_path = PROJECT_ROOT / app.docker_compose
        
        if not docker_compose_path.exists():
            self._log(f"Docker compose file not found: {docker_compose_path}", level="WARNING")
            self._log("Assuming containers are already running...")
            return True
        
        self._log(f"Starting Docker containers: {docker_compose_path}")
        
        try:
            # Stop any existing containers first
            subprocess.run(
                ["docker-compose", "-f", str(docker_compose_path), "down", "--remove-orphans"],
                cwd=str(docker_compose_path.parent),
                capture_output=True,
                timeout=60
            )
            
            # Start containers
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_path), "up", "-d"],
                cwd=str(docker_compose_path.parent),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                self._log(f"Docker start failed: {result.stderr}", level="ERROR")
                return False
            
            # Wait for health check
            self._log(f"Waiting for application to be ready at {app.url}...")
            return self._wait_for_health(app)
            
        except subprocess.TimeoutExpired:
            self._log("Docker start timed out", level="ERROR")
            return False
        except Exception as e:
            self._log(f"Docker start error: {str(e)}", level="ERROR")
            return False
    
    def _start_petclinic_container(self, app: ApplicationConfig) -> bool:
        """
        Start the PetClinic container (uses simple docker run, not docker-compose).
        
        Returns:
            True if container started successfully
        """
        container_name = 'petclinic'
        image_name = 'petclinic-frontend'
        
        try:
            # Check if container already exists and is running
            result = subprocess.run(
                ['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            status = result.stdout.strip()
            
            if status:
                if 'Up' in status:
                    self._log(f"PetClinic container already running")
                else:
                    # Container exists but stopped, start it
                    self._log(f"Starting existing PetClinic container...")
                    subprocess.run(
                        ['docker', 'start', container_name],
                        capture_output=True,
                        timeout=60
                    )
            else:
                # No container exists, check if image exists and create container
                img_result = subprocess.run(
                    ['docker', 'images', image_name, '--format', '{{.Repository}}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if not img_result.stdout.strip():
                    # Need to build the image first
                    self._log(f"Building PetClinic image...")
                    dockerfile_dir = PROJECT_ROOT / 'benchmark' / 'pet-clinic' / 'spring-petclinic-angular'
                    subprocess.run(
                        ['docker', 'build', '-t', image_name, '.'],
                        cwd=str(dockerfile_dir),
                        capture_output=True,
                        timeout=300
                    )
                
                # Run new container
                self._log(f"Creating and starting PetClinic container...")
                subprocess.run(
                    ['docker', 'run', '-d', '-p', '8080:8080', f'--name={container_name}', image_name],
                    capture_output=True,
                    timeout=60
                )
            
            # Wait for health check
            self._log(f"Waiting for PetClinic to be healthy...")
            return self._wait_for_health(app)
                
        except Exception as e:
            self._log(f"Error starting PetClinic container: {e}", level="ERROR")
            return False
    
    def _stop_docker(self, app: ApplicationConfig):
        """Stop Docker containers for the application."""
        # Handle PetClinic separately
        if app.name == 'petclinic':
            try:
                self._log("Stopping PetClinic container...")
                subprocess.run(
                    ['docker', 'stop', 'petclinic'],
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:
                self._log(f"Docker stop error: {str(e)}", level="WARNING")
            return
        
        docker_compose_path = PROJECT_ROOT / app.docker_compose
        
        if not docker_compose_path.exists():
            return
        
        self._log(f"Stopping Docker containers...")
        
        try:
            subprocess.run(
                ["docker-compose", "-f", str(docker_compose_path), "down"],
                cwd=str(docker_compose_path.parent),
                capture_output=True,
                timeout=60
            )
        except Exception as e:
            self._log(f"Docker stop error: {str(e)}", level="WARNING")
    
    def _wait_for_health(self, app: ApplicationConfig, check_interval: float = 2.0) -> bool:
        """
        Wait for application to be healthy.
        
        Args:
            app: Application configuration
            check_interval: Seconds between health checks
            
        Returns:
            True if application is healthy
        """
        import urllib.request
        import urllib.error
        
        url = f"{app.url}{app.health_check_endpoint}"
        timeout = app.health_check_timeout_seconds
        elapsed = 0
        
        while elapsed < timeout:
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status == 200:
                        self._log(f"Application is ready!")
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                pass
            
            time.sleep(check_interval)
            elapsed += check_interval
            if elapsed % 10 == 0:
                self._log(f"Still waiting... ({elapsed}s / {timeout}s)")
        
        self._log(f"Health check timed out after {timeout}s", level="ERROR")
        return False
    
    def _run_experiment(self, run_config: RunConfig) -> Tuple[bool, RunMetrics]:
        """
        Execute a single experiment run.
        
        Returns:
            Tuple of (success, metrics)
        """
        start_time = datetime.now()
        docker_started = False
        
        try:
            # Start Docker containers for the application
            self._log(f"Starting application: {run_config.application.name}")
            if not self.dry_run:
                docker_started = self._start_docker(run_config.application)
                if not docker_started:
                    raise RuntimeError(f"Failed to start Docker for {run_config.application.name}")
            
            # Create components from ablation config
            components = ComponentFactory.create_all(run_config.ablation_config)
            
            self._log(f"Components configured:")
            self._log(ComponentFactory.get_component_summary(components))
            
            # Generate ablation config file
            ablation_config_file = self._generate_run_config_file(run_config, components)
            self._log(f"Ablation config written to: {ablation_config_file}")
            
            # Load application config
            app_config_file = self._load_app_config(run_config.application)
            self._log(f"Application config: {app_config_file}")
            
            if self.dry_run:
                self._log("[DRY RUN] Would execute experiment")
                end_time = datetime.now()
                return True, RunMetrics(
                    ablation_id=run_config.ablation_config.id,
                    application=run_config.application.name,
                    run_id=run_config.run_id,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=0.0,
                    total_features=run_config.application.feature_count,
                    detected_features=0,
                    feature_coverage=0.0,
                    states_explored=0,
                    actions_executed=0,
                    unique_pages=0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    success=True
                )
            
            # Build the command to run main.py with ablation config
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--config", str(app_config_file),
                "--ablation-config", str(ablation_config_file),
                "--output-dir", str(run_config.output_dir)
            ]
            
            if run_config.headless:
                cmd.append("--headless")
            
            self._log(f"Executing: {' '.join(cmd)}")
            
            # Set environment variables for limits
            # main.py reads these: MAX_RUNTIME_MINUTES, MAX_TOTAL_STATES
            params = self.config_loader.get_experiment_params()
            env = os.environ.copy()
            env['MAX_RUNTIME_MINUTES'] = str(params.timeout_minutes)
            env['MAX_TOTAL_STATES'] = str(params.max_states)
            
            # Set the model from ablation config (critical for llm_api_call.py)
            model_obj = self.config_loader.get_model()
            model_name = self.model if self.model else model_obj.ollama_model
            env['OLLAMA_MODEL'] = model_name
            
            # Set other Ollama environment variables from config
            ollama_settings = self.config_loader.get_ollama_settings()
            env['OLLAMA_BASE_URL'] = ollama_settings.get('base_url', 'http://localhost:11434')
            env['OLLAMA_TEMPERATURE'] = str(ollama_settings.get('temperature', 0))
            env['OLLAMA_NUM_PREDICT'] = str(ollama_settings.get('num_predict', 4096))
            env['OLLAMA_TIMEOUT'] = str(ollama_settings.get('timeout', 120))
            
            self._log(f"Limits: timeout={params.timeout_minutes} min (0=unlimited), max_states={params.max_states} (0=unlimited)")
            self._log(f"Using model: {model_name}")
            
            # Run the experiment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                env=env
            )
            
            # Stream output
            log_file = run_config.output_dir / "experiment.log"
            with open(log_file, 'w') as f:
                for line in process.stdout:
                    f.write(line)
                    if self.verbose:
                        print(line, end='')
            
            process.wait()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if process.returncode != 0:
                raise RuntimeError(f"Experiment failed with return code {process.returncode}")
            
            # Run post-experiment evaluation
            eval_results = self._run_evaluation(run_config)
            
            # Load results from the run (with evaluation data)
            metrics = self._load_run_results(run_config, start_time, end_time, duration, eval_results)
            
            # Stop Docker containers
            if docker_started:
                self._stop_docker(run_config.application)
            
            return True, metrics
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self._log(f"Experiment failed: {str(e)}", level="ERROR")
            self._log(traceback.format_exc(), level="ERROR")
            
            # Stop Docker containers on failure too
            if docker_started:
                self._stop_docker(run_config.application)
            
            return False, RunMetrics(
                ablation_id=run_config.ablation_config.id,
                application=run_config.application.name,
                run_id=run_config.run_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                total_features=run_config.application.feature_count,
                detected_features=0,
                feature_coverage=0.0,
                states_explored=0,
                actions_executed=0,
                unique_pages=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                success=False,
                error=str(e)
            )
    
    def _run_evaluation(self, run_config: RunConfig) -> Dict[str, Any]:
        """
        Run post-experiment evaluation using evaluate_autoe2e.py logic.
        
        Returns evaluation metrics (precision, recall, f1, coverage).
        """
        try:
            # Import evaluation components
            from evaluate_autoe2e import (
                connect_to_mongodb, load_data, 
                ActionChainReconstructor, FeatureCoverageEvaluator,
                BENCHMARKS
            )
            
            app_name = run_config.application.config_name
            
            self._log("Running post-experiment evaluation...")
            
            # Connect to MongoDB
            client, db = connect_to_mongodb()
            
            try:
                # Load data from MongoDB
                func_records, action_func_records, func_lookup = load_data(db, app_name)
                
                if not action_func_records:
                    self._log("No action-functionality data found", level="WARNING")
                    return {}
                
                # Get benchmark grammar
                grammar = BENCHMARKS.get(app_name.upper(), {})
                if not grammar:
                    self._log(f"No benchmark grammar for {app_name}", level="WARNING")
                    return {}
                
                # Reconstruct action chains
                reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
                chains = reconstructor.get_all_chains()
                
                # Deduplicate chains
                seq_best = {}
                for chain in chains:
                    seq, test_ids, record, func_text = chain
                    func_pointer = record.get('func_pointer', '')
                    score = func_lookup.get(func_pointer, {}).get('score', 0)
                    if seq not in seq_best or score > seq_best[seq][1]:
                        seq_best[seq] = (chain, score)
                chains = [item[0] for item in seq_best.values()]
                
                # Evaluate coverage
                evaluator = FeatureCoverageEvaluator(grammar, func_lookup)
                results = evaluator.evaluate(chains)
                
                metrics = results.get('metrics', {})
                
                self._log(f"Evaluation: precision={metrics.get('precision', 0):.2%}, "
                         f"recall={metrics.get('recall', 0):.2%}, "
                         f"f1={metrics.get('f1', 0):.2%}")
                
                return {
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1', 0),
                    'detected_features': metrics.get('covered', 0),
                    'total_generated': metrics.get('total_generated', 0)
                }
                
            finally:
                client.close()
                
        except Exception as e:
            self._log(f"Evaluation failed: {e}", level="ERROR")
            return {}
    
    def _load_run_results(
        self,
        run_config: RunConfig,
        start_time: datetime,
        end_time: datetime,
        duration: float,
        eval_results: Dict[str, Any] = None
    ) -> RunMetrics:
        """Load results from a completed run."""
        results_file = run_config.output_dir / "results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            # Start with empty results
            results = {
                'detected_features': 0,
                'states_explored': 0,
                'actions_executed': 0,
                'unique_pages': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Merge evaluation results if provided
        if eval_results:
            results.update(eval_results)
        
        # Try to load crawl stats from run summary
        app_name = run_config.application.config_name
        report_dir = PROJECT_ROOT / "report"
        summary_files = list(report_dir.glob(f"{app_name}_run_summary_*.json"))
        if summary_files:
            latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_summary, 'r') as f:
                    run_summary = json.load(f)
                    crawl_stats = run_summary.get('crawl_stats', {})
                    results['states_explored'] = crawl_stats.get('total_states', 0)
                    results['actions_executed'] = crawl_stats.get('total_actions_processed', 0)
            except:
                pass
        
        total_features = run_config.application.feature_count
        detected = results.get('detected_features', 0)
        coverage = detected / total_features if total_features > 0 else 0.0
        
        return RunMetrics(
            ablation_id=run_config.ablation_config.id,
            application=run_config.application.name,
            run_id=run_config.run_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_features=total_features,
            detected_features=detected,
            feature_coverage=coverage,
            states_explored=results.get('states_explored', 0),
            actions_executed=results.get('actions_executed', 0),
            unique_pages=results.get('unique_pages', 0),
            precision=results.get('precision', 0.0),
            recall=results.get('recall', 0.0),
            f1_score=results.get('f1_score', 0.0),
            success=True
        )
    
    def run_single(
        self,
        ablation_id: str,
        app_name: str,
        run_id: int,
        force: bool = False
    ) -> bool:
        """
        Run a single experiment.
        
        Args:
            ablation_id: Ablation ID (e.g., "A1.1")
            app_name: Application name
            run_id: Run number
            force: Run even if already completed
            
        Returns:
            True if successful
        """
        # Check checkpoint
        if not force and self.checkpoint.is_completed(ablation_id, app_name, run_id):
            self._log(f"Skipping {ablation_id}/{app_name}/run_{run_id} (already completed)")
            return True
        
        # Load configs
        ablation_config = self.config_loader.get_ablation(ablation_id)
        if ablation_config is None:
            self._log(f"Unknown ablation: {ablation_id}", level="ERROR")
            return False
        
        app_config = self.config_loader.get_application(app_name)
        if app_config is None:
            self._log(f"Unknown application: {app_name}", level="ERROR")
            return False
        
        params = self.config_loader.get_experiment_params()
        
        # Create run config
        output_dir = self._create_run_dir(ablation_id, app_name, run_id)
        run_config = RunConfig(
            ablation_config=ablation_config,
            application=app_config,
            run_id=run_id,
            output_dir=output_dir,
            timeout_minutes=params.timeout_minutes,
            headless=params.headless
        )
        
        self._log("=" * 70)
        self._log(f"STARTING RUN: {ablation_id} / {app_name} / run_{run_id}")
        self._log(f"Description: {ablation_config.description}")
        self._log("=" * 70)
        
        # Execute
        success, metrics = self._run_experiment(run_config)
        
        # Record results
        self.metrics.record_run(metrics)
        self.checkpoint.mark_completed(
            ablation_id, app_name, run_id,
            success=success,
            duration_seconds=metrics.duration_seconds,
            error=metrics.error
        )
        
        if success:
            self._log(f"Run completed: coverage={metrics.feature_coverage:.2%}, "
                     f"states={metrics.states_explored}")
        else:
            self._log(f"Run failed: {metrics.error}", level="ERROR")
        
        return success
    
    def run_ablation(
        self,
        ablation_id: str,
        applications: Optional[List[str]] = None,
        repetitions: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Run all experiments for an ablation.
        
        Args:
            ablation_id: Ablation ID
            applications: List of apps (default: all recommended)
            repetitions: Number of runs per app (default: from config)
            force: Run even if completed
            
        Returns:
            Summary of results
        """
        params = self.config_loader.get_experiment_params()
        
        if applications is None:
            applications = self.config_loader.get_recommended_apps()
        if repetitions is None:
            repetitions = params.repetitions
        
        results = {
            'ablation_id': ablation_id,
            'applications': applications,
            'repetitions': repetitions,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'runs': []
        }
        
        for app_name in applications:
            for run_id in range(1, repetitions + 1):
                if not force and self.checkpoint.is_completed(ablation_id, app_name, run_id):
                    results['skipped'] += 1
                    continue
                
                success = self.run_single(ablation_id, app_name, run_id, force=force)
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                results['runs'].append({
                    'application': app_name,
                    'run_id': run_id,
                    'success': success
                })
                
                # Delay between runs
                if not self.dry_run:
                    time.sleep(params.inter_run_delay_seconds)
        
        return results
    
    def run_all(
        self,
        ablation_ids: Optional[List[str]] = None,
        applications: Optional[List[str]] = None,
        repetitions: Optional[int] = None,
        force: bool = False,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete ablation study.
        
        Args:
            ablation_ids: List of ablations (default: all)
            applications: List of apps (default: recommended)
            repetitions: Runs per combination
            force: Ignore checkpoints
            resume: Skip completed runs (default: True)
            
        Returns:
            Summary of all results
        """
        params = self.config_loader.get_experiment_params()
        
        if ablation_ids is None:
            ablation_ids = self.config_loader.get_ablation_ids()
        if applications is None:
            applications = self.config_loader.get_recommended_apps()
        if repetitions is None:
            repetitions = params.repetitions
        
        total_runs = len(ablation_ids) * len(applications) * repetitions
        
        self._log("=" * 70)
        self._log("STARTING ABLATION STUDY")
        self._log("=" * 70)
        self._log(f"Ablations: {len(ablation_ids)}")
        self._log(f"Applications: {len(applications)}")
        self._log(f"Repetitions: {repetitions}")
        self._log(f"Total planned runs: {total_runs}")
        
        if resume and not force:
            remaining = self.checkpoint.get_remaining_runs(
                ablation_ids, applications, repetitions
            )
            self._log(f"Remaining runs: {len(remaining)}")
        
        self._log("=" * 70)
        
        results = {
            'start_time': datetime.now().isoformat(),
            'ablation_ids': ablation_ids,
            'applications': applications,
            'repetitions': repetitions,
            'total_runs': total_runs,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'by_ablation': {}
        }
        
        for ablation_id in ablation_ids:
            ablation_results = self.run_ablation(
                ablation_id,
                applications=applications,
                repetitions=repetitions,
                force=force
            )
            
            results['successful'] += ablation_results['successful']
            results['failed'] += ablation_results['failed']
            results['skipped'] += ablation_results['skipped']
            results['by_ablation'][ablation_id] = ablation_results
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save summary
        summary_file = self.results_dir / f"study_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self._log("=" * 70)
        self._log("ABLATION STUDY COMPLETE")
        self._log(f"Successful: {results['successful']}")
        self._log(f"Failed: {results['failed']}")
        self._log(f"Skipped: {results['skipped']}")
        self._log(f"Summary saved to: {summary_file}")
        self._log("=" * 70)
        
        return results
    
    def show_plan(
        self,
        ablation_ids: Optional[List[str]] = None,
        applications: Optional[List[str]] = None,
        repetitions: Optional[int] = None
    ):
        """Show what would be run without executing."""
        params = self.config_loader.get_experiment_params()
        
        if ablation_ids is None:
            ablation_ids = self.config_loader.get_ablation_ids()
        if applications is None:
            applications = self.config_loader.get_recommended_apps()
        if repetitions is None:
            repetitions = params.repetitions
        
        remaining = self.checkpoint.get_remaining_runs(
            ablation_ids, applications, repetitions
        )
        
        total = len(ablation_ids) * len(applications) * repetitions
        completed = total - len(remaining)
        
        print("\n" + "=" * 70)
        print("ABLATION STUDY PLAN")
        print("=" * 70)
        print(f"\nAblations ({len(ablation_ids)}):")
        for aid in ablation_ids:
            ablation = self.config_loader.get_ablation(aid)
            print(f"  - {aid}: {ablation.description if ablation else 'Unknown'}")
        
        print(f"\nApplications ({len(applications)}):")
        for app in applications:
            print(f"  - {app}")
        
        print(f"\nRepetitions per combination: {repetitions}")
        print(f"\nTotal planned runs: {total}")
        print(f"Already completed: {completed}")
        print(f"Remaining: {len(remaining)}")
        
        if remaining:
            print(f"\nNext 10 runs:")
            for ablation_id, app, run_id in remaining[:10]:
                print(f"  - {ablation_id} / {app} / run_{run_id}")
        
        # Estimate time
        avg_duration = 30  # minutes per run estimate
        estimated_hours = (len(remaining) * avg_duration) / 60
        print(f"\nEstimated time remaining: {estimated_hours:.1f} hours")
        print("=" * 70 + "\n")
