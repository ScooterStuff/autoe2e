"""
Experiment Runner Module
========================

Main orchestration logic for running AutoE2E experiments.
Handles the complete lifecycle of experiment execution including:
- Environment setup
- Docker container management
- Browser initialization
- AutoE2E exploration and inference
- Result collection
- Cleanup
"""

import os
import sys
import time
import json
import subprocess
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from .config_loader import ConfigLoader, ModelConfig, ApplicationConfig, ExperimentParams
from .checkpoint_manager import CheckpointManager
from .result_collector import ResultCollector, ExperimentMetrics


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    success: bool
    metrics: Optional[ExperimentMetrics]
    error_message: Optional[str]
    logs_dir: Optional[str]
    duration_seconds: float


class ExperimentRunner:
    """
    Orchestrates experiment execution for AutoE2E replication study.
    
    Usage:
        runner = ExperimentRunner()
        
        # Run single experiment
        result = runner.run_experiment(model_config, app_config, run_id=1)
        
        # Run all experiments
        runner.run_all_experiments()
    """
    
    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        result_collector: Optional[ResultCollector] = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            config_loader: Configuration loader (uses default if None)
            checkpoint_manager: Checkpoint manager (uses default if None)
            result_collector: Result collector (uses default if None)
        """
        self.config = config_loader or ConfigLoader()
        self.checkpoint = checkpoint_manager or CheckpointManager()
        self.results = result_collector or ResultCollector()
        
        self.params = self.config.get_experiment_params()
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Track running processes for cleanup
        self._docker_process: Optional[subprocess.Popen] = None
        self._current_run: Optional[Tuple[str, str, int]] = None
    
    def setup_environment(self, model: ModelConfig, app: ApplicationConfig) -> Dict[str, str]:
        """
        Set up environment variables for the experiment.
        
        Args:
            model: Model configuration
            app: Application configuration
            
        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        
        # Provider-aware model settings
        if model.provider == 'openai':
            env['LLM_PROVIDER'] = 'openai'
            env['OPENAI_MODEL'] = model.ollama_model
            env['OPENAI_TEMPERATURE'] = str(self.params.inference.temperature)
            # OPENAI_API_KEY is read from os.environ (already in env copy)
        else:
            env['LLM_PROVIDER'] = 'ollama'
            env['OLLAMA_MODEL'] = model.ollama_model
            env['OLLAMA_EMBEDDING_MODEL'] = self.config.get_embedding_model()
            env['OLLAMA_TEMPERATURE'] = str(self.params.inference.temperature)
            env['OLLAMA_TIMEOUT'] = str(self.params.execution.query_timeout_seconds)
            env['OLLAMA_NUM_PREDICT'] = '4096'
        
        # Seed for reproducibility (model-specific overrides global)
        seed = model.seed if model.seed is not None else self.params.inference.seed
        if seed is not None:
            env['OLLAMA_SEED'] = str(seed)
        
        # Application settings
        env['APP_NAME'] = app.config_name
        
        # Exploration limits
        env['MAX_RUNTIME_MINUTES'] = str(self.params.exploration.timeout_minutes)
        env['MAX_TOTAL_STATES'] = str(self.params.exploration.max_states)
        
        return env
    
    def check_ollama_status(self, models_to_check: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Check if Ollama is running and required models are available.
        
        Args:
            models_to_check: Specific model names to check. If None, only checks embedding model.
        
        Returns:
            Tuple of (is_ready, missing_models)
        """
        import requests
        
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return False, ["Ollama not responding"]
            
            available = [m['name'] for m in response.json().get('models', [])]
            
            missing = []
            
            # Only check specified models (not all configured models)
            if models_to_check:
                for model_name in models_to_check:
                    model_base = model_name.split(':')[0]
                    found = any(model_base in avail for avail in available)
                    if not found:
                        missing.append(model_name)
            
            # Check embedding model
            embedding_model = self.config.get_embedding_model()
            embed_base = embedding_model.split(':')[0]
            if not any(embed_base in avail for avail in available):
                missing.append(embedding_model)
            
            return len(missing) == 0, missing
            
        except requests.exceptions.ConnectionError:
            return False, ["Cannot connect to Ollama"]
        except Exception as e:
            return False, [str(e)]
    
    def check_application_health(
        self, 
        app: ApplicationConfig,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Check if application is running and healthy.
        
        Args:
            app: Application configuration
            timeout: Health check timeout (uses app config if None)
            
        Returns:
            True if application is healthy
        """
        import requests
        
        timeout = timeout or app.health_check_timeout_seconds
        url = f"{app.url.rstrip('/')}{app.health_check_endpoint}"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:
                    return True
            except:
                pass
            time.sleep(2)
        
        return False
    
    def start_docker_containers(self, app: ApplicationConfig) -> bool:
        """
        Start Docker containers for the application.
        
        Args:
            app: Application configuration
            
        Returns:
            True if containers started successfully
        """
        docker_compose_path = self.project_root / app.docker_compose
        
        # Handle PetClinic separately - uses simple docker run, not docker-compose
        if app.name == 'petclinic':
            return self._start_petclinic_container(app)
        
        if not docker_compose_path.exists():
            print(f"Warning: Docker compose file not found: {docker_compose_path}")
            print("Assuming containers are already running...")
            return True
        
        try:
            # For MantisBT, don't restart containers to preserve installation
            # Just ensure they're running
            if app.name == 'mantisbt':
                print(f"MantisBT: Checking if containers are running (preserving state)...")
                result = subprocess.run(
                    ['docker-compose', '-f', str(docker_compose_path), 'ps', '-q'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if not result.stdout.strip():
                    print("MantisBT containers not running, starting them...")
                    subprocess.run(
                        ['docker-compose', '-f', str(docker_compose_path), 'up', '-d'],
                        capture_output=True,
                        timeout=120
                    )
            else:
                # For other apps, do full restart
                # Stop any existing containers first
                subprocess.run(
                    ['docker-compose', '-f', str(docker_compose_path), 'down'],
                    capture_output=True,
                    timeout=60
                )
                
                # Start containers
                self._docker_process = subprocess.Popen(
                    ['docker-compose', '-f', str(docker_compose_path), 'up', '-d'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Wait for health check
            print(f"Waiting for {app.name} to be healthy...")
            if self.check_application_health(app):
                print(f"{app.name} is ready!")
                return True
            else:
                print(f"Health check failed for {app.name}")
                return False
                
        except Exception as e:
            print(f"Error starting Docker containers: {e}")
            return False
    
    def _start_petclinic_container(self, app: ApplicationConfig) -> bool:
        """
        Start the PetClinic container (uses simple docker run, not docker-compose).
        
        Args:
            app: Application configuration
            
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
                    print(f"PetClinic container already running")
                else:
                    # Container exists but stopped, start it
                    print(f"Starting existing PetClinic container...")
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
                    print(f"Building PetClinic image...")
                    dockerfile_dir = self.project_root / 'benchmark' / 'pet-clinic' / 'spring-petclinic-angular'
                    subprocess.run(
                        ['docker', 'build', '-t', image_name, '.'],
                        cwd=str(dockerfile_dir),
                        capture_output=True,
                        timeout=900
                    )
                
                # Run new container
                print(f"Creating and starting PetClinic container...")
                subprocess.run(
                    ['docker', 'run', '-d', '-p', '8080:8080', f'--name={container_name}', image_name],
                    capture_output=True,
                    timeout=60
                )
            
            # Wait for health check
            print(f"Waiting for PetClinic to be healthy...")
            if self.check_application_health(app):
                print(f"PetClinic is ready!")
                return True
            else:
                print(f"Health check failed for PetClinic")
                return False
                
        except Exception as e:
            print(f"Error starting PetClinic container: {e}")
            return False
    
    def stop_docker_containers(self, app: ApplicationConfig):
        """Stop Docker containers for the application."""
        # Handle PetClinic separately
        if app.name == 'petclinic':
            try:
                subprocess.run(
                    ['docker', 'stop', 'petclinic'],
                    capture_output=True,
                    timeout=60
                )
                subprocess.run(
                    ['docker', 'rm', 'petclinic'],
                    capture_output=True,
                    timeout=30
                )
            except Exception as e:
                print(f"Warning: Error stopping PetClinic container: {e}")
            return
            
        docker_compose_path = self.project_root / app.docker_compose
        
        if docker_compose_path.exists():
            try:
                subprocess.run(
                    ['docker-compose', '-f', str(docker_compose_path), 'down'],
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:
                print(f"Warning: Error stopping containers: {e}")
    
    def clear_mongodb_collections(self, app_name: str):
        """
        Clear MongoDB collections for the application.
        
        This ensures each run starts fresh.
        """
        try:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            
            load_dotenv()
            
            client = MongoClient(os.getenv("ATLAS_URI"))
            db = client.myDatabase
            
            # Clear app-specific data
            db["action-functionality"].delete_many({'app': app_name})
            db["functionality"].delete_many({'app': app_name})
            
            print(f"Cleared MongoDB collections for {app_name}")
            
        except Exception as e:
            print(f"Warning: Could not clear MongoDB: {e}")
    
    def run_main_script(
        self,
        env: Dict[str, str],
        timeout_minutes: int,
        verbose: bool = False
    ) -> Tuple[bool, str, Dict]:
        """
        Run the main AutoE2E script.
        
        Args:
            env: Environment variables
            timeout_minutes: Maximum runtime
            verbose: Show real-time output (default: False)
            
        Returns:
            Tuple of (success, error_message, run_summary)
        """
        main_script = self.project_root / "main.py"
        timeout_seconds = timeout_minutes * 60
        
        try:
            # Run main.py with unbuffered output if verbose
            env_to_use = env.copy()
            if verbose:
                env_to_use['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
            
            # Use -u flag only if verbose
            cmd = [sys.executable, '-u', str(main_script)] if verbose else [sys.executable, str(main_script)]
            
            process = subprocess.Popen(
                cmd,
                env=env_to_use,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1 if verbose else -1  # Line buffered if verbose
            )
            
            # Read output with timeout-aware approach using threads
            import threading
            import queue
            
            output_lines = []
            output_queue = queue.Queue()
            
            def reader_thread(pipe, q):
                """Thread to read from pipe without blocking main loop."""
                try:
                    for line in iter(pipe.readline, ''):
                        q.put(line)
                    pipe.close()
                except:
                    pass
            
            reader = threading.Thread(target=reader_thread, args=(process.stdout, output_queue))
            reader.daemon = True
            reader.start()
            
            start_time = time.time()
            
            while True:
                # Check timeout FIRST
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"\n⏰ TIMEOUT: {timeout_minutes} minutes elapsed, killing process...")
                    process.kill()
                    process.wait()  # Ensure process is dead
                    return False, f"Timeout after {timeout_minutes} minutes", {}
                
                # Check if process finished
                if process.poll() is not None:
                    # Drain remaining output
                    while not output_queue.empty():
                        try:
                            line = output_queue.get_nowait()
                            output_lines.append(line.strip())
                            if verbose:
                                print(line, end='', flush=True)
                        except queue.Empty:
                            break
                    break
                
                # Read available output (non-blocking)
                try:
                    line = output_queue.get(timeout=1.0)  # Wait max 1 second
                    output_lines.append(line.strip())
                    if verbose:
                        print(line, end='', flush=True)
                except queue.Empty:
                    pass  # No output available, loop again to check timeout
            
            # Read any remaining output
            remaining = process.stdout.read()
            if remaining:
                if verbose:
                    print(remaining, end='', flush=True)
                output_lines.extend(remaining.strip().split('\n'))
            
            # Check exit code
            if process.returncode != 0:
                return False, f"Process exited with code {process.returncode}", {}
            
            # Load run summary
            app_name = env.get('APP_NAME', 'UNKNOWN')
            report_dir = self.project_root / "report"
            
            # Find the latest run summary
            run_summary = {}
            summary_files = list(report_dir.glob(f"{app_name}_run_summary_*.json"))
            if summary_files:
                latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
                with open(latest_summary, 'r') as f:
                    run_summary = json.load(f)
            
            return True, "", run_summary
            
        except Exception as e:
            return False, str(e), {}
    
    def collect_metrics_from_run(
        self,
        model: ModelConfig,
        app: ApplicationConfig,
        run_id: int,
        run_summary: Dict,
        duration_seconds: float
    ) -> ExperimentMetrics:
        """
        Collect metrics from a completed run.
        
        Combines run summary with evaluation results.
        """
        metrics = ExperimentMetrics(
            model_name=model.name,
            application_name=app.name,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            total_features=app.feature_count
        )
        
        # Extract from run summary
        if run_summary:
            crawl_stats = run_summary.get('crawl_stats', {})
            llm_stats = run_summary.get('llm_stats', {})
            duration = run_summary.get('duration', {})
            
            metrics.states_explored = crawl_stats.get('total_states', 0)
            metrics.actions_processed = crawl_stats.get('total_actions_processed', 0)
            metrics.completed = crawl_stats.get('completed', False)
            metrics.stopped_by_time_limit = crawl_stats.get('stopped_by_time_limit', False)
            metrics.stopped_by_state_limit = crawl_stats.get('stopped_by_state_limit', False)
            metrics.error = crawl_stats.get('error')
            
            metrics.llm_queries = llm_stats.get('llm_calls', 0)
            metrics.total_tokens = llm_stats.get('estimated_total_tokens', 0)
            metrics.total_time_seconds = duration.get('total_seconds', duration_seconds)
        else:
            metrics.total_time_seconds = duration_seconds
        
        return metrics
    
    def run_post_experiment_evaluation(
        self,
        model: ModelConfig,
        app: ApplicationConfig,
        run_id: int,
        verbose: bool = False
    ) -> Optional[Dict]:
        """
        Run post-experiment evaluation to calculate feature coverage metrics.
        
        Args:
            model: Model configuration
            app: Application configuration
            run_id: Run identifier
            verbose: Show detailed output
            
        Returns:
            Evaluation results dictionary, or None if evaluation failed
        """
        try:
            from experiments.scripts.post_experiment_evaluation import PostExperimentEvaluator
            
            print("\n" + "=" * 70)
            print("RUNNING POST-EXPERIMENT EVALUATION")
            print("=" * 70)
            
            evaluator = PostExperimentEvaluator()
            result = evaluator.run_full_evaluation(
                model_name=model.name,
                app_name=app.name,
                run_id=run_id,
                verbose=verbose
            )
            
            return result.to_dict() if result else None
            
        except Exception as e:
            print(f"\n⚠️ Post-experiment evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_experiment(
        self,
        model: ModelConfig,
        app: ApplicationConfig,
        run_id: int = 1,
        skip_docker: bool = False,
        verbose: bool = False
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            model: Model configuration
            app: Application configuration
            run_id: Run identifier (1-based)
            skip_docker: Skip Docker container management
            verbose: Show real-time output (default: False)
            
        Returns:
            ExperimentResult with metrics and status
        """
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {model.name} / {app.name} / Run {run_id}")
        print("=" * 70)
        
        self._current_run = (model.name, app.name, run_id)
        start_time = time.time()
        
        try:
            # Mark as running
            self.checkpoint.mark_running(model.name, app.name, run_id)
            
            # Setup environment
            env = self.setup_environment(model, app)
            
            # Clear previous data
            self.clear_mongodb_collections(app.config_name)
            
            # Start Docker if needed
            if not skip_docker:
                if not self.start_docker_containers(app):
                    raise RuntimeError("Failed to start Docker containers")
            
            # Run main script
            success, error, run_summary = self.run_main_script(
                env,
                self.params.exploration.timeout_minutes,
                verbose=verbose
            )
            
            if not success:
                raise RuntimeError(error)
            
            # Collect metrics
            duration = time.time() - start_time
            metrics = self.collect_metrics_from_run(
                model, app, run_id, run_summary, duration
            )
            
            # Save initial metrics
            metrics_file = self.results.save_metrics(metrics)
            
            # Run post-experiment evaluation BEFORE marking as complete
            print("\n" + "-" * 70)
            print("Running post-experiment evaluation...")
            print("-" * 70)
            eval_result = self.run_post_experiment_evaluation(
                model, app, run_id, verbose=verbose
            )
            
            # Update metrics with evaluation results if available
            if eval_result:
                metrics.feature_coverage = eval_result.get('feature_coverage', 0.0)
                metrics.total_features_covered = eval_result.get('covered', 0)
                metrics.precision = eval_result.get('precision', 0.0)
                metrics.recall = eval_result.get('recall', 0.0)
                metrics.f1_score = eval_result.get('f1', 0.0)
                metrics.inferred_features = eval_result.get('total_generated', 0)
                metrics.correct_inferences = eval_result.get('correct', 0)
                
                # Re-save metrics with evaluation data
                metrics_file = self.results.save_metrics(metrics)
            
            # Mark as completed only AFTER evaluation is done
            self.checkpoint.mark_completed(model.name, app.name, run_id, metrics_file)
            
            print(f"\n✓ Experiment completed successfully in {duration:.1f}s")
            
            return ExperimentResult(
                success=True,
                metrics=metrics,
                error_message=None,
                logs_dir=str(self.project_root / "autoe2e" / "logs"),
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            print(f"\n✗ Experiment failed: {error_msg}")
            traceback.print_exc()
            
            # Mark as failed
            self.checkpoint.mark_failed(model.name, app.name, run_id, error_msg)
            
            return ExperimentResult(
                success=False,
                metrics=None,
                error_message=error_msg,
                logs_dir=str(self.project_root / "autoe2e" / "logs"),
                duration_seconds=duration
            )
        
        finally:
            self._current_run = None
            
            # Cleanup
            if not skip_docker:
                self.stop_docker_containers(app)
    
    def run_all_experiments(
        self,
        models: Optional[List[str]] = None,
        apps: Optional[List[str]] = None,
        resume: bool = True,
        dry_run: bool = False,
        verbose: bool = False
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run all experiments across models and applications.
        
        Args:
            models: List of model names (all if None)
            apps: List of application names (all if None)
            resume: Skip completed runs (uses checkpoint)
            dry_run: Only show what would be run
            verbose: Show real-time output (default: False)
            
        Returns:
            Dictionary mapping "model:app" to list of results
        """
        # Get configurations
        model_configs = self.config.get_models()
        app_configs = self.config.get_applications()
        repetitions = self.params.execution.repetitions
        
        # Filter if specified
        if models:
            model_configs = [m for m in model_configs if m.name in models]
        if apps:
            app_configs = [a for a in app_configs if a.name in apps]
        
        # Calculate total runs
        total_runs = len(model_configs) * len(app_configs) * repetitions
        
        print("\n" + "=" * 70)
        print("EXPERIMENT PLAN")
        print("=" * 70)
        print(f"Models: {[m.name for m in model_configs]}")
        print(f"Applications: {[a.name for a in app_configs]}")
        print(f"Repetitions: {repetitions}")
        print(f"Total planned runs: {total_runs}")
        
        # Get pending runs
        pending = self.checkpoint.get_pending_runs(
            [m.name for m in model_configs],
            [a.name for a in app_configs],
            repetitions
        )
        
        print(f"Pending runs: {len(pending)}")
        
        if resume:
            completed = self.checkpoint.get_completed_runs()
            print(f"Skipping {len(completed)} completed runs")
        
        if dry_run:
            print("\nDRY RUN - would execute:")
            for run in pending[:20]:  # Show first 20
                print(f"  - {run.model_name} / {run.app_name} / run_{run.run_id}")
            if len(pending) > 20:
                print(f"  ... and {len(pending) - 20} more")
            return {}
        
        print("=" * 70 + "\n")
        
        # Check Ollama status - only check ollama-provider models that will actually be used
        ollama_models = [m.ollama_model for m in model_configs if m.provider != 'openai']
        if ollama_models:
            ollama_ready, missing = self.check_ollama_status(ollama_models)
            if not ollama_ready:
                print("ERROR: Ollama is not ready")
                print(f"Issues: {missing}")
                print("\nTo fix:")
                print("1. Start Ollama: ollama serve")
                print("2. Pull missing models:")
                for m in missing:
                    print(f"   ollama pull {m}")
                return {}
        
        # Run experiments
        results: Dict[str, List[ExperimentResult]] = {}
        
        for model in model_configs:
            for app in app_configs:
                key = f"{model.name}:{app.name}"
                results[key] = []
                
                for run_id in range(1, repetitions + 1):
                    # Check if should run
                    if resume and self.checkpoint.is_completed(model.name, app.name, run_id):
                        print(f"Skipping completed: {model.name}/{app.name}/run_{run_id}")
                        continue
                    
                    # Run experiment
                    result = self.run_experiment(model, app, run_id, verbose=verbose)
                    results[key].append(result)
                    
                    # Delay between runs
                    if self.params.execution.inter_run_delay_seconds > 0:
                        print(f"Waiting {self.params.execution.inter_run_delay_seconds}s before next run...")
                        time.sleep(self.params.execution.inter_run_delay_seconds)
        
        # Generate summary reports
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY REPORTS")
        print("=" * 70)
        
        self.results.generate_summary_csv()
        self.results.generate_aggregated_json()
        
        # Print final summary
        self.checkpoint.print_status()
        
        return results


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\nInterrupt received, cleaning up...")
    sys.exit(1)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Test the experiment runner
    runner = ExperimentRunner()
    
    # Print configuration summary
    runner.config.print_summary()
    
    # Check Ollama
    ready, missing = runner.check_ollama_status()
    print(f"Ollama ready: {ready}")
    if missing:
        print(f"Missing models: {missing}")
