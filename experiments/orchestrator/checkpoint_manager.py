"""
Checkpoint Manager Module
=========================

Manages experiment progress tracking and resumption.
Saves completed runs and allows experiments to be resumed after failures.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import threading


@dataclass
class RunIdentifier:
    """Unique identifier for an experiment run."""
    model_name: str
    app_name: str
    run_id: int
    
    def to_tuple(self) -> Tuple[str, str, int]:
        return (self.model_name, self.app_name, self.run_id)
    
    def to_key(self) -> str:
        return f"{self.model_name}:{self.app_name}:{self.run_id}"
    
    @classmethod
    def from_key(cls, key: str) -> 'RunIdentifier':
        parts = key.split(':')
        return cls(
            model_name=parts[0],
            app_name=parts[1],
            run_id=int(parts[2])
        )


@dataclass
class RunStatus:
    """Status information for a run."""
    identifier: RunIdentifier
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    metrics_file: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.identifier.model_name,
            'app_name': self.identifier.app_name,
            'run_id': self.identifier.run_id,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'error_message': self.error_message,
            'metrics_file': self.metrics_file
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RunStatus':
        return cls(
            identifier=RunIdentifier(
                model_name=data['model_name'],
                app_name=data['app_name'],
                run_id=data['run_id']
            ),
            status=data['status'],
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            error_message=data.get('error_message'),
            metrics_file=data.get('metrics_file')
        )


class CheckpointManager:
    """
    Manages experiment checkpoints for resumption support.
    
    Features:
    - Track completed, running, and failed runs
    - Save progress after each successful run
    - Load checkpoint on startup to skip completed experiments
    - Thread-safe operations
    
    Usage:
        checkpoint = CheckpointManager()
        
        # Check if run is needed
        if checkpoint.should_run(model, app, run_id):
            checkpoint.mark_running(model, app, run_id)
            try:
                # Run experiment
                checkpoint.mark_completed(model, app, run_id, metrics_file)
            except Exception as e:
                checkpoint.mark_failed(model, app, run_id, str(e))
    """
    
    def __init__(self, checkpoint_dir: Optional[str] = None, checkpoint_file: str = "progress.json"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_file: Name of the checkpoint file
        """
        if checkpoint_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            checkpoint_dir = project_root / "experiments" / "results" / "checkpoints"
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / checkpoint_file
        
        self._lock = threading.Lock()
        self._progress: Dict[str, RunStatus] = {}
        self._metadata: Dict[str, Any] = {}
        
        # Load existing checkpoint if available
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint from disk if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._metadata = data.get('metadata', {})
                runs = data.get('runs', {})
                
                for key, run_data in runs.items():
                    self._progress[key] = RunStatus.from_dict(run_data)
                
                print(f"Loaded checkpoint with {len(self._progress)} runs")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                self._progress = {}
                self._metadata = {}
    
    def _save_checkpoint(self):
        """Save checkpoint to disk."""
        data = {
            'metadata': {
                **self._metadata,
                'last_updated': datetime.now().isoformat(),
                'total_runs': len(self._progress)
            },
            'runs': {key: status.to_dict() for key, status in self._progress.items()}
        }
        
        # Write to temporary file first, then rename (atomic operation)
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(self.checkpoint_file)
    
    def _get_key(self, model_name: str, app_name: str, run_id: int) -> str:
        """Generate unique key for a run."""
        return f"{model_name}:{app_name}:{run_id}"
    
    def should_run(self, model_name: str, app_name: str, run_id: int) -> bool:
        """
        Check if a run should be executed.
        
        Returns True if the run has not been completed yet.
        """
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            if key not in self._progress:
                return True
            status = self._progress[key].status
            return status not in ['completed']
    
    def is_completed(self, model_name: str, app_name: str, run_id: int) -> bool:
        """Check if a specific run is completed."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            if key not in self._progress:
                return False
            return self._progress[key].status == 'completed'
    
    def is_running(self, model_name: str, app_name: str, run_id: int) -> bool:
        """Check if a specific run is currently running."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            if key not in self._progress:
                return False
            return self._progress[key].status == 'running'
    
    def mark_running(self, model_name: str, app_name: str, run_id: int):
        """Mark a run as currently running."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            self._progress[key] = RunStatus(
                identifier=RunIdentifier(model_name, app_name, run_id),
                status='running',
                start_time=datetime.now().isoformat()
            )
            self._save_checkpoint()
    
    def mark_completed(
        self, 
        model_name: str, 
        app_name: str, 
        run_id: int,
        metrics_file: Optional[str] = None
    ):
        """Mark a run as completed."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            existing = self._progress.get(key)
            start_time = existing.start_time if existing else None
            
            self._progress[key] = RunStatus(
                identifier=RunIdentifier(model_name, app_name, run_id),
                status='completed',
                start_time=start_time,
                end_time=datetime.now().isoformat(),
                metrics_file=metrics_file
            )
            self._save_checkpoint()
    
    def mark_failed(
        self, 
        model_name: str, 
        app_name: str, 
        run_id: int,
        error_message: str
    ):
        """Mark a run as failed."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            existing = self._progress.get(key)
            start_time = existing.start_time if existing else None
            
            self._progress[key] = RunStatus(
                identifier=RunIdentifier(model_name, app_name, run_id),
                status='failed',
                start_time=start_time,
                end_time=datetime.now().isoformat(),
                error_message=error_message
            )
            self._save_checkpoint()
    
    def get_status(self, model_name: str, app_name: str, run_id: int) -> Optional[RunStatus]:
        """Get status for a specific run."""
        with self._lock:
            key = self._get_key(model_name, app_name, run_id)
            return self._progress.get(key)
    
    def get_completed_runs(self) -> List[RunIdentifier]:
        """Get all completed runs."""
        with self._lock:
            return [
                status.identifier 
                for status in self._progress.values() 
                if status.status == 'completed'
            ]
    
    def get_failed_runs(self) -> List[Tuple[RunIdentifier, str]]:
        """Get all failed runs with their error messages."""
        with self._lock:
            return [
                (status.identifier, status.error_message or "Unknown error")
                for status in self._progress.values() 
                if status.status == 'failed'
            ]
    
    def get_pending_runs(
        self, 
        models: List[str], 
        apps: List[str], 
        repetitions: int
    ) -> List[RunIdentifier]:
        """
        Get list of runs that still need to be executed.
        
        Args:
            models: List of model names
            apps: List of application names
            repetitions: Number of repetitions per model-app pair
            
        Returns:
            List of RunIdentifiers for runs that are not completed
        """
        pending = []
        with self._lock:
            for model in models:
                for app in apps:
                    for run_id in range(1, repetitions + 1):
                        key = self._get_key(model, app, run_id)
                        if key not in self._progress or self._progress[key].status != 'completed':
                            pending.append(RunIdentifier(model, app, run_id))
        return pending
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of experiment progress."""
        with self._lock:
            completed = sum(1 for s in self._progress.values() if s.status == 'completed')
            failed = sum(1 for s in self._progress.values() if s.status == 'failed')
            running = sum(1 for s in self._progress.values() if s.status == 'running')
            pending = sum(1 for s in self._progress.values() if s.status == 'pending')
            
            return {
                'total_tracked': len(self._progress),
                'completed': completed,
                'failed': failed,
                'running': running,
                'pending': pending,
                'last_updated': self._metadata.get('last_updated')
            }
    
    def reset(self):
        """Reset all checkpoint data."""
        with self._lock:
            self._progress = {}
            self._metadata = {}
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            print("Checkpoint data reset")
    
    def set_experiment_metadata(self, **kwargs):
        """Set metadata about the experiment."""
        with self._lock:
            self._metadata.update(kwargs)
            self._save_checkpoint()
    
    def print_status(self):
        """Print current checkpoint status."""
        summary = self.get_progress_summary()
        
        print("\n" + "=" * 50)
        print("CHECKPOINT STATUS")
        print("=" * 50)
        print(f"  Completed: {summary['completed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Running: {summary['running']}")
        print(f"  Pending: {summary['pending']}")
        print(f"  Last Updated: {summary['last_updated']}")
        print("=" * 50 + "\n")
        
        # Print failed runs if any
        failed_runs = self.get_failed_runs()
        if failed_runs:
            print("Failed runs:")
            for run, error in failed_runs[:10]:  # Show first 10
                print(f"  - {run.model_name}/{run.app_name}/run_{run.run_id}: {error[:50]}...")


if __name__ == "__main__":
    # Test the checkpoint manager
    manager = CheckpointManager()
    
    # Test operations
    manager.mark_running("qwen3:8b", "petclinic", 1)
    manager.mark_completed("qwen3:8b", "petclinic", 1, "metrics/run1.json")
    manager.mark_failed("qwen3:8b", "petclinic", 2, "Connection timeout")
    
    # Print status
    manager.print_status()
    
    # Check should_run
    print(f"Should run qwen3:8b/petclinic/1: {manager.should_run('qwen3:8b', 'petclinic', 1)}")
    print(f"Should run qwen3:8b/petclinic/2: {manager.should_run('qwen3:8b', 'petclinic', 2)}")
    print(f"Should run qwen3:8b/petclinic/3: {manager.should_run('qwen3:8b', 'petclinic', 3)}")
