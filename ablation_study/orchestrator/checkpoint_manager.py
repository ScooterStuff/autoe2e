"""
Ablation Checkpoint Manager
===========================

Manages checkpoints for ablation study experiments.
Tracks completed runs and enables resumption.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CheckpointEntry:
    """Entry for a completed ablation run."""
    ablation_id: str
    application: str
    run_id: int
    completed_at: str
    success: bool
    duration_seconds: float
    error: Optional[str] = None


class AblationCheckpointManager:
    """
    Manages checkpoints for ablation study experiments.
    
    Features:
    - Track completed runs
    - Resume from checkpoint
    - Mark runs as completed or failed
    - Get remaining runs
    
    Usage:
        checkpoint = AblationCheckpointManager()
        
        # Check if run is completed
        if not checkpoint.is_completed("A1.1", "petclinic", 1):
            # Run experiment
            ...
            checkpoint.mark_completed("A1.1", "petclinic", 1, success=True)
    """
    
    def __init__(self, checkpoint_file: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint file.
                           Defaults to ablation_study/results/checkpoint.json
        """
        if checkpoint_file is None:
            current = Path(__file__).resolve()
            checkpoint_file = current.parent.parent / "results" / "checkpoint.json"
        
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._completed: Dict[str, CheckpointEntry] = {}
        self._load()
    
    def _get_key(self, ablation_id: str, application: str, run_id: int) -> str:
        """Generate unique key for a run."""
        return f"{ablation_id}:{application}:{run_id}"
    
    def _load(self):
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    for key, entry in data.get('completed', {}).items():
                        self._completed[key] = CheckpointEntry(**entry)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint file: {e}")
                self._completed = {}
    
    def _save(self):
        """Save checkpoint to file."""
        data = {
            'last_updated': datetime.now().isoformat(),
            'completed': {k: asdict(v) for k, v in self._completed.items()}
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_completed(self, ablation_id: str, application: str, run_id: int) -> bool:
        """Check if a run is completed."""
        key = self._get_key(ablation_id, application, run_id)
        return key in self._completed and self._completed[key].success
    
    def mark_completed(
        self,
        ablation_id: str,
        application: str,
        run_id: int,
        success: bool = True,
        duration_seconds: float = 0.0,
        error: Optional[str] = None
    ):
        """Mark a run as completed."""
        key = self._get_key(ablation_id, application, run_id)
        self._completed[key] = CheckpointEntry(
            ablation_id=ablation_id,
            application=application,
            run_id=run_id,
            completed_at=datetime.now().isoformat(),
            success=success,
            duration_seconds=duration_seconds,
            error=error
        )
        self._save()
    
    def get_completed_runs(
        self,
        ablation_id: Optional[str] = None,
        application: Optional[str] = None
    ) -> List[CheckpointEntry]:
        """
        Get list of completed runs, optionally filtered.
        
        Args:
            ablation_id: Filter by ablation ID
            application: Filter by application
            
        Returns:
            List of completed runs
        """
        runs = list(self._completed.values())
        
        if ablation_id:
            runs = [r for r in runs if r.ablation_id == ablation_id]
        if application:
            runs = [r for r in runs if r.application == application]
        
        return runs
    
    def get_remaining_runs(
        self,
        ablation_ids: List[str],
        applications: List[str],
        repetitions: int
    ) -> List[Tuple[str, str, int]]:
        """
        Get list of remaining (not completed) runs.
        
        Args:
            ablation_ids: List of ablation IDs to run
            applications: List of applications to run
            repetitions: Number of repetitions per combination
            
        Returns:
            List of (ablation_id, application, run_id) tuples
        """
        remaining = []
        
        for ablation_id in ablation_ids:
            for app in applications:
                for run_id in range(1, repetitions + 1):
                    if not self.is_completed(ablation_id, app, run_id):
                        remaining.append((ablation_id, app, run_id))
        
        return remaining
    
    def get_statistics(self) -> Dict:
        """Get checkpoint statistics."""
        successful = sum(1 for e in self._completed.values() if e.success)
        failed = sum(1 for e in self._completed.values() if not e.success)
        
        # Get unique ablations and applications
        ablations = set(e.ablation_id for e in self._completed.values())
        applications = set(e.application for e in self._completed.values())
        
        return {
            'total_runs': len(self._completed),
            'successful': successful,
            'failed': failed,
            'unique_ablations': len(ablations),
            'unique_applications': len(applications),
            'ablations': list(ablations),
            'applications': list(applications)
        }
    
    def reset(self):
        """Reset all checkpoints."""
        self._completed = {}
        self._save()
    
    def remove_entry(self, ablation_id: str, application: str, run_id: int):
        """Remove a specific checkpoint entry."""
        key = self._get_key(ablation_id, application, run_id)
        if key in self._completed:
            del self._completed[key]
            self._save()
    
    def print_status(self):
        """Print checkpoint status."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("ABLATION STUDY CHECKPOINT STATUS")
        print("=" * 50)
        print(f"Total runs: {stats['total_runs']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Unique ablations: {stats['unique_ablations']}")
        print(f"Unique applications: {stats['unique_applications']}")
        
        if stats['ablations']:
            print(f"\nCompleted ablations: {', '.join(sorted(stats['ablations']))}")
        if stats['applications']:
            print(f"Completed applications: {', '.join(sorted(stats['applications']))}")
        
        print("=" * 50 + "\n")
