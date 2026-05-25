# AutoE2E Ablation Study Framework

A comprehensive ablation study framework for evaluating the contribution of each component in the AutoE2E automated end-to-end testing system.

## Overview

This framework enables systematic evaluation of **30 ablations** across 7 components:

| Component                | # Ablations | Description                                                                   |
| ------------------------ | ----------- | ----------------------------------------------------------------------------- |
| C1 - Context Extraction  | 4           | What context is included in prompts (screenshot, prev state, prev action)     |
| C2 - Prompting Strategy  | 3           | SINGLE/DOUBLE prompt strategy (dual, single_action, action_pair, merged)      |
| C3 - Scoring Function    | 9           | Scoring method (geometric/uniform/linear/binary) + p parameter + R candidates |
| C4 - Feature Aggregation | 4           | How similar features are merged (semantic, embedding_only, exact_match, none) |
| C5 - Score Accumulation  | 3           | How scores combine (differential, simple_sum, maximum, final_only)            |
| C6 - Finality Detection  | 3           | When features are marked final (llm, heuristic, all_final, disabled)          |
| C7 - Score Threshold     | 3           | Filtering features by accumulated score cutoff                                |

**Total: 30 ablations + 1 baseline = 31 configurations**

## Quick Start

```bash
# 1. View available ablations
python -m ablation_study.scripts.run_ablation_study --list-ablations

# 2. View execution plan
python -m ablation_study.scripts.run_ablation_study --show-plan

# 3. Run a single ablation for testing
python -m ablation_study.scripts.run_single_ablation baseline petclinic --dry-run

# 4. Run the full ablation study
python -m ablation_study.scripts.run_ablation_study

# 5. Analyze results
python -m ablation_study.scripts.analyze_ablations
```

## Directory Structure

```
ablation_study/
├── __init__.py
├── README.md                   # This file
├── INTEGRATION_GUIDE.md        # How to integrate with AutoE2E
├── config/
│   ├── ablations.yaml          # All 31 ablation definitions
│   ├── model.yaml              # LLM model configuration
│   ├── applications.yaml       # Benchmark applications
│   └── experiment_params.yaml  # Experiment parameters
├── components/
│   ├── __init__.py
│   ├── context_extractor.py    # C1: Context extraction
│   ├── prompt_manager.py       # C2: Prompting strategy
│   ├── scoring_function.py     # C3: Scoring function
│   ├── score_accumulator.py    # C4: Score accumulation
│   └── score_threshold.py      # C5: Score threshold
├── orchestrator/
│   ├── __init__.py
│   ├── config_loader.py        # Load YAML configurations
│   ├── component_factory.py    # Create components from config
│   ├── ablation_runner.py      # Main experiment orchestrator
│   ├── checkpoint_manager.py   # Track progress/resume
│   └── metrics_collector.py    # Collect and analyze metrics
├── scripts/
│   ├── __init__.py
│   ├── run_ablation_study.py   # Main entry point
│   ├── run_single_ablation.py  # Run individual ablation
│   └── analyze_ablations.py    # Analyze results
└── results/
    └── README.md               # Results directory info
```

## Usage

### Command Line Interface

#### Run Full Study

```bash
# All ablations on recommended apps (3 runs each)
python -m ablation_study.scripts.run_ablation_study

# Specific ablations
python -m ablation_study.scripts.run_ablation_study -A baseline A1.1 A1.2

# Specific applications
python -m ablation_study.scripts.run_ablation_study -a petclinic realworld

# Custom repetitions
python -m ablation_study.scripts.run_ablation_study -R 5
```

#### Run Single Experiment

```bash
# Run baseline on petclinic
python -m ablation_study.scripts.run_single_ablation baseline petclinic

# With specific run ID
python -m ablation_study.scripts.run_single_ablation A1.1 realworld --run-id 2

# Show configuration only
python -m ablation_study.scripts.run_single_ablation A1.1 petclinic --show-config
```

#### Analyze Results

```bash
# Summary table
python -m ablation_study.scripts.analyze_ablations

# Compare against baseline
python -m ablation_study.scripts.analyze_ablations --compare A1.1 A1.2 A1.3

# Statistical significance
python -m ablation_study.scripts.analyze_ablations --significance

# Export formats
python -m ablation_study.scripts.analyze_ablations --latex
python -m ablation_study.scripts.analyze_ablations --csv results.csv
python -m ablation_study.scripts.analyze_ablations --json analysis.json
```

### Programmatic Usage

```python
from ablation_study.orchestrator import (
    AblationConfigLoader,
    AblationRunner,
    ComponentFactory,
    AblationMetricsCollector
)

# Load configurations
loader = AblationConfigLoader()
loader.print_summary()

# Get specific ablation
ablation = loader.get_ablation("A1.1")
print(f"Description: {ablation.description}")

# Create components
components = ComponentFactory.create_all(ablation)
print(ComponentFactory.get_component_summary(components))

# Run experiments
runner = AblationRunner(dry_run=True)
runner.show_plan()

# Analyze results
collector = AblationMetricsCollector()
collector.print_summary()
```

## Ablation Details

### C1 - Context Extraction (4 ablations)

| ID   | Name               | Description                        |
| ---- | ------------------ | ---------------------------------- |
| A1.1 | No Screenshot      | Text-only HTML, no visual context  |
| A1.2 | No Previous State  | Remove previous state from prompt  |
| A1.3 | No Previous Action | Remove previous action from prompt |
| A1.4 | Minimal Context    | No screenshot + no history         |

### C2 - Prompting Strategy (3 ablations)

| ID   | Name        | Description                           |
| ---- | ----------- | ------------------------------------- |
| A2.1 | Single Only | Only SINGLE action type extraction    |
| A2.2 | Pair Only   | Only DOUBLE action type extraction    |
| A2.3 | Merged      | Single prompt combining both contexts |

### C3 - Scoring Function (9 ablations)

| ID      | Name           | Description                         |
| ------- | -------------- | ----------------------------------- |
| A3.1    | Uniform        | 1/R for all top-R features          |
| A3.2    | Linear         | (R-rank+1)/R decay                  |
| A3.3    | Binary         | 1 if rank=1 else 0                  |
| A3.4a-d | Geometric p    | p=0.2, 0.3, 0.7, 0.8 (baseline=0.5) |
| A3.5a-c | Max Candidates | R=3, 5, 15 (baseline=10)            |

Baseline uses log-probability formula: `(rank-1)*log(1-p) + log(p)`

### C4 - Score Accumulation (3 ablations)

| ID   | Name       | Description                 |
| ---- | ---------- | --------------------------- |
| A5.1 | Final Only | Last observation score only |
| A5.2 | Simple Sum | Σ scores (no differential)  |
| A5.3 | Maximum    | max(scores) across chain    |

Baseline uses differential: `curr_score - prev_score`

### C5 - Score Threshold (3 ablations)

| ID   | Name         | Threshold    |
| ---- | ------------ | ------------ |
| A7.1 | Non-negative | score >= 0   |
| A7.2 | Moderate     | score >= 1.0 |
| A7.3 | High         | score >= 2.0 |

Filters final features by accumulated differential score. Higher scores indicate stronger evidence across observations.

## Metrics Collected

For each experiment run:

- **Feature Coverage**: % of ground truth features detected
- **Precision**: Correct features / Total detected
- **Recall**: Correct features / Ground truth features
- **F1 Score**: Harmonic mean of precision and recall
- **States Explored**: Number of unique states visited
- **Duration**: Total experiment time

## Resuming Experiments

The framework automatically saves progress:

```bash
# Resume from checkpoint (default behavior)
python -m ablation_study.scripts.run_ablation_study

# Force re-run all
python -m ablation_study.scripts.run_ablation_study --force

# Check status
python -m ablation_study.scripts.run_ablation_study --status
```

## Configuration

### Modify Ablations

Edit `config/ablations.yaml` to add or modify ablations:

```yaml
A1.5_custom:
  id: "A1.5"
  description: "Custom context configuration"
  parent: baseline
  overrides:
    context:
      include_screenshot: true
      include_previous_state: false
```

### Change Applications

Edit `config/applications.yaml`:

```yaml
applications:
  - name: myapp
    config_name: MYAPP
    url: http://localhost:8080
    feature_count: 25
    # ...
```

### Adjust Parameters

Edit `config/experiment_params.yaml`:

```yaml
execution:
  repetitions: 5 # More runs for higher confidence
  retry_on_failure: 3
```

## Integration

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed instructions on integrating this framework with the main AutoE2E codebase.

## Requirements

- Python 3.8+
- PyYAML
- scipy (for statistical tests)
- All AutoE2E dependencies

Install:

```bash
pip install pyyaml scipy
```

## License

Part of the AutoE2E research project.
