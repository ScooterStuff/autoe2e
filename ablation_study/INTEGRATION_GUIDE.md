# Ablation Study Integration Guide

This guide explains how to integrate the ablation study framework with the main AutoE2E codebase.

## Overview

The ablation study framework provides configurable components that modify the behavior of AutoE2E.
Integration requires modifying a few key files to read ablation configurations and use the
configurable components.

## Current Codebase Architecture

Based on the current implementation:

### Key Files

- **main.py**: Main crawl loop, state processing, action extraction
- **autoe2e/infer_utils.py**: Core inference functions
  - `extract_state_context()`: Vision-based state understanding
  - `extract_action_functionalities()`: Feature inference (SINGLE/DOUBLE types)
  - `insert_functionalities()`: Database storage with embeddings
  - `update_functionality_score()`: Differential score accumulation
  - `mark_final_functionalities()`: LLM-based finality detection
- **autoe2e/prompts.py**: All prompt templates
- **autoe2e/llm_api_call.py**: Ollama LLM chains (sonnet_chain, haiku_chain)
- **autoe2e/utils/**init**.py**: `geometric_score(rank, p=0.5, R=10)`

### Current Behavior (Baseline)

1. **Context Extraction**: Uses screenshot + previous state + previous action
2. **Prompting**: Dual strategy (SINGLE + DOUBLE action types)
3. **Scoring**: Geometric log-probability: `(rank-1)*log(1-p) + log(p)` with p=0.5, R=10
4. **Aggregation**: Embedding similarity + LLM validation via `map_similar_func_to_exact_match()`
5. **Accumulation**: Differential scoring in `update_functionality_score()`
6. **Finality**: LLM-based detection in `mark_final_functionalities()`

## Integration Points

### 1. main.py - Entry Point Modification

Add command-line argument for ablation configuration:

```python
# Add to imports
import argparse

# In argument parsing, add:
parser = argparse.ArgumentParser()
parser.add_argument('--ablation-config', type=str, default=None,
                    help='Path to ablation configuration JSON file')
args = parser.parse_args()
```

Load ablation configuration when starting:

```python
# After loading main config, add:
from ablation_study.orchestrator import ComponentFactory
from ablation_study.orchestrator.config_loader import AblationConfig

ablation_components = None
if args.ablation_config:
    with open(args.ablation_config, 'r') as f:
        ablation_config = json.load(f)
    ablation_components = ComponentFactory.create_from_dict(ablation_config['components'])
    print(f"Ablation mode: {ablation_config['ablation_id']}")
```

### 2. autoe2e/infer_utils.py - Core Modifications

#### Context Extraction (extract_state_context)

Current signature:

```python
def extract_state_context(
    crawl_context: CrawlContext,
    state: State,
    prev_state: State | None = None,
    prev_action: Action | None = None
) -> str:
```

Modified version with ablation support:

```python
def extract_state_context(
    crawl_context: CrawlContext,
    state: State,
    prev_state: State | None = None,
    prev_action: Action | None = None,
    context_extractor = None  # NEW: ablation component
) -> str:
    """Extract context from application state using vision model."""
    state_id = state.get_id(StateIdEvaluator.BY_ACTIONS)

    screenshot_path = f'{crawl_context.config.temp_dir}/screenshot_{state_id}.png'
    save_screenshot(crawl_context.driver, screenshot_path)

    # Prepare context inputs based on ablation config
    if context_extractor:
        text_inputs = context_extractor.prepare_context_inputs(
            description="None",
            previous_state_context=prev_state.get_context() if prev_state else None,
            previous_action_html=prev_action.element.outerHTML if prev_action else None
        )
        # Use placeholder image if screenshots disabled
        if context_extractor.should_include_screenshot():
            image_data = png_to_base64(screenshot_path)
        else:
            image_data = context_extractor.get_placeholder_image()
    else:
        # Original behavior
        text_inputs = {
            "description": "None",
            "previous_state": "None. This is the first state." if prev_state is None else prev_state.get_context(),
            "previous_action": "None. This is the first state." if prev_action is None else prev_action.element.outerHTML,
        }
        image_data = png_to_base64(screenshot_path)

    context_text = sonnet_chain(
        CONTEXT_EXTRACTION_SYSTEM_PROMPT,
        create_context_user_messages(text_inputs, image_data)
    )
    return get_llm_text(context_text)
```

#### Scoring Function (geometric_score in utils/**init**.py)

Current implementation uses log-probability:

```python
@lru_cache(maxsize=32)
def geometric_score(rank, p=0.5, R=10):
    if rank is not None and rank >= 1:
        return (rank - 1) * np.log(1 - p) + np.log(p)
    return R * np.log(1 - p) + np.log(p)
```

To support ablation, modify or create wrapper:

```python
def get_score(rank, scoring_function=None):
    """Get score with configurable scoring function."""
    if scoring_function:
        return scoring_function.score(rank)
    return geometric_score(rank)
```

#### Feature Aggregation (map_similar_func_to_exact_match)

Current behavior: embedding + LLM validation via `SIMILARITY_SYSTEM_PROMPT`

Modified version:

```python
def map_similar_func_to_exact_match(func_info, aggregator=None):
    """Map a functionality to existing similar ones or mark as new."""
    rank, text, embedding, similar_funcs = func_info

    # Use aggregator if provided
    if aggregator:
        if aggregator.method == 'none':
            return {'match': False, 'rank': rank, 'text': text, 'embedding': embedding}
        elif aggregator.method == 'exact_match':
            exact_indices = get_exact_match_indices(text, similar_funcs)
            if exact_indices:
                return {'match': True, 'match_index': exact_indices,
                        'combined_text': text, 'rank': rank, 'text': text, 'embedding': embedding}
            return {'match': False, 'rank': rank, 'text': text, 'embedding': embedding}
        elif aggregator.method == 'embedding_only':
            # Skip LLM validation, just use embedding similarity
            # ... implementation
            pass

    # Original behavior with LLM validation
    # ... existing code
```

#### Score Accumulation (update_functionality_score)

Current behavior: differential scoring `curr_score - prev_score`

Modified version:

```python
def update_functionality_score(prev_state, prev_action, curr_state, curr_action,
                               accumulator=None):
    """Update functionality scores based on action sequence."""
    # ... get curr_action_funcs and prev_action_funcs from DB ...

    for curr_func in curr_action_funcs:
        corresponding = [x for x in prev_action_funcs
                        if x['func_pointer'] == curr_func['func_pointer']]
        prev_score = geometric_score(None) if not corresponding else corresponding[0]['rank_score']

        if accumulator:
            # Use configured accumulation method
            new_score = accumulator.accumulate(prev_score, curr_func['rank_score'])
        else:
            # Original differential scoring
            new_score = curr_func['rank_score'] - prev_score

        func_db.update_one(
            filter={'app': APP_NAME, '_id': ObjectId(curr_func['func_pointer']), 'final': False},
            update={'$inc': {'score': new_score}}
        )
```

### 3. Creating a Wrapper Module

For cleaner integration, create `autoe2e/ablation_wrapper.py`:

```python
"""
Ablation Wrapper
================

Wraps AutoE2E inference functions with ablation configuration support.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ablation_study.orchestrator import ComponentFactory


class AblationWrapper:
    """Wraps inference functions with ablation configuration."""

    def __init__(self, ablation_config_path: Optional[str] = None):
        """Initialize with ablation configuration."""
        self.config = None
        self.components = None

        if ablation_config_path:
            self.load_config(ablation_config_path)

    def load_config(self, config_path: str):
        """Load ablation configuration from file."""
        with open(config_path, 'r') as f:
            full_config = json.load(f)

        self.config = full_config.get('components', {})
        self.components = ComponentFactory.create_from_dict(self.config)

    def is_enabled(self) -> bool:
        """Check if ablation mode is enabled."""
        return self.components is not None

    def get_context_extractor(self):
        """Get configured context extractor."""
        if self.components:
            return self.components['context_extractor']
        from ablation_study.components import ContextExtractor
        return ContextExtractor()

    def get_prompt_manager(self):
        """Get configured prompt manager."""
        if self.components:
            return self.components['prompt_manager']
        from ablation_study.components import PromptManager
        return PromptManager()

    def get_scoring_function(self):
        """Get configured scoring function."""
        if self.components:
            return self.components['scoring_function']
        from ablation_study.components import ScoringFunction
        return ScoringFunction()

    def get_feature_aggregator(self):
        """Get configured feature aggregator."""
        if self.components:
            return self.components['feature_aggregator']
        from ablation_study.components import FeatureAggregator
        return FeatureAggregator()

    def get_score_accumulator(self):
        """Get configured score accumulator."""
        if self.components:
            return self.components['score_accumulator']
        from ablation_study.components import ScoreAccumulator
        return ScoreAccumulator()

    def get_score_threshold(self):
        """Get configured score threshold."""
        if self.components:
            return self.components['score_threshold']
        from ablation_study.components import ScoreThreshold
        return ScoreThreshold()


# Global instance (can be set at startup)
_wrapper: Optional[AblationWrapper] = None


def init_ablation(config_path: Optional[str] = None):
    """Initialize ablation configuration globally."""
    global _wrapper
    _wrapper = AblationWrapper(config_path)


def get_wrapper() -> AblationWrapper:
    """Get the global ablation wrapper."""
    global _wrapper
    if _wrapper is None:
        _wrapper = AblationWrapper()
    return _wrapper
```

### 4. Usage in main.py

```python
# At startup
from autoe2e.ablation_wrapper import init_ablation, get_wrapper

def main():
    args = parse_args()

    # Initialize ablation configuration if provided
    if args.ablation_config:
        init_ablation(args.ablation_config)
        print(f"Ablation mode enabled: {args.ablation_config}")

    # Get components
    wrapper = get_wrapper()
    context_extractor = wrapper.get_context_extractor()
    prompt_manager = wrapper.get_prompt_manager()
    # ... etc

    # Use throughout the crawl loop
    ...
```

## Running Ablation Experiments

### Quick Start

```bash
# View available ablations
python -m ablation_study.scripts.run_ablation_study --list-ablations

# View available applications
python -m ablation_study.scripts.run_ablation_study --list-apps

# Dry run to see plan
python -m ablation_study.scripts.run_ablation_study --dry-run

# Run specific ablation
python -m ablation_study.scripts.run_single_ablation A1.1 petclinic

# Run all ablations on recommended apps
python -m ablation_study.scripts.run_ablation_study
```

### Analyzing Results

```bash
# View summary
python -m ablation_study.scripts.analyze_ablations

# Compare specific ablations
python -m ablation_study.scripts.analyze_ablations --compare A1.1 A1.2 A1.3

# Statistical significance
python -m ablation_study.scripts.analyze_ablations --significance

# Export to LaTeX
python -m ablation_study.scripts.analyze_ablations --latex

# Export to CSV
python -m ablation_study.scripts.analyze_ablations --csv results.csv
```

## Component Details

These map to the components in `ablations.yaml`:

### C1 - Context Extraction (4 ablations)

Controls what information is included in the LLM prompt:

- **A1.1**: No screenshot (text-only HTML)
- **A1.2**: No previous state context
- **A1.3**: No previous action context
- **A1.4**: Minimal - no screenshot, no history

Maps to `autoe2e/infer_utils.py::extract_state_context()` which uses:

- `png_to_base64(screenshot_path)` for screenshot
- `prev_state.get_context()` for previous state
- `prev_action.element.outerHTML` for previous action

### C2 - Prompting Strategy (3 ablations)

Controls how functionalities are inferred:

- **A2.1**: Single-action query only (SINGLE type)
- **A2.2**: Action-pair query only (DOUBLE type)
- **A2.3**: Merged query (single combined prompt)

Maps to `main.py` which currently does dual prompting:

1. First call for SINGLE action type (every action)
2. Second call for DOUBLE action type (when prev_action exists)

### C3 - Scoring Function (9 ablations)

Controls how ranks are converted to scores:

- **A3.1**: Uniform scoring (1/R for all ranks)
- **A3.2**: Linear decay ((R-rank+1)/R)
- **A3.3**: Binary (1 if rank=1, else 0)
- **A3.4a-d**: Geometric with p=0.2, 0.3, 0.7, 0.8 (baseline p=0.5)
- **A3.5a-c**: Max candidates R=3, 5, 15 (baseline R=10)

Maps to `autoe2e/utils/__init__.py::geometric_score(rank, p=0.5, R=10)`:

```python
(rank - 1) * np.log(1 - p) + np.log(p)  # log-probability formula
```

### C4 - Score Accumulation (3 ablations)

Controls how scores combine over multiple observations:

- **A5.1**: Final action score only (no accumulation)
- **A5.2**: Simple sum (no differential)
- **A5.3**: Maximum score across action chain

Maps to `autoe2e/infer_utils.py::update_functionality_score()`:

```python
diff = curr_func['rank_score'] - prev_score  # differential scoring
func_db.update_one(..., update={'$inc': {'score': diff}})
```

### C5 - Score Threshold (3 ablations)

Controls filtering of features by accumulated score:

- **A7.1**: Keep features with score >= 0 (non-negative)
- **A7.2**: Keep features with score >= 1.0 (moderate confidence)
- **A7.3**: Keep features with score >= 2.0 (high confidence)

The geometric scoring function produces negative values per-rank:

```
With p=0.5: rank 1 = -0.693, rank 2 = -1.386, rank 3 = -2.079, ...
```

After differential accumulation, positive scores indicate the feature appeared
at better ranks in the current action than the previous action. Higher accumulated
scores = stronger evidence for the feature.

#### Integration Point

Apply threshold when querying final features:

```python
# In evaluation or final feature retrieval
if score_threshold and score_threshold.is_enabled():
    features = func_db.find({
        'app': APP_NAME,
        'final': True,
        'score': {'$gte': score_threshold.get_min_score()}
    })
else:
    features = func_db.find({'app': APP_NAME, 'final': True})
```

## Expected Results Structure

After running experiments, results are stored as:

```json
{
  "ablation_id": "A1.1",
  "application": "petclinic",
  "run_id": 1,
  "feature_coverage": 0.75,
  "precision": 0.82,
  "recall": 0.75,
  "f1_score": 0.78,
  "states_explored": 45,
  "duration_seconds": 1234.5
}
```

## Best Practices

1. **Run baseline first**: Always run baseline on all apps before ablations
2. **Use checkpoints**: The system automatically saves progress
3. **3 repetitions minimum**: For statistical validity
4. **Monitor resources**: Experiments can be resource-intensive
5. **Check logs**: Review experiment.log for errors

## Troubleshooting

### Experiment fails to start

- Check if Docker containers are running for the application
- Verify config file paths in applications.yaml

### Low coverage scores

- Check if the LLM model is responding correctly
- Review prompts in experiment logs

### Resume not working

- Verify checkpoint.json exists and is valid
- Use --force to override and re-run

## Contact

For issues with the ablation study framework, check the logs in:

- `ablation_study/results/<ablation_id>/<app>/run_<id>/experiment.log`
