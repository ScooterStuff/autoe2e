# AutoE2E Experiment Runner Guide

This guide covers all commands for running AutoE2E replication experiments with local LLMs via Ollama.

## Prerequisites

1. **Ollama running**: `ollama serve`
2. **Models pulled**:
   ```bash
   ollama pull qwen3:8b           # or your preferred model
   ollama pull nomic-embed-text   # required for embeddings
   ```
3. **MongoDB running** (for storing results)
4. **Application Docker containers** (or use `--skip-docker`)

---

## Quick Start

```powershell
# Simplest run: 1 repetition, quiet mode
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic

# With verbose output to see progress
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic -v

# Limit exploration to 30 states (faster for testing)
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic -S 30 -v
```

---

## Commands

### Single Experiment

Run one model on one application:

```powershell
python -m experiments.scripts.run_single_experiment [OPTIONS]
```

| Flag              | Short | Required | Default | Description                          |
| ----------------- | ----- | -------- | ------- | ------------------------------------ |
| `--model`         | `-m`  | ✅       | -       | Model name (e.g., `qwen3:8b`)        |
| `--app`           | `-a`  | ✅       | -       | Application name (e.g., `petclinic`) |
| `--run`           | `-r`  | ❌       | 1       | Run ID number                        |
| `--max-states`    | `-S`  | ❌       | 1000    | Max states to explore                |
| `--timeout`       | `-T`  | ❌       | 720     | Max runtime in minutes               |
| `--verbose`       | `-v`  | ❌       | off     | Show real-time output                |
| `--skip-docker`   | -     | ❌       | off     | Skip Docker container management     |
| `--force`         | -     | ❌       | off     | Re-run even if completed             |
| `--no-checkpoint` | -     | ❌       | off     | Don't save to checkpoint             |

**Examples:**

```powershell
# Basic run
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic

# Verbose with state limit
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic -S 50 -v

# Force re-run of completed experiment
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic --force

# Skip Docker (containers already running)
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic --skip-docker
```

---

### All Experiments

Run multiple models/applications with repetitions:

```powershell
python -m experiments.scripts.run_all_experiments [OPTIONS]
```

| Flag                 | Short | Required | Default | Description                             |
| -------------------- | ----- | -------- | ------- | --------------------------------------- |
| `--model`            | `-m`  | ❌       | all     | Model(s) to run (space-separated)       |
| `--app`              | `-a`  | ❌       | all     | Application(s) to run (space-separated) |
| `--runs`             | `-R`  | ❌       | 1       | Number of repetitions                   |
| `--max-states`       | `-S`  | ❌       | 1000    | Max states to explore                   |
| `--timeout`          | `-T`  | ❌       | 720     | Max runtime in minutes                  |
| `--verbose`          | `-v`  | ❌       | off     | Show real-time output                   |
| `--skip-docker`      | -     | ❌       | off     | Skip Docker container management        |
| `--dry-run`          | `-n`  | ❌       | off     | Show plan without executing             |
| `--resume`           | `-r`  | ❌       | on      | Resume from checkpoint                  |
| `--no-resume`        | -     | ❌       | off     | Run all (ignore checkpoint)             |
| `--reset-checkpoint` | -     | ❌       | off     | Clear checkpoint before running         |
| `--list-models`      | -     | ❌       | -       | List available models and exit          |
| `--list-apps`        | -     | ❌       | -       | List available applications and exit    |
| `--status`           | -     | ❌       | -       | Show checkpoint status and exit         |

**Examples:**

```powershell
# Run one model on one app, 1 repetition (default)
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic

# Run 3 repetitions with verbose output
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic -R 3 -v

# Run multiple models on multiple apps
python -m experiments.scripts.run_all_experiments -m qwen3:8b llama3.1:8b -a petclinic realworld

# Dry run - see what would execute
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic --dry-run

# Check what's available
python -m experiments.scripts.run_all_experiments --list-models
python -m experiments.scripts.run_all_experiments --list-apps

# Check progress
python -m experiments.scripts.run_all_experiments --status

# Reset and start fresh
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic --reset-checkpoint
```

---

### Generate Reports

```powershell
python -m experiments.scripts.generate_report [OPTIONS]
```

---

## Available Models

Configured in `experiments/config/models.yaml`:

| Model              | Context Length | Description                         |
| ------------------ | -------------- | ----------------------------------- |
| `qwen3:8b`         | 32K            | 8B general model (baseline)         |
| `qwen3:32b`        | 32K            | 32B general model (high capability) |
| `devstral:24b`     | 128K           | 24B code-specialized model          |
| `mistral-nemo:12b` | 128K           | 12B model (medium capability)       |
| `llama3.1:8b`      | 128K           | 8B model (alternative baseline)     |
| `llava-llama3:8b`  | 8K             | 8B multimodal model with vision     |

---

## Available Applications

Configured in `experiments/config/applications.yaml`:

| App         | Features | Description                              |
| ----------- | -------- | ---------------------------------------- |
| `petclinic` | 17       | Spring PetClinic (veterinary management) |
| `realworld` | 10       | Conduit blog platform                    |
| `dimeshift` | 12       | Personal finance tracker                 |
| `mantisbt`  | 25       | Bug tracking system                      |
| `traduora`  | 15       | Translation management                   |
| `saleor`    | 20       | E-commerce platform                      |
| `taskcafe`  | 14       | Project management                       |
| `phoenix`   | 18       | Admin dashboard                          |

---

## How Model Selection Works

The `--model` flag sets the `OLLAMA_MODEL` environment variable, which **overrides** any default in the code:

```
Your command: --model qwen3:8b
    ↓
Experiment runner sets: OLLAMA_MODEL=qwen3:8b
    ↓
Subprocess runs main.py with this env var
    ↓
llm_api_call.py reads: os.getenv("OLLAMA_MODEL", "llava-llama3:8b")
    ↓
Uses qwen3:8b (env var takes precedence over default)
```

---

## Typical Workflow

```powershell
# 1. Check Ollama is ready
ollama list

# 2. List available options
python -m experiments.scripts.run_all_experiments --list-models
python -m experiments.scripts.run_all_experiments --list-apps

# 3. Do a quick test run (limited states, verbose)
python -m experiments.scripts.run_single_experiment -m qwen3:8b -a petclinic -S 30 -v

# 4. Run full experiment with 3 repetitions
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic -R 3

# 5. Check progress
python -m experiments.scripts.run_all_experiments --status

# 6. Generate reports
python -m experiments.scripts.generate_report
```

---

## Troubleshooting

### "Model not found"

```powershell
ollama pull <model-name>
```

### "Ollama not running"

```powershell
ollama serve
```

### "MongoDB connection error"

Check your `.env` file has correct `ATLAS_URI`

### Resume interrupted run

```powershell
python -m experiments.scripts.run_all_experiments -m qwen3:8b -a petclinic --resume
```

### Start fresh (clear checkpoint)

```powershell
python -m experiments.scripts.run_all_experiments --reset-checkpoint
```
