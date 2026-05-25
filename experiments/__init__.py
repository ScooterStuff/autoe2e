"""
AutoE2E Experiments Package
===========================

This package provides the experiment configurator and orchestration system
for running AutoE2E replication studies with local LLMs via Ollama.

Main Components:
- orchestrator: Experiment execution and management
- config: Configuration files (YAML)
- scripts: Entry point scripts

The experiment runner calls main.py as a subprocess, which uses the existing
llm_api_call.py for Ollama integration (ChatOllama, OllamaEmbeddings via LangChain).

Usage:
    # Run all experiments
    python -m experiments.scripts.run_all_experiments
    
    # Run single experiment  
    python -m experiments.scripts.run_single_experiment --model qwen3:8b --app petclinic
    
    # Generate reports
    python -m experiments.scripts.generate_report
"""

__version__ = "1.0.0"
