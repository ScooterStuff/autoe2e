#!/usr/bin/env python3
"""
Post-Experiment Evaluation Module
=================================

Handles post-experiment evaluation tasks including:
1. Exporting data from MongoDB (functionality and action-functionality collections)
2. Running feature coverage evaluation using benchmark grammar
3. Saving evaluation results

Usage:
    python -m experiments.scripts.post_experiment_evaluation --model gemma3-12b --app petclinic --run 1
    
    # Or integrate into experiment runner
    from experiments.scripts.post_experiment_evaluation import PostExperimentEvaluator
    evaluator = PostExperimentEvaluator()
    results = evaluator.run_full_evaluation("gemma3-12b", "petclinic", 1)
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Please install required packages:")
    print("  pip install pymongo python-dotenv")
    sys.exit(1)


# ============================================================
# BENCHMARK GRAMMARS
# ============================================================

BENCHMARKS = {
    "PETCLINIC": {
        "view a list of owners": "c2c3",
        "find an owner": "c2c3t1+c10",
        "add owner": "c2(c4|c3c12)(t15|t16|t17|t18|t19)+c31",
        "view owner details": "c2c3c11",
        "add a new pet to the owner": "c2c3c11c15(t9|t10|s2)+c21",
        "edit owner details": "c2c3c11c14(t2|t3|t4|t5|t6)+c17",
        "edit pet details": "c2c3c11c22(t7|t8|s1)+c19",
        "delete pet": "c2c3c11c23",
        "add visit for pet": "c2c3c11c24(t11|t12)+c28",
        "edit details of a visit": "c2c3c11c25(t13|t14)+c29",
        "delete a visit": "c2c3c11c26",
        "edit details of a vet": "c5c6c32(t20|t21|s3)+c37",
        "delete a vet": "c5c6c33",
        "add a vet": "c5(c7|c6c35)(t22|t23|s4)+c39",
        "edit a pet type": "c8c40t24+c44",
        "delete a pet type": "c8c41",
        "add a pet type": "c8c43t25+c46",
        "edit a specialty": "c9c47t26+c51",
        "add a specialty": "c9c50t27+c53",
        "delete a specialty": "c9c48"
    },
    
    # RealWorld (Conduit) - Blog platform features (17 features)
    "REALWORLD": {
        "view home feed": "c1",
        "view global feed": "c1c2",
        "view article": "c1c2c3",
        "register user": "c4(t1|t2|t3)+c5",
        "login user": "c6(t4|t5)+c7",
        "logout user": "c8c9",
        "view profile": "c10c11",
        "edit profile settings": "c8c12(t6|t7|t8)+c13",
        "create article": "c8c14(t9|t10|t11|t12)+c15",
        "edit article": "c1c2c3c16(t13|t14|t15)+c17",
        "delete article": "c1c2c3c18",
        "add comment": "c1c2c3(t16)+c19",
        "delete comment": "c1c2c3c20",
        "favorite article": "c1c2c3c21",
        "unfavorite article": "c1c2c3c22",
        "follow user": "c10c11c23",
        "unfollow user": "c10c11c24"
    },
    
    # TaskCafe - Task management features (32 features)
    "TASKCAFE": {
        "view projects": "c1",
        "create project": "c1c2(t1)+c3",
        "view project": "c1c4",
        "edit project": "c1c4c5(t2)+c6",
        "delete project": "c1c4c7",
        "view task list": "c1c4c8",
        "create task list": "c1c4c9(t3)+c10",
        "edit task list": "c1c4c8c11(t4)+c12",
        "delete task list": "c1c4c8c13",
        "view task": "c1c4c8c14",
        "create task": "c1c4c8c15(t5|t6)+c16",
        "edit task title": "c1c4c8c14c17(t7)+c18",
        "edit task description": "c1c4c8c14c19(t8)+c20",
        "set task due date": "c1c4c8c14c21(t9)+c22",
        "assign task": "c1c4c8c14c23(s1)+c24",
        "unassign task": "c1c4c8c14c25",
        "add task label": "c1c4c8c14c26(s2)+c27",
        "remove task label": "c1c4c8c14c28",
        "move task": "c1c4c8c14c29(s3)+c30",
        "delete task": "c1c4c8c14c31",
        "add checklist": "c1c4c8c14c32(t10)+c33",
        "add checklist item": "c1c4c8c14c34(t11)+c35",
        "toggle checklist item": "c1c4c8c14c36",
        "delete checklist": "c1c4c8c14c37",
        "add comment": "c1c4c8c14(t12)+c38",
        "delete comment": "c1c4c8c14c39",
        "view team": "c40",
        "invite member": "c40c41(t13)+c42",
        "remove member": "c40c43",
        "change member role": "c40c44(s4)+c45",
        "view my tasks": "c46",
        "filter tasks": "c46c47(s5)+c48"
    },
    
    # Dimeshift - Expense tracking features (21 features)
    "DIMESHIFT": {
        "view dashboard": "c1",
        "view transactions": "c1c2",
        "add income": "c1c2c3(t1|t2|t3|s1)+c4",
        "add expense": "c1c2c5(t4|t5|t6|s2)+c6",
        "edit transaction": "c1c2c7(t7|t8)+c8",
        "delete transaction": "c1c2c9",
        "view categories": "c10",
        "add category": "c10c11(t9|s3)+c12",
        "edit category": "c10c13(t10)+c14",
        "delete category": "c10c15",
        "view accounts": "c16",
        "add account": "c16c17(t11|t12|s4)+c18",
        "edit account": "c16c19(t13)+c20",
        "delete account": "c16c21",
        "transfer between accounts": "c16c22(t14|t15|s5|s6)+c23",
        "view reports": "c24",
        "filter by date range": "c24c25(t16|t17)+c26",
        "filter by category": "c24c27(s7)+c28",
        "export report": "c24c29",
        "view budget": "c30",
        "set budget": "c30c31(t18|s8)+c32"
    },
    
    # MantisBT - Bug tracking features (27 features)
    "MANTISBT": {
        "view issues": "c1",
        "view issue details": "c1c2",
        "create issue": "c3(t1|t2|t3|s1|s2)+c4",
        "edit issue": "c1c2c5(t4|t5)+c6",
        "delete issue": "c1c2c7",
        "assign issue": "c1c2c8(s3)+c9",
        "change issue status": "c1c2c10(s4)+c11",
        "change issue priority": "c1c2c12(s5)+c13",
        "change issue severity": "c1c2c14(s6)+c15",
        "add issue note": "c1c2(t6)+c16",
        "delete issue note": "c1c2c17",
        "attach file": "c1c2c18(t7)+c19",
        "delete attachment": "c1c2c20",
        "add issue relationship": "c1c2c21(s7|t8)+c22",
        "delete issue relationship": "c1c2c23",
        "view projects": "c24",
        "create project": "c24c25(t9|t10|s8)+c26",
        "edit project": "c24c27(t11)+c28",
        "delete project": "c24c29",
        "view users": "c30",
        "create user": "c30c31(t12|t13|t14|s9)+c32",
        "edit user": "c30c33(t15)+c34",
        "delete user": "c30c35",
        "view categories": "c24c27c36",
        "add category": "c24c27c36c37(t16)+c38",
        "edit category": "c24c27c36c39(t17)+c40",
        "delete category": "c24c27c36c41"
    },
    
    # EverTraduora - Translation management features (41 features)
    "EVERTRADUORA": {
        "view projects": "c1",
        "create project": "c1c2(t1|t2)+c3",
        "view project": "c1c4",
        "edit project": "c1c4c5(t3)+c6",
        "delete project": "c1c4c7",
        "view locales": "c1c4c8",
        "add locale": "c1c4c8c9(s1)+c10",
        "remove locale": "c1c4c8c11",
        "set default locale": "c1c4c8c12(s2)+c13",
        "view terms": "c1c4c14",
        "add term": "c1c4c14c15(t4)+c16",
        "edit term": "c1c4c14c17(t5)+c18",
        "delete term": "c1c4c14c19",
        "add translation": "c1c4c14c20(t6)+c21",
        "edit translation": "c1c4c14c22(t7)+c23",
        "delete translation": "c1c4c14c24",
        "import terms": "c1c4c25(t8)+c26",
        "export terms": "c1c4c27(s3)+c28",
        "view team": "c1c4c29",
        "invite member": "c1c4c29c30(t9)+c31",
        "remove member": "c1c4c29c32",
        "change member role": "c1c4c29c33(s4)+c34",
        "view labels": "c1c4c35",
        "add label": "c1c4c35c36(t10|t11)+c37",
        "edit label": "c1c4c35c38(t12)+c39",
        "delete label": "c1c4c35c40",
        "assign label to term": "c1c4c14c41(s5)+c42",
        "remove label from term": "c1c4c14c43",
        "search terms": "c1c4c14(t13)+c44",
        "filter by locale": "c1c4c14c45(s6)+c46",
        "filter by label": "c1c4c14c47(s7)+c48",
        "filter untranslated": "c1c4c14c49",
        "view api keys": "c1c4c50",
        "create api key": "c1c4c50c51(t14)+c52",
        "delete api key": "c1c4c50c53",
        "view activity": "c1c4c54",
        "view user settings": "c55",
        "edit user settings": "c55c56(t15|t16)+c57",
        "change password": "c55c58(t17|t18)+c59",
        "view organization": "c60",
        "edit organization": "c60c61(t19)+c62"
    },
    
    # Saleor Storefront - E-commerce frontend features (13 features)
    "SALEOR": {
        "view products": "c1",
        "view product details": "c1c2",
        "search products": "c1(t1)+c3",
        "filter by category": "c1c4(s1)+c5",
        "add to cart": "c1c2c6",
        "view cart": "c7",
        "update cart quantity": "c7c8(t2)+c9",
        "remove from cart": "c7c10",
        "checkout": "c7c11(t3|t4|t5|t6|t7)+c12",
        "login": "c13(t8|t9)+c14",
        "register": "c15(t10|t11|t12)+c16",
        "view account": "c17",
        "view order history": "c17c18"
    },
    
    # Saleor Dashboard - E-commerce admin (simplified subset - 20 key features)
    # Note: Full dashboard has 130 features, this is a representative subset
    "SALEOR_DASHBOARD": {
        "view dashboard": "c1",
        "view orders": "c1c2",
        "view order details": "c1c2c3",
        "update order status": "c1c2c3c4(s1)+c5",
        "view products": "c6",
        "create product": "c6c7(t1|t2|t3|s2)+c8",
        "edit product": "c6c9(t4)+c10",
        "delete product": "c6c11",
        "view categories": "c12",
        "create category": "c12c13(t5|t6)+c14",
        "edit category": "c12c15(t7)+c16",
        "delete category": "c12c17",
        "view customers": "c18",
        "view customer details": "c18c19",
        "view staff": "c20",
        "create staff": "c20c21(t8|t9|t10)+c22",
        "edit staff": "c20c23(t11)+c24",
        "view discounts": "c25",
        "create discount": "c25c26(t12|t13|s3)+c27",
        "delete discount": "c25c28"
    }
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single run."""
    # Metrics
    total_features: int
    total_generated: int
    correct: int
    covered: int
    precision: float
    recall: float
    feature_coverage: float
    f1: float
    
    # Details
    covered_features: List[str]
    uncovered_features: List[str]
    matched_sequences: List[Dict]
    unmatched_sequences: List[Dict]
    
    # Metadata
    model_name: str
    app_name: str
    run_id: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvaluationResult':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# CORE EVALUATION CLASSES
# ============================================================

class FeatureGrammarParser:
    """Parses and matches feature grammar rules."""
    
    def __init__(self, grammar: Dict[str, str]):
        self.grammar = grammar
        self.compiled_patterns = {
            name: re.compile(f"^{rule}$") 
            for name, rule in grammar.items()
        }
    
    def find_matching_features(self, action_sequence: str) -> List[str]:
        """Find all features that match the given action sequence."""
        return [
            name for name, pattern in self.compiled_patterns.items()
            if pattern.match(action_sequence)
        ]


class ActionChainReconstructor:
    """
    Reconstructs action chains from database records.
    
    The key is the test_id field which contains identifiers like:
    - "c3-navbar-owners-search" -> extract "c3"
    - "t1-search-input" -> extract "t1"
    
    For FORM type actions, it also handles:
    - form_fields: list of field identifiers that were filled (e.g., ['t2', 't3', 't4'])
    - submit_prefix: the submit button identifier (e.g., 'c31')
    
    We trace back through prev_state/prev_action to build the full chain.
    """
    
    def __init__(self, records: List[dict], func_lookup: Dict[str, dict] = None):
        self.records = records
        self.func_lookup = func_lookup or {}
        # Build index for efficient parent lookup
        self.index = {}
        for r in records:
            key = (r.get('state'), r.get('action'))
            self.index[key] = r
    
    @staticmethod
    def extract_prefix(test_id: str) -> Optional[str]:
        """Extract action identifier (c2, t1, s3) from test_id."""
        if not test_id:
            return None
        match = re.match(r'^([cts]\d+)', test_id)
        return match.group(1) if match else None
    
    def get_functionality_text(self, func_pointer: str) -> Optional[str]:
        """Look up the functionality text from func_pointer."""
        if not func_pointer:
            return None
        func = self.func_lookup.get(str(func_pointer))
        if func:
            return func.get('text')
        return None
    
    def reconstruct_chain(self, record: dict) -> Tuple[str, List[str]]:
        """
        Build action sequence by tracing back through prev_state/prev_action.
        
        For FORM type actions, includes form fields and submit button:
        - Form fields are concatenated: t2t3t4t5t6 (to match regex like (t2|t3|t4|t5|t6)+)
        - Submit button is appended: c31
        
        Returns:
            Tuple of (action_sequence_string, list_of_test_ids)
        """
        prefixes = []
        test_ids = []
        visited = set()
        current = record
        is_first = True  # Track if this is the first (final) action
        
        # Store form suffix to append at the end (form fields + submit)
        form_suffix = ""
        form_suffix_test_ids = []
        
        while current:
            state = current.get('state')
            if state in visited:
                break
            visited.add(state)
            
            action_type = current.get('type', 'SINGLE')
            
            # Handle FORM type actions specially (only for the leaf/final action)
            if is_first and action_type in ['FORM', 'FORM_DOUBLE']:
                form_fields = current.get('form_fields', [])
                submit_prefix = current.get('submit_prefix')
                
                # Concatenate form fields (e.g., "t2t3t4t5t6") 
                # This will match regex patterns like (t2|t3|t4|t5|t6)+
                if form_fields:
                    form_suffix = ''.join(form_fields)
                    form_suffix_test_ids.append(f"form_fields:{','.join(form_fields)}")
                
                # Append submit button (e.g., "c31")
                if submit_prefix:
                    form_suffix += submit_prefix
                    form_suffix_test_ids.append(f"submit:{submit_prefix}")
            else:
                # Regular action - extract test_id prefix
                test_id = current.get('test_id')
                prefix = self.extract_prefix(test_id)
                
                if prefix:
                    prefixes.insert(0, prefix)
                    test_ids.insert(0, test_id or '')
            
            is_first = False
            prev_state = current.get('prev_state')
            prev_action = current.get('prev_action')
            
            if not prev_state or not prev_action:
                break
            
            current = self.index.get((prev_state, prev_action))
        
        # Combine: navigation prefixes + form suffix
        final_sequence = ''.join(prefixes) + form_suffix
        final_test_ids = test_ids + form_suffix_test_ids
        
        return final_sequence, final_test_ids
    
    def get_all_chains(self) -> List[Tuple[str, List[str], dict, Optional[str]]]:
        """
        Get all unique action chains.
        
        Returns:
            List of tuples: (sequence_string, test_id_chain, original_record, functionality_text)
        """
        seen = set()
        chains = []
        
        for record in self.records:
            seq, test_ids = self.reconstruct_chain(record)
            if seq and seq not in seen:
                seen.add(seq)
                func_pointer = record.get('func_pointer')
                func_text = self.get_functionality_text(func_pointer)
                chains.append((seq, test_ids, record, func_text))
        
        return chains


class FeatureCoverageEvaluator:
    """Main evaluator that computes all metrics."""
    
    def __init__(self, grammar: Dict[str, str]):
        self.parser = FeatureGrammarParser(grammar)
        self.total_features = len(grammar)
    
    def evaluate(self, chains: List[Tuple[str, List[str], dict, Optional[str]]]) -> Dict:
        """
        Evaluate feature coverage.
        
        Args:
            chains: List of (sequence, test_ids, record, func_text) tuples
            
        Returns:
            Dictionary with all metrics
        """
        covered = set()
        correct = []
        incorrect = []
        
        for seq, test_ids, record, func_text in chains:
            matches = self.parser.find_matching_features(seq)
            if matches:
                correct.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'features': matches,
                    'func_pointer': str(record.get('func_pointer', '')),
                    'inferred_feature': func_text
                })
                covered.update(matches)
            else:
                incorrect.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'func_pointer': str(record.get('func_pointer', '')),
                    'inferred_feature': func_text
                })
        
        n_gen = len(chains)
        n_correct = len(correct)
        n_covered = len(covered)
        
        precision = n_correct / n_gen if n_gen > 0 else 0
        recall = n_covered / self.total_features if self.total_features > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'metrics': {
                'total_features': self.total_features,
                'total_generated': n_gen,
                'correct': n_correct,
                'covered': n_covered,
                'precision': precision,
                'recall': recall,
                'feature_coverage': recall,
                'f1': f1
            },
            'covered_features': sorted(covered),
            'uncovered_features': sorted([
                f for f in self.parser.grammar.keys() if f not in covered
            ]),
            'matched': correct,
            'unmatched': incorrect
        }


# ============================================================
# POST-EXPERIMENT EVALUATOR
# ============================================================

class PostExperimentEvaluator:
    """
    Main class for post-experiment evaluation.
    
    Handles:
    1. Exporting data from MongoDB
    2. Running feature coverage evaluation
    3. Saving results to appropriate directories
    """
    
    def __init__(self, results_base_dir: Optional[Path] = None):
        """
        Initialize the evaluator.
        
        Args:
            results_base_dir: Base directory for results (default: experiments/results/replication)
        """
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.results_base_dir = results_base_dir or (
            self.project_root / "experiments" / "results" / "replication"
        )
    
    def connect_to_mongodb(self):
        """Connect to MongoDB Atlas."""
        uri = os.getenv("ATLAS_URI")
        if not uri:
            raise ValueError("ATLAS_URI not found in environment. Create a .env file.")
        
        client = MongoClient(uri)
        db = client.myDatabase
        return client, db
    
    @staticmethod
    def sanitize_model_name(model_name: str) -> str:
        """
        Sanitize model name for use in file/directory paths.
        
        Replaces characters that are invalid in Windows paths (like ':').
        
        Args:
            model_name: Original model name (e.g., "gemma3:27b")
            
        Returns:
            Sanitized name safe for file paths (e.g., "gemma3-27b")
        """
        # Replace colon with hyphen (common in Ollama model names like "gemma3:27b")
        return model_name.replace(':', '-')
    
    def get_run_directory(self, model_name: str, app_name: str, run_id: int) -> Path:
        """
        Get the directory path for a specific run.
        
        Args:
            model_name: Name of the model (e.g., "gemma3:12b" or "gemma3-12b")
            app_name: Name of the application (e.g., "petclinic")
            run_id: Run number
            
        Returns:
            Path to the run directory
        """
        # Sanitize model name for Windows compatibility (colons not allowed)
        safe_model_name = self.sanitize_model_name(model_name)
        return self.results_base_dir / safe_model_name / app_name / f"run_{run_id}"
    
    def export_mongodb_data(
        self, 
        app_name: str, 
        model_name: str, 
        run_id: int
    ) -> Tuple[Path, Path, List[dict], List[dict]]:
        """
        Export functionality and action-functionality data from MongoDB.
        
        Args:
            app_name: Application name (e.g., "PETCLINIC")
            model_name: Model name for directory structure
            run_id: Run number
            
        Returns:
            Tuple of (func_file_path, action_func_file_path, func_records, action_func_records)
        """
        print(f"📡 Connecting to MongoDB...")
        client, db = self.connect_to_mongodb()
        
        try:
            # Get run directory
            run_dir = self.get_run_directory(model_name, app_name.lower(), run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Export functionality collection
            print(f"📥 Exporting functionality data for {app_name}...")
            func_records = list(db["functionality"].find({"app": app_name}))
            
            # Convert ObjectId to string for JSON serialization
            func_serializable = []
            for r in func_records:
                r_copy = dict(r)
                r_copy['_id'] = str(r_copy['_id'])
                func_serializable.append(r_copy)
            
            func_file = run_dir / "functionality_data.json"
            with open(func_file, 'w', encoding='utf-8') as f:
                json.dump(func_serializable, f, indent=2, default=str)
            print(f"   ✓ Saved {len(func_records)} functionality records to {func_file}")
            
            # Export action-functionality collection
            print(f"📥 Exporting action-functionality data for {app_name}...")
            action_func_records = list(db["action-functionality"].find({"app": app_name}))
            
            # Convert ObjectId to string for JSON serialization
            action_func_serializable = []
            for r in action_func_records:
                r_copy = dict(r)
                r_copy['_id'] = str(r_copy['_id'])
                if 'func_pointer' in r_copy and r_copy['func_pointer']:
                    r_copy['func_pointer'] = str(r_copy['func_pointer'])
                action_func_serializable.append(r_copy)
            
            action_func_file = run_dir / "action_functionality_data.json"
            with open(action_func_file, 'w', encoding='utf-8') as f:
                json.dump(action_func_serializable, f, indent=2, default=str)
            print(f"   ✓ Saved {len(action_func_records)} action-functionality records to {action_func_file}")
            
            return func_file, action_func_file, func_records, action_func_records
            
        finally:
            client.close()
    
    def run_evaluation(
        self, 
        func_records: List[dict], 
        action_func_records: List[dict],
        app_name: str,
        verbose: bool = True
    ) -> Dict:
        """
        Run feature coverage evaluation.
        
        Args:
            func_records: Functionality records from MongoDB
            action_func_records: Action-functionality records from MongoDB
            app_name: Application name for benchmark lookup
            verbose: Print detailed output
            
        Returns:
            Evaluation results dictionary
        """
        # Get benchmark grammar
        benchmark_key = app_name.upper()
        if benchmark_key not in BENCHMARKS:
            raise ValueError(f"No benchmark grammar for {app_name}. Available: {list(BENCHMARKS.keys())}")
        
        grammar = BENCHMARKS[benchmark_key]
        print(f"   ✓ Loaded benchmark with {len(grammar)} features")
        
        # Build functionality lookup
        func_lookup = {}
        for func in func_records:
            func_id = str(func.get('_id'))
            func_lookup[func_id] = func
        
        # Reconstruct action chains
        print(f"🔄 Reconstructing action chains...")
        reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
        chains = reconstructor.get_all_chains()
        print(f"   ✓ Found {len(chains)} unique action chains")
        
        if verbose and chains:
            print(f"\n📋 Action chains with inferred features:")
            print("-" * 90)
            for seq, test_ids, record, func_text in chains[:10]:  # Show first 10
                func_display = func_text if func_text else "(no functionality linked)"
                print(f"   Sequence:  {seq}")
                print(f"   Inferred:  {func_display}")
                print("-" * 90)
            if len(chains) > 10:
                print(f"   ... and {len(chains) - 10} more chains")
        
        # Run evaluation
        print(f"\n🎯 Evaluating against benchmark grammar...")
        evaluator = FeatureCoverageEvaluator(grammar)
        results = evaluator.evaluate(chains)
        
        return results
    
    def save_evaluation_results(
        self, 
        results: Dict, 
        model_name: str, 
        app_name: str, 
        run_id: int
    ) -> Path:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            model_name: Model name
            app_name: Application name
            run_id: Run number
            
        Returns:
            Path to saved file
        """
        run_dir = self.get_run_directory(model_name, app_name.lower(), run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create export data
        export_data = {
            'metadata': {
                'model_name': model_name,
                'app_name': app_name,
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'evaluation_version': '1.0'
            },
            'metrics': results['metrics'],
            'covered_features': results['covered_features'],
            'uncovered_features': results['uncovered_features'],
            'matched_count': len(results['matched']),
            'unmatched_count': len(results['unmatched']),
            'matched_sequences': [
                {
                    'sequence': m['sequence'],
                    'matched_features': m['features'],
                    'inferred_feature': m.get('inferred_feature'),
                    'test_ids': m.get('test_ids', [])
                }
                for m in results['matched']
            ],
            'unmatched_sequences': [
                {
                    'sequence': m['sequence'],
                    'inferred_feature': m.get('inferred_feature'),
                    'test_ids': m.get('test_ids', [])
                }
                for m in results['unmatched']
            ]
        }
        
        # Save to run directory
        eval_file = run_dir / f"{app_name.upper()}_evaluation.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Evaluation results saved to {eval_file}")
        
        return eval_file
    
    def update_metrics_file(
        self, 
        results: Dict, 
        model_name: str, 
        app_name: str, 
        run_id: int
    ) -> Optional[Path]:
        """
        Update the existing metrics.json file with evaluation results.
        
        Args:
            results: Evaluation results dictionary
            model_name: Model name
            app_name: Application name
            run_id: Run number
            
        Returns:
            Path to updated metrics file, or None if not found
        """
        run_dir = self.get_run_directory(model_name, app_name.lower(), run_id)
        metrics_file = run_dir / "metrics.json"
        
        if not metrics_file.exists():
            print(f"⚠️  No existing metrics.json found at {metrics_file}")
            return None
        
        # Load existing metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Update with evaluation results
        m = results['metrics']
        metrics.update({
            'feature_coverage': m['feature_coverage'],
            'total_features_covered': m['covered'],
            'total_features': m['total_features'],
            'inferred_features': m['total_generated'],
            'correct_inferences': m['correct'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1_score': m['f1'],
            'evaluation_timestamp': datetime.now().isoformat()
        })
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✓ Updated metrics.json with evaluation results")
        
        return metrics_file
    
    def print_results(self, results: Dict, verbose: bool = True):
        """Print formatted evaluation results."""
        m = results['metrics']
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                     EVALUATION RESULTS                         ║
╠════════════════════════════════════════════════════════════════╣
║  Total Features in Benchmark:           {m['total_features']:>20}  ║
║  Total Test Sequences Generated:        {m['total_generated']:>20}  ║
║  Correct Test Sequences:                {m['correct']:>20}  ║
║  Features Covered:                      {m['covered']:>20}  ║
╠════════════════════════════════════════════════════════════════╣
║  PRECISION:                             {m['precision']:>19.2%}  ║
║  RECALL (FEATURE COVERAGE):             {m['recall']:>19.2%}  ║
║  F1 SCORE:                              {m['f1']:>19.2%}  ║
╚════════════════════════════════════════════════════════════════╝
""")
        
        if verbose:
            print("✅ COVERED FEATURES:")
            for f in results['covered_features']:
                print(f"   • {f}")
            
            print("\n❌ UNCOVERED FEATURES:")
            for f in results['uncovered_features']:
                print(f"   • {f}")
            
            if results['matched']:
                print(f"\n📋 MATCHED SEQUENCES ({len(results['matched'])} total):")
                print("-" * 95)
                for item in results['matched'][:10]:  # Show first 10
                    inferred = item.get('inferred_feature') or "(no functionality linked)"
                    print(f"   Sequence: {item['sequence']}")
                    print(f"   Matched:  {item['features']}")
                    print(f"   Inferred: {inferred}")
                    print("-" * 95)
                if len(results['matched']) > 10:
                    print(f"   ... and {len(results['matched']) - 10} more matched sequences")
            
            if results['unmatched']:
                print(f"\n⚠️  UNMATCHED SEQUENCES ({len(results['unmatched'])} total):")
                print("-" * 95)
                for item in results['unmatched'][:5]:  # Show first 5
                    inferred = item.get('inferred_feature') or "(no functionality linked)"
                    print(f"   Sequence: {item['sequence']}")
                    print(f"   Inferred: {inferred}")
                    print("-" * 95)
                if len(results['unmatched']) > 5:
                    print(f"   ... and {len(results['unmatched']) - 5} more unmatched sequences")
    
    def run_full_evaluation(
        self, 
        model_name: str, 
        app_name: str, 
        run_id: int,
        verbose: bool = True
    ) -> EvaluationResult:
        """
        Run the complete post-experiment evaluation pipeline.
        
        Args:
            model_name: Name of the model (e.g., "gemma3-12b")
            app_name: Name of the application (e.g., "petclinic")
            run_id: Run number
            verbose: Print detailed output
            
        Returns:
            EvaluationResult with all metrics and details
        """
        print("\n" + "=" * 70)
        print(f"POST-EXPERIMENT EVALUATION")
        print(f"Model: {model_name} | App: {app_name} | Run: {run_id}")
        print("=" * 70)
        
        # Normalize app name for MongoDB query (uppercase)
        app_name_upper = app_name.upper()
        
        # Step 1: Export data from MongoDB
        print("\n📤 STEP 1: Exporting data from MongoDB")
        print("-" * 50)
        func_file, action_func_file, func_records, action_func_records = self.export_mongodb_data(
            app_name_upper, model_name, run_id
        )
        
        if not action_func_records:
            print(f"\n⚠️  No action-functionality data found for {app_name_upper}")
            # Return empty result
            return EvaluationResult(
                total_features=len(BENCHMARKS.get(app_name_upper, {})),
                total_generated=0,
                correct=0,
                covered=0,
                precision=0.0,
                recall=0.0,
                feature_coverage=0.0,
                f1=0.0,
                covered_features=[],
                uncovered_features=list(BENCHMARKS.get(app_name_upper, {}).keys()),
                matched_sequences=[],
                unmatched_sequences=[],
                model_name=model_name,
                app_name=app_name,
                run_id=run_id,
                timestamp=datetime.now().isoformat()
            )
        
        # Step 2: Run evaluation
        print("\n📊 STEP 2: Running feature coverage evaluation")
        print("-" * 50)
        results = self.run_evaluation(
            func_records, action_func_records, app_name_upper, verbose=verbose
        )
        
        # Step 3: Print results
        print("\n📈 STEP 3: Evaluation Results")
        print("-" * 50)
        self.print_results(results, verbose=verbose)
        
        # Step 4: Save results
        print("\n💾 STEP 4: Saving results")
        print("-" * 50)
        eval_file = self.save_evaluation_results(results, model_name, app_name, run_id)
        self.update_metrics_file(results, model_name, app_name, run_id)
        
        print("\n" + "=" * 70)
        print("✅ POST-EXPERIMENT EVALUATION COMPLETE")
        print("=" * 70)
        
        # Create and return EvaluationResult
        m = results['metrics']
        return EvaluationResult(
            total_features=m['total_features'],
            total_generated=m['total_generated'],
            correct=m['correct'],
            covered=m['covered'],
            precision=m['precision'],
            recall=m['recall'],
            feature_coverage=m['feature_coverage'],
            f1=m['f1'],
            covered_features=results['covered_features'],
            uncovered_features=results['uncovered_features'],
            matched_sequences=results['matched'],
            unmatched_sequences=results['unmatched'],
            model_name=model_name,
            app_name=app_name,
            run_id=run_id,
            timestamp=datetime.now().isoformat()
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run post-experiment evaluation for AutoE2E",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help="Model name (e.g., gemma3-12b)"
    )
    
    parser.add_argument(
        '--app', '-a',
        type=str,
        required=True,
        help="Application name (e.g., petclinic)"
    )
    
    parser.add_argument(
        '--run', '-r',
        type=int,
        default=1,
        help="Run ID (default: 1)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Minimal output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    evaluator = PostExperimentEvaluator()
    result = evaluator.run_full_evaluation(
        model_name=args.model,
        app_name=args.app,
        run_id=args.run,
        verbose=not args.quiet
    )
    
    # Return success/failure based on whether we got any results
    return 0 if result.total_generated > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
