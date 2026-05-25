"""
AUTOE2E Feature Coverage Evaluator (Local JSON Version)
=========================================================

Evaluate feature coverage using local JSON files instead of MongoDB.

Usage:
    python evaluate_local.py                                    # List available saves
    python evaluate_local.py petclinic_gemma3_27b               # Evaluate specific save
    python evaluate_local.py petclinic_gemma3_27b --export out.json
    python evaluate_local.py --all                              # Evaluate all saves

The script reads from:
    saves/<model_folder>/functionality.json
    saves/<model_folder>/action-functionality.json

"""

import json
import math
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# AUTOE2E SCORE THRESHOLD
# =============================================================================
DEFAULT_SCORE_THRESHOLD = math.log(0.5)  # -0.693 (default from paper)

def get_active_score_threshold() -> float:
    """Get the active score threshold, considering ablation settings."""
    try:
        from autoe2e.ablation_integration import ABLATION_MODE, get_score_threshold
        if ABLATION_MODE:
            return get_score_threshold()
    except ImportError:
        pass
    return DEFAULT_SCORE_THRESHOLD


# ============================================================
# BENCHMARK GRAMMARS
# ============================================================

BENCHMARKS = {
    "PETCLINICS": {
        "view a list of owners": "c2c3",
        "view a list of vets": "c5c6",
        "view a list of pet types": "c8",
        "view a list of specialties": "c9",
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
}


# ============================================================
# CORE EVALUATION CLASSES
# ============================================================

class FeatureGrammarParser:
    """Parses and matches feature grammar rules"""
    
    def __init__(self, grammar: Dict[str, str], partial_match: bool = False):
        self.grammar = grammar
        self.partial_match = partial_match
        self.compiled_patterns = {
            name: re.compile(f"^{rule}$") 
            for name, rule in grammar.items()
        }
        
        # For partial matching, compile patterns without the last action
        if partial_match:
            self.almost_complete_patterns = self._build_almost_complete_patterns()
        else:
            self.almost_complete_patterns = {}
    
    def _remove_last_action(self, rule: str) -> Optional[str]:
        """
        Remove the last action from a rule to create an 'almost complete' pattern.
        
        Examples:
        - "c2c3c11c14(t2|t3|t4|t5|t6)+c17" -> "c2c3c11c14(t2|t3|t4|t5|t6)+"
        - "c2c3" -> "c2"
        - "c8c41" -> "c8"
        - "c5c6c32(t20|t21|s3)+c37" -> "c5c6c32(t20|t21|s3)+"
        - "c9c48" -> "c9"
        
        Returns None if the rule is too simple (single action).
        """
        # Pattern ends with a clickable like c17, c31, etc. after form fields
        # Match: ...)+c17 or ...+c17
        form_submit_match = re.match(r'^(.+\)+)([cs]\d+)$', rule)
        if form_submit_match:
            return form_submit_match.group(1)
        
        # Pattern is just navigation clicks like c2c3c11c14
        nav_only_match = re.match(r'^((?:[cs]\d+)+)$', rule)
        if nav_only_match:
            nav_part = nav_only_match.group(1)
            # Extract all clickables and remove the last one
            clickables = re.findall(r'[cs]\d+', nav_part)
            if len(clickables) > 1:
                return ''.join(clickables[:-1])
            return None  # Single action, can't make partial
        
        # Pattern with alternation at start like "(c1|c2)" - can't make partial
        if rule.startswith('(') and not re.search(r'\+', rule):
            return None
        
        # Pattern ends with simple clickable after navigation
        # Like "c5c6c33" -> "c5c6"
        simple_end_match = re.match(r'^(.+?)([cs]\d+)$', rule)
        if simple_end_match:
            prefix = simple_end_match.group(1)
            # Make sure prefix is not empty and ends properly
            if prefix and (prefix.endswith(')') or re.match(r'.*[cs]\d+$', prefix)):
                return prefix
        
        return None
    
    def _build_almost_complete_patterns(self) -> Dict[str, re.Pattern]:
        """Build patterns that match sequences missing only the last action"""
        almost_complete = {}
        for name, rule in self.grammar.items():
            without_last = self._remove_last_action(rule)
            if without_last:
                try:
                    almost_complete[name] = re.compile(f"^{without_last}$")
                except re.error:
                    pass  # Skip invalid regex
        return almost_complete
    
    def find_matching_features(self, action_sequence: str) -> List[str]:
        """Find all features that match the given action sequence"""
        matches = []
        
        # Check exact matches first
        for name, pattern in self.compiled_patterns.items():
            if pattern.match(action_sequence):
                matches.append(name)
        
        # If partial matching is enabled and no exact matches, check almost-complete patterns
        if self.partial_match and not matches:
            for name, pattern in self.almost_complete_patterns.items():
                if pattern.match(action_sequence):
                    matches.append(name)
        
        return matches
    
    def find_matching_features_with_type(self, action_sequence: str) -> List[Tuple[str, str]]:
        """
        Find all features that match, returning (feature_name, match_type).
        match_type is 'exact' or 'partial'
        """
        matches = []
        
        # Check exact matches first
        for name, pattern in self.compiled_patterns.items():
            if pattern.match(action_sequence):
                matches.append((name, 'exact'))
        
        # If partial matching is enabled and no exact matches, check almost-complete patterns
        if self.partial_match and not matches:
            for name, pattern in self.almost_complete_patterns.items():
                if pattern.match(action_sequence):
                    matches.append((name, 'partial'))
        
        return matches


class ActionChainReconstructor:
    """
    Reconstructs action chains from database records following AUTOE2E paper logic.
    """
    
    def __init__(self, records: List[dict], func_lookup: Dict[str, dict] = None):
        self.records = records
        self.func_lookup = func_lookup or {}
        
        # Build index for efficient parent lookup: (state, action) -> record
        self.index = {}
        for r in records:
            key = (r.get('state'), r.get('action'))
            self.index[key] = r
        
        # Group AFD records by func_pointer for efficient feature-based lookup
        self.records_by_feature = defaultdict(list)
        for r in records:
            fp = r.get('func_pointer')
            if fp:
                self.records_by_feature[fp].append(r)
    
    @staticmethod
    def extract_prefix(test_id: str) -> Optional[str]:
        """Extract action identifier (c2, t1, s3) from test_id"""
        if not test_id:
            return None
        match = re.match(r'^([cts]\d+)', test_id)
        return match.group(1) if match else None
    
    def get_functionality_text(self, func_pointer: str) -> Optional[str]:
        """Look up the functionality text from func_pointer"""
        if not func_pointer:
            return None
        func = self.func_lookup.get(str(func_pointer))
        if func:
            return func.get('text')
        return None
    
    def reconstruct_chain(self, record: dict) -> Tuple[str, List[str]]:
        """
        Build action sequence by tracing back through prev_state/prev_action.
        
        Returns:
            Tuple of (action_sequence_string, list_of_test_ids)
        """
        prefixes = []
        test_ids = []
        visited = set()
        current = record
        is_first = True
        
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
                
                if form_fields:
                    form_suffix = ''.join(form_fields)
                    form_suffix_test_ids.append(f"form_fields:{','.join(form_fields)}")
                
                if submit_prefix:
                    form_suffix += submit_prefix
                    form_suffix_test_ids.append(f"submit:{submit_prefix}")
            else:
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
        
        final_sequence = ''.join(prefixes) + form_suffix
        final_test_ids = test_ids + form_suffix_test_ids
        
        return final_sequence, final_test_ids
    
    def find_leaf_records_for_feature(self, feature_id: str) -> List[dict]:
        """Find the best AFD records for a feature to reconstruct chains from."""
        feature_records = self.records_by_feature.get(feature_id, [])
        if not feature_records:
            return []
        
        records_with_test_id = [r for r in feature_records if r.get('test_id')]
        candidates = records_with_test_id if records_with_test_id else feature_records
        
        if not candidates:
            return []
        
        max_depth = max(r.get('depth', 0) for r in candidates)
        
        if max_depth > 0:
            return [r for r in candidates if r.get('depth', 0) >= max_depth - 1]
        else:
            return candidates
    
    def get_all_chains(self) -> List[Tuple[str, List[str], dict, Optional[str]]]:
        """Get all unique action chains following AUTOE2E algorithm."""
        chains = []
        seen = set()
        
        final_features = {
            fid: f for fid, f in self.func_lookup.items() 
            if f.get('final', False)
        }
        
        if len(final_features) < 5:
            non_final = {fid: f for fid, f in self.func_lookup.items() if not f.get('final', False)}
            sorted_non_final = sorted(non_final.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:10]
            final_features = {**final_features, **dict(sorted_non_final)}
        
        for feature_id, feature in final_features.items():
            feature_text = feature.get('text', '')
            
            candidate_records = self.find_leaf_records_for_feature(feature_id)
            
            if not candidate_records:
                continue
            
            for record in candidate_records:
                seq, test_ids = self.reconstruct_chain(record)
                
                if not seq:
                    continue
                
                key = (feature_id, seq)
                if key in seen:
                    continue
                seen.add(key)
                
                chains.append((seq, test_ids, record, feature_text))
        
        chains.sort(key=lambda x: self.func_lookup.get(x[2].get('func_pointer', ''), {}).get('score', 0), reverse=True)
        
        return chains


class FeatureCoverageEvaluator:
    """Main evaluator that computes all metrics"""
    
    def __init__(self, grammar: Dict[str, str], func_lookup: Dict[str, dict] = None, partial_match: bool = False, total_functionalities: int = 0):
        self.parser = FeatureGrammarParser(grammar, partial_match=partial_match)
        self.total_features = len(grammar)
        self.func_lookup = func_lookup or {}
        self.partial_match = partial_match
        self.total_functionalities = total_functionalities
    
    def _get_score(self, func_pointer: str) -> float:
        """Get the score for a feature by its func_pointer"""
        if not func_pointer or not self.func_lookup:
            return 0.0
        func = self.func_lookup.get(func_pointer, {})
        return func.get('score', 0.0)
    
    def _compute_precision_at_k(self, ranked_relevance: List[int], k_values: List[int]) -> Dict[int, float]:
        """
        Compute Precision@K for multiple K values.
        
        Args:
            ranked_relevance: List of 1s (relevant/matched) and 0s (not relevant) in ranked order
            k_values: List of K values to compute precision for
        
        Returns:
            Dict mapping K -> Precision@K
        """
        results = {}
        n = len(ranked_relevance)
        for k in k_values:
            if k <= 0:
                results[k] = 0.0
            elif k > n:
                # If k > n, use all items
                relevant_in_k = sum(ranked_relevance)
                results[k] = relevant_in_k / k  # Still divide by k, not n
            else:
                relevant_in_k = sum(ranked_relevance[:k])
                results[k] = relevant_in_k / k
        return results
    
    def _compute_ndcg(self, ranked_relevance: List[int], k: int = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG).
        
        DCG = sum(rel_i / log2(i+1)) for i = 1 to n (1-indexed)
        IDCG = ideal DCG where all relevant items are ranked first
        NDCG = DCG / IDCG
        
        Args:
            ranked_relevance: List of relevance scores (1 for match, 0 for no match) in ranked order
            k: Optional cutoff (compute NDCG@k). If None, compute for full list.
        
        Returns:
            NDCG score between 0 and 1
        """
        if not ranked_relevance:
            return 0.0
        
        # Apply cutoff if specified
        if k is not None and k > 0:
            ranked_relevance = ranked_relevance[:k]
        
        n = len(ranked_relevance)
        
        # Compute DCG
        dcg = 0.0
        for i, rel in enumerate(ranked_relevance):
            # i is 0-indexed, so position is i+1
            dcg += rel / math.log2(i + 2)  # log2(i+2) because position is 1-indexed
        
        # Compute IDCG (ideal ranking: all 1s come first)
        total_relevant = sum(ranked_relevance)
        ideal_relevance = [1] * total_relevant + [0] * (n - total_relevant)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(self, chains: List[Tuple[str, List[str], dict, Optional[str]]]) -> Dict:
        """Evaluate feature coverage."""
        covered = set()
        covered_exact = set()
        covered_partial = set()
        correct = []
        incorrect = []
        
        for seq, test_ids, record, func_text in chains:
            func_pointer = record.get('func_pointer')
            score = self._get_score(func_pointer)
            
            # Use typed matching to track exact vs partial matches
            if self.partial_match:
                matches_with_type = self.parser.find_matching_features_with_type(seq)
                matches = [m[0] for m in matches_with_type]
                match_types = {m[0]: m[1] for m in matches_with_type}
            else:
                matches = self.parser.find_matching_features(seq)
                match_types = {m: 'exact' for m in matches}
            
            if matches:
                correct.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'features': matches,
                    'match_types': match_types,
                    'func_pointer': func_pointer,
                    'inferred_feature': func_text,
                    'score': score
                })
                covered.update(matches)
                for m in matches:
                    if match_types.get(m) == 'exact':
                        covered_exact.add(m)
                    else:
                        covered_partial.add(m)
            else:
                incorrect.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'func_pointer': func_pointer,
                    'inferred_feature': func_text,
                    'score': score
                })
        
        n_gen = len(chains)
        n_correct_seq = len(correct)  # Number of sequences that matched (can have duplicates)
        n_covered = len(covered)      # Number of unique features covered
        
        # Precision = unique correctly covered features / total generated features
        # (NOT n_correct_seq, which double-counts when multiple sequences match the same benchmark feature)
        precision = n_covered / n_gen if n_gen > 0 else 0
        recall = n_covered / self.total_features if self.total_features > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Matthews Correlation Coefficient (MCC)
        # Universe = all features in the FD (total_functionalities)
        # TP = covered benchmark features (generated AND in benchmark)
        # FP = generated features not in benchmark
        # FN = benchmark features not generated
        # TN = FD features correctly filtered out (not generated AND not in benchmark)
        tp = n_covered
        fp = n_gen - n_covered
        fn = self.total_features - n_covered
        tn = self.total_functionalities - tp - fp - fn if self.total_functionalities > 0 else 0
        
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        numerator = (tp * tn) - (fp * fn)
        denominator_terms = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denominator_terms > 0:
            mcc = numerator / math.sqrt(denominator_terms)
        else:
            mcc = 0.0
        
        # Compute ranking metrics (Precision@K and NDCG)
        # Build ranked relevance list: chains are already sorted by score in get_all_chains
        # We need to create a binary relevance list (1 if matched, 0 if not)
        all_items = correct + incorrect
        # Sort by score descending to ensure proper ranking
        all_items_sorted = sorted(all_items, key=lambda x: x.get('score', 0), reverse=True)
        ranked_relevance = [1 if 'features' in item and item['features'] else 0 for item in all_items_sorted]
        
        # Precision@K for common k values
        k_values = [1, 3, 5, 10, 20]
        precision_at_k = self._compute_precision_at_k(ranked_relevance, k_values)
        
        # NDCG for various cutoffs
        ndcg_full = self._compute_ndcg(ranked_relevance)
        ndcg_at_k = {}
        for k in k_values:
            ndcg_at_k[k] = self._compute_ndcg(ranked_relevance, k)
        
        return {
            'metrics': {
                'total_features': self.total_features,
                'total_generated': n_gen,
                'correct': n_covered,  # Unique features covered (no duplicates)
                'correct_sequences': n_correct_seq,  # Sequences that matched (may have duplicates)
                'covered': n_covered,
                'covered_exact': len(covered_exact),
                'covered_partial': len(covered_partial),
                'precision': precision,
                'recall': recall,
                'feature_coverage': recall,
                'f1': f1,
                'mcc': mcc,
                'precision_at_k': precision_at_k,
                'ndcg': ndcg_full,
                'ndcg_at_k': ndcg_at_k
            },
            'covered_features': sorted(covered),
            'covered_features_exact': sorted(covered_exact),
            'covered_features_partial': sorted(covered_partial),
            'uncovered_features': sorted([
                f for f in self.parser.grammar.keys() if f not in covered
            ]),
            'matched': correct,
            'unmatched': incorrect
        }


# ============================================================
# LOCAL JSON LOADING FUNCTIONS
# ============================================================

def get_saves_dir() -> Path:
    """Get the saves directory path"""
    return Path(__file__).parent / "saves"


def list_available_saves() -> List[str]:
    """List all available save folders"""
    saves_dir = get_saves_dir()
    if not saves_dir.exists():
        return []
    
    folders = []
    for item in saves_dir.iterdir():
        if item.is_dir():
            func_file = item / "functionality.json"
            afd_file = item / "action-functionality.json"
            if func_file.exists() and afd_file.exists():
                folders.append(item.name)
    
    return sorted(folders)


def detect_app_from_folder(folder_name: str) -> Optional[str]:
    """Detect the app name from the folder name"""
    folder_lower = folder_name.lower()
    for app_name in BENCHMARKS.keys():
        if app_name.lower() in folder_lower:
            return app_name
    return None


def load_local_data(save_folder: str, apply_score_threshold: bool = True, score_threshold: float = None):
    """
    Load data from local JSON files.
    
    Args:
        save_folder: Name of the folder in saves/
        apply_score_threshold: If True, filter features by score >= threshold
        score_threshold: Explicit threshold value (uses get_active_score_threshold() if None)
    
    Returns:
        Tuple of (func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold)
    """
    saves_dir = get_saves_dir()
    folder_path = saves_dir / save_folder
    
    func_file = folder_path / "functionality.json"
    afd_file = folder_path / "action-functionality.json"
    
    if not func_file.exists():
        raise FileNotFoundError(f"functionality.json not found in {folder_path}")
    if not afd_file.exists():
        raise FileNotFoundError(f"action-functionality.json not found in {folder_path}")
    
    active_threshold = score_threshold if score_threshold is not None else get_active_score_threshold()
    
    print(f"   Loading {func_file.name}...")
    with open(func_file, 'r', encoding='utf-8') as f:
        func_records = json.load(f)
    
    print(f"   Loading {afd_file.name}...")
    with open(afd_file, 'r', encoding='utf-8') as f:
        action_func_records = json.load(f)
    
    # Build lookup dictionaries for functionalities by their _id
    func_lookup = {}
    filtered_lookup = {}
    filtered_count = 0
    
    for func in func_records:
        func_id = str(func.get('_id'))
        
        if apply_score_threshold:
            score = func.get('score', 0)
            if score < active_threshold:
                filtered_count += 1
                filtered_lookup[func_id] = func
                continue
        
        func_lookup[func_id] = func
    
    return func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold


def explore_saves():
    """Explore available save folders"""
    print("\n" + "="*60)
    print("📁 AVAILABLE SAVES")
    print("="*60)
    
    folders = list_available_saves()
    
    if not folders:
        print("\n⚠️  No saves found in 'saves/' directory")
        print("   Expected structure:")
        print("     saves/<model_name>/functionality.json")
        print("     saves/<model_name>/action-functionality.json")
        return
    
    print(f"\nFound {len(folders)} save folder(s):\n")
    
    for folder in folders:
        app = detect_app_from_folder(folder)
        app_str = f"(detected: {app})" if app else "(unknown app)"
        print(f"   • {folder} {app_str}")
    
    print("\n" + "-"*60)
    print("Usage: python evaluate_local.py <folder_name>")
    print("Example: python evaluate_local.py petclinic_gemma3_27b")


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_save(save_folder: str, grammar: Dict[str, str] = None, verbose: bool = True, dedup_by_feature: bool = True, partial_match: bool = False):
    """
    Complete evaluation pipeline for a saved run.
    
    Args:
        save_folder: Name of the folder in saves/
        grammar: Optional custom grammar. If None, auto-detects from folder name
        verbose: Whether to print detailed output
        dedup_by_feature: Whether to deduplicate unmatched sequences by inferred feature name
        partial_match: If True, also match sequences that are prefixes of expected patterns
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print(f"📊 AUTOE2E EVALUATION: {save_folder}")
    print("="*60)
    
    # Auto-detect app
    app_name = detect_app_from_folder(save_folder)
    if app_name:
        print(f"   Detected app: {app_name}")
    else:
        print(f"   ⚠️  Could not detect app from folder name")
    
    # Load data
    print(f"\n📥 Loading data from saves/{save_folder}/...")
    try:
        func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold = load_local_data(save_folder)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None
    
    print(f"   ✓ Total Functionalities:               {len(func_records)}")
    print(f"   ✓ Passed score threshold (≥{active_threshold:.3f}): {len(func_lookup)}")
    print(f"   ✗ Filtered out (below threshold):      {filtered_count}")
    print(f"   ✓ Action-Functionality mappings (AFD): {len(action_func_records)}")
    
    # Show breakdown of valid features
    final_features = [f for f in func_lookup.values() if f.get('final', False)]
    non_final_features = [f for f in func_lookup.values() if not f.get('final', False)]
    print(f"   ✓ Features marked as final (testable): {len(final_features)}")
    print(f"   ℹ️  Features not yet final: {len(non_final_features)}")
    
    if not action_func_records:
        print(f"\n⚠️  No action-functionality data found")
        return None
    
    # Get grammar
    if grammar is None:
        if app_name and app_name in BENCHMARKS:
            grammar = BENCHMARKS[app_name]
        else:
            print(f"\n⚠️  No benchmark grammar found for {app_name or save_folder}")
            print(f"   Available: {list(BENCHMARKS.keys())}")
            return None
    
    print(f"   ✓ Benchmark features to match: {len(grammar)}")
    
    # Reconstruct chains
    print(f"\n🔄 Reconstructing action chains (AUTOE2E algorithm)...")
    print(f"   Step 1: Filter to final features only")
    print(f"   Step 2: Find leaf AFD records (max depth per feature)")
    print(f"   Step 3: Trace back through prev_state/prev_action")
    print(f"   Step 4: Filter out single-action navigation")
    
    reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
    chains = reconstructor.get_all_chains()
    print(f"   ✓ Valid feature chains extracted: {len(chains)}")
    
    # Deduplicate: keep only highest-scoring chain for each sequence
    print(f"   Step 5: Deduplicate sequences (keep highest score)")
    seq_best = {}
    for chain in chains:
        seq, test_ids, record, func_text = chain
        func_pointer = record.get('func_pointer', '')
        score = func_lookup.get(func_pointer, {}).get('score', 0)
        if seq not in seq_best or score > seq_best[seq][1]:
            seq_best[seq] = (chain, score)
    chains = [item[0] for item in seq_best.values()]
    print(f"   ✓ Unique sequences after deduplication: {len(chains)}")
    
    # Optional: Deduplicate by inferred feature name
    if dedup_by_feature:
        print(f"   Step 6: Deduplicate by feature (matched by benchmark feature, unmatched by inferred name)")
        
        temp_parser = FeatureGrammarParser(grammar, partial_match=partial_match)
        
        matched_chains = []
        unmatched_chains = []
        
        for chain in chains:
            seq, test_ids, record, func_text = chain
            matches = temp_parser.find_matching_features(seq)
            if matches:
                matched_chains.append(chain)
            else:
                unmatched_chains.append(chain)
        
        # Dedup matched chains by benchmark feature name (1 per covered feature)
        benchmark_feature_best = {}
        for chain in matched_chains:
            seq, test_ids, record, func_text = chain
            matches = temp_parser.find_matching_features(seq)
            func_pointer = record.get('func_pointer', '')
            score = func_lookup.get(func_pointer, {}).get('score', 0)
            for feature_name in matches:
                if feature_name not in benchmark_feature_best or score > benchmark_feature_best[feature_name][1]:
                    benchmark_feature_best[feature_name] = (chain, score)
        
        # Collect unique chains (a chain might be the best for multiple benchmark features)
        seen_matched_seqs = set()
        deduped_matched = []
        for feature_name, (chain, score) in benchmark_feature_best.items():
            seq = chain[0]
            if seq not in seen_matched_seqs:
                seen_matched_seqs.add(seq)
                deduped_matched.append(chain)
        
        # Dedup unmatched chains by inferred feature name
        feature_best = {}
        for chain in unmatched_chains:
            seq, test_ids, record, func_text = chain
            feature_key = (func_text or "").strip().lower()
            if not feature_key:
                feature_key = f"__unnamed_{seq}"
            func_pointer = record.get('func_pointer', '')
            score = func_lookup.get(func_pointer, {}).get('score', 0)
            if feature_key not in feature_best or score > feature_best[feature_key][1]:
                feature_best[feature_key] = (chain, score)
        
        deduplicated_unmatched = [item[0] for item in feature_best.values()]
        
        chains = deduped_matched + deduplicated_unmatched
        print(f"   ✓ Matched sequences (before dedup): {len(matched_chains)}")
        print(f"   ✓ Matched after dedup by benchmark feature: {len(deduped_matched)}")
        print(f"   ✓ Unmatched after dedup by feature: {len(deduplicated_unmatched)}")
        print(f"   ✓ Total unique features: {len(chains)}")
    else:
        print(f"   Step 6: Skipped (deduplication by feature disabled)")
    
    if verbose:
        print(f"\n📋 Extracted Feature Chains:")
        print("-" * 90)
        for seq, test_ids, record, func_text in chains:
            func_display = func_text if func_text else "(no functionality linked)"
            depth = record.get('depth', 0)
            func_pointer = record.get('func_pointer', '')
            score = func_lookup.get(func_pointer, {}).get('score', 0)
            print(f"   Feature:   {func_display}")
            print(f"   Score:     {score:.3f}")
            print(f"   Sequence:  {seq}")
            print(f"   Depth:     {depth}")
            print(f"   Test IDs:  {test_ids}")
            print("-" * 90)
    
    # Check below-threshold features
    below_threshold_chains = []
    if filtered_lookup:
        print(f"\n🔍 Checking below-threshold features for potential matches...")
        below_reconstructor = ActionChainReconstructor(action_func_records, filtered_lookup)
        below_threshold_chains = below_reconstructor.get_all_chains()
        print(f"   ✓ Below-threshold chains: {len(below_threshold_chains)}")
    
    # Evaluate
    print(f"\n🎯 Evaluating against benchmark...")
    if partial_match:
        print(f"   ℹ️  Partial matching enabled (prefix sequences count as matches)")
    evaluator = FeatureCoverageEvaluator(grammar, func_lookup, partial_match=partial_match, total_functionalities=len(func_records))
    results = evaluator.evaluate(chains)
    
    # Find matches in below-threshold chains
    below_threshold_matches = []
    if below_threshold_chains:
        for seq, test_ids, record, func_text in below_threshold_chains:
            func_pointer = record.get('func_pointer')
            score = filtered_lookup.get(func_pointer, {}).get('score', 0)
            matches = evaluator.parser.find_matching_features(seq)
            if matches:
                below_threshold_matches.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'features': matches,
                    'func_pointer': func_pointer,
                    'inferred_feature': func_text,
                    'score': score
                })
    
    results['below_threshold_matches'] = below_threshold_matches
    results['active_threshold'] = active_threshold
    results['grammar'] = grammar
    results['save_folder'] = save_folder
    results['app_name'] = app_name
    results['partial_match'] = partial_match
    
    # Add exploration statistics
    final_features = [f for f in func_lookup.values() if f.get('final', False)]
    results['exploration_stats'] = {
        'total_functionalities': len(func_records),
        'passed_threshold': len(func_lookup),
        'filtered_out': filtered_count,
        'afd_mappings': len(action_func_records),
        'final_features': len(final_features),
        'benchmark_features': len(grammar)
    }
    
    # Print results
    print_results(results, verbose)
    
    return results


def print_results(results: Dict, verbose: bool = True):
    """Print formatted results"""
    m = results['metrics']
    active_threshold = results.get('active_threshold', DEFAULT_SCORE_THRESHOLD)
    partial_match = results.get('partial_match', False)
    
    # Build coverage breakdown string
    if partial_match and m.get('covered_exact', 0) + m.get('covered_partial', 0) > 0:
        coverage_breakdown = f" (exact: {m.get('covered_exact', 0)}, partial: {m.get('covered_partial', 0)})"
    else:
        coverage_breakdown = ""
    
    # Format Precision@K
    p_at_k = m.get('precision_at_k', {})
    
    # Format NDCG@K
    ndcg_at_k = m.get('ndcg_at_k', {})
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                     EVALUATION RESULTS                         ║
╠════════════════════════════════════════════════════════════════╣
║  Total Features in Benchmark:           {m['total_features']:>20}  ║
║  Total Test Sequences Generated:        {m['total_generated']:>20}  ║
║  Correct Test Sequences:                {m['correct']:>20}  ║
║  Features Covered:                      {m['covered']:>20}{coverage_breakdown}  ║
╠════════════════════════════════════════════════════════════════╣
║  PRECISION:                             {m['precision']:>19.2%}  ║
║  RECALL (FEATURE COVERAGE):             {m['recall']:>19.2%}  ║
║  F1 SCORE:                              {m['f1']:>19.2%}  ║
║  MCC:                                   {m['mcc']:>19.4f}  ║
╠════════════════════════════════════════════════════════════════╣
║  NDCG (full):                           {m.get('ndcg', 0):>19.4f}  ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    # Print ranking metrics in a cleaner format below the box
    print("📊 RANKING METRICS:")
    # Precision@K
    p_at_k_items = [(k, v) for k, v in sorted(p_at_k.items()) if k <= m['total_generated']]
    if p_at_k_items:
        p_str = "   Precision@K:  " + "  ".join([f"P@{k}: {v:.1%}" for k, v in p_at_k_items])
        print(p_str)
    # NDCG@K
    ndcg_at_k_items = [(k, v) for k, v in sorted(ndcg_at_k.items()) if k <= m['total_generated']]
    if ndcg_at_k_items:
        ndcg_str = "   NDCG@K:       " + "  ".join([f"@{k}: {v:.3f}" for k, v in ndcg_at_k_items])
        print(ndcg_str)
    print()
    
    # Always show matched sequences
    print("📋 MATCHED SEQUENCES:")
    print("-" * 95)
    for item in results['matched']:
        score = item.get('score', 0)
        inferred = item.get('inferred_feature') or "(no functionality linked)"
        match_types = item.get('match_types', {})
        # Show match type indicator if partial matching is on
        type_indicators = []
        for feat in item['features']:
            mtype = match_types.get(feat, 'exact')
            if mtype == 'partial':
                type_indicators.append(f"{feat} [PARTIAL]")
            else:
                type_indicators.append(feat)
        print(f"   [{score:.3f}] {item['sequence']}  →  {type_indicators}")
        print(f"            Inferred: {inferred}")
    print("-" * 95)
    
    # Always show uncovered features
    grammar = results.get('grammar', {})
    print("\n❌ UNCOVERED FEATURES (from benchmark):")
    for f in results['uncovered_features']:
        expected_seq = grammar.get(f, '?')
        print(f"   • {f}  (expected: {expected_seq})")
    
    if verbose:
        print("\n✅ COVERED FEATURES (from benchmark):")
        for f in results['covered_features']:
            print(f"   • {f}")
        
        print("\n📋 MATCHED SEQUENCES (with inferred functionality):")
        print("-" * 95)
        for item in results['matched']:
            inferred = item.get('inferred_feature') or "(no functionality linked)"
            score = item.get('score', 0)
            print(f"   Sequence: {item['sequence']}")
            print(f"   Score:    {score:.3f}")
            print(f"   Matched:  {item['features']}")
            print(f"   Inferred: {inferred}")
            print("-" * 95)
        
        if results['unmatched']:
            print(f"\n⚠️  UNMATCHED SEQUENCES ({len(results['unmatched'])} total):")
            print("-" * 95)
            for item in results['unmatched']:
                inferred = item.get('inferred_feature') or "(no functionality linked)"
                score = item.get('score', 0)
                print(f"   Sequence: {item['sequence']}")
                print(f"   Score:    {score:.3f}")
                print(f"   Inferred: {inferred}")
                print("-" * 95)
        
        below_threshold = results.get('below_threshold_matches', [])
        if below_threshold:
            print(f"\n🔻 MATCHED SEQUENCES BELOW THRESHOLD ({len(below_threshold)} total):")
            print(f"   (These match benchmark features but have score < {active_threshold:.3f})")
            print("-" * 95)
            for item in below_threshold:
                inferred = item.get('inferred_feature') or "(no functionality linked)"
                score = item.get('score', 0)
                print(f"   Sequence: {item['sequence']}")
                print(f"   Score:    {score:.3f} ❌ (below {active_threshold:.3f})")
                print(f"   Matched:  {item['features']}")
                print(f"   Inferred: {inferred}")
                print("-" * 95)


def export_results(results: Dict, filename: str):
    """Export results to JSON file"""
    export_data = {
        'save_folder': results.get('save_folder'),
        'app_name': results.get('app_name'),
        'metrics': results['metrics'],
        'covered_features': results['covered_features'],
        'uncovered_features': results['uncovered_features'],
        'matched_count': len(results['matched']),
        'unmatched_count': len(results['unmatched']),
        'matched_sequences': [
            {
                'sequence': m['sequence'], 
                'matched_features': m['features'],
                'inferred_feature': m.get('inferred_feature')
            } 
            for m in results['matched']
        ],
        'unmatched_sequences': [
            {
                'sequence': m['sequence'],
                'inferred_feature': m.get('inferred_feature')
            }
            for m in results['unmatched']
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Results exported to {filename}")


def evaluate_all_saves(verbose: bool = False, partial_match: bool = False) -> Dict[str, Dict]:
    """Evaluate all available saves and return summary"""
    folders = list_available_saves()
    
    if not folders:
        print("No saves found!")
        return {}
    
    print("\n" + "="*80)
    print("📊 EVALUATING ALL SAVES")
    if partial_match:
        print("   (Partial matching enabled)")
    print("="*80)
    print(f"Found {len(folders)} save folder(s)\n")
    
    all_results = {}
    
    for folder in folders:
        print(f"\n{'─'*80}")
        results = evaluate_save(folder, verbose=verbose, dedup_by_feature=True, partial_match=partial_match)
        if results:
            all_results[folder] = results
    
    # Print summary table
    print("\n" + "="*170)
    print("📈 SUMMARY")
    print("="*170)
    print(f"\n{'Folder':<35} {'Funcs':>8} {'Final':>8} {'Gen':>6} {'Correct':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'MCC':>8} {'NDCG':>8} {'P@5':>8} {'P@10':>8}")
    print("-"*170)
    
    for folder, results in all_results.items():
        m = results['metrics']
        es = results.get('exploration_stats', {})
        p_at_k = m.get('precision_at_k', {})
        print(f"{folder:<35} {es.get('total_functionalities', 0):>8} {es.get('final_features', 0):>8} {m['total_generated']:>6} {m['correct']:>8} {m['precision']:>7.1%} {m['recall']:>7.1%} {m['f1']:>7.1%} {m['mcc']:>7.4f} {m.get('ndcg', 0):>7.3f} {p_at_k.get(5, 0):>7.1%} {p_at_k.get(10, 0):>7.1%}")
    
    print("-"*150)
    
    return all_results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AUTOE2E Feature Coverage Evaluator (Local JSON)")
    parser.add_argument("save_folder", nargs="?", default=None, 
                        help="Save folder name to evaluate (e.g., petclinic_gemma3_27b)")
    parser.add_argument("--list", action="store_true",
                        help="List available save folders")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all available saves")
    parser.add_argument("--export", type=str, metavar="FILE",
                        help="Export results to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication by inferred feature name")
    parser.add_argument("--partial", action="store_true",
                        help="Enable partial matching (sequences matching navigation prefix count as matches)")
    
    args = parser.parse_args()
    
    if args.list or (args.save_folder is None and not args.all):
        explore_saves()
    elif args.all:
        all_results = evaluate_all_saves(verbose=not args.quiet, partial_match=args.partial)
        
        if args.export:
            summary = {
                folder: {
                    'metrics': r['metrics'],
                    'exploration_stats': r.get('exploration_stats', {}),
                    'covered_features': r['covered_features'],
                    'covered_features_exact': r.get('covered_features_exact', []),
                    'covered_features_partial': r.get('covered_features_partial', []),
                    'uncovered_features': r['uncovered_features']
                }
                for folder, r in all_results.items()
            }
            with open(args.export, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Summary exported to {args.export}")
    else:
        results = evaluate_save(args.save_folder, verbose=not args.quiet, dedup_by_feature=not args.no_dedup, partial_match=args.partial)
        
        if results and args.export:
            export_results(results, args.export)
