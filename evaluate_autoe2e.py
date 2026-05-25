"""
AUTOE2E Feature Coverage Evaluator
===================================

Run this script on your local machine to evaluate feature coverage.

Usage:
    python evaluate_autoe2e.py                    # Evaluate PETCLINIC
    python evaluate_autoe2e.py PETCLINIC          # Evaluate specific app
    python evaluate_autoe2e.py --explore          # Just explore the database

Requirements:
    pip install pymongo python-dotenv

"""

import os
import re
import sys
import json
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# =============================================================================
# AUTOE2E SCORE THRESHOLD
# =============================================================================
# The paper uses accumulated scores to identify real features vs noise.
# Score = sum of geometric_score(rank) over all observations
# geometric_score(rank=1) = log(0.5) = -0.693 (highest single observation)
# geometric_score(rank=2) = -1.386, rank=3 = -2.079, etc.
#
# Features that consistently appear at high ranks accumulate positive scores.
# Features that appear inconsistently or at low ranks stay near/below threshold.
#
# Default threshold: log(0.5) = -0.693
# - A feature seen once at rank 1 barely passes
# - A feature seen multiple times at rank 1 has score >> 0
# - Increase threshold to be more selective (e.g., 0, 1, 3, 5)
#
# Ablation Support (A7):
# - A7.1: Score threshold >= 0 (non-negative scores)
# - A7.2: Score threshold >= 1.0 (moderate confidence)
# - A7.3: Score threshold >= 2.0 (high confidence)
# =============================================================================
DEFAULT_SCORE_THRESHOLD = math.log(0.5)  # -0.693 (default from paper)

def get_active_score_threshold() -> float:
    """
    Get the active score threshold, considering ablation settings.
    
    Returns the ablation-configured threshold if in ablation mode,
    otherwise returns the default threshold.
    
    Returns:
        Active score threshold value
    """
    try:
        from autoe2e.ablation_integration import ABLATION_MODE, get_score_threshold
        if ABLATION_MODE:
            score_threshold = get_score_threshold()
            if score_threshold and score_threshold.is_enabled():
                return score_threshold.get_min_score()
    except ImportError:
        pass
    return DEFAULT_SCORE_THRESHOLD

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
}


# ============================================================
# CORE EVALUATION CLASSES
# ============================================================

class FeatureGrammarParser:
    """Parses and matches feature grammar rules"""
    
    def __init__(self, grammar: Dict[str, str]):
        self.grammar = grammar
        self.compiled_patterns = {
            name: re.compile(f"^{rule}$") 
            for name, rule in grammar.items()
        }
    
    def find_matching_features(self, action_sequence: str) -> List[str]:
        """Find all features that match the given action sequence"""
        return [
            name for name, pattern in self.compiled_patterns.items()
            if pattern.match(action_sequence)
        ]


class ActionChainReconstructor:
    """
    Reconstructs action chains from database records following AUTOE2E paper logic.
    
    AUTOE2E CHAIN EXTRACTION ALGORITHM:
    1. Filter FD (functionality DB) by score >= threshold to get VALID features
    2. For each VALID feature that has final=True in FD:
       a. Find all AFD records with func_pointer pointing to this feature
       b. Find the "leaf" records - those at maximum depth for this feature
       c. From each leaf, trace backwards using prev_state/prev_action
    3. The chain represents the sequence of actions to execute the feature
    
    The key is the test_id field which contains identifiers like:
    - "c3-navbar-owners-search" -> extract "c3"
    - "t1-search-input" -> extract "t1"
    
    For FORM type actions, it also handles:
    - form_fields: list of field identifiers that were filled (e.g., ['t2', 't3', 't4'])
    - submit_prefix: the submit button identifier (e.g., 'c31')
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
    
    def reconstruct_chain(self, record: dict) -> Tuple[str, List[str], Optional[str]]:
        """
        Build action sequence by tracing back through prev_state/prev_action.
        
        For FORM type actions, includes form fields and submit button:
        - Form fields are concatenated: t2t3t4t5t6 (to match regex like (t2|t3|t4|t5|t6)+)
        - Submit button is appended: c31
        
        Returns:
            Tuple of (action_sequence_string, list_of_test_ids, root_state_url)
            root_state_url is the starting point of the chain (where BFS began)
        """
        prefixes = []
        test_ids = []
        visited = set()
        current = record
        is_first = True  # Track if this is the first (final) action
        root_state = None  # Track the root of the chain (BFS starting point)
        
        # Store form suffix to append at the end (form fields + submit)
        form_suffix = ""
        form_suffix_test_ids = []
        
        while current:
            state = current.get('state')
            if state in visited:
                break
            visited.add(state)
            
            # Track the root state (will be overwritten until we reach the root)
            root_state = state
            
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
        
        return final_sequence, final_test_ids, root_state
    
    def find_leaf_records_for_feature(self, feature_id: str) -> List[dict]:
        """
        Find the best AFD records for a feature to reconstruct chains from.
        
        Since BFS doesn't have natural termination points, we use heuristics:
        1. Prefer records at higher depth (longer chains = more context)
        2. Prefer records with valid test_id (can be traced)
        3. If multiple records at same depth, include all (different paths to same feature)
        
        Returns records that are good candidates for chain reconstruction.
        """
        feature_records = self.records_by_feature.get(feature_id, [])
        if not feature_records:
            return []
        
        # Filter to records that have a valid test_id (can contribute to chain)
        records_with_test_id = [r for r in feature_records if r.get('test_id')]
        
        # If no records have test_id, fall back to all records
        candidates = records_with_test_id if records_with_test_id else feature_records
        
        if not candidates:
            return []
        
        # Find the maximum depth among candidates
        max_depth = max(r.get('depth', 0) for r in candidates)
        
        # Return records at max depth (these have the longest chains)
        # Also include records at depth-1 if max_depth > 0, to catch alternative paths
        if max_depth > 0:
            return [r for r in candidates if r.get('depth', 0) >= max_depth - 1]
        else:
            return candidates
    
    def get_all_chains(self, include_non_final: bool = False) -> List[Tuple[str, List[str], dict, Optional[str], Optional[str]]]:
        """
        Get all unique action chains following AUTOE2E algorithm.
        
        AUTOE2E SCORING MECHANISM:
        - Each AFD record has rank_score = geometric_score(LLM_rank)
        - FD features accumulate scores: features consistently ranked high get high scores
        - Score threshold filters real features from noise
        
        CHAIN SELECTION:
        1. Filter FD by score >= threshold (keeps real features)
        2. For each valid feature with final=True:
           - Find AFD records pointing to this feature
           - Select deepest records (longest chains)
           - Reconstruct chain by tracing prev_state/prev_action
        
        The score IS the filter - no arbitrary chain length/depth thresholds needed.
        
        Args:
            include_non_final: If True, include ALL features (not just final=True)
        
        Returns:
            List of tuples: (sequence_string, test_id_chain, original_record, functionality_text, root_state)
        """
        chains = []
        seen = set()  # (feature_id, sequence) pairs to avoid duplicates
        
        if include_non_final:
            # Include ALL features that passed the score threshold
            selected_features = self.func_lookup
        else:
            # Filter to features with final=True (LLM confirmed these complete a user goal)
            # The score threshold was already applied when building func_lookup
            final_features = {
                fid: f for fid, f in self.func_lookup.items() 
                if f.get('final', False)
            }
            
            # If very few final features, also include top-scoring non-final ones
            if len(final_features) < 5:
                non_final = {fid: f for fid, f in self.func_lookup.items() if not f.get('final', False)}
                sorted_non_final = sorted(non_final.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:10]
                final_features = {**final_features, **dict(sorted_non_final)}
            
            selected_features = final_features
        
        for feature_id, feature in selected_features.items():
            feature_text = feature.get('text', '')
            feature_score = feature.get('score', 0)
            
            # Find AFD records for this feature
            candidate_records = self.find_leaf_records_for_feature(feature_id)
            
            if not candidate_records:
                continue
            
            for record in candidate_records:
                seq, test_ids, root_state = self.reconstruct_chain(record)
                
                # Skip empty sequences
                if not seq:
                    continue
                
                # Deduplicate by (feature_id, sequence)
                key = (feature_id, seq)
                if key in seen:
                    continue
                seen.add(key)
                
                chains.append((seq, test_ids, record, feature_text, root_state))
        
        # Sort by feature score (higher score = more confident feature)
        chains.sort(key=lambda x: self.func_lookup.get(x[2].get('func_pointer', ''), {}).get('score', 0), reverse=True)
        
        return chains
    
    def get_all_chains_legacy(self) -> List[Tuple[str, List[str], dict, Optional[str], Optional[str]]]:
        """
        LEGACY: Get all unique action chains (old behavior for comparison).
        
        This is the old method that iterates all AFD records.
        Kept for debugging/comparison purposes.
        """
        seen = set()
        chains = []
        
        for record in self.records:
            # Only include records whose functionality passed the score threshold
            func_pointer = record.get('func_pointer')
            if func_pointer and func_pointer not in self.func_lookup:
                continue  # Skip - feature was filtered out due to low score
            
            seq, test_ids, root_state = self.reconstruct_chain(record)
            if seq and seq not in seen:
                seen.add(seq)
                func_text = self.get_functionality_text(func_pointer)
                chains.append((seq, test_ids, record, func_text, root_state))
        
        return chains


class FeatureCoverageEvaluator:
    """Main evaluator that computes all metrics"""
    
    def __init__(self, grammar: Dict[str, str], func_lookup: Dict[str, dict] = None):
        self.parser = FeatureGrammarParser(grammar)
        self.total_features = len(grammar)
        self.func_lookup = func_lookup or {}
    
    def _get_score(self, func_pointer: str) -> float:
        """Get the score for a feature by its func_pointer"""
        if not func_pointer or not self.func_lookup:
            return 0.0
        func = self.func_lookup.get(func_pointer, {})
        return func.get('score', 0.0)
    
    def evaluate(self, chains: List[Tuple]) -> Dict:
        """
        Evaluate feature coverage.
        
        Args:
            chains: List of (sequence, test_ids, record, func_text, [root_state]) tuples
            
        Returns:
            Dictionary with all metrics
        """
        covered = set()
        correct = []
        incorrect = []
        
        for chain in chains:
            # Handle both old 4-tuple and new 5-tuple format
            if len(chain) >= 5:
                seq, test_ids, record, func_text, root_state = chain[:5]
            else:
                seq, test_ids, record, func_text = chain[:4]
                root_state = None
            func_pointer = record.get('func_pointer')
            score = self._get_score(func_pointer)
            
            matches = self.parser.find_matching_features(seq)
            if matches:
                correct.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'features': matches,
                    'func_pointer': func_pointer,
                    'inferred_feature': func_text,
                    'score': score
                })
                covered.update(matches)
            else:
                incorrect.append({
                    'sequence': seq,
                    'test_ids': test_ids,
                    'func_pointer': func_pointer,
                    'inferred_feature': func_text,
                    'score': score
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
                'feature_coverage': recall,  # Alias for clarity
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
# HELPER FUNCTIONS
# ============================================================

def _extract_url_path(state_hash_or_url: str) -> str:
    """
    Extract a readable path from a state identifier.
    
    The state could be:
    - A full URL: http://localhost:3000/projects/123/terms -> /projects/.../terms
    - A hash: abc123def456 -> (hash)
    - None -> (unknown)
    """
    if not state_hash_or_url:
        return "(unknown)"
    
    # Check if it looks like a URL
    if state_hash_or_url.startswith(('http://', 'https://')):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(state_hash_or_url)
            path = parsed.path or '/'
            # Truncate long paths
            if len(path) > 50:
                parts = path.split('/')
                if len(parts) > 4:
                    path = '/'.join(parts[:2]) + '/.../' + '/'.join(parts[-2:])
            return path
        except:
            pass
    
    # Check if it looks like a hash (alphanumeric, no slashes)
    if len(state_hash_or_url) > 20 and '/' not in state_hash_or_url:
        return f"(state:{state_hash_or_url[:8]}...)"
    
    # Return as-is if short enough, otherwise truncate
    if len(state_hash_or_url) > 40:
        return state_hash_or_url[:37] + "..."
    return state_hash_or_url


# ============================================================
# MONGODB FUNCTIONS
# ============================================================

def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    uri = os.getenv("ATLAS_URI")
    if not uri:
        raise ValueError("ATLAS_URI not found in environment. Create a .env file.")
    
    client = MongoClient(uri)
    db = client.myDatabase
    return client, db


def load_data(db, app_name: str, apply_score_threshold: bool = True, score_threshold: float = None):
    """
    Load data from MongoDB collections.
    
    Args:
        db: MongoDB database connection
        app_name: Name of the application
        apply_score_threshold: If True, filter features by score >= threshold
        score_threshold: Explicit threshold value (uses get_active_score_threshold() if None)
    
    Returns:
        Tuple of (func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold)
    """
    # Get active threshold (from ablation or default)
    active_threshold = score_threshold if score_threshold is not None else get_active_score_threshold()
    
    func_records = list(db["functionality"].find({"app": app_name}))
    action_func_records = list(db["action-functionality"].find({"app": app_name}))
    
    # Build lookup dictionaries for functionalities by their _id
    func_lookup = {}  # Features that pass threshold
    filtered_lookup = {}  # Features below threshold (for reporting)
    filtered_count = 0
    for func in func_records:
        # Handle both ObjectId and string _id formats
        func_id = str(func.get('_id'))
        
        # Apply score threshold filtering if enabled
        if apply_score_threshold:
            score = func.get('score', 0)
            if score < active_threshold:
                filtered_count += 1
                filtered_lookup[func_id] = func  # Keep for below-threshold reporting
                continue  # Skip features below threshold
        
        func_lookup[func_id] = func
    
    return func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold


def explore_database():
    """Explore the database structure"""
    print("\n" + "="*60)
    print("🔍 EXPLORING DATABASE")
    print("="*60)
    
    client, db = connect_to_mongodb()
    
    try:
        # List collections
        collections = db.list_collection_names()
        print(f"\n📁 Collections in 'myDatabase':")
        for coll in collections:
            count = db[coll].count_documents({})
            print(f"   • {coll}: {count} documents")
        
        # Apps in functionality
        if "functionality" in collections:
            apps = db["functionality"].distinct("app")
            print(f"\n📱 Apps with functionality data:")
            for app in apps:
                count = db["functionality"].count_documents({"app": app})
                print(f"   • {app}: {count} functionalities")
        
        # Apps in action-functionality
        if "action-functionality" in collections:
            apps = db["action-functionality"].distinct("app")
            print(f"\n🔗 Apps with action-functionality data:")
            for app in apps:
                count = db["action-functionality"].count_documents({"app": app})
                print(f"   • {app}: {count} mappings")
                
                # Show sample test_ids
                sample_test_ids = db["action-functionality"].distinct(
                    "test_id", 
                    {"app": app, "test_id": {"$ne": None}}
                )[:10]
                if sample_test_ids:
                    print(f"      Sample test_ids: {sample_test_ids}")
    
    finally:
        client.close()


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_app(app_name: str, grammar: Dict[str, str] = None, verbose: bool = True, dedup_by_feature: bool = True):
    """
    Complete evaluation pipeline for an app.
    
    Args:
        app_name: Name of the app to evaluate (e.g., "PETCLINIC")
        grammar: Optional custom grammar. If None, uses BENCHMARKS[app_name]
        verbose: Whether to print detailed output
        dedup_by_feature: Whether to deduplicate unmatched sequences by inferred feature name
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print(f"📊 AUTOE2E EVALUATION: {app_name}")
    print("="*60)
    
    # Connect
    print("\n📡 Connecting to MongoDB...")
    client, db = connect_to_mongodb()
    
    try:
        # Load data
        print(f"📥 Loading data...")
        func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold = load_data(db, app_name)
        print(f"   ✓ Total Functionalities in DB:         {len(func_records)}")
        print(f"   ✓ Passed score threshold (≥{active_threshold:.3f}): {len(func_lookup)}")
        print(f"   ✗ Filtered out (below threshold):      {filtered_count}")
        print(f"   ✓ Action-Functionality mappings (AFD): {len(action_func_records)}")
        
        # Show breakdown of valid features
        final_features = [f for f in func_lookup.values() if f.get('final', False)]
        non_final_features = [f for f in func_lookup.values() if not f.get('final', False)]
        print(f"   ✓ Features marked as final (testable): {len(final_features)}")
        print(f"   ℹ️  Features not yet final: {len(non_final_features)}")
        
        if not action_func_records:
            print(f"\n⚠️  No data found for {app_name}")
            return None
        
        # Get grammar
        if grammar is None:
            if app_name not in BENCHMARKS:
                print(f"\n⚠️  No benchmark for {app_name}")
                print(f"   Available: {list(BENCHMARKS.keys())}")
                return None
            grammar = BENCHMARKS[app_name]
        
        print(f"   ✓ Benchmark features to match: {len(grammar)}")
        
        # Reconstruct chains (now with func_lookup for feature text)
        # Include ALL features by default (not just final ones)
        print(f"\n🔄 Reconstructing action chains (AUTOE2E algorithm)...")
        print(f"   Step 1: Include ALL features (final + non-final)")
        print(f"   Step 2: Find leaf AFD records (max depth per feature)")
        print(f"   Step 3: Trace back through prev_state/prev_action")
        print(f"   Step 4: Filter out single-action navigation")
        
        reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
        chains = reconstructor.get_all_chains(include_non_final=True)  # Include all features
        print(f"   ✓ Valid feature chains extracted: {len(chains)}")
        
        # Deduplicate: keep only highest-scoring chain for each sequence
        print(f"   Step 5: Deduplicate sequences (keep highest score)")
        seq_best = {}
        for chain in chains:
            seq, test_ids, record, func_text, root_state = chain
            func_pointer = record.get('func_pointer', '')
            score = func_lookup.get(func_pointer, {}).get('score', 0)
            if seq not in seq_best or score > seq_best[seq][1]:
                seq_best[seq] = (chain, score)
        chains = [item[0] for item in seq_best.values()]
        print(f"   ✓ Unique sequences after deduplication: {len(chains)}")
        
        # Optional: Deduplicate by inferred feature name
        if dedup_by_feature:
            # Deduplicate: keep only highest-scoring chain for each inferred feature
            # BUT keep all sequences that match the benchmark (even if same inferred feature)
            print(f"   Step 6: Deduplicate by inferred feature (keep matched + highest score unmatched)")
            
            # Create a temporary parser to check which sequences match benchmark
            temp_parser = FeatureGrammarParser(grammar)
            
            matched_chains = []  # Keep all that match benchmark
            unmatched_chains = []  # Will be deduplicated by feature name
            
            for chain in chains:
                seq, test_ids, record, func_text, root_state = chain
                matches = temp_parser.find_matching_features(seq)
                if matches:
                    matched_chains.append(chain)
                else:
                    unmatched_chains.append(chain)
            
            # Deduplicate only unmatched chains by inferred feature name
            feature_best = {}
            for chain in unmatched_chains:
                seq, test_ids, record, func_text, root_state = chain
                # Normalize feature name for comparison (lowercase, strip whitespace)
                feature_key = (func_text or "").strip().lower()
                if not feature_key:
                    feature_key = f"__unnamed_{seq}"  # Keep unnamed features separate
                func_pointer = record.get('func_pointer', '')
                score = func_lookup.get(func_pointer, {}).get('score', 0)
                if feature_key not in feature_best or score > feature_best[feature_key][1]:
                    feature_best[feature_key] = (chain, score)
            
            deduplicated_unmatched = [item[0] for item in feature_best.values()]
            
            # Combine: all matched + deduplicated unmatched
            chains = matched_chains + deduplicated_unmatched
            print(f"   ✓ Matched sequences (kept all): {len(matched_chains)}")
            print(f"   ✓ Unmatched after dedup by feature: {len(deduplicated_unmatched)}")
            print(f"   ✓ Total unique features: {len(chains)}")
        else:
            print(f"   Step 6: Skipped (deduplication by feature disabled)")
        
        if verbose:
            print(f"\n📋 Extracted Feature Chains:")
            print("-" * 90)
            for chain in chains:
                seq, test_ids, record, func_text, root_state = chain
                func_display = func_text if func_text else "(no functionality linked)"
                depth = record.get('depth', 0)
                func_pointer = record.get('func_pointer', '')
                score = func_lookup.get(func_pointer, {}).get('score', 0)
                # Extract a short version of root_state for display
                root_display = _extract_url_path(root_state) if root_state else "(unknown)"
                print(f"   Feature:   {func_display}")
                print(f"   Score:     {score:.3f}")
                print(f"   Sequence:  {seq}")
                print(f"   Depth:     {depth}")
                print(f"   Root:      {root_display}")
                print(f"   Test IDs:  {test_ids}")
                print("-" * 90)
        
        # Also reconstruct chains from below-threshold features (for reporting)
        below_threshold_chains = []
        if filtered_lookup:
            print(f"\n🔍 Checking below-threshold features for potential matches...")
            below_reconstructor = ActionChainReconstructor(action_func_records, filtered_lookup)
            below_threshold_chains = below_reconstructor.get_all_chains()
            print(f"   ✓ Below-threshold chains: {len(below_threshold_chains)}")
        
        # Evaluate
        print(f"\n🎯 Evaluating against benchmark...")
        evaluator = FeatureCoverageEvaluator(grammar, func_lookup)
        results = evaluator.evaluate(chains)
        
        # Find matches in below-threshold chains
        below_threshold_matches = []
        if below_threshold_chains:
            for chain in below_threshold_chains:
                seq, test_ids, record, func_text, root_state = chain
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
        
        # Print results
        print_results(results, verbose)
        
        return results
        
    finally:
        client.close()


def print_results(results: Dict, verbose: bool = True):
    """Print formatted results"""
    m = results['metrics']
    active_threshold = results.get('active_threshold', DEFAULT_SCORE_THRESHOLD)
    
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
    
    # Always show matched sequences (even in non-verbose mode)
    print("📋 MATCHED SEQUENCES:")
    print("-" * 95)
    for item in results['matched']:
        score = item.get('score', 0)
        inferred = item.get('inferred_feature') or "(no functionality linked)"
        print(f"   [{score:.3f}] {item['sequence']}  →  {item['features']}")
        print(f"            Inferred: {inferred}")
    print("-" * 95)
    
    # Always show uncovered features (even in non-verbose mode)
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
        
        # New section: matched sequences below threshold
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


def show_all_chains(app_name: str, show_below_threshold: bool = False, limit: int = None, final_only: bool = False):
    """
    Display all extracted chain sequences for an app without running evaluation.
    
    Args:
        app_name: Name of the app to show chains for
        show_below_threshold: Also show chains from features below score threshold
        limit: Maximum number of chains to display (None for all)
        final_only: If True, only include final features. Default False = include all.
    """
    print("\n" + "="*60)
    print(f"🔗 ALL CHAIN SEQUENCES: {app_name}")
    print("="*60)
    
    # Connect
    print("\n📡 Connecting to MongoDB...")
    client, db = connect_to_mongodb()
    
    try:
        # Load data
        print(f"📥 Loading data...")
        func_records, action_func_records, func_lookup, filtered_lookup, filtered_count, active_threshold = load_data(db, app_name)
        print(f"   ✓ Total Functionalities in DB:         {len(func_records)}")
        print(f"   ✓ Passed score threshold (≥{active_threshold:.3f}): {len(func_lookup)}")
        print(f"   ✗ Filtered out (below threshold):      {filtered_count}")
        print(f"   ✓ Action-Functionality mappings (AFD): {len(action_func_records)}")
        
        # Show score distribution
        scores = [f.get('score', 0) for f in func_lookup.values()]
        if scores:
            final_count = sum(1 for f in func_lookup.values() if f.get('final', False))
            non_final_count = len(func_lookup) - final_count
            print(f"\n📊 Score Distribution (above threshold):")
            print(f"   ✓ Final features:     {final_count}")
            print(f"   ⏳ Non-final features: {non_final_count}")
            print(f"   📈 Score range: {min(scores):.3f} to {max(scores):.3f}")
            # Show distribution buckets
            buckets = {'0-1': 0, '1-2': 0, '2-3': 0, '3-4': 0, '4-5': 0, '5-6': 0, '6+': 0}
            for s in scores:
                if s >= 6: buckets['6+'] += 1
                elif s >= 5: buckets['5-6'] += 1
                elif s >= 4: buckets['4-5'] += 1
                elif s >= 3: buckets['3-4'] += 1
                elif s >= 2: buckets['2-3'] += 1
                elif s >= 1: buckets['1-2'] += 1
                else: buckets['0-1'] += 1
            print(f"   📊 Distribution: {buckets}")
        
        if not action_func_records:
            print(f"\n⚠️  No data found for {app_name}")
            return
        
        # Reconstruct chains
        mode = "final features only" if final_only else "ALL features"
        print(f"\n🔄 Reconstructing action chains ({mode})...")
        reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
        chains = reconstructor.get_all_chains(include_non_final=not final_only)
        
        # Sort by score descending
        chains_with_scores = []
        for chain in chains:
            seq, test_ids, record, func_text, root_state = chain
            func_pointer = record.get('func_pointer', '')
            score = func_lookup.get(func_pointer, {}).get('score', 0)
            is_final = func_lookup.get(func_pointer, {}).get('final', False)
            chains_with_scores.append((seq, test_ids, record, func_text, root_state, score, is_final))
        
        chains_with_scores.sort(key=lambda x: x[5], reverse=True)
        
        # Apply limit
        if limit:
            chains_with_scores = chains_with_scores[:limit]
        
        print(f"\n📋 EXTRACTED CHAINS ({len(chains_with_scores)} sequences):")
        print("=" * 100)
        
        for i, (seq, test_ids, record, func_text, root_state, score, is_final) in enumerate(chains_with_scores, 1):
            func_display = func_text if func_text else "(no functionality linked)"
            depth = record.get('depth', 0)
            final_marker = "✅" if is_final else "⏳"
            root_display = _extract_url_path(root_state) if root_state else "(unknown)"
            
            print(f"\n[{i:3d}] {final_marker} Score: {score:>7.3f}  |  Depth: {depth}")
            print(f"      Feature:  {func_display}")
            print(f"      Sequence: {seq}")
            print(f"      Root:     {root_display}  ⬅️ chain starts here (BFS root)")
            print(f"      Test IDs: {test_ids}")
        
        print("\n" + "=" * 100)
        print(f"Total: {len(chains_with_scores)} chains above threshold")
        
        # Optionally show below-threshold chains
        if show_below_threshold and filtered_lookup:
            print(f"\n🔻 BELOW-THRESHOLD CHAINS (score < {active_threshold:.3f}):")
            print("-" * 100)
            
            below_reconstructor = ActionChainReconstructor(action_func_records, filtered_lookup)
            below_chains = below_reconstructor.get_all_chains()
            
            below_with_scores = []
            for chain in below_chains:
                seq, test_ids, record, func_text, root_state = chain
                func_pointer = record.get('func_pointer', '')
                score = filtered_lookup.get(func_pointer, {}).get('score', 0)
                is_final = filtered_lookup.get(func_pointer, {}).get('final', False)
                below_with_scores.append((seq, test_ids, record, func_text, root_state, score, is_final))
            
            below_with_scores.sort(key=lambda x: x[5], reverse=True)
            
            if limit:
                below_with_scores = below_with_scores[:limit]
            
            for i, (seq, test_ids, record, func_text, root_state, score, is_final) in enumerate(below_with_scores, 1):
                func_display = func_text if func_text else "(no functionality linked)"
                depth = record.get('depth', 0)
                final_marker = "✅" if is_final else "⏳"
                root_display = _extract_url_path(root_state) if root_state else "(unknown)"
                
                print(f"\n[{i:3d}] {final_marker} Score: {score:>7.3f} ❌  |  Depth: {depth}")
                print(f"      Feature:  {func_display}")
                print(f"      Sequence: {seq}")
                print(f"      Root:     {root_display}")
            
            print("\n" + "-" * 100)
            print(f"Total below threshold: {len(below_with_scores)} chains")
        
    finally:
        client.close()


def export_results(results: Dict, filename: str):
    """Export results to JSON file"""
    # Convert for JSON serialization
    export_data = {
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


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AUTOE2E Feature Coverage Evaluator")
    parser.add_argument("app", nargs="?", default="PETCLINIC", 
                        help="App name to evaluate (default: PETCLINIC)")
    parser.add_argument("--explore", action="store_true",
                        help="Just explore the database")
    parser.add_argument("--chains", action="store_true",
                        help="Show all extracted chain sequences without evaluation")
    parser.add_argument("--final", action="store_true",
                        help="With --chains, only include final features (default: include all)")
    parser.add_argument("--below-threshold", action="store_true",
                        help="With --chains, also show chains below score threshold")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Limit number of chains to display")
    parser.add_argument("--export", type=str, metavar="FILE",
                        help="Export results to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication by inferred feature name")
    
    args = parser.parse_args()
    
    if args.explore:
        explore_database()
    elif args.chains:
        show_all_chains(args.app, show_below_threshold=args.below_threshold, limit=args.limit, final_only=args.final)
    else:
        results = evaluate_app(args.app, verbose=not args.quiet, dedup_by_feature=not args.no_dedup)
        
        if results and args.export:
            export_results(results, args.export)