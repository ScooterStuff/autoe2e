"""
Analyze matched sequences to verify semantic correctness.
"""
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_local import evaluate_save, BENCHMARKS, list_available_saves

def analyze_all_matches():
    """Collect and analyze all matched sequences from all saves."""
    
    folders = list_available_saves()
    
    all_matches = []
    
    for folder in folders:
        results = evaluate_save(folder, verbose=False, dedup_by_feature=True)
        if not results:
            continue
        
        app_name = results.get('app_name', 'UNKNOWN')
        
        for match in results.get('matched', []):
            for feature in match.get('features', []):
                all_matches.append({
                    'folder': folder,
                    'app': app_name,
                    'sequence': match.get('sequence', ''),
                    'score': match.get('score', 0),
                    'benchmark_feature': feature,
                    'inferred_feature': match.get('inferred_feature', ''),
                })
    
    return all_matches

def print_analysis_table(matches):
    """Print markdown table of matches with semantic analysis."""
    
    print("\n## All Matched Sequences Analysis\n")
    print("| Folder | Sequence | Score | Benchmark Feature | Inferred Feature | Semantic Match? |")
    print("|--------|----------|-------|-------------------|------------------|-----------------|")
    
    for m in matches:
        benchmark = m['benchmark_feature']
        inferred = m['inferred_feature']
        
        # Simple semantic analysis
        semantic_match = analyze_semantic_match(benchmark, inferred)
        
        folder_short = m['folder'][:25] + "..." if len(m['folder']) > 25 else m['folder']
        inferred_short = inferred[:40] + "..." if len(inferred) > 40 else inferred
        
        print(f"| {folder_short} | {m['sequence']} | {m['score']:.2f} | {benchmark} | {inferred_short} | {semantic_match} |")

def analyze_semantic_match(benchmark: str, inferred: str) -> str:
    """
    Analyze if benchmark and inferred features are semantically equivalent.
    Returns: '✓ Yes', '~ Partial', or '✗ No'
    """
    benchmark_lower = benchmark.lower()
    inferred_lower = inferred.lower()
    
    # Extract key action words
    action_words = {
        'view': ['view', 'display', 'show', 'see', 'list', 'navigate to', 'access'],
        'add': ['add', 'create', 'new', 'insert'],
        'edit': ['edit', 'update', 'modify', 'change', 'save'],
        'delete': ['delete', 'remove', 'destroy'],
        'find': ['find', 'search', 'lookup', 'filter'],
    }
    
    entity_words = {
        'owner': ['owner', 'owners'],
        'pet': ['pet', 'pets'],
        'vet': ['vet', 'vets', 'veterinarian', 'veterinarians'],
        'specialty': ['specialty', 'specialties', 'speciality'],
        'visit': ['visit', 'visits', 'appointment'],
        'pet type': ['pet type', 'pet types', 'type'],
    }
    
    # Check for exact/very close match
    if benchmark_lower == inferred_lower:
        return '✓ Yes'
    
    # Check if key words overlap
    benchmark_action = None
    inferred_action = None
    
    for action, synonyms in action_words.items():
        for syn in synonyms:
            if syn in benchmark_lower:
                benchmark_action = action
            if syn in inferred_lower:
                inferred_action = action
    
    benchmark_entity = None
    inferred_entity = None
    
    for entity, synonyms in entity_words.items():
        for syn in synonyms:
            if syn in benchmark_lower:
                benchmark_entity = entity
            if syn in inferred_lower:
                inferred_entity = entity
    
    # Both action and entity match
    if benchmark_action == inferred_action and benchmark_entity == inferred_entity:
        if benchmark_action and benchmark_entity:
            return '✓ Yes'
    
    # Action matches but entity unclear or matches
    if benchmark_action == inferred_action and benchmark_action:
        if benchmark_entity == inferred_entity:
            return '✓ Yes'
        return '~ Partial'
    
    # Entity matches but action is different
    if benchmark_entity == inferred_entity and benchmark_entity:
        return '~ Partial'
    
    # Check for navigation-like inferred with view-like benchmark
    if 'navigate' in inferred_lower and ('view' in benchmark_lower or 'list' in benchmark_lower):
        # Check entity match
        for entity, synonyms in entity_words.items():
            bench_has = any(s in benchmark_lower for s in synonyms)
            inf_has = any(s in inferred_lower for s in synonyms)
            if bench_has and inf_has:
                return '✓ Yes'
        return '~ Partial'
    
    return '✗ No'


if __name__ == "__main__":
    print("Collecting all matched sequences...")
    matches = analyze_all_matches()
    print(f"\nFound {len(matches)} total matches across all saves.\n")
    
    # Group by folder for cleaner output
    by_folder = {}
    for m in matches:
        folder = m['folder']
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(m)
    
    # Print summary per folder
    print("\n## Summary by Save Folder\n")
    for folder, folder_matches in by_folder.items():
        yes_count = sum(1 for m in folder_matches if analyze_semantic_match(m['benchmark_feature'], m['inferred_feature']).startswith('✓'))
        partial_count = sum(1 for m in folder_matches if analyze_semantic_match(m['benchmark_feature'], m['inferred_feature']).startswith('~'))
        no_count = sum(1 for m in folder_matches if analyze_semantic_match(m['benchmark_feature'], m['inferred_feature']).startswith('✗'))
        print(f"**{folder}**: {len(folder_matches)} matches (✓ {yes_count}, ~ {partial_count}, ✗ {no_count})")
    
    print_analysis_table(matches)
