"""
Dump all generated test sequences for each save folder into text files.
Outputs to gen_dumps/<save_folder>.txt

Shows each generated chain with:
- Whether it matched a benchmark feature or not
- The action sequence
- The inferred functionality text
- The score
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Make project root importable so we can reuse evaluate_local
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse evaluation logic
from evaluate_local import (
    BENCHMARKS, 
    ActionChainReconstructor, 
    FeatureGrammarParser,
    get_active_score_threshold,
    detect_app_from_folder,
    get_saves_dir,
    list_available_saves,
    DEFAULT_SCORE_THRESHOLD,
)


def dump_save(save_folder: str, output_dir: Path, partial_match: bool = True):
    """Dump all generated chains for a save folder to a text file."""
    
    saves_dir = get_saves_dir()
    folder_path = saves_dir / save_folder
    func_file = folder_path / "functionality.json"
    afd_file = folder_path / "action-functionality.json"
    
    if not func_file.exists() or not afd_file.exists():
        print(f"  SKIP {save_folder}: missing files")
        return
    
    app_name = detect_app_from_folder(save_folder)
    if not app_name or app_name not in BENCHMARKS:
        print(f"  SKIP {save_folder}: no benchmark grammar")
        return
    
    grammar = BENCHMARKS[app_name]
    active_threshold = get_active_score_threshold()
    
    # Load data
    print(f"  Loading {save_folder}...")
    with open(func_file, 'r', encoding='utf-8') as f:
        func_records = json.load(f)
    with open(afd_file, 'r', encoding='utf-8') as f:
        action_func_records = json.load(f)
    
    # Build func_lookup (apply score threshold)
    func_lookup = {}
    filtered_count = 0
    for func in func_records:
        func_id = str(func.get('_id'))
        score = func.get('score', 0)
        if score < active_threshold:
            filtered_count += 1
            continue
        func_lookup[func_id] = func
    
    total_funcs = len(func_records)
    final_features = sum(1 for f in func_lookup.values() if f.get('final', False))
    
    # Reconstruct chains
    reconstructor = ActionChainReconstructor(action_func_records, func_lookup)
    chains = reconstructor.get_all_chains()
    
    # Deduplicate by sequence (keep highest score)
    seq_best = {}
    for chain in chains:
        seq, test_ids, record, func_text = chain
        func_pointer = record.get('func_pointer', '')
        score = func_lookup.get(func_pointer, {}).get('score', 0)
        if seq not in seq_best or score > seq_best[seq][1]:
            seq_best[seq] = (chain, score)
    chains = [item[0] for item in seq_best.values()]
    
    # Deduplicate by feature name
    parser = FeatureGrammarParser(grammar, partial_match=partial_match)
    
    matched_chains = []
    unmatched_chains = []
    
    for chain in chains:
        seq, test_ids, record, func_text = chain
        matches = parser.find_matching_features(seq)
        if matches:
            matched_chains.append(chain)
        else:
            unmatched_chains.append(chain)
    
    # Dedup matched by benchmark feature
    benchmark_feature_best = {}
    for chain in matched_chains:
        seq, test_ids, record, func_text = chain
        matches = parser.find_matching_features(seq)
        func_pointer = record.get('func_pointer', '')
        score = func_lookup.get(func_pointer, {}).get('score', 0)
        for feature_name in matches:
            if feature_name not in benchmark_feature_best or score > benchmark_feature_best[feature_name][1]:
                benchmark_feature_best[feature_name] = (chain, score)
    
    seen_matched_seqs = set()
    deduped_matched = []
    for feature_name, (chain, score) in benchmark_feature_best.items():
        seq = chain[0]
        if seq not in seen_matched_seqs:
            seen_matched_seqs.add(seq)
            deduped_matched.append(chain)
    
    # Dedup unmatched by inferred feature name
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
    
    all_chains = deduped_matched + deduplicated_unmatched
    
    # Sort all by score descending
    def get_score(chain):
        _, _, record, _ = chain
        fp = record.get('func_pointer', '')
        return func_lookup.get(fp, {}).get('score', 0)
    
    all_chains.sort(key=get_score, reverse=True)
    
    # Write output file
    output_file = output_dir / f"{save_folder}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"{'='*100}\n")
        out.write(f"ALL GENERATED TEST SEQUENCES: {save_folder}\n")
        out.write(f"{'='*100}\n\n")
        out.write(f"Total Functionalities (FD):  {total_funcs}\n")
        out.write(f"Passed threshold:            {len(func_lookup)}\n")
        out.write(f"Filtered (below threshold):  {filtered_count}\n")
        out.write(f"Final features:              {final_features}\n")
        out.write(f"Score threshold:             {active_threshold:.3f}\n\n")
        out.write(f"Total generated (after dedup): {len(all_chains)}\n")
        out.write(f"  Matched:   {len(deduped_matched)}\n")
        out.write(f"  Unmatched: {len(deduplicated_unmatched)}\n\n")
        
        # Covered features
        covered = set()
        for chain in deduped_matched:
            seq = chain[0]
            matches = parser.find_matching_features(seq)
            covered.update(matches)
        
        out.write(f"Benchmark features: {len(grammar)}\n")
        out.write(f"Covered: {len(covered)} / {len(grammar)}\n")
        out.write(f"Precision: {len(covered)/len(all_chains)*100:.1f}%\n")
        out.write(f"Recall: {len(covered)/len(grammar)*100:.1f}%\n\n")
        
        # Covered
        out.write(f"COVERED BENCHMARK FEATURES:\n")
        for f in sorted(covered):
            out.write(f"  + {f}\n")
        out.write(f"\nUNCOVERED BENCHMARK FEATURES:\n")
        for f in sorted(grammar.keys()):
            if f not in covered:
                out.write(f"  - {f}  (expected: {grammar[f]})\n")
        
        # === MATCHED ===
        out.write(f"\n\n{'='*100}\n")
        out.write(f"MATCHED SEQUENCES ({len(deduped_matched)})\n")
        out.write(f"{'='*100}\n\n")
        
        for i, chain in enumerate(sorted(deduped_matched, key=get_score, reverse=True), 1):
            seq, test_ids, record, func_text = chain
            fp = record.get('func_pointer', '')
            score = func_lookup.get(fp, {}).get('score', 0)
            matches = parser.find_matching_features(seq)
            
            out.write(f"#{i}\n")
            out.write(f"  Sequence:   {seq}\n")
            out.write(f"  Matches:    {matches}\n")
            out.write(f"  Inferred:   {func_text}\n")
            out.write(f"  Score:      {score:.3f}\n")
            out.write(f"  Test IDs:   {test_ids}\n")
            out.write(f"\n")
        
        # === UNMATCHED ===
        out.write(f"\n{'='*100}\n")
        out.write(f"UNMATCHED SEQUENCES ({len(deduplicated_unmatched)})\n")
        out.write(f"{'='*100}\n\n")
        
        for i, chain in enumerate(sorted(deduplicated_unmatched, key=get_score, reverse=True), 1):
            seq, test_ids, record, func_text = chain
            fp = record.get('func_pointer', '')
            score = func_lookup.get(fp, {}).get('score', 0)
            
            out.write(f"#{i}\n")
            out.write(f"  Sequence:   {seq}\n")
            out.write(f"  Inferred:   {func_text}\n")
            out.write(f"  Score:      {score:.3f}\n")
            out.write(f"  Test IDs:   {test_ids}\n")
            out.write(f"\n")
    
    print(f"  Done: {output_file}  ({len(all_chains)} total: {len(deduped_matched)} matched, {len(deduplicated_unmatched)} unmatched)")


def main():
    output_dir = Path(__file__).resolve().parent.parent / "gen_dumps"
    output_dir.mkdir(exist_ok=True)
    
    folders = list_available_saves()
    if not folders:
        print("No saves found!")
        return
    
    print(f"Dumping generated chains for {len(folders)} saves...\n")
    
    for folder in folders:
        dump_save(folder, output_dir)
    
    print(f"\nAll files written to {output_dir}/")


if __name__ == "__main__":
    main()
