"""Quick analysis: why petclinic_main_interactive has highest precision."""
import json, math, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluate_local import ActionChainReconstructor, FeatureGrammarParser, BENCHMARKS, DEFAULT_SCORE_THRESHOLD

SAVES = PROJECT_ROOT / "saves"

# Focus on key petclinic folders for comparison
FOLDERS = [
    "petclinic_main_interactive",
    "petclinic_main_32b",
    "petclinic_gpt4",
    "petclinic_qwen2.5coder_32b",
    "petclinic_qwen2.5vl_32b",
    "petclinic_gemma3_27b",
]

def analyze_folder(name):
    folder = SAVES / name
    if not (folder / "functionality.json").exists():
        return None
    
    bench_key = "PETCLINIC" if "petclinic" in name.lower() else None
    if not bench_key:
        return None
    # Use PETCLINICS if available (with submit buttons)
    if "PETCLINICS" in BENCHMARKS:
        bench_key = "PETCLINICS"
    grammar = BENCHMARKS[bench_key]
    
    func_records = json.loads((folder / "functionality.json").read_text(encoding="utf-8"))
    afd_records = json.loads((folder / "action-functionality.json").read_text(encoding="utf-8"))
    
    func_lookup = {}
    for f in func_records:
        fid = str(f.get("_id"))
        if f.get("score", -999) >= DEFAULT_SCORE_THRESHOLD:
            func_lookup[fid] = f
    
    reconstructor = ActionChainReconstructor(afd_records, func_lookup)
    chains = reconstructor.get_all_chains()
    
    # Dedup by sequence
    seq_best = {}
    for chain in chains:
        seq, _, record, _ = chain
        fp = record.get("func_pointer", "")
        score = func_lookup.get(fp, {}).get("score", 0)
        if seq not in seq_best or score > seq_best[seq][1]:
            seq_best[seq] = (chain, score)
    chains = [v[0] for v in seq_best.values()]
    
    parser = FeatureGrammarParser(grammar, partial_match=True)
    
    matched, unmatched = [], []
    for chain in chains:
        if parser.find_matching_features(chain[0]):
            matched.append(chain)
        else:
            unmatched.append(chain)
    
    # Dedup matched by benchmark feature
    bench_best = {}
    for chain in matched:
        fp = chain[2].get("func_pointer", "")
        score = func_lookup.get(fp, {}).get("score", 0)
        for feat in parser.find_matching_features(chain[0]):
            if feat not in bench_best or score > bench_best[feat][1]:
                bench_best[feat] = (chain, score)
    
    seen = set()
    deduped_matched = []
    for feat, (chain, _) in bench_best.items():
        if chain[0] not in seen:
            seen.add(chain[0])
            deduped_matched.append(chain)
    
    # Dedup unmatched
    feat_best = {}
    for chain in unmatched:
        key = (chain[3] or "").strip().lower() or f"__unnamed_{chain[0]}"
        fp = chain[2].get("func_pointer", "")
        score = func_lookup.get(fp, {}).get("score", 0)
        if key not in feat_best or score > feat_best[key][1]:
            feat_best[key] = (chain, score)
    deduped_unmatched = [v[0] for v in feat_best.values()]
    
    gen = len(deduped_matched) + len(deduped_unmatched)
    correct = len(bench_best)
    precision = correct / gen * 100 if gen else 0
    total_bench = len(grammar)
    recall = correct / total_bench * 100 if total_bench else 0
    
    # Score distribution of all chains
    all_scores = []
    for chain in chains:
        fp = chain[2].get("func_pointer", "")
        score = func_lookup.get(fp, {}).get("score", 0)
        all_scores.append(score)
    
    # Unmatched feature names for inspection
    unmatched_names = []
    for chain in deduped_unmatched:
        unmatched_names.append((chain[0], chain[3]))
    
    return {
        "name": name,
        "total_funcs": len(func_records),
        "above_threshold": len(func_lookup),
        "total_raw_chains": len(list(ActionChainReconstructor(afd_records, func_lookup).get_all_chains())),
        "unique_seqs": len(seq_best),
        "gen": gen,
        "correct": correct,
        "matched_chains": len(deduped_matched),
        "unmatched_chains": len(deduped_unmatched),
        "precision": precision,
        "recall": recall,
        "bench_features_found": list(bench_best.keys()),
        "unmatched_names": unmatched_names,
        "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "min_score": min(all_scores) if all_scores else 0,
    }


print("=" * 100)
print("PRECISION ANALYSIS: Why petclinic_main_interactive has highest precision")
print("=" * 100)

all_results = {}
for f in FOLDERS:
    r = analyze_folder(f)
    if r:
        all_results[f] = r

# Summary table
print(f"\n{'Folder':<40} {'Gen':>5} {'Corr':>5} {'Prec%':>7} {'Rec%':>7} {'Match':>6} {'Unmtch':>6} {'Funcs':>7} {'AbvThr':>7} {'RawCh':>7}")
print("-" * 110)
for name, r in all_results.items():
    print(f"{r['name']:<40} {r['gen']:>5} {r['correct']:>5} {r['precision']:>7.1f} {r['recall']:>7.1f} "
          f"{r['matched_chains']:>6} {r['unmatched_chains']:>6} {r['total_funcs']:>7} {r['above_threshold']:>7} {r['total_raw_chains']:>7}")

# Detailed analysis of main_interactive
if "petclinic_main_interactive" in all_results:
    mi = all_results["petclinic_main_interactive"]
    print(f"\n{'='*80}")
    print("DETAIL: petclinic_main_interactive")
    print(f"{'='*80}")
    print(f"\nBenchmark features FOUND ({mi['correct']}):")
    for feat in sorted(mi['bench_features_found']):
        print(f"  + {feat}")
    
    total_bench = len(BENCHMARKS.get("PETCLINICS", BENCHMARKS["PETCLINIC"]))
    missing = set(BENCHMARKS.get("PETCLINICS", BENCHMARKS["PETCLINIC"]).keys()) - set(mi['bench_features_found'])
    print(f"\nBenchmark features MISSING ({len(missing)}):")
    for feat in sorted(missing):
        print(f"  - {feat}")
    
    print(f"\nUnmatched generated features ({mi['unmatched_chains']}):")
    for seq, fname in sorted(mi['unmatched_names'], key=lambda x: x[0]):
        print(f"  {seq:<30} {fname}")

# Compare unmatched counts (key driver of precision)
print(f"\n{'='*80}")
print("KEY INSIGHT: Unmatched chain breakdown")
print(f"{'='*80}")
for name, r in all_results.items():
    unmatched_ratio = r['unmatched_chains'] / r['gen'] * 100 if r['gen'] else 0
    print(f"\n{name}:")
    print(f"  Gen={r['gen']}, Matched={r['matched_chains']}, Unmatched={r['unmatched_chains']} ({unmatched_ratio:.1f}% of gen)")
    print(f"  Precision = {r['correct']}/{r['gen']} = {r['precision']:.1f}%")
    print(f"  Unique sequences from chain reconstruction: {r['unique_seqs']}")
    print(f"  Avg chain score: {r['avg_score']:.3f}")
    if r['unmatched_names']:
        print(f"  Sample unmatched features:")
        for seq, fname in r['unmatched_names'][:5]:
            print(f"    {seq:<25} {fname}")
