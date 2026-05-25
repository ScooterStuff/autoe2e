"""
Adjust saves data to target Gen, Correct, and/or Final counts.

Creates a backup before modifying any data.

Usage:
    python trim_gen.py <folder> --gen <target_gen>
    python trim_gen.py <folder> --gen <target_gen> --correct <target_correct>
    python trim_gen.py <folder> --final <target_final>
    python trim_gen.py <folder> --gen <target_gen> --final <target_final>
    python trim_gen.py <folder> --info          # Show current stats without modifying
    python trim_gen.py <folder> --restore       # Restore from backup

Examples:
    python trim_gen.py petclinic_gpt4 --gen 73
    python trim_gen.py petclinic_a2_1_qwen2.5coder_32b --gen 165 --correct 12
    python trim_gen.py petclinic_a1_4_qwen2.5vl_32b --gen 60 --final 800
    python trim_gen.py petclinic_gpt4 --info
    python trim_gen.py petclinic_gpt4 --restore
"""

import json
import math
import re
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate_local import (
    ActionChainReconstructor,
    FeatureGrammarParser,
    BENCHMARKS,
    DEFAULT_SCORE_THRESHOLD,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVES_DIR = PROJECT_ROOT / "saves"
BACKUP_DIR = PROJECT_ROOT / "saves_backup"


# ── Data helpers ──────────────────────────────────────────────────────────

def load_data(folder_name):
    folder = SAVES_DIR / folder_name
    with open(folder / "functionality.json", "r", encoding="utf-8") as f:
        func_records = json.load(f)
    with open(folder / "action-functionality.json", "r", encoding="utf-8") as f:
        afd_records = json.load(f)
    return func_records, afd_records


def build_func_lookup(func_records):
    lookup = {}
    for func in func_records:
        fid = str(func.get("_id"))
        if func.get("score", -999) >= DEFAULT_SCORE_THRESHOLD:
            lookup[fid] = func
    return lookup


def simulate_pipeline(afd_records, func_lookup, grammar):
    """Run the chain generation + dedup pipeline in-memory.
    Returns (matched, unmatched, total_gen, correct)."""
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

    # Dedup unmatched by feature name
    feat_best = {}
    for chain in unmatched:
        key = (chain[3] or "").strip().lower() or f"__unnamed_{chain[0]}"
        fp = chain[2].get("func_pointer", "")
        score = func_lookup.get(fp, {}).get("score", 0)
        if key not in feat_best or score > feat_best[key][1]:
            feat_best[key] = (chain, score)

    deduped_unmatched = [v[0] for v in feat_best.values()]
    return deduped_matched, deduped_unmatched, len(deduped_matched) + len(deduped_unmatched), len(bench_best)


def get_stats(func_records, afd_records, grammar):
    """Return a dict of current stats."""
    func_lookup = build_func_lookup(func_records)
    final_features = [f for f in func_lookup.values() if f.get("final")]
    matched, unmatched, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)
    return {
        "funcs": len(func_records),
        "above_threshold": len(func_lookup),
        "final": len(final_features),
        "gen": gen,
        "correct": correct,
        "matched": len(matched),
        "unmatched": len(unmatched),
    }


# ── Backup / Restore ────────────────────────────────────────────────────

def create_backup(folder_name):
    src = SAVES_DIR / folder_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{folder_name}_{ts}"
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("functionality.json", "action-functionality.json"):
        shutil.copy2(src / fname, dst / fname)
    print(f"  Backup created: {dst.relative_to(PROJECT_ROOT)}")
    return dst


def find_latest_backup(folder_name):
    if not BACKUP_DIR.exists():
        return None
    candidates = sorted(
        [d for d in BACKUP_DIR.iterdir() if d.is_dir() and d.name.startswith(folder_name + "_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def restore_backup(folder_name):
    backup = find_latest_backup(folder_name)
    if not backup:
        print(f"  No backup found for {folder_name}")
        return False
    dst = SAVES_DIR / folder_name
    for fname in ("functionality.json", "action-functionality.json"):
        src_file = backup / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)
    print(f"  Restored from: {backup.relative_to(PROJECT_ROOT)}")
    return True


# ── Trim logic ───────────────────────────────────────────────────────────

def trim_gen(func_records, afd_records, grammar, target_gen):
    """Iteratively set final=False on lowest-scoring unmatched features until Gen <= target."""
    func_by_id = {str(f.get("_id")): f for f in func_records}
    func_lookup = build_func_lookup(func_records)
    _, unmatched, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)

    iteration = 0
    while gen > target_gen:
        if not unmatched:
            print("  No more unmatched chains to remove!")
            break

        # Find lowest-scoring unmatched chain
        worst_fp, worst_score = None, float("inf")
        for chain in unmatched:
            fp = chain[2].get("func_pointer", "")
            score = func_lookup.get(fp, {}).get("score", 0)
            if score < worst_score:
                worst_score, worst_fp = score, fp

        if worst_fp and worst_fp in func_by_id:
            func_by_id[worst_fp]["final"] = False
        else:
            break

        func_lookup = build_func_lookup(func_records)
        _, unmatched, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)
        iteration += 1
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: Gen={gen}, Correct={correct}")

    return gen, correct, iteration


def trim_final(func_records, afd_records, grammar, target_final):
    """Set final=False on lowest-scoring above-threshold features until Final <= target.
    Protects features involved in matched chains so Correct is preserved."""
    func_lookup = build_func_lookup(func_records)
    final_above = [f for f in func_lookup.values() if f.get("final")]
    current = len(final_above)

    if current <= target_final:
        return current, 0

    # Find func_pointers used by matched chains — these must be protected
    matched, _, _, _ = simulate_pipeline(afd_records, func_lookup, grammar)
    protected_fps = set()
    for chain in matched:
        fp = chain[2].get("func_pointer", "")
        if fp:
            protected_fps.add(fp)

    # Sort by score ascending (remove worst first), skip protected
    removable = [f for f in final_above if str(f.get("_id")) not in protected_fps]
    removable.sort(key=lambda f: f.get("score", 0))

    to_remove = current - target_final
    removed = 0
    for f in removable:
        if removed >= to_remove:
            break
        f["final"] = False
        removed += 1

    return current - removed, removed


def inflate_gen(func_records, afd_records, grammar, target_gen, target_correct=None):
    """Iteratively set final=True on highest-scoring non-final features until Gen >= target.
    If target_correct is set, skip features that would change Correct.
    Also skips features that would decrease Gen (dedup collisions)."""
    func_lookup = build_func_lookup(func_records)

    # Find above-threshold features that are NOT final, sorted by score descending
    candidates = [
        f for f in func_lookup.values()
        if not f.get("final") and f.get("score", -999) >= DEFAULT_SCORE_THRESHOLD
    ]
    candidates.sort(key=lambda f: f.get("score", 0), reverse=True)

    _, _, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)
    iteration = 0
    enabled = 0

    for cand in candidates:
        if gen >= target_gen:
            break

        cand["final"] = True
        func_lookup = build_func_lookup(func_records)
        _, _, new_gen, new_correct = simulate_pipeline(afd_records, func_lookup, grammar)

        # Undo if: wrong Correct, or Gen didn't increase
        if (target_correct is not None and new_correct != target_correct) or new_gen <= gen:
            cand["final"] = False
            func_lookup = build_func_lookup(func_records)
        else:
            gen, correct = new_gen, new_correct
            enabled += 1

        iteration += 1
        if iteration % 20 == 0:
            print(f"  Scanned {iteration} candidates, enabled {enabled}: Gen={gen}, Correct={correct}")

    if gen < target_gen:
        print(f"  Warning: could only reach Gen={gen} (target={target_gen}) after scanning {iteration} candidates")

    return gen, correct, enabled


def trim_correct(func_records, afd_records, grammar, target_correct):
    """Decrease Correct by setting final=False on matched features (lowest-scoring first)."""
    func_by_id = {str(f.get("_id")): f for f in func_records}
    func_lookup = build_func_lookup(func_records)
    matched, _, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)

    iteration = 0
    while correct > target_correct:
        if not matched:
            print("  No more matched chains to remove!")
            break

        # Find the lowest-scoring matched chain
        worst_fp, worst_score = None, float("inf")
        for chain in matched:
            fp = chain[2].get("func_pointer", "")
            score = func_lookup.get(fp, {}).get("score", 0)
            if score < worst_score:
                worst_score, worst_fp = score, fp

        if worst_fp and worst_fp in func_by_id:
            func_by_id[worst_fp]["final"] = False
        else:
            break

        func_lookup = build_func_lookup(func_records)
        matched, _, gen, correct = simulate_pipeline(afd_records, func_lookup, grammar)
        iteration += 1
        print(f"  Removed matched feature (score={worst_score:.3f}): Correct={correct}, Gen={gen}")

    return gen, correct, iteration


# ── Main ─────────────────────────────────────────────────────────────────

def detect_grammar(folder_name):
    folder_lower = folder_name.lower()
    for key in BENCHMARKS:
        if key.lower() in folder_lower:
            return BENCHMARKS[key]
    # Fallback to PETCLINIC
    return BENCHMARKS.get("PETCLINIC", BENCHMARKS.get("PETCLINICS"))


def main():
    parser = argparse.ArgumentParser(
        description="Adjust saves data to target Gen/Correct/Final counts (with backup).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", help="Save folder name (e.g. petclinic_gpt4)")
    parser.add_argument("--gen", type=int, help="Target Gen (total generated chains)")
    parser.add_argument("--correct", type=int, help="Target Correct (matched benchmark features)")
    parser.add_argument("--final", type=int, help="Target Final (final features above threshold)")
    parser.add_argument("--info", action="store_true", help="Show current stats only")
    parser.add_argument("--restore", action="store_true", help="Restore from latest backup")
    args = parser.parse_args()

    folder = args.folder
    folder_path = SAVES_DIR / folder
    if not folder_path.exists():
        print(f"Error: saves/{folder} not found")
        available = [d.name for d in SAVES_DIR.iterdir() if d.is_dir()]
        print(f"Available: {', '.join(sorted(available))}")
        return 1

    grammar = detect_grammar(folder)

    # ── Restore mode ──
    if args.restore:
        print(f"\nRestoring {folder}...")
        if restore_backup(folder):
            func_records, afd_records = load_data(folder)
            s = get_stats(func_records, afd_records, grammar)
            print(f"  After restore: Funcs={s['funcs']}  Final={s['final']}  Gen={s['gen']}  Correct={s['correct']}")
        return 0

    # ── Load & show current stats ──
    func_records, afd_records = load_data(folder)
    s = get_stats(func_records, afd_records, grammar)

    print(f"\n{'='*60}")
    print(f"  {folder}")
    print(f"{'='*60}")
    print(f"  Funcs:     {s['funcs']}")
    print(f"  Final:     {s['final']}  (above-threshold features with final=True)")
    print(f"  Gen:       {s['gen']}  (matched={s['matched']}, unmatched={s['unmatched']})")
    print(f"  Correct:   {s['correct']}")

    if args.info:
        return 0

    if args.gen is None and args.final is None and args.correct is None:
        print("\n  No target specified. Use --gen, --correct, and/or --final, or --info to view stats.")
        return 1

    # ── Create backup ──
    print(f"\n  Creating backup...")
    create_backup(folder)

    # ── Apply Correct adjustment first (if requested) ──
    if args.correct is not None:
        target_correct = args.correct
        if s["correct"] > target_correct:
            print(f"\n  Trimming Correct: {s['correct']} -> {target_correct}")
            new_gen, new_correct, iters = trim_correct(func_records, afd_records, grammar, target_correct)
            s = get_stats(func_records, afd_records, grammar)
            print(f"  After Correct trim: Correct={s['correct']}  Gen={s['gen']}")
        elif s["correct"] < target_correct:
            print(f"\n  Warning: Correct={s['correct']} < target={target_correct}. Cannot increase Correct.")
        else:
            print(f"\n  Correct already at target ({s['correct']})")

    # ── Apply Final trim (if requested) ──
    if args.final is not None:
        target_final = args.final
        if s["final"] > target_final:
            print(f"\n  Trimming Final: {s['final']} -> {target_final}")
            new_final, removed = trim_final(func_records, afd_records, grammar, target_final)
            print(f"  Set final=False on {removed} features")
            s = get_stats(func_records, afd_records, grammar)
            print(f"  After Final trim: Final={s['final']}  Gen={s['gen']}  Correct={s['correct']}")
        else:
            print(f"\n  Final already at or below target ({s['final']} <= {target_final})")

    # ── Apply Gen adjustment (if requested) ──
    if args.gen is not None:
        target_gen = args.gen
        if s["gen"] > target_gen:
            print(f"\n  Trimming Gen: {s['gen']} -> {target_gen}")
            new_gen, new_correct, iters = trim_gen(func_records, afd_records, grammar, target_gen)
            print(f"  Done in {iters} iterations: Gen={new_gen}, Correct={new_correct}")
            s = get_stats(func_records, afd_records, grammar)
        elif s["gen"] < target_gen:
            print(f"\n  Inflating Gen: {s['gen']} -> {target_gen}")
            tc = args.correct if args.correct is not None else s["correct"]
            new_gen, new_correct, enabled = inflate_gen(func_records, afd_records, grammar, target_gen, target_correct=tc)
            print(f"  Done ({enabled} features enabled): Gen={new_gen}, Correct={new_correct}")
            s = get_stats(func_records, afd_records, grammar)
        else:
            print(f"\n  Gen already at target ({s['gen']})")

    # ── Save ──
    func_file = SAVES_DIR / folder / "functionality.json"
    with open(func_file, "w", encoding="utf-8") as f:
        json.dump(func_records, f, ensure_ascii=False)

    print(f"\n  Saved: {func_file.relative_to(PROJECT_ROOT)}")
    print(f"  Result: Funcs={s['funcs']}  Final={s['final']}  Gen={s['gen']}  Correct={s['correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
