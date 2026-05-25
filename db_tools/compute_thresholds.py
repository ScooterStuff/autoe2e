import math

# Constants from the dump
TOTAL_FUNCTIONALITIES = 6099
FINAL_FEATURES = 1441
TOTAL_BENCHMARK = 23

# Matched sequences: (score, feature_name)
matched = [
    (6.238, "view a list of owners"),
    (6.238, "view a list of vets"),
    (6.238, "view a list of pet types"),
    (6.238, "view owner details"),
    (6.238, "add owner"),
    (6.238, "add a vet"),
    (6.238, "edit a specialty"),
    (6.238, "delete pet"),
    (6.238, "edit details of a vet"),
    (6.238, "add a new pet to the owner"),
    (6.238, "view a list of specialties"),
    (4.852, "edit owner details"),
    (-0.693, "delete a visit"),
]

# Unmatched sequences by score (score, count)
unmatched_by_score = [
    (6.238, 73),
    (4.852, 43),
    (3.466, 3),
    (2.079, 5),
    (-0.693, 18),
]

def compute_ndcg(ranked_relevance):
    if not ranked_relevance:
        return 0.0
    n = len(ranked_relevance)
    dcg = 0.0
    for i, rel in enumerate(ranked_relevance):
        dcg += rel / math.log2(i + 2)
    total_relevant = sum(ranked_relevance)
    ideal_relevance = [1] * total_relevant + [0] * (n - total_relevant)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / math.log2(i + 2)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def compute_metrics(threshold_label, min_score):
    # Filter matched sequences above threshold
    filtered_matched = [(s, f) for s, f in matched if s > min_score]
    n_covered = len(set(f for _, f in filtered_matched))
    
    # Filter unmatched sequences above threshold
    n_unmatched = sum(count for score, count in unmatched_by_score if score > min_score)
    
    n_gen = len(filtered_matched) + n_unmatched
    
    if n_gen == 0:
        return None
    
    precision = n_covered / n_gen if n_gen > 0 else 0
    recall = n_covered / TOTAL_BENCHMARK if TOTAL_BENCHMARK > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    tp = n_covered
    fp = n_gen - n_covered
    fn = TOTAL_BENCHMARK - n_covered
    tn = TOTAL_FUNCTIONALITIES - tp - fp - fn
    
    numerator = (tp * tn) - (fp * fn)
    denom_terms = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = numerator / math.sqrt(denom_terms) if denom_terms > 0 else 0.0
    
    # Build ranked relevance for NDCG
    # All items sorted by score descending; matched=1, unmatched=0
    all_items = []
    for s, f in filtered_matched:
        all_items.append((s, 1))
    for score, count in unmatched_by_score:
        if score > min_score:
            for _ in range(count):
                all_items.append((score, 0))
    all_items.sort(key=lambda x: x[0], reverse=True)
    ranked_relevance = [rel for _, rel in all_items]
    ndcg = compute_ndcg(ranked_relevance)
    
    return {
        'threshold': threshold_label,
        'funcs': TOTAL_FUNCTIONALITIES,
        'final': FINAL_FEATURES,
        'gen': n_gen,
        'correct': n_covered,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'ndcg': ndcg,
    }

# Compute for each threshold
thresholds = [
    ("score > 1", 1),
    ("score > 2", 2),
    ("score > 3", 3),
    ("score > 4", 4),
    ("score > 5", 5),
    ("all (baseline)", -999),  # include everything
]

print(f"\n{'Folder':<40} {'Funcs':>8} {'Final':>8} {'Gen':>6} {'Correct':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'MCC':>8} {'NDCG':>8}")
print("-" * 145)

for label, min_score in thresholds:
    m = compute_metrics(label, min_score)
    if m:
        print(f"petclinic_qwen2.5vl_32b ({m['threshold']:<15}) "
              f"{m['funcs']:>8} "
              f"{m['final']:>8} "
              f"{m['gen']:>6} "
              f"{m['correct']:>8} "
              f"{m['precision']:>7.1%} "
              f"{m['recall']:>7.1%} "
              f"{m['f1']:>7.1%} "
              f"{m['mcc']:>7.4f} "
              f"{m['ndcg']:>7.3f}")
