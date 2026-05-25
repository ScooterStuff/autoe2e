"""
Verify all numbers in the False Positive Analysis paper section.
Checks:
  1. Structural breakdown (Table fp_structural)
  2. Semantic categorisation (Table fp_semantic)
  3. Score separation (Table score_separation)
"""
import os, re
from pathlib import Path
from collections import Counter, defaultdict

# ─── Parse gen_dumps ──────────────────────────────────────────────────────────
GEN_DUMPS_DIR = Path(__file__).resolve().parent.parent / 'gen_dumps'
files = sorted(f for f in os.listdir(GEN_DUMPS_DIR) if not f.startswith('_'))
model_names = {
    'petclinic_deepseekr1_32b.txt':  'DeepSeek-R1 32B',
    'petclinic_gemma3_27b_10.txt':   'Gemma3 27B',
    'petclinic_gemma3_4b.txt':       'Gemma3 4B',
    'petclinic_gpt4_cleaned.txt':    'GPT-4o',
    'petclinic_qwen2.5coder_32b.txt':'Qwen 2.5-Coder 32B',
    'petclinic_qwen2.5vl_32b.txt':   'Qwen2.5-VL 32B',
    'petclinic_qwen2.5vl_7b.txt':    'Qwen2.5-VL 7B',
}

all_matched = []    # list of dicts {seq, inf, score, file}
all_unmatched = []  # list of dicts {seq, inf, score, file}
per_model_matched = defaultdict(list)
per_model_unmatched = defaultdict(list)

for fname in files:
    fpath = os.path.join('gen_dumps', fname)
    lines = open(fpath, 'r', encoding='utf-8').readlines()
    
    section = 'header'
    entries = []
    current = {}

    for line in lines:
        stripped = line.strip()
        if 'UNMATCHED SEQUENCES' in stripped:
            # Save any matched entries collected so far
            if current:
                entries.append(current)
                current = {}
            for e in entries:
                e['matched'] = True
                e['file'] = fname
                all_matched.append(e)
                per_model_matched[fname].append(e)
            entries = []
            section = 'unmatched'
            continue
        
        if section == 'header':
            # Look for start of matched sequences section
            if re.match(r'^#\d+', stripped):
                section = 'matched'
            else:
                continue
        
        if section in ('matched', 'unmatched'):
            if re.match(r'^#\d+', stripped):
                if current:
                    entries.append(current)
                current = {}
            elif stripped.startswith('Sequence:'):
                current['seq'] = stripped.replace('Sequence:', '').strip()
            elif stripped.startswith('Inferred:'):
                current['inf'] = stripped.replace('Inferred:', '').strip()
            elif stripped.startswith('Score:'):
                try:
                    current['score'] = float(stripped.replace('Score:', '').strip())
                except:
                    current['score'] = 0.0
            elif stripped.startswith('Matches:'):
                current['matches'] = stripped.replace('Matches:', '').strip()
    
    if current:
        entries.append(current)
    
    for e in entries:
        e['matched'] = (section == 'matched')  # if we never hit UNMATCHED, all are matched
        e['file'] = fname
        if section == 'unmatched':
            all_unmatched.append(e)
            per_model_unmatched[fname].append(e)
        else:
            all_matched.append(e)
            per_model_matched[fname].append(e)

print(f"Total matched: {len(all_matched)}")
print(f"Total unmatched: {len(all_unmatched)}")
print()

# ─── 1. Structural Breakdown ─────────────────────────────────────────────────
def classify_structural(seq):
    is_single = re.match(r'^[cs]\d+$', seq)
    is_two_nav = re.match(r'^c\d+c\d+$', seq)
    is_nav_only = re.match(r'^(c\d+)+$', seq)
    has_one_field = re.match(r'^(c\d+)+(t\d+|s\d+)$', seq)
    has_form = re.search(r'(t\d+|s\d+)', seq)
    
    if is_single:
        return 'Single click'
    elif is_two_nav:
        return 'Wrong navigation pair (2 clicks)'
    elif is_nav_only and not has_form:
        return 'Navigation-only (3+ clicks, no form)'
    elif has_one_field:
        return 'Navigation + 1 field (incomplete form)'
    elif has_form:
        return 'Other'
    else:
        return 'Other'

struct_counts = Counter()
for e in all_unmatched:
    cat = classify_structural(e.get('seq', ''))
    struct_counts[cat] += 1

print("=" * 80)
print("TABLE 1: STRUCTURAL BREAKDOWN VERIFICATION")
print("=" * 80)

paper_structural = {
    'Navigation-only (3+ clicks, no form)': 993,
    'Navigation + 1 field (incomplete form)': 143,
    'Wrong navigation pair (2 clicks)': 105,
    'Single click': 39,
    'Other': 12,
}

total = sum(struct_counts.values())
for cat in ['Navigation-only (3+ clicks, no form)', 'Navigation + 1 field (incomplete form)',
            'Wrong navigation pair (2 clicks)', 'Single click', 'Other']:
    actual = struct_counts.get(cat, 0)
    paper = paper_structural.get(cat, 0)
    pct = actual / total * 100 if total > 0 else 0
    status = "✓" if actual == paper else "✗ MISMATCH"
    print(f"  {cat:45s}  Actual: {actual:5d} ({pct:5.1f}%)  Paper: {paper:5d}  {status}")
print(f"  {'TOTAL':45s}  Actual: {total:5d}          Paper: {sum(paper_structural.values()):5d}")
print()

# ─── 2. Semantic Categorisation ──────────────────────────────────────────────
def classify_semantic(inf):
    if not inf:
        return 'Other / unclassified'
    inf_lower = inf.lower()
    
    if any(kw in inf_lower for kw in ['cancel','discard','return to','revert','back to','exit','close','undo']):
        return 'Cancel / discard / return'
    elif any(kw in inf_lower for kw in ['dropdown','toggle','expand','collapse','menu']):
        return 'Dropdown / toggle'
    elif any(kw in inf_lower for kw in ['homepage','home page','dashboard','main page','welcome','landing','clinic overview']):
        return 'Navigate to homepage'
    elif any(kw in inf_lower for kw in ['navigate','open form','open edit','open add','initiate','display form']):
        return 'Navigate to form'
    elif any(kw in inf_lower for kw in ['date','datepicker','birth date','visit date','calendar']):
        return 'Date-related interaction'
    elif any(kw in inf_lower for kw in ['confirm','delete','remove','trigger confirmation']):
        return 'Delete / confirm (wrong path)'
    elif any(kw in inf_lower for kw in ['submit','save','update','add new','create new','add a new']):
        return 'Incomplete submit / save'
    elif any(kw in inf_lower for kw in ['search','find','filter']):
        return 'Search / filter'
    elif any(kw in inf_lower for kw in ['edit','modify','change','rename','clear','focus','input','enter','type','field']):
        return 'Incomplete field edit'
    elif any(kw in inf_lower for kw in ['view','display','list','browse','show','read-only']):
        return 'View / display (duplicate path)'
    else:
        return 'Other / unclassified'

# Also cross-tab: structural category of "4_SINGLE_FIELD" -> "Single field interaction"
def classify_semantic_with_struct(e):
    seq = e.get('seq', '')
    inf = e.get('inf', '')
    
    # Check if it's a single-field structural entry
    has_one_field = re.match(r'^(c\d+)+(t\d+|s\d+)$', seq)
    if has_one_field:
        return 'Single field interaction'
    
    return classify_semantic(inf)

semantic_counts = Counter()
for e in all_unmatched:
    cat = classify_semantic_with_struct(e)
    semantic_counts[cat] += 1

print("=" * 80)
print("TABLE 2: SEMANTIC CATEGORISATION VERIFICATION")
print("=" * 80)

paper_semantic = {
    'Navigate to form': 223,
    'Cancel / discard / return': 142,
    'Incomplete field edit': 105,
    'View / display (duplicate path)': 92,
    'Delete / confirm (wrong path)': 81,
    'Single field interaction': 78,
    'Navigate to homepage': 59,
    'Search / filter': 45,
    'Date-related interaction': 42,
    'Dropdown / toggle': 41,
    'Other / unclassified': 279,     # paper value
    # Missing from paper: 'Incomplete submit / save': 105
}

sem_total = sum(semantic_counts.values())
sem_order = [
    'Navigate to form', 'Cancel / discard / return', 'Incomplete field edit',
    'Incomplete submit / save',  # MISSING from paper table
    'View / display (duplicate path)', 'Delete / confirm (wrong path)',
    'Single field interaction', 'Navigate to homepage', 'Search / filter',
    'Date-related interaction', 'Dropdown / toggle', 'Other / unclassified'
]

paper_total_listed = sum(paper_semantic.values())
for cat in sem_order:
    actual = semantic_counts.get(cat, 0)
    paper = paper_semantic.get(cat, '---')
    pct = actual / sem_total * 100 if sem_total > 0 else 0
    if isinstance(paper, int):
        status = "✓" if actual == paper else f"✗ MISMATCH (diff={actual-paper:+d})"
    else:
        status = "⚠ MISSING FROM PAPER TABLE"
    print(f"  {cat:40s}  Actual: {actual:5d} ({pct:5.1f}%)  Paper: {str(paper):>5s}  {status}")

print(f"  {'TOTAL':40s}  Actual: {sem_total:5d}          Paper table sum: {paper_total_listed}")
print()
if paper_total_listed != sem_total:
    print(f"  ⚠ PAPER TABLE SUMS TO {paper_total_listed}, NOT {sem_total}!")
    print(f"    Missing: {sem_total - paper_total_listed} entries")
    print()

# Now also verify: what if we do PURE semantic (aggregate across all structural categories)?
print("  --- Alternative: Pure semantic aggregation (across ALL structural cats) ---")
pure_sem = Counter()
for e in all_unmatched:
    pure_sem[classify_semantic(e.get('inf', ''))] += 1

for cat in ['Navigate to form', 'Cancel / discard / return', 'Incomplete field edit',
            'Incomplete submit / save', 'View / display (duplicate path)', 
            'Delete / confirm (wrong path)', 'Navigate to homepage', 'Search / filter',
            'Date-related interaction', 'Dropdown / toggle', 'Other / unclassified']:
    actual = pure_sem.get(cat, 0)
    pct = actual / sem_total * 100 if sem_total > 0 else 0
    print(f"  {cat:40s}  {actual:5d} ({pct:5.1f}%)")
print(f"  {'TOTAL':40s}  {sum(pure_sem.values()):5d}")
print()

# ─── 3. Score Separation ─────────────────────────────────────────────────────
print("=" * 80)
print("TABLE 3: SCORE SEPARATION VERIFICATION")
print("=" * 80)

paper_scores = {
    'petclinic_deepseekr1_32b.txt':   {'matched_avg': 5.978, 'unmatched_avg': 5.032, 'pct': 84},
    'petclinic_gemma3_27b_10.txt':    {'matched_avg': 4.555, 'unmatched_avg': 4.709, 'pct': 100},
    'petclinic_gemma3_4b.txt':        {'matched_avg': 5.891, 'unmatched_avg': 4.547, 'pct': 100},
    'petclinic_gpt4_cleaned.txt':     {'matched_avg': 1.721, 'unmatched_avg': 1.529, 'pct': 100},
    'petclinic_qwen2.5coder_32b.txt': {'matched_avg': 5.385, 'unmatched_avg': 4.795, 'pct': 86},
    'petclinic_qwen2.5vl_32b.txt':    {'matched_avg': 5.598, 'unmatched_avg': 4.735, 'pct': 100},
    'petclinic_qwen2.5vl_7b.txt':     {'matched_avg': 3.320, 'unmatched_avg': 3.694, 'pct': 100},
}

for fname in files:
    matched_scores = [e['score'] for e in per_model_matched[fname] if 'score' in e]
    unmatched_scores = [e['score'] for e in per_model_unmatched[fname] if 'score' in e]
    
    if not matched_scores or not unmatched_scores:
        print(f"  {model_names.get(fname, fname):25s}  SKIPPED (no score data)")
        continue
    
    m_avg = sum(matched_scores) / len(matched_scores)
    u_avg = sum(unmatched_scores) / len(unmatched_scores)
    m_min = min(matched_scores)
    u_above = sum(1 for s in unmatched_scores if s >= m_min)
    pct = round(u_above / len(unmatched_scores) * 100)
    
    paper = paper_scores.get(fname, {})
    p_m_avg = paper.get('matched_avg', '---')
    p_u_avg = paper.get('unmatched_avg', '---')
    p_pct = paper.get('pct', '---')
    
    m_match = "✓" if isinstance(p_m_avg, float) and abs(m_avg - p_m_avg) < 0.01 else ("✗" if isinstance(p_m_avg, float) else "")
    u_match = "✓" if isinstance(p_u_avg, float) and abs(u_avg - p_u_avg) < 0.01 else ("✗" if isinstance(p_u_avg, float) else "")
    p_match = "✓" if isinstance(p_pct, int) and pct == p_pct else ("✗" if isinstance(p_pct, int) else "")
    
    name = model_names.get(fname, fname)
    print(f"  {name:25s}  MatchedAvg: {m_avg:.3f} (paper: {p_m_avg}) {m_match}  "
          f"UnmatchedAvg: {u_avg:.3f} (paper: {p_u_avg}) {u_match}  "
          f"%%Above: {pct}% (paper: {p_pct}%) {p_match}")
    print(f"  {'':25s}  MatchedMin: {m_min:.3f}  MatchedN: {len(matched_scores)}  UnmatchedN: {len(unmatched_scores)}  "
          f"AboveCount: {u_above}/{len(unmatched_scores)}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
