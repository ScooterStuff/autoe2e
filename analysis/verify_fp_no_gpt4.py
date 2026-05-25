"""Recompute all FP analysis tables excluding GPT-4o."""
import os, re
from pathlib import Path
from collections import Counter, defaultdict

GEN_DUMPS_DIR = Path(__file__).resolve().parent.parent / 'gen_dumps'
EXCLUDE = {'petclinic_gpt4_cleaned.txt'}
files = sorted(f for f in os.listdir(GEN_DUMPS_DIR) if not f.startswith('_') and f not in EXCLUDE)

model_names = {
    'petclinic_deepseekr1_32b.txt':  'DeepSeek-R1 32B',
    'petclinic_gemma3_27b_10.txt':   'Gemma3 27B',
    'petclinic_gemma3_4b.txt':       'Gemma3 4B',
    'petclinic_qwen2.5coder_32b.txt':'Qwen 2.5-Coder 32B',
    'petclinic_qwen2.5vl_32b.txt':   'Qwen2.5-VL 32B',
    'petclinic_qwen2.5vl_7b.txt':    'Qwen2.5-VL 7B',
}

all_unmatched = []
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
            if current: entries.append(current); current = {}
            for e in entries: e['file'] = fname; per_model_matched[fname].append(e)
            entries = []; section = 'unmatched'; continue
        if section == 'header':
            if re.match(r'^#\d+', stripped): section = 'matched'
            else: continue
        if section in ('matched', 'unmatched'):
            if re.match(r'^#\d+', stripped):
                if current: entries.append(current); current = {}
            elif stripped.startswith('Sequence:'): current['seq'] = stripped.replace('Sequence:','').strip()
            elif stripped.startswith('Inferred:'): current['inf'] = stripped.replace('Inferred:','').strip()
            elif stripped.startswith('Score:'):
                try: current['score'] = float(stripped.replace('Score:','').strip())
                except: current['score'] = 0.0
    if current: entries.append(current)
    for e in entries:
        e['file'] = fname
        if section == 'unmatched': all_unmatched.append(e); per_model_unmatched[fname].append(e)
        else: per_model_matched[fname].append(e)

total_unmatched = len(all_unmatched)
print(f"Total unmatched (excl GPT-4o): {total_unmatched}\n")

# --- Structural ---
def classify_structural(seq):
    if re.match(r'^[cs]\d+$', seq): return 'Single click'
    elif re.match(r'^c\d+c\d+$', seq): return 'Wrong navigation pair (2 clicks)'
    elif re.match(r'^(c\d+)+$', seq): return 'Navigation-only (3+ clicks, no form)'
    elif re.match(r'^(c\d+)+(t\d+|s\d+)$', seq): return 'Navigation + 1 field (incomplete form)'
    else: return 'Other'

struct = Counter(classify_structural(e.get('seq','')) for e in all_unmatched)
print("STRUCTURAL BREAKDOWN (excl GPT-4o):")
for cat in ['Navigation-only (3+ clicks, no form)', 'Navigation + 1 field (incomplete form)',
            'Wrong navigation pair (2 clicks)', 'Single click', 'Other']:
    c = struct.get(cat, 0)
    print(f"  {cat:45s} {c:5d} ({c/total_unmatched*100:.1f}%)")
print(f"  {'TOTAL':45s} {total_unmatched:5d}")

# --- Semantic (using the original analyze_precision.py's subcategory approach) ---
# We replicate the subcategory logic from the original script to get exact same bucketing
def classify_semantic_subcat(seq, inf):
    inf_lower = (inf or '').lower()
    # Structural: single-field -> "Single field interaction"
    if re.match(r'^(c\d+)+(t\d+|s\d+)$', seq):
        return 'Single field interaction'
    # Keyword-based
    if any(kw in inf_lower for kw in ['cancel','discard','return to','revert','back to','exit','close','undo']):
        return 'Cancel / discard / return'
    if any(kw in inf_lower for kw in ['dropdown','toggle','expand','collapse','menu']):
        return 'Dropdown / toggle'
    if any(kw in inf_lower for kw in ['homepage','home page','dashboard','main page','welcome','landing','clinic overview']):
        return 'Navigate to homepage'
    if any(kw in inf_lower for kw in ['navigate','open form','open edit','open add','initiate','display form']):
        return 'Navigate to form'
    if any(kw in inf_lower for kw in ['date','datepicker','birth date','visit date','calendar']):
        return 'Date-related interaction'
    if any(kw in inf_lower for kw in ['confirm','delete','remove','trigger confirmation']):
        return 'Delete / confirm (wrong path)'
    if any(kw in inf_lower for kw in ['submit','save','update','add new','create new','add a new']):
        return 'Incomplete submit / save'
    if any(kw in inf_lower for kw in ['search','find','filter']):
        return 'Search / filter'
    if any(kw in inf_lower for kw in ['edit','modify','change','rename','clear','focus','input','enter','type','field']):
        return 'Incomplete field edit'
    if any(kw in inf_lower for kw in ['view','display','list','browse','show','read-only']):
        return 'View / display (duplicate path)'
    return 'Other / unclassified'

sem = Counter(classify_semantic_subcat(e.get('seq',''), e.get('inf','')) for e in all_unmatched)
print("\nSEMANTIC CATEGORISATION (excl GPT-4o):")
for cat, c in sem.most_common():
    print(f"  {cat:40s} {c:5d} ({c/total_unmatched*100:.1f}%)")
print(f"  {'TOTAL':40s} {total_unmatched:5d}")
print(f"  Sum check: {sum(sem.values())}")

# --- Score separation ---
print("\nSCORE SEPARATION (excl GPT-4o):")
for fname in files:
    ms = [e['score'] for e in per_model_matched[fname] if 'score' in e]
    us = [e['score'] for e in per_model_unmatched[fname] if 'score' in e]
    if not ms or not us: continue
    m_avg = sum(ms)/len(ms); u_avg = sum(us)/len(us); m_min = min(ms)
    above = sum(1 for s in us if s >= m_min)
    pct = round(above/len(us)*100)
    print(f"  {model_names[fname]:25s} M_avg={m_avg:.3f} U_avg={u_avg:.3f} Min_m={m_min:.3f} Above={pct}%")
