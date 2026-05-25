"""Categorize unmatched sequences to find precision killers."""
import os, re
from pathlib import Path
from collections import Counter

GEN_DUMPS_DIR = Path(__file__).resolve().parent.parent / 'gen_dumps'

all_categories = Counter()
all_details = {}

for fname in sorted(os.listdir(GEN_DUMPS_DIR)):
    if fname.startswith('_'): continue
    fpath = os.path.join(GEN_DUMPS_DIR, fname)
    lines = open(fpath, 'r', encoding='utf-8').readlines()
    in_unmatched = False
    entries = []
    current = {}
    for line in lines:
        if 'UNMATCHED SEQUENCES' in line:
            in_unmatched = True
            continue
        if in_unmatched:
            if line.strip().startswith('Sequence:'):
                if current: entries.append(current)
                current = {'seq': line.strip().replace('Sequence:','').strip()}
            elif line.strip().startswith('Inferred:'):
                current['inf'] = line.strip().replace('Inferred:','').strip()
    if current: entries.append(current)
    
    categories = Counter()
    for e in entries:
        seq = e.get('seq','')
        inf = e.get('inf','').lower()
        
        cat = 'OTHER'
        
        # Categorize based on sequence structure AND inferred text
        is_single = re.match(r'^[cs]\d+$', seq)
        is_two_nav = re.match(r'^c\d+c\d+$', seq)
        is_nav_only = re.match(r'^(c\d+)+$', seq)  # Only click actions, no text/select
        has_one_field = re.match(r'^(c\d+)+(t\d+|s\d+)$', seq)  # Nav + single field
        has_form = re.search(r'(t\d+|s\d+)', seq)  # Has any text/select field
        
        # 1. Single nav (just one click - e.g. c2, c5, c8, c9)
        if is_single:
            cat = '1_SINGLE_NAV'
        
        # 2. Wrong navigation pair (e.g. c5c8, c8c9 - navbar hops that aren't in grammar)
        elif is_two_nav:
            cat = '2_WRONG_NAV_PAIR'
        
        # 3. Navigation-only (3+ clicks, no form fields) - going somewhere but not doing anything testable
        elif is_nav_only and not has_form:
            cat = '3_NAV_ONLY_DEEP'
        
        # 4. Single field fill (nav + one text/select but not enough for a complete form)
        elif has_one_field:
            cat = '4_SINGLE_FIELD'
        
        # 5. Form with multiple fields but wrong sequence
        elif has_form:
            cat = '5_WRONG_FORM_SEQ'
        
        else:
            cat = '6_OTHER'
        
        # Sub-categorize by inferred text content
        subcat = ''
        if any(kw in inf for kw in ['cancel','discard','return to','revert','back to','exit','close','undo']):
            subcat = '_cancel'
        elif any(kw in inf for kw in ['dropdown','toggle','expand','collapse','menu']):
            subcat = '_dropdown'
        elif any(kw in inf for kw in ['homepage','home page','dashboard','main page','welcome','landing','clinic overview']):
            subcat = '_homepage'
        elif any(kw in inf for kw in ['navigate','open form','open edit','open add','initiate','display form']):
            subcat = '_nav_to_form'
        elif any(kw in inf for kw in ['date','datepicker','birth date','visit date','calendar']):
            subcat = '_date'
        elif any(kw in inf for kw in ['confirm','delete','remove','trigger confirmation']):
            subcat = '_delete'
        elif any(kw in inf for kw in ['submit','save','update','add new','create new','add a new']):
            subcat = '_submit'
        elif any(kw in inf for kw in ['search','find','filter']):
            subcat = '_search'
        elif any(kw in inf for kw in ['edit','modify','change','rename','clear','focus','input','enter','type','field']):
            subcat = '_field_edit'
        elif any(kw in inf for kw in ['view','display','list','browse','show','read-only']):
            subcat = '_view'
        
        full_cat = cat + subcat
        categories[full_cat] += 1
        all_categories[full_cat] += 1
        if full_cat not in all_details:
            all_details[full_cat] = []
        all_details[full_cat].append((fname.replace('.txt',''), seq, inf))
    
    print(f'\n{fname}: {len(entries)} unmatched')
    for cat, count in categories.most_common():
        print(f'  {cat:30s}: {count:4d}  ({count/len(entries)*100:.0f}%)')

print(f'\n\n{"="*80}')
print(f'OVERALL CATEGORIES (total across all saves):')
print(f'{"="*80}')
total = sum(all_categories.values())
for cat, count in all_categories.most_common():
    pct = count/total*100
    print(f'  {cat:30s}: {count:4d}  ({pct:.1f}%)')
print(f'  {"TOTAL":30s}: {total:4d}')

# Show examples for top categories
print(f'\n\n{"="*80}')
print('TOP CATEGORY EXAMPLES:')
print(f'{"="*80}')
for cat, count in all_categories.most_common(15):
    print(f'\n--- {cat} ({count} occurrences) ---')
    examples = all_details[cat][:8]
    for save, seq, inf in examples:
        print(f'  [{save}] {seq:30s} | {inf[:70]}')
