"""
Deep Investigation: Why Navigation Features Have High Scores
=============================================================

This script investigates WHY navigation features are accumulating high scores
in AUTOE2E when they theoretically shouldn't according to the paper's design.

Hypotheses to investigate:
1. Score aggregation may be counting duplicates
2. Navigation is appearing at rank 1 consistently (high prior)
3. The similarity merging is not working properly for navigation
4. PETCLINIC benchmark may have navigation-heavy UI patterns
"""

import os
import re
import math
from collections import defaultdict, Counter

try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Please install required packages")
    exit(1)


def connect_to_mongodb():
    uri = os.getenv("ATLAS_URI")
    client = MongoClient(uri)
    return client, client.myDatabase


def is_navigation_feature(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    nav_keywords = [
        'navigate to', 'go to', 'navigates to', 'goes to',
        'access the', 'visit', 'open the', 'opens the',
        'view the', 'display the', 'show the',
        'switch to', 'switches to', 'return to', 'returns to',
        'back to', 'redirect to'
    ]
    for keyword in nav_keywords:
        if keyword in text_lower:
            return True
    view_patterns = [
        r'^view\s+\w+',
        r'^(display|show|open)\s+',
        r'^go\s+to\s+',
    ]
    for pattern in view_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def investigate_score_accumulation(app_name: str):
    """Deep dive into how scores are accumulated."""
    
    print("\n" + "=" * 80)
    print(f"🔬 DEEP INVESTIGATION: Score Accumulation for {app_name}")
    print("=" * 80)
    
    client, db = connect_to_mongodb()
    
    try:
        func_records = list(db["functionality"].find({"app": app_name}))
        afd_records = list(db["action-functionality"].find({"app": app_name}))
        
        # INVESTIGATION 1: Check for duplicate/similar navigation text
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 1: Navigation Text Duplication")
        print("-" * 80)
        
        nav_texts = []
        non_nav_texts = []
        for f in func_records:
            text = f.get('text', '').strip().lower()
            if is_navigation_feature(text):
                nav_texts.append(text)
            else:
                non_nav_texts.append(text)
        
        nav_text_counts = Counter(nav_texts)
        print(f"\n   Total navigation features: {len(nav_texts)}")
        print(f"   Unique navigation texts:   {len(nav_text_counts)}")
        print(f"   Duplication ratio:         {len(nav_texts)/max(1, len(nav_text_counts)):.2f}x")
        
        print(f"\n   Most common navigation texts (with counts):")
        for text, count in nav_text_counts.most_common(10):
            print(f"      [{count:3d}x] {text[:60]}")
        
        # INVESTIGATION 2: Check score calculation for top navigation features
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 2: Score Breakdown for Top Navigation Features")
        print("-" * 80)
        
        # Build lookup
        func_by_id = {str(f['_id']): f for f in func_records}
        afd_by_func = defaultdict(list)
        for afd in afd_records:
            fp = afd.get('func_pointer')
            if fp:
                afd_by_func[fp].append(afd)
        
        # Get top scoring navigation features
        nav_features = [(str(f['_id']), f) for f in func_records if is_navigation_feature(f.get('text', ''))]
        nav_features.sort(key=lambda x: x[1].get('score', 0), reverse=True)
        
        print(f"\n   Top 5 Navigation Features - Score Decomposition:")
        for func_id, func in nav_features[:5]:
            text = func.get('text', '')
            score = func.get('score', 0)
            afd_list = afd_by_func.get(func_id, [])
            
            print(f"\n   Feature: \"{text}\"")
            print(f"   Stored Score: {score:.3f}")
            print(f"   AFD Records: {len(afd_list)}")
            
            # Check what rank_scores contributed
            rank_scores = [afd.get('rank_score', 0) for afd in afd_list]
            if rank_scores:
                print(f"   Rank scores from AFD: {[f'{s:.3f}' for s in rank_scores]}")
                print(f"   Sum of rank_scores: {sum(rank_scores):.3f}")
                
                # Check the actual ranks (reverse engineering from geometric score)
                # geometric_score(rank) = log(1/(rank+1))
                # rank = (1/exp(score)) - 1
                ranks = []
                for rs in rank_scores:
                    try:
                        rank = round((1/math.exp(rs)) - 1)
                        ranks.append(rank)
                    except:
                        ranks.append("?")
                print(f"   Implied ranks: {ranks}")
        
        # INVESTIGATION 3: Check if navigation gets rank 1 more often
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 3: Rank Distribution Analysis")
        print("-" * 80)
        
        def rank_from_score(score):
            try:
                return round((1/math.exp(score)) - 1)
            except:
                return 99
        
        nav_ranks = []
        term_ranks = []
        
        for afd in afd_records:
            func_id = afd.get('func_pointer')
            if not func_id or func_id not in func_by_id:
                continue
            
            text = func_by_id[func_id].get('text', '')
            rank_score = afd.get('rank_score', -3.93)
            rank = rank_from_score(rank_score)
            
            if is_navigation_feature(text):
                nav_ranks.append(rank)
            else:
                term_ranks.append(rank)
        
        if nav_ranks:
            nav_rank_dist = Counter(nav_ranks)
            print(f"\n   Navigation Features Rank Distribution:")
            print(f"      Total observations: {len(nav_ranks)}")
            for rank in sorted(nav_rank_dist.keys())[:10]:
                pct = nav_rank_dist[rank] / len(nav_ranks) * 100
                bar = "█" * int(pct / 2)
                print(f"      Rank {rank:2d}: {nav_rank_dist[rank]:4d} ({pct:5.1f}%) {bar}")
        
        if term_ranks:
            term_rank_dist = Counter(term_ranks)
            print(f"\n   Non-Navigation Features Rank Distribution:")
            print(f"      Total observations: {len(term_ranks)}")
            for rank in sorted(term_rank_dist.keys())[:10]:
                pct = term_rank_dist[rank] / len(term_ranks) * 100
                bar = "█" * int(pct / 2)
                print(f"      Rank {rank:2d}: {term_rank_dist[rank]:4d} ({pct:5.1f}%) {bar}")
        
        # INVESTIGATION 4: Check score update mechanism
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 4: Score Update Audit")
        print("-" * 80)
        
        # The score in FD should be:
        # Initial: geometric_score(first_rank)
        # Updates: sum of (rank(F | A_i, A_i-1) - rank(F | A_i-1))
        #
        # BUT looking at the code, I see that:
        # - no_match_insert: score = geometric_score(rank + 1)
        # - Updates come from update_functionality_score which does:
        #   diff = curr_func['rank_score'] - prev_score
        #   func_db.update_one(..., '$inc': {'score': diff})
        
        # This means score ACCUMULATES rank_scores.
        # If a feature appears at rank 1 multiple times, it gets -0.693 added each time
        # BUT -0.693 is NEGATIVE, so adding it should DECREASE the score!
        
        print("\n   Score Update Logic Analysis:")
        print("   - Initial score: geometric_score(initial_rank)")
        print("   - Update rule: score += (curr_rank_score - prev_rank_score)")
        print("   - geometric_score(rank=1) = log(0.5) = -0.693")
        print("   - geometric_score(rank=2) = log(0.333) = -1.099")
        print("   - geometric_score(rank=50) = log(0.0196) = -3.932")
        print("")
        print("   Expected behavior:")
        print("   - Feature at rank 1 every time: score stays low (around -0.693)")
        print("   - Feature improving rank: score increases")
        print("   - Feature worsening rank: score decreases")
        
        # Let's check if there's something wrong with how scores are stored
        print("\n   Actual scores vs expected from AFD:")
        for func_id, func in nav_features[:5]:
            text = func.get('text', '')
            stored_score = func.get('score', 0)
            afd_list = afd_by_func.get(func_id, [])
            
            # Calculate what the score SHOULD be
            # The initial insert gives geometric_score(rank+1)
            # Then updates add diffs
            if afd_list:
                # First AFD record would have set initial score
                initial_rank_score = afd_list[0].get('rank_score', 0)
                # For simplicity, assume initial score = initial_rank_score
                # (This is approximately true)
                expected_score = initial_rank_score
                # No subsequent updates tracked in AFD alone
            else:
                expected_score = 0
            
            print(f"\n   \"{text[:40]}...\"")
            print(f"      Stored: {stored_score:.3f}, Expected (from 1st AFD): {expected_score:.3f}")
            print(f"      Difference: {stored_score - expected_score:.3f}")
            
            if abs(stored_score - expected_score) > 1:
                print(f"      ⚠️ LARGE DISCREPANCY - Investigate score accumulation!")
        
        # INVESTIGATION 5: Check for score being set vs incremented
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 5: Score Value Distribution")
        print("-" * 80)
        
        all_scores = [f.get('score', 0) for f in func_records]
        nav_scores = [f.get('score', 0) for f in func_records if is_navigation_feature(f.get('text', ''))]
        
        # Check if scores are clustered around specific values
        score_histogram = Counter([round(s, 1) for s in all_scores])
        print(f"\n   Score Distribution (all features):")
        for score_bucket in sorted(score_histogram.keys(), reverse=True)[:15]:
            count = score_histogram[score_bucket]
            bar = "█" * min(50, count)
            print(f"      {score_bucket:>6.1f}: {count:4d} {bar}")
        
        # Check the maximum theoretical score
        # If only 1 observation at rank 1: -0.693
        # If score is much higher, something is accumulating
        max_score = max(all_scores) if all_scores else 0
        print(f"\n   Maximum score in database: {max_score:.3f}")
        print(f"   Single rank-1 observation: -0.693")
        print(f"   If max >> -0.693, scores are being accumulated (possibly incorrectly)")
        
        if max_score > 1:
            print(f"\n   ⚠️ FINDING: Scores are accumulating beyond single-observation values!")
            print(f"   This suggests either:")
            print(f"   1. Multiple positive score updates (rank improvements)")
            print(f"   2. A bug in score initialization or update")
            print(f"   3. Intended cumulative scoring not matching paper description")
        
        # INVESTIGATION 6: What's happening with similarity merging?
        print("\n" + "-" * 80)
        print("📋 INVESTIGATION 6: Similarity Merging Impact")
        print("-" * 80)
        
        # Count features with similar texts (indicating merging should have happened)
        similar_nav = defaultdict(list)
        for f in func_records:
            text = f.get('text', '').strip().lower()
            if is_navigation_feature(text):
                # Normalize text for grouping
                normalized = re.sub(r'[^\w\s]', '', text)
                normalized = ' '.join(normalized.split())
                similar_nav[normalized].append(f)
        
        print(f"\n   Navigation features that should potentially be merged:")
        merged_groups = [(k, v) for k, v in similar_nav.items() if len(v) > 1]
        merged_groups.sort(key=lambda x: len(x[1]), reverse=True)
        
        for text, features in merged_groups[:10]:
            scores = [f.get('score', 0) for f in features]
            print(f"      \"{text[:50]}\"")
            print(f"         Count: {len(features)}, Scores: {[f'{s:.2f}' for s in scores]}")
        
        if merged_groups:
            print(f"\n   ⚠️ FINDING: {len(merged_groups)} navigation text groups exist with duplicates!")
            print(f"   Similarity merging may not be combining them properly.")
        
        return {
            'nav_text_duplication': len(nav_texts) / max(1, len(nav_text_counts)),
            'max_score': max_score,
            'merged_groups_count': len(merged_groups)
        }
        
    finally:
        client.close()


if __name__ == "__main__":
    import sys
    app_name = sys.argv[1] if len(sys.argv) > 1 else "PETCLINIC"
    results = investigate_score_accumulation(app_name)
