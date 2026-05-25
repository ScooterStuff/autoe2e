"""
Navigation Feature Analysis for AUTOE2E
========================================

This script investigates whether navigation actions should be retained as features
under the AUTOE2E methodology by:

1. Identifying navigation-related features in the database
2. Tracking their confidence score updates over time
3. Analyzing terminal vs non-terminal occurrences
4. Evaluating against Definition 2 criteria
5. Comparing against other feature types

Key Questions:
- Do navigation features accumulate consistent positive evidence?
- Do their scores decay due to high prior probability?
- Are they filtered out after aggregation and cutoff?
"""

import os
import re
import math
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

try:
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Please install required packages:")
    print("  pip install pymongo python-dotenv")
    exit(1)


# AUTOE2E Score Threshold (from paper)
SCORE_THRESHOLD = math.log(0.5)  # -0.693


def geometric_score(rank: int) -> float:
    """
    Compute geometric score for a rank.
    rank=1 gives highest score (log(0.5) = -0.693)
    rank=None gives lowest score (log(1/51) ≈ -3.93)
    """
    if rank is None or rank > 50:
        return math.log(1/51)
    return math.log(1/(rank + 1))


def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    uri = os.getenv("ATLAS_URI")
    if not uri:
        raise ValueError("ATLAS_URI not found in environment")
    client = MongoClient(uri)
    db = client.myDatabase
    return client, db


def is_navigation_feature(text: str) -> bool:
    """
    Determine if a feature text describes navigation.
    
    Navigation patterns:
    - "navigate to X", "go to X"
    - "view X page", "open X page"
    - "access X", "visit X"
    - Single clicks that just change pages
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Explicit navigation keywords
    nav_keywords = [
        'navigate to', 'go to', 'navigates to', 'goes to',
        'access the', 'visit', 'open the', 'opens the',
        'view the', 'display the', 'show the',
        'switch to', 'switches to',
        'return to', 'returns to',
        'back to', 'redirect to'
    ]
    
    # Check for explicit navigation patterns
    for keyword in nav_keywords:
        if keyword in text_lower:
            return True
    
    # Patterns that suggest pure page viewing (not data operations)
    view_patterns = [
        r'^view\s+\w+\s+(page|list|section|panel|tab)$',
        r'^(display|show|open)\s+\w+\s+(page|list|section)$',
        r'^go\s+to\s+',
        r'page\s*$',  # ends with "page"
    ]
    
    for pattern in view_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def is_terminal_feature(text: str) -> bool:
    """
    Determine if a feature is likely terminal (completes a user goal).
    
    Terminal features typically involve:
    - Data modification (add, edit, delete, save, update)
    - Form submissions
    - Explicit actions with side effects
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    terminal_keywords = [
        'add ', 'create ', 'delete ', 'remove ', 'edit ',
        'update ', 'save ', 'submit ', 'confirm ', 'cancel ',
        'register', 'login', 'logout', 'sign in', 'sign out',
        'upload ', 'download ', 'export ', 'import ',
        'send ', 'post ', 'publish ', 'assign ', 'unassign '
    ]
    
    for keyword in terminal_keywords:
        if keyword in text_lower:
            return True
    
    return False


def analyze_navigation_features(app_name: str, verbose: bool = True):
    """
    Main analysis function for navigation features.
    """
    print("\n" + "=" * 70)
    print(f"🔍 NAVIGATION FEATURE ANALYSIS: {app_name}")
    print("=" * 70)
    
    client, db = connect_to_mongodb()
    
    try:
        # Load all functionalities for the app
        func_records = list(db["functionality"].find({"app": app_name}))
        afd_records = list(db["action-functionality"].find({"app": app_name}))
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total functionalities: {len(func_records)}")
        print(f"   Total AFD records: {len(afd_records)}")
        
        if not func_records:
            print(f"   ⚠️ No data found for {app_name}")
            return None
        
        # Build lookup dictionaries
        func_by_id = {str(f['_id']): f for f in func_records}
        afd_by_func = defaultdict(list)
        for afd in afd_records:
            fp = afd.get('func_pointer')
            if fp:
                afd_by_func[fp].append(afd)
        
        # Classify features
        nav_features = []
        terminal_features = []
        other_features = []
        
        for func in func_records:
            func_id = str(func['_id'])
            text = func.get('text', '')
            score = func.get('score', 0)
            is_final = func.get('final', False)
            
            feature_info = {
                'id': func_id,
                'text': text,
                'score': score,
                'is_final': is_final,
                'afd_count': len(afd_by_func.get(func_id, [])),
                'afd_records': afd_by_func.get(func_id, [])
            }
            
            if is_navigation_feature(text):
                nav_features.append(feature_info)
            elif is_terminal_feature(text):
                terminal_features.append(feature_info)
            else:
                other_features.append(feature_info)
        
        # Sort by score (descending)
        nav_features.sort(key=lambda x: x['score'], reverse=True)
        terminal_features.sort(key=lambda x: x['score'], reverse=True)
        other_features.sort(key=lambda x: x['score'], reverse=True)
        
        # =================================================================
        # ANALYSIS 1: Feature Distribution
        # =================================================================
        print("\n" + "-" * 70)
        print("📋 ANALYSIS 1: Feature Classification Distribution")
        print("-" * 70)
        
        total = len(func_records)
        print(f"   Navigation features:  {len(nav_features):3d} ({len(nav_features)/total*100:.1f}%)")
        print(f"   Terminal features:    {len(terminal_features):3d} ({len(terminal_features)/total*100:.1f}%)")
        print(f"   Other features:       {len(other_features):3d} ({len(other_features)/total*100:.1f}%)")
        
        # =================================================================
        # ANALYSIS 2: Score Distribution
        # =================================================================
        print("\n" + "-" * 70)
        print("📈 ANALYSIS 2: Score Distribution by Feature Type")
        print("-" * 70)
        
        def score_stats(features):
            if not features:
                return {'min': 0, 'max': 0, 'avg': 0, 'above_threshold': 0, 'final_count': 0}
            scores = [f['score'] for f in features]
            return {
                'min': min(scores),
                'max': max(scores),
                'avg': sum(scores) / len(scores),
                'above_threshold': sum(1 for s in scores if s >= SCORE_THRESHOLD),
                'final_count': sum(1 for f in features if f['is_final'])
            }
        
        nav_stats = score_stats(nav_features)
        term_stats = score_stats(terminal_features)
        other_stats = score_stats(other_features)
        
        print(f"\n   Threshold: {SCORE_THRESHOLD:.3f}")
        print(f"\n   {'Type':<20} {'Min':>8} {'Max':>8} {'Avg':>8} {'≥Thresh':>8} {'Final':>8}")
        print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"   {'Navigation':<20} {nav_stats['min']:>8.3f} {nav_stats['max']:>8.3f} {nav_stats['avg']:>8.3f} {nav_stats['above_threshold']:>8} {nav_stats['final_count']:>8}")
        print(f"   {'Terminal':<20} {term_stats['min']:>8.3f} {term_stats['max']:>8.3f} {term_stats['avg']:>8.3f} {term_stats['above_threshold']:>8} {term_stats['final_count']:>8}")
        print(f"   {'Other':<20} {other_stats['min']:>8.3f} {other_stats['max']:>8.3f} {other_stats['avg']:>8.3f} {other_stats['above_threshold']:>8} {other_stats['final_count']:>8}")
        
        # =================================================================
        # ANALYSIS 3: Top Features Overall (to see where navigation ranks)
        # =================================================================
        print("\n" + "-" * 70)
        print("🏆 ANALYSIS 3: Top 20 Features by Score (All Types)")
        print("-" * 70)
        
        all_features = nav_features + terminal_features + other_features
        all_features.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n   {'Rank':<5} {'Score':>8} {'Type':<12} {'Final':<6} {'Feature Text'}")
        print(f"   {'-'*5} {'-'*8} {'-'*12} {'-'*6} {'-'*50}")
        
        nav_count_in_top20 = 0
        for i, f in enumerate(all_features[:20], 1):
            ftype = 'NAV' if is_navigation_feature(f['text']) else ('TERM' if is_terminal_feature(f['text']) else 'OTHER')
            if ftype == 'NAV':
                nav_count_in_top20 += 1
            final_str = '✓' if f['is_final'] else ''
            text_truncated = f['text'][:50] + '...' if len(f['text']) > 50 else f['text']
            print(f"   {i:<5} {f['score']:>8.3f} {ftype:<12} {final_str:<6} {text_truncated}")
        
        print(f"\n   Navigation features in top 20: {nav_count_in_top20}")
        
        # =================================================================
        # ANALYSIS 4: Navigation Features Detail
        # =================================================================
        print("\n" + "-" * 70)
        print("🧭 ANALYSIS 4: All Navigation Features (Detailed)")
        print("-" * 70)
        
        passing_nav = [f for f in nav_features if f['score'] >= SCORE_THRESHOLD]
        failing_nav = [f for f in nav_features if f['score'] < SCORE_THRESHOLD]
        
        print(f"\n   Navigation features passing threshold: {len(passing_nav)}/{len(nav_features)}")
        print(f"   Navigation features marked final:      {sum(1 for f in nav_features if f['is_final'])}/{len(nav_features)}")
        
        if verbose and nav_features:
            print(f"\n   📌 PASSING (score ≥ {SCORE_THRESHOLD:.3f}):")
            for f in passing_nav[:10]:
                final_str = '✓FINAL' if f['is_final'] else ''
                print(f"      • [{f['score']:>6.3f}] {f['text'][:60]} {final_str}")
            
            print(f"\n   ❌ FAILING (score < {SCORE_THRESHOLD:.3f}):")
            for f in failing_nav[:10]:
                print(f"      • [{f['score']:>6.3f}] {f['text'][:60]}")
        
        # =================================================================
        # ANALYSIS 5: Score Evolution Analysis
        # =================================================================
        print("\n" + "-" * 70)
        print("📉 ANALYSIS 5: Score Update Pattern Analysis")
        print("-" * 70)
        
        def analyze_score_updates(features, feature_type: str):
            """Analyze how scores evolved based on AFD records."""
            total_updates = 0
            positive_updates = 0
            negative_updates = 0
            terminal_occurrences = 0
            non_terminal_occurrences = 0
            
            for f in features:
                afd_list = f.get('afd_records', [])
                for afd in afd_list:
                    total_updates += 1
                    rank_score = afd.get('rank_score', 0)
                    
                    # Terminal = FORM type or high depth with final=True
                    is_terminal = afd.get('type') in ['FORM', 'FORM_DOUBLE'] or afd.get('final', False)
                    
                    if is_terminal:
                        terminal_occurrences += 1
                    else:
                        non_terminal_occurrences += 1
                    
                    # Check if this observation contributed positively
                    # A rank 1 observation has score -0.693, rank 2 has -1.386, etc.
                    # Positive contribution = better than average rank
                    if rank_score > geometric_score(3):  # Better than rank 3
                        positive_updates += 1
                    else:
                        negative_updates += 1
            
            return {
                'total_updates': total_updates,
                'positive_updates': positive_updates,
                'negative_updates': negative_updates,
                'terminal_occurrences': terminal_occurrences,
                'non_terminal_occurrences': non_terminal_occurrences
            }
        
        nav_updates = analyze_score_updates(nav_features, 'Navigation')
        term_updates = analyze_score_updates(terminal_features, 'Terminal')
        
        print(f"\n   Navigation Features Score Pattern:")
        print(f"      Total AFD observations:    {nav_updates['total_updates']}")
        print(f"      High-rank observations:    {nav_updates['positive_updates']} ({nav_updates['positive_updates']/max(1,nav_updates['total_updates'])*100:.1f}%)")
        print(f"      Low-rank observations:     {nav_updates['negative_updates']} ({nav_updates['negative_updates']/max(1,nav_updates['total_updates'])*100:.1f}%)")
        print(f"      Terminal occurrences:      {nav_updates['terminal_occurrences']}")
        print(f"      Non-terminal occurrences:  {nav_updates['non_terminal_occurrences']}")
        
        print(f"\n   Terminal Features Score Pattern (for comparison):")
        print(f"      Total AFD observations:    {term_updates['total_updates']}")
        print(f"      High-rank observations:    {term_updates['positive_updates']} ({term_updates['positive_updates']/max(1,term_updates['total_updates'])*100:.1f}%)")
        print(f"      Low-rank observations:     {term_updates['negative_updates']} ({term_updates['negative_updates']/max(1,term_updates['total_updates'])*100:.1f}%)")
        print(f"      Terminal occurrences:      {term_updates['terminal_occurrences']}")
        print(f"      Non-terminal occurrences:  {term_updates['non_terminal_occurrences']}")
        
        # =================================================================
        # ANALYSIS 6: Definition 2 Evaluation
        # =================================================================
        print("\n" + "-" * 70)
        print("📖 ANALYSIS 6: Definition 2 Criteria Evaluation")
        print("-" * 70)
        
        print("""
   Definition 2 from AUTOE2E Paper:
   A functionality F is an "essential operation" if:
   1. It produces a user-visible outcome beyond mere state transition
   2. It would be written as an E2E test case by developers
   3. It represents a final user goal (not just a step)
   
   Navigation Analysis Against Definition 2:
""")
        
        # Criterion 1: User-visible outcome beyond state transition
        nav_with_visible_outcome = 0
        for f in nav_features:
            text = f['text'].lower()
            # Navigation typically only produces state transition
            # unless it loads data or displays information
            if any(x in text for x in ['load', 'display', 'show', 'list', 'view all']):
                nav_with_visible_outcome += 1
        
        # Criterion 2: Would developers write E2E tests?
        nav_testable = 0
        for f in nav_features:
            text = f['text'].lower()
            # Simple navigation is typically not independently tested
            # unless it's testing route guards or access control
            if any(x in text for x in ['verify', 'check', 'ensure', 'restricted', 'authorized']):
                nav_testable += 1
        
        # Criterion 3: Final user goal
        nav_final_goal = sum(1 for f in nav_features if f['is_final'])
        
        print(f"   Criterion 1 (visible outcome beyond transition):")
        print(f"      Navigation features meeting criteria: {nav_with_visible_outcome}/{len(nav_features)}")
        print(f"      ANALYSIS: Most navigation only changes URL/view state")
        
        print(f"\n   Criterion 2 (developers would write E2E tests):")
        print(f"      Navigation features meeting criteria: {nav_testable}/{len(nav_features)}")
        print(f"      ANALYSIS: Navigation is tested implicitly as part of other features")
        
        print(f"\n   Criterion 3 (represents final user goal):")
        print(f"      Navigation features marked final:     {nav_final_goal}/{len(nav_features)}")
        print(f"      ANALYSIS: Navigation is typically a means to an end")
        
        # =================================================================
        # SUMMARY AND RECOMMENDATION
        # =================================================================
        print("\n" + "=" * 70)
        print("📋 SUMMARY AND RECOMMENDATION")
        print("=" * 70)
        
        # Calculate key metrics
        nav_pass_rate = len(passing_nav) / len(nav_features) if nav_features else 0
        term_pass_rate = len([f for f in terminal_features if f['score'] >= SCORE_THRESHOLD]) / len(terminal_features) if terminal_features else 0
        
        nav_avg_score = nav_stats['avg']
        term_avg_score = term_stats['avg']
        
        print(f"""
   Key Findings:
   
   1. SCORE FILTERING EFFECTIVENESS:
      • Navigation pass rate:  {nav_pass_rate*100:.1f}%
      • Terminal pass rate:    {term_pass_rate*100:.1f}%
      • Navigation avg score:  {nav_avg_score:.3f}
      • Terminal avg score:    {term_avg_score:.3f}
      
   2. RANKING POSITION:
      • Navigation features in top 20: {nav_count_in_top20}/20
      
   3. TERMINAL VS NON-TERMINAL:
      • Navigation terminal occurrences:     {nav_updates['terminal_occurrences']}
      • Navigation non-terminal occurrences: {nav_updates['non_terminal_occurrences']}
      
   4. DEFINITION 2 COMPLIANCE:
      • Navigation features fully meeting criteria: ~{min(nav_with_visible_outcome, nav_testable, nav_final_goal)}/{len(nav_features)}
""")
        
        # Determine recommendation
        if nav_pass_rate < 0.3 and nav_avg_score < term_avg_score - 0.5:
            recommendation = """
   RECOMMENDATION: Navigation features are NATURALLY FILTERED OUT
   
   The AUTOE2E scoring mechanism effectively deprioritizes navigation because:
   • They rarely appear at high ranks (users navigate to DO something else)
   • Their scores decay as they appear as non-terminal intermediary steps
   • They don't satisfy Definition 2 criteria for essential operations
   
   The paper's methodology already handles this - no special filtering needed.
   Navigation features that somehow accumulate high scores likely represent
   important routing/access features that ARE worth testing (e.g., "view owner list").
"""
        elif nav_pass_rate > 0.5:
            recommendation = """
   RECOMMENDATION: INVESTIGATE - High navigation retention rate
   
   Many navigation features are passing the threshold, which may indicate:
   • The application has many standalone view pages worth testing
   • The threshold may need adjustment
   • Some "navigation" is actually "viewing data" which IS valuable
   
   Consider manual review of passing navigation features.
"""
        else:
            recommendation = """
   RECOMMENDATION: MIXED - Some navigation retained
   
   The scoring mechanism partially filters navigation. Features like
   "view list of X" or "display X page" may legitimately represent
   valuable test scenarios if they verify data display.
   
   Consider:
   • "View list" features where the list content matters -> KEEP
   • Pure "navigate to page" without verification -> FILTER
"""
        
        print(recommendation)
        
        # Return analysis data for programmatic use
        return {
            'app_name': app_name,
            'total_features': len(func_records),
            'navigation_features': len(nav_features),
            'terminal_features': len(terminal_features),
            'nav_passing': len(passing_nav),
            'nav_pass_rate': nav_pass_rate,
            'nav_avg_score': nav_avg_score,
            'term_avg_score': term_avg_score,
            'nav_in_top20': nav_count_in_top20,
            'nav_features_detail': nav_features,
            'recommendation': recommendation.strip()
        }
        
    finally:
        client.close()


def analyze_multi_run_score_evolution(app_name: str):
    """
    Analyze how navigation features' scores evolved across exploration.
    Track the score update rule: rank(F | Aᵢ, Aᵢ₋₁) − rank(F | Aᵢ₋₁)
    """
    print("\n" + "=" * 70)
    print(f"📊 MULTI-CONTEXT SCORE EVOLUTION ANALYSIS: {app_name}")
    print("=" * 70)
    
    client, db = connect_to_mongodb()
    
    try:
        # Get all AFD records sorted by depth (proxy for temporal order)
        afd_records = list(db["action-functionality"].find({"app": app_name}).sort("depth", 1))
        func_records = list(db["functionality"].find({"app": app_name}))
        
        func_by_id = {str(f['_id']): f for f in func_records}
        
        # Track score contributions per feature
        feature_score_history = defaultdict(list)
        
        # Group AFD records by action chain (state + action)
        action_observations = defaultdict(list)
        for afd in afd_records:
            key = (afd.get('state'), afd.get('action'))
            action_observations[key].append(afd)
        
        # For each feature, track its rank at each observation
        for afd in afd_records:
            func_id = afd.get('func_pointer')
            if not func_id or func_id not in func_by_id:
                continue
            
            feature = func_by_id[func_id]
            text = feature.get('text', '')
            
            feature_score_history[func_id].append({
                'depth': afd.get('depth', 0),
                'rank_score': afd.get('rank_score', 0),
                'type': afd.get('type', 'SINGLE'),
                'state': afd.get('state', ''),
                'text': text,
                'is_nav': is_navigation_feature(text)
            })
        
        # Analyze score accumulation patterns
        print("\n   Score Contribution Patterns by Feature Type:")
        print("-" * 70)
        
        nav_contributions = []
        term_contributions = []
        
        for func_id, history in feature_score_history.items():
            text = func_by_id[func_id].get('text', '')
            total_contribution = sum(h['rank_score'] for h in history)
            
            if is_navigation_feature(text):
                nav_contributions.append({
                    'text': text,
                    'observations': len(history),
                    'total_contribution': total_contribution,
                    'avg_rank_score': total_contribution / len(history) if history else 0,
                    'history': history
                })
            elif is_terminal_feature(text):
                term_contributions.append({
                    'text': text,
                    'observations': len(history),
                    'total_contribution': total_contribution,
                    'avg_rank_score': total_contribution / len(history) if history else 0,
                    'history': history
                })
        
        # Sort by total contribution
        nav_contributions.sort(key=lambda x: x['total_contribution'], reverse=True)
        term_contributions.sort(key=lambda x: x['total_contribution'], reverse=True)
        
        print(f"\n   Top 5 Navigation Features by Score Contribution:")
        for i, nc in enumerate(nav_contributions[:5], 1):
            print(f"   {i}. {nc['text'][:50]}")
            print(f"      Observations: {nc['observations']}, Total: {nc['total_contribution']:.3f}, Avg: {nc['avg_rank_score']:.3f}")
        
        print(f"\n   Top 5 Terminal Features by Score Contribution:")
        for i, tc in enumerate(term_contributions[:5], 1):
            print(f"   {i}. {tc['text'][:50]}")
            print(f"      Observations: {tc['observations']}, Total: {tc['total_contribution']:.3f}, Avg: {tc['avg_rank_score']:.3f}")
        
        # Analyze score decay for navigation
        print(f"\n   Navigation Score Decay Analysis:")
        print("-" * 70)
        
        decay_count = 0
        growth_count = 0
        for nc in nav_contributions:
            if len(nc['history']) >= 2:
                first_half = nc['history'][:len(nc['history'])//2]
                second_half = nc['history'][len(nc['history'])//2:]
                first_avg = sum(h['rank_score'] for h in first_half) / len(first_half) if first_half else 0
                second_avg = sum(h['rank_score'] for h in second_half) / len(second_half) if second_half else 0
                if second_avg < first_avg:
                    decay_count += 1
                else:
                    growth_count += 1
        
        print(f"   Features showing score decay over time: {decay_count}")
        print(f"   Features showing score growth over time: {growth_count}")
        
        if decay_count > growth_count:
            print(f"\n   FINDING: Navigation features tend to DECAY in score")
            print(f"   This suggests they appear at lower ranks as exploration continues,")
            print(f"   likely because they are prerequisites to more specific features.")
        else:
            print(f"\n   FINDING: Navigation features maintain or grow scores")
            print(f"   This may indicate persistent high-rank appearances.")
        
        return {
            'nav_contributions': nav_contributions,
            'term_contributions': term_contributions,
            'decay_vs_growth': {'decay': decay_count, 'growth': growth_count}
        }
        
    finally:
        client.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze navigation features in AUTOE2E")
    parser.add_argument("app", nargs="?", default="PETCLINIC",
                        help="App name to analyze")
    parser.add_argument("--evolution", action="store_true",
                        help="Also analyze score evolution over time")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Less verbose output")
    
    args = parser.parse_args()
    
    results = analyze_navigation_features(args.app, verbose=not args.quiet)
    
    if args.evolution:
        evolution_results = analyze_multi_run_score_evolution(args.app)
