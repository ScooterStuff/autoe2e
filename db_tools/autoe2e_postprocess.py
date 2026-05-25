"""
AUTOE2E Post-Processing Script

This script addresses the low precision problem when using smaller LLMs by:
1. Clustering similar feature descriptions using aggressive text normalization
2. Consolidating scores across duplicate features
3. Re-ranking and filtering test sequences

Usage:
    python autoe2e_postprocess.py <APP_NAME> [--threshold SCORE_THRESHOLD]
"""

import os
import re
import sys
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# Similarity threshold for text clustering (0.0-1.0)
CLUSTER_THRESHOLD = 0.6  # More aggressive than default 0.85

# Score threshold for final filtering
DEFAULT_SCORE_THRESHOLD = -0.693  # log(0.5)

# =============================================================================
# Database Connection (matches evaluate_autoe2e.py)
# =============================================================================

def connect_to_mongodb():
    """Connect to MongoDB Atlas (same as evaluate_autoe2e.py)"""
    uri = os.getenv("ATLAS_URI")
    if not uri:
        raise ValueError(
            "ATLAS_URI not found in environment.\n"
            "Create a .env file with your MongoDB Atlas connection string:\n"
            "  ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/"
        )
    
    client = MongoClient(uri)
    db = client.myDatabase  # Same database name as evaluate_autoe2e.py
    return client, db

# Initialize connection lazily
_client = None
_db = None
func_db = None
action_func_db = None

def get_db():
    """Get database connection (lazy initialization)"""
    global _client, _db, func_db, action_func_db
    if _db is None:
        _client, _db = connect_to_mongodb()
        func_db = _db["functionality"]  # Match collection name from evaluate script
        action_func_db = _db["action-functionality"]
    return _db, func_db, action_func_db


# =============================================================================
# Text Normalization (Aggressive)
# =============================================================================

def normalize_aggressive(text: str) -> str:
    """
    Aggressively normalize text for clustering.
    Much more aggressive than the standard normalization.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove common filler words that don't change meaning
    filler_words = {
        'the', 'a', 'an', 'to', 'of', 'for', 'in', 'on', 'at', 'by',
        'with', 'from', 'this', 'that', 'these', 'those',
        'specific', 'particular', 'certain', 'given',
        'page', 'section', 'area', 'screen', 'view',
        'button', 'link', 'element', 'item', 'option',
        'click', 'clicking', 'select', 'selecting',
        'navigate', 'navigating', 'navigation', 'go', 'going',
        'access', 'accessing', 'open', 'opening',
        'perform', 'performing', 'execute', 'executing',
        'feature', 'functionality', 'function', 'action',
        'within', 'inside', 'into', 'onto',
        'current', 'new', 'existing'
    }
    
    words = text.split()
    words = [w for w in words if w not in filler_words and len(w) > 1]
    
    # Sort words to make "add owner" == "owner add"
    words = sorted(set(words))
    
    return ' '.join(words)


def extract_key_concepts(text: str) -> set:
    """
    Extract key concepts/entities from feature description.
    """
    if not text:
        return set()
    
    text = text.lower()
    
    # Common entity patterns in web apps
    entities = set()
    
    # CRUD operations
    crud_patterns = [
        (r'\b(add|create|new|insert)\b', 'create'),
        (r'\b(edit|update|modify|change)\b', 'update'),
        (r'\b(delete|remove|destroy)\b', 'delete'),
        (r'\b(view|show|display|list|see|read)\b', 'read'),
        (r'\b(search|find|filter|query)\b', 'search'),
    ]
    
    for pattern, concept in crud_patterns:
        if re.search(pattern, text):
            entities.add(concept)
    
    # Common web app entities (customize for your app)
    entity_patterns = [
        (r'\bowner[s]?\b', 'owner'),
        (r'\bpet[s]?\b', 'pet'),
        (r'\bvet[s]?\b', 'vet'),
        (r'\bvisit[s]?\b', 'visit'),
        (r'\bspecialt(?:y|ies)\b', 'specialty'),
        (r'\btype[s]?\b', 'type'),
        (r'\bpet\s*type[s]?\b', 'pettype'),
        (r'\buser[s]?\b', 'user'),
        (r'\baccount[s]?\b', 'account'),
        (r'\bprofile[s]?\b', 'profile'),
        (r'\bsetting[s]?\b', 'setting'),
        (r'\bform[s]?\b', 'form'),
        (r'\bdetail[s]?\b', 'detail'),
        (r'\binfo(?:rmation)?\b', 'info'),
    ]
    
    for pattern, concept in entity_patterns:
        if re.search(pattern, text):
            entities.add(concept)
    
    return entities


def features_are_similar(text1: str, text2: str, threshold: float = CLUSTER_THRESHOLD) -> bool:
    """
    Determine if two feature descriptions are similar enough to cluster.
    """
    # Aggressive normalization
    norm1 = normalize_aggressive(text1)
    norm2 = normalize_aggressive(text2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # If either is empty after normalization, they might be junk
    if not norm1 or not norm2:
        return False
    
    # Word-based Jaccard similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0
    
    # Also check key concepts
    concepts1 = extract_key_concepts(text1)
    concepts2 = extract_key_concepts(text2)
    
    if concepts1 and concepts2:
        concept_intersection = len(concepts1 & concepts2)
        concept_union = len(concepts1 | concepts2)
        concept_similarity = concept_intersection / concept_union if concept_union > 0 else 0
        
        # If concepts match well, lower the word threshold
        if concept_similarity >= 0.8:
            return jaccard >= 0.4 or concept_similarity >= 0.8
    
    return jaccard >= threshold


# =============================================================================
# Clustering Functions
# =============================================================================

def cluster_features(features: list) -> dict:
    """
    Cluster similar features together.
    Returns a dict mapping cluster_id -> list of feature docs
    """
    if not features:
        return {}
    
    clusters = {}
    cluster_id = 0
    assigned = set()
    
    for i, feat1 in enumerate(features):
        if feat1['_id'] in assigned:
            continue
        
        # Start a new cluster
        cluster = [feat1]
        assigned.add(feat1['_id'])
        
        # Find all similar features
        for j, feat2 in enumerate(features):
            if i == j or feat2['_id'] in assigned:
                continue
            
            if features_are_similar(feat1['text'], feat2['text']):
                cluster.append(feat2)
                assigned.add(feat2['_id'])
        
        clusters[cluster_id] = cluster
        cluster_id += 1
    
    return clusters


def get_cluster_representative(cluster: list) -> dict:
    """
    Get the best representative from a cluster.
    Chooses the one with highest score, aggregating scores from all members.
    """
    if not cluster:
        return None
    
    # Sum all scores
    total_score = sum(f.get('score', 0) for f in cluster)
    
    # Pick the feature with highest individual score as representative
    best = max(cluster, key=lambda f: f.get('score', 0))
    
    # Return with aggregated score
    return {
        '_id': best['_id'],
        'text': best['text'],
        'original_score': best.get('score', 0),
        'aggregated_score': total_score,
        'cluster_size': len(cluster),
        'cluster_members': [f['text'] for f in cluster]
    }


# =============================================================================
# Main Post-Processing
# =============================================================================

def postprocess_features(app_name: str, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    """
    Post-process features for an app:
    1. Cluster similar features
    2. Aggregate scores
    3. Report statistics
    """
    print(f"\n{'='*60}")
    print(f"POST-PROCESSING: {app_name}")
    print(f"{'='*60}\n")
    
    # Get database connection
    db, func_db, action_func_db = get_db()
    
    # Load all features
    all_features = list(func_db.find({'app': app_name}))
    print(f"Total features in database: {len(all_features)}")
    
    # Filter by score
    above_threshold = [f for f in all_features if f.get('score', 0) >= score_threshold]
    print(f"Features above threshold ({score_threshold:.3f}): {len(above_threshold)}")
    
    # Cluster features
    print(f"\nClustering features (threshold={CLUSTER_THRESHOLD})...")
    clusters = cluster_features(above_threshold)
    print(f"Number of clusters: {len(clusters)}")
    
    # Get representatives
    representatives = []
    for cluster_id, cluster in clusters.items():
        rep = get_cluster_representative(cluster)
        if rep:
            representatives.append(rep)
    
    # Sort by aggregated score
    representatives.sort(key=lambda x: x['aggregated_score'], reverse=True)
    
    # Report
    print(f"\n{'='*60}")
    print("CLUSTERED FEATURES (sorted by aggregated score)")
    print(f"{'='*60}\n")
    
    for i, rep in enumerate(representatives[:30]):  # Top 30
        print(f"[{i+1}] Score: {rep['aggregated_score']:.3f} (original: {rep['original_score']:.3f})")
        print(f"    Text: {rep['text'][:80]}...")
        print(f"    Cluster size: {rep['cluster_size']}")
        if rep['cluster_size'] > 1:
            print(f"    Members:")
            for member in rep['cluster_members'][:3]:
                print(f"      - {member[:60]}...")
            if rep['cluster_size'] > 3:
                print(f"      ... and {rep['cluster_size'] - 3} more")
        print()
    
    # Statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}\n")
    
    print(f"Original feature count: {len(above_threshold)}")
    print(f"After clustering: {len(representatives)}")
    print(f"Reduction: {100 * (1 - len(representatives)/len(above_threshold)):.1f}%")
    
    # Size distribution
    sizes = [rep['cluster_size'] for rep in representatives]
    print(f"\nCluster size distribution:")
    print(f"  Singletons (size=1): {sum(1 for s in sizes if s == 1)}")
    print(f"  Small (2-5): {sum(1 for s in sizes if 2 <= s <= 5)}")
    print(f"  Medium (6-10): {sum(1 for s in sizes if 6 <= s <= 10)}")
    print(f"  Large (>10): {sum(1 for s in sizes if s > 10)}")
    
    return representatives


def update_database_with_clusters(app_name: str, representatives: list, dry_run: bool = True):
    """
    Optionally update the database to merge clustered features.
    
    WARNING: This modifies the database. Use dry_run=True first!
    """
    if dry_run:
        print("\n[DRY RUN] Would update database with clustered features")
        print("Set dry_run=False to actually apply changes")
        return
    
    # Get database connection
    db, func_db, action_func_db = get_db()
    
    print("\nUpdating database...")
    
    for rep in representatives:
        # Update the representative's score
        func_db.update_one(
            {'_id': rep['_id']},
            {'$set': {'score': rep['aggregated_score']}}
        )
    
    print(f"Updated {len(representatives)} feature scores")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autoe2e_postprocess.py <APP_NAME> [--threshold SCORE]")
        sys.exit(1)
    
    app_name = sys.argv[1]
    threshold = DEFAULT_SCORE_THRESHOLD
    
    if '--threshold' in sys.argv:
        idx = sys.argv.index('--threshold')
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])
    
    representatives = postprocess_features(app_name, threshold)
    
    # Optionally update database
    # update_database_with_clusters(app_name, representatives, dry_run=False)