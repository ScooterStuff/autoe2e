"""Debug script to explore the database structure"""
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import math

load_dotenv()

client = MongoClient(os.getenv('ATLAS_URI'))
db = client.myDatabase

SCORE_THRESHOLD = math.log(0.5)  # -0.693

print("="*60)
print("DATABASE EXPLORATION FOR PETCLINIC")
print("="*60)

# Count records
func_count = db['functionality'].count_documents({'app': 'PETCLINIC'})
afd_count = db['action-functionality'].count_documents({'app': 'PETCLINIC'})
print(f"\nTotal functionality records: {func_count}")
print(f"Total action-functionality records: {afd_count}")

# Score distribution
print(f"\n=== SCORE THRESHOLD: {SCORE_THRESHOLD:.3f} ===")
above_threshold = db['functionality'].count_documents({
    'app': 'PETCLINIC',
    'score': {'$gte': SCORE_THRESHOLD}
})
below_threshold = db['functionality'].count_documents({
    'app': 'PETCLINIC',
    'score': {'$lt': SCORE_THRESHOLD}
})
print(f"Features above threshold (valid): {above_threshold}")
print(f"Features below threshold (filtered): {below_threshold}")

# Sample functionality records
print("\n=== SAMPLE FUNCTIONALITY RECORDS (FD) ===")
funcs = list(db['functionality'].find({'app': 'PETCLINIC'}).sort('score', -1).limit(10))
for f in funcs:
    fid = str(f['_id'])
    text = f.get('text', 'N/A')
    score = f.get('score', 0)
    final = f.get('final', False)
    status = "✓ VALID" if score >= SCORE_THRESHOLD else "✗ FILTERED"
    print(f"{status} | score={score:+.3f} | final={final} | {text}")

# Sample action-functionality records  
print("\n=== SAMPLE ACTION-FUNCTIONALITY RECORDS (AFD) ===")
afd = list(db['action-functionality'].find({'app': 'PETCLINIC'}).limit(10))
for a in afd:
    test_id = a.get('test_id') or 'N/A'
    func_ptr = a.get('func_pointer') or 'N/A'
    final = a.get('final', False)
    depth = a.get('depth', 0)
    atype = a.get('type') or 'N/A'
    prev_state = a.get('prev_state')
    prev_action = a.get('prev_action')
    has_prev = "has_prev" if prev_state and prev_action else "NO_prev"
    print(f"test_id={str(test_id):30s} | depth={depth} | type={str(atype):8s} | final={final} | {has_prev} | func_ptr={str(func_ptr)[:12]}...")

# Check for AFD records with final=True
print("\n=== AFD RECORDS WITH FINAL=TRUE ===")
final_afd = list(db['action-functionality'].find({
    'app': 'PETCLINIC',
    'final': True
}).limit(10))
print(f"Found {db['action-functionality'].count_documents({'app': 'PETCLINIC', 'final': True})} final AFD records")
for a in final_afd[:5]:
    test_id = a.get('test_id', 'N/A')
    func_ptr = a.get('func_pointer', 'N/A')
    print(f"  test_id={test_id} | func_ptr={func_ptr[:12]}...")

# Check what func_pointers are actually linked to valid features
print("\n=== CHECKING FUNC_POINTER LINKAGE ===")
valid_func_ids = set()
for f in db['functionality'].find({'app': 'PETCLINIC', 'score': {'$gte': SCORE_THRESHOLD}}):
    valid_func_ids.add(str(f['_id']))

print(f"Valid feature IDs: {len(valid_func_ids)}")

# Count AFD records pointing to valid features
afd_with_valid_ptr = 0
afd_with_invalid_ptr = 0
for a in db['action-functionality'].find({'app': 'PETCLINIC'}):
    if a.get('func_pointer') in valid_func_ids:
        afd_with_valid_ptr += 1
    else:
        afd_with_invalid_ptr += 1

print(f"AFD records pointing to valid features: {afd_with_valid_ptr}")
print(f"AFD records pointing to invalid/filtered features: {afd_with_invalid_ptr}")

client.close()

# Additional analysis - show depth distribution and chain structure
client = MongoClient(os.getenv('ATLAS_URI'))
db = client.myDatabase

print("\n=== DEPTH DISTRIBUTION ===")
depths = db['action-functionality'].aggregate([
    {'$match': {'app': 'PETCLINIC'}},
    {'$group': {'_id': '$depth', 'count': {'$sum': 1}}},
    {'$sort': {'_id': 1}}
])
for d in depths:
    print(f"  depth={d['_id']}: {d['count']} records")

print("\n=== RECORDS WITH PREV_STATE (potential chains) ===")
with_prev = list(db['action-functionality'].find({
    'app': 'PETCLINIC',
    'prev_state': {'$ne': None}
}).limit(5))
print(f"Found {db['action-functionality'].count_documents({'app': 'PETCLINIC', 'prev_state': {'$ne': None}})} records with prev_state")
for a in with_prev:
    test_id = a.get('test_id') or 'N/A'
    depth = a.get('depth', 0)
    print(f"  test_id={test_id}, depth={depth}")

print("\n=== UNIQUE TEST_IDS ===")
test_ids = db['action-functionality'].distinct('test_id', {'app': 'PETCLINIC'})
print(f"Total unique test_ids: {len(test_ids)}")
print("Sample test_ids:", [t for t in test_ids if t][:20])

print("\n=== RECORDS BY TYPE ===")
types = db['action-functionality'].aggregate([
    {'$match': {'app': 'PETCLINIC'}},
    {'$group': {'_id': '$type', 'count': {'$sum': 1}}}
])
for t in types:
    print(f"  type={t['_id']}: {t['count']} records")

client.close()
