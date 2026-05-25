"""Check test_id values in the MongoDB database for DIMESHIFT."""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv('MONGO_DB_URL'))
db = client['autoe2e']

# Get actions for DIMESHIFT
actions = list(db['actions'].find({'config': 'DIMESHIFT'}).limit(20))

print(f"Found {len(actions)} actions for DIMESHIFT:")
print("=" * 80)

for a in actions:
    test_id = a.get('test_id', None)
    tag = a.get('tag', 'unknown')
    href = a.get('attrs', {}).get('href', '')
    text = a.get('text', '')[:30] if a.get('text') else ''
    
    print(f"  test_id: {test_id or 'NULL':<25} | tag: {tag:<8} | href: {href:<20} | text: {text}")

print("\n" + "=" * 80)
print("Statistics:")
with_testid = sum(1 for a in actions if a.get('test_id'))
without_testid = sum(1 for a in actions if not a.get('test_id'))
print(f"  Actions with test_id: {with_testid}")
print(f"  Actions without test_id (NULL): {without_testid}")
