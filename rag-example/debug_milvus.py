#!/usr/bin/env python3
"""Debug Milvus collection issues - why stats show 0 but UI shows data."""

from pymilvus import MilvusClient, connections, utility

MILVUS_HOST = "milvus-milvus.milvus.svc.cluster.local"
MILVUS_PORT = 19530
COLLECTION_NAME = "paper_rag_documents"

print("=" * 60)
print("Milvus Collection Debug")
print("=" * 60)

# Connect using MilvusClient
milvus_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
print(f"\n1. Connecting to: {milvus_uri}")
client = MilvusClient(uri=milvus_uri, db_name="default")

# Check if collection exists
print(f"\n2. Checking if collection '{COLLECTION_NAME}' exists...")
has_collection = client.has_collection(COLLECTION_NAME)
print(f"   Collection exists: {has_collection}")

if not has_collection:
    print("\n❌ Collection not found in 'default' database.")
    print("\n   Checking all databases...")
    # Use connections API to check other databases
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=str(MILVUS_PORT),
    )
    dbs = utility.list_database()
    print(f"   Available databases: {dbs}")

    for db in dbs:
        print(f"\n   Checking database: {db}")
        client_db = MilvusClient(uri=milvus_uri, db_name=db)
        if client_db.has_collection(COLLECTION_NAME):
            print(f"   ✅ Found collection in database: {db}")
            client = client_db
            break
else:
    print(f"   ✅ Collection found in 'default' database")

# Try to load the collection
print(f"\n3. Loading collection '{COLLECTION_NAME}'...")
try:
    client.load_collection(COLLECTION_NAME)
    print("   Collection loaded successfully")
except Exception as e:
    print(f"   Error loading collection: {e}")

# Get collection stats
print(f"\n4. Getting collection stats...")
stats = client.get_collection_stats(COLLECTION_NAME)
row_count = stats.get("row_count", 0)
print(f"   Stats: {stats}")
print(f"   Row count from stats: {row_count}")

# Try to query the collection directly
print(f"\n5. Attempting direct query...")
try:
    # Query without filter to get count
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="",  # No filter
        output_fields=["count(*)"],
        limit=1,
    )
    print(f"   Query results: {results}")
except Exception as e:
    print(f"   Error querying: {e}")

# Try to get a few sample records
print(f"\n6. Attempting to fetch sample records...")
try:
    samples = client.query(
        collection_name=COLLECTION_NAME,
        filter="chunk_index >= 0",
        output_fields=["source_file", "chunk_index", "text"],
        limit=5,
    )
    print(f"   Found {len(samples)} sample records")
    if samples:
        print(f"   First record: {samples[0].get('source_file', 'N/A')}")
except Exception as e:
    print(f"   Error fetching samples: {e}")

# Check collection info using connections API
print(f"\n7. Getting detailed collection info...")
try:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=str(MILVUS_PORT),
    )
    from pymilvus import Collection

    collection = Collection(COLLECTION_NAME)
    print(f"   Collection schema: {collection.schema}")
    print(f"   Collection num entities: {collection.num_entities}")

    # Flush to ensure data is persisted
    print(f"\n8. Flushing collection...")
    collection.flush()
    print("   Collection flushed")

    # Check again after flush
    print(f"\n9. Checking count after flush...")
    print(f"   Collection num entities: {collection.num_entities}")

except Exception as e:
    print(f"   Error getting collection info: {e}")

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)
