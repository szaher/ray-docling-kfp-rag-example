"""Read chunked JSONL files from S3, embed, and insert into Milvus.

Reads JSONL chunk files produced by docling_chunk_process.py from an
S3-compatible bucket, generates embeddings (via a TEI/OpenAI-compatible
endpoint or a local sentence-transformers model), and inserts into Milvus.

All parameters are read from environment variables.
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# Parameters (environment variables)
# ---------------------------------------------------------------------------

# S3 input
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000")
S3_BUCKET = os.environ.get("S3_BUCKET", "rag-chunks")
S3_PREFIX = os.environ.get("S3_PREFIX", "chunks")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

# Milvus
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus-milvus.milvus.svc.cluster.local")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_DB = os.environ.get("MILVUS_DB", "default")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "rag_documents")

# Embedding — either via endpoint or local model
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

# Batch sizes
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
MILVUS_BATCH_SIZE = int(os.environ.get("MILVUS_BATCH_SIZE", "256"))


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
    )


def read_chunks_from_s3(s3_client, bucket: str, prefix: str) -> list[dict]:
    """Read all JSONL files from S3 and return a list of chunk dicts."""
    all_chunks = []
    paginator = s3_client.get_paginator("list_objects_v2")

    file_count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".jsonl"):
                continue

            resp = s3_client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read().decode("utf-8")
            for line in body.strip().split("\n"):
                if line:
                    all_chunks.append(json.loads(line))
            file_count += 1

    print(f"Read {len(all_chunks)} chunks from {file_count} JSONL files in s3://{bucket}/{prefix}/")
    return all_chunks


# ---------------------------------------------------------------------------
# Milvus collection setup
# ---------------------------------------------------------------------------


def setup_milvus_collection():
    """Create or recreate the Milvus collection with an IVF_FLAT index."""
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    client = MilvusClient(uri=uri, db_name=MILVUS_DB)

    if client.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection '{COLLECTION_NAME}'")
        client.drop_collection(COLLECTION_NAME)

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(
                name="source_file", dtype=DataType.VARCHAR, max_length=512
            ),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=EMBEDDING_DIM,
            ),
        ],
        description="RAG document chunks with embeddings",
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(
        collection_name=COLLECTION_NAME, index_params=index_params
    )

    print(f"Collection '{COLLECTION_NAME}' created (dim={EMBEDDING_DIM}).")
    return client


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embed_via_endpoint(texts: list[str]) -> list[list[float]]:
    """Call a TEI/OpenAI-compatible embedding endpoint."""
    import requests

    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = requests.post(
            f"{EMBEDDING_ENDPOINT}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_embeddings.extend([d["embedding"] for d in data])
    return all_embeddings


def embed_via_local(texts: list[str], model) -> list[list[float]]:
    """Embed using a local sentence-transformers model."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=EMBED_BATCH_SIZE,
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run():
    s3 = _get_s3_client()
    all_chunks = read_chunks_from_s3(s3, S3_BUCKET, S3_PREFIX)

    if not all_chunks:
        print("No chunks found. Exiting.")
        return

    # Setup Milvus collection
    from pymilvus import MilvusClient

    setup_milvus_collection()

    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    milvus_client = MilvusClient(uri=uri, db_name=MILVUS_DB)

    # Setup embedding
    use_endpoint = bool(EMBEDDING_ENDPOINT)
    local_model = None
    if use_endpoint:
        print(f"Using embedding endpoint: {EMBEDDING_ENDPOINT}")
    else:
        print(f"Using local embedding model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer

        local_model = SentenceTransformer(EMBEDDING_MODEL)

    # Process in batches: embed -> insert into Milvus
    start_time = time.time()
    total_inserted = 0

    for i in range(0, len(all_chunks), MILVUS_BATCH_SIZE):
        batch = all_chunks[i : i + MILVUS_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        if use_endpoint:
            embeddings = embed_via_endpoint(texts)
        else:
            embeddings = embed_via_local(texts, local_model)

        data = [
            {
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "embedding": emb,
            }
            for c, emb in zip(batch, embeddings)
        ]

        milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
        total_inserted += len(data)

        if (i // MILVUS_BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Inserted {total_inserted}/{len(all_chunks)} chunks "
                f"({elapsed:.1f}s elapsed)"
            )

    # Load collection for searching
    milvus_client.load_collection(COLLECTION_NAME)
    stats = milvus_client.get_collection_stats(COLLECTION_NAME)

    wall_clock = time.time() - start_time

    print("\n" + "=" * 60)
    print("MILVUS INGESTION REPORT")
    print("=" * 60)
    print(f"Source:            s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Embedding:         {'endpoint' if use_endpoint else 'local'} ({EMBEDDING_MODEL})")
    print(f"Collection:        {COLLECTION_NAME}")
    print(f"Total chunks:      {len(all_chunks)}")
    print(f"Total inserted:    {total_inserted}")
    print(f"Wall clock:        {wall_clock:.1f}s")
    if wall_clock > 0:
        print(f"Chunks/second:     {total_inserted / wall_clock:.2f}")
    print(f"Collection stats:  {stats}")
    print("=" * 60)


if __name__ == "__main__":
    run()
