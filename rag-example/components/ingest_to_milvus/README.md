# ingest_to_milvus Component

KFP component that reads chunked JSONL files from S3, generates embeddings, and inserts them into a Milvus vector database.

## Overview

Reads JSONL chunk files from an S3-compatible bucket (produced by the `parse_and_chunk` component), generates embedding vectors using either a deployed embedding service endpoint or a local sentence-transformers model, creates a Milvus collection with an IVF_FLAT index, and batch-inserts all vectors.

No PVC is required — all input comes from S3 and output goes to Milvus.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `s3_endpoint` | str | — | S3-compatible endpoint URL (e.g. MinIO) |
| `s3_bucket` | str | — | S3 bucket containing chunk files |
| `s3_access_key` | str | — | S3 access key |
| `s3_secret_key` | str | — | S3 secret key |
| `milvus_host` | str | — | Milvus service hostname |
| `s3_prefix` | str | `chunks` | Key prefix for chunk files in S3 |
| `embedding_endpoint` | str | `""` | Embedding service URL. If empty, uses local model |
| `embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | Model name (for API or local) |
| `embedding_dim` | int | `384` | Embedding vector dimension |
| `milvus_port` | int | `19530` | Milvus gRPC port |
| `milvus_db` | str | `default` | Milvus database name |
| `collection_name` | str | `rag_documents` | Milvus collection name |
| `embed_batch_size` | int | `64` | Batch size for embedding requests |
| `milvus_batch_size` | int | `256` | Batch size for Milvus inserts |

## Returns

A string in the format `collection_name:total_inserted` (e.g., `rag_documents:15432`).

## Embedding Modes

**Endpoint mode** (`embedding_endpoint` provided): Calls an OpenAI-compatible `/v1/embeddings` API. Works with TEI (Text Embeddings Inference), OpenAI, or any compatible embedding service.

**Local mode** (`embedding_endpoint` empty): Loads the sentence-transformers model directly in the component pod. Simpler but requires more memory.

## Milvus Collection Schema

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Primary key, auto-generated |
| `source_file` | VARCHAR(512) | Source PDF filename |
| `chunk_index` | INT64 | Chunk position in document |
| `text` | VARCHAR(8192) | Chunk text |
| `embedding` | FLOAT_VECTOR | Normalized embedding |

Index: `IVF_FLAT`, metric: `COSINE`, `nlist=128`.

## Dependencies

```
pymilvus>=2.4.0
sentence-transformers>=2.2.0
requests>=2.28.0
boto3>=1.28.0
```

## Usage

```python
from components.ingest_to_milvus import ingest_to_milvus

@dsl.pipeline(name="My Pipeline")
def my_pipeline():
    ingest = ingest_to_milvus(
        s3_endpoint="http://minio-service.default.svc.cluster.local:9000",
        s3_bucket="rag-chunks",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin",
        milvus_host="milvus-milvus.milvus.svc.cluster.local",
        embedding_endpoint="http://all-minilm-l6-v2:8080",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
```
