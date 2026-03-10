# pdf_to_milvus Component

KFP component that processes PDFs with Docling, chunks with `HybridChunker`, generates embeddings, and inserts directly into Milvus. No intermediate files are written.

## Overview

This component submits a RayJob to an OpenShift cluster using the CodeFlare SDK. The RayJob runs `docling_milvus_process.py`, which uses Ray Data's actor pool to distribute PDF processing across worker pods.

### Processing Pipeline (per file)

```
PDF file (from PVC)
  |-> Docling DocumentConverter (parse)
  |-> HybridChunker (structure-aware chunking)
  |-> SentenceTransformer.encode() (embed)
  |-> MilvusClient.insert() (ingest)
  |-> Status report (success/error/timeout)
```

Each Ray actor runs a long-lived subprocess that holds:
- Docling `DocumentConverter` with PDF pipeline options
- `HybridChunker` initialized with the embedding model's tokenizer
- `SentenceTransformer` model for embeddings
- `MilvusClient` connection for inserts

This avoids model reload overhead between files. If a subprocess crashes or hangs (timeout), the actor restarts it automatically.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvc_name` | str | — | Name of the ReadWriteMany PVC with input PDFs |
| `pvc_mount_path` | str | — | Mount path for the PVC inside pods |
| `input_path` | str | — | Relative path to input PDFs under the PVC |
| `ray_image` | str | — | Container image with Ray + Docling pre-installed |
| `namespace` | str | — | OpenShift namespace for the RayJob |
| `milvus_host` | str | — | Milvus service hostname |
| `milvus_port` | int | `19530` | Milvus gRPC port |
| `milvus_db` | str | `default` | Milvus database name |
| `collection_name` | str | `rag_documents` | Milvus collection name |
| `embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `embedding_dim` | int | `384` | Embedding vector dimension |
| `chunk_max_tokens` | int | `256` | Max tokens per chunk |
| `num_workers` | int | `2` | Number of Ray worker pods |
| `worker_cpus` | int | `8` | CPUs per worker pod |
| `worker_memory_gb` | int | `16` | Memory (GB) per worker pod |
| `head_cpus` | int | `2` | CPUs for the Ray head pod |
| `head_memory_gb` | int | `4` | Memory (GB) for the Ray head pod |
| `cpus_per_actor` | int | `4` | CPUs per Docling processing actor |
| `min_actors` | int | `2` | Minimum actor pool size |
| `max_actors` | int | `4` | Maximum actor pool size |
| `batch_size` | int | `4` | Files per batch per actor |
| `num_files` | int | `0` | Number of PDFs to process (0 = all) |
| `timeout_seconds` | int | `600` | Per-file processing timeout |

## Returns

A string with the Milvus collection name (e.g., `rag_documents`).

## How It Works

1. **Submit RayJob** via CodeFlare SDK with the configured cluster spec (head + workers), PVC volume mount, and runtime environment variables.

2. **Patch head `num-cpus=0`** so that no processing actors are scheduled on the head node, reserving it for coordination.

3. **Poll for completion** by checking the RayJob's `.status.jobStatus` field every 30 seconds. Raises `RuntimeError` on failure.

### Inside the RayJob (`docling_milvus_process.py`)

1. **Driver** discovers PDFs on the PVC, creates/recreates the Milvus collection with an `IVF_FLAT` index, repartitions the file list into blocks, and launches the Ray Data actor pool.

2. **Actors** each spawn a subprocess that loads Docling, the chunker, the embedding model, and a Milvus client. Files are sent one at a time through a multiprocessing queue.

3. **Subprocess** for each file: parse with Docling, chunk with `HybridChunker`, embed all chunks, batch-insert into Milvus. Returns status (success/error/timeout) with metrics.

4. **Driver** collects results and prints a processing report with throughput stats, error summary, and actor distribution.

## Dependencies

### Component runtime (KFP pod)

```
codeflare-sdk>=0.25.0
kubernetes>=28.1.0
ray[default]>=2.44.1
```

### RayJob worker image

Must include:
- `docling` (PDF parsing, HybridChunker)
- `sentence-transformers` (embeddings)
- `pymilvus` (Milvus client)
- `opencv-python-headless`, `pypdfium2` (Docling PDF backend)
- `ray[default]` (Ray Data)
- `pandas` (Ray Data format)

### External services

- **KubeRay Operator** (>= 1.1.0) — manages RayJob CRDs
- **Milvus** (>= 2.4.0) — vector database

## Usage

### In a KFP pipeline

```python
from components.pdf_to_milvus import pdf_to_milvus

@dsl.pipeline(name="My RAG Pipeline")
def my_pipeline():
    ingest = pdf_to_milvus(
        pvc_name="data-pvc",
        pvc_mount_path="/mnt/data",
        input_path="input/pdfs",
        ray_image="quay.io/rhoai-szaher/docling-ray:latest",
        namespace="rag-example",
        milvus_host="milvus-milvus.milvus.svc.cluster.local",
    )
```

### Standalone compilation

```bash
python component.py
# Output: component_component.yaml
```

## Milvus Collection Schema

Created by the RayJob driver before processing starts:

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Primary key, auto-generated |
| `source_file` | VARCHAR(512) | Source PDF filename |
| `chunk_index` | INT64 | Chunk position in document |
| `text` | VARCHAR(8192) | Chunk text |
| `embedding` | FLOAT_VECTOR(384) | Normalized embedding |

Index: `IVF_FLAT`, metric: `COSINE`, `nlist=128`.

## Scaling Guidelines

- **More files:** Increase `max_actors` and `num_workers`. Each worker can host multiple actors if it has enough CPUs (`worker_cpus / cpus_per_actor`).
- **Larger PDFs:** Increase `timeout_seconds` and `worker_memory_gb`. Docling parsing of large documents with many tables can be memory-intensive.
- **Higher throughput:** Increase `batch_size` to send more files per actor call, reducing scheduling overhead. Monitor memory usage.
- **Chunk quality:** Adjust `chunk_max_tokens`. Smaller chunks give more precise retrieval but increase the number of vectors. Larger chunks provide more context per result.
