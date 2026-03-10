# parse_and_chunk Component

KFP component that parses PDFs with Docling and chunks them using `HybridChunker` via a Ray Data actor pool. Writes JSONL chunk files to S3 (MinIO).

## Overview

Submits a RayJob that distributes PDF processing across worker pods. Each actor runs Docling in a subprocess for crash isolation, parses PDFs, and chunks them using Docling's `HybridChunker` (structure-aware chunking that respects headings, paragraphs, tables). The output is one JSONL file per source PDF, uploaded to an S3-compatible bucket.

The PVC is mounted **read-only** — it's only used for reading input PDFs. All output goes to S3.

### Output Format

Each JSONL file in S3 contains one JSON object per line:

```json
{"source_file": "document.pdf", "chunk_index": 0, "text": "First chunk text..."}
{"source_file": "document.pdf", "chunk_index": 1, "text": "Second chunk text..."}
```

Files are written to `s3://<bucket>/<prefix>/<stem>.jsonl`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pvc_name` | str | — | Name of the PVC with input PDFs |
| `pvc_mount_path` | str | — | Mount path for the PVC |
| `input_path` | str | — | Relative path to input PDFs |
| `ray_image` | str | — | Container image with Ray + Docling |
| `namespace` | str | — | OpenShift namespace for the RayJob |
| `s3_endpoint` | str | — | S3-compatible endpoint URL (e.g. MinIO) |
| `s3_bucket` | str | — | S3 bucket for output chunks |
| `s3_access_key` | str | — | S3 access key |
| `s3_secret_key` | str | — | S3 secret key |
| `s3_prefix` | str | `chunks` | Key prefix for chunk files in S3 |
| `tokenizer` | str | `sentence-transformers/all-MiniLM-L6-v2` | Tokenizer for HybridChunker |
| `chunk_max_tokens` | int | `256` | Max tokens per chunk |
| `num_workers` | int | `2` | Ray worker pods |
| `worker_cpus` | int | `8` | CPUs per worker |
| `worker_memory_gb` | int | `16` | Memory (GB) per worker |
| `head_cpus` | int | `2` | CPUs for Ray head |
| `head_memory_gb` | int | `4` | Memory (GB) for Ray head |
| `cpus_per_actor` | int | `4` | CPUs per Docling actor |
| `min_actors` | int | `2` | Minimum actor pool size |
| `max_actors` | int | `4` | Maximum actor pool size |
| `batch_size` | int | `4` | Files per batch per actor |
| `num_files` | int | `0` | PDFs to process (0 = all) |
| `timeout_seconds` | int | `600` | Per-file timeout |

## Returns

The S3 URI where JSONL chunk files were written (e.g., `s3://rag-chunks/chunks`).

## Dependencies

### Component runtime (KFP pod)

```
codeflare-sdk>=0.25.0
kubernetes>=28.1.0
ray[default]>=2.44.1
```

### RayJob worker image

Must include: `docling`, `opencv-python-headless`, `pypdfium2`, `boto3`, `ray[default]`, `pandas`.

### External services

- KubeRay Operator (>= 1.1.0)
- S3-compatible object store (MinIO)

## Usage

```python
from components.parse_and_chunk import parse_and_chunk

@dsl.pipeline(name="My Pipeline")
def my_pipeline():
    chunks = parse_and_chunk(
        pvc_name="data-pvc",
        pvc_mount_path="/mnt/data",
        input_path="input/pdfs",
        ray_image="quay.io/rhoai-szaher/docling-ray:latest",
        namespace="rag-example",
        s3_endpoint="http://minio-service.default.svc.cluster.local:9000",
        s3_bucket="rag-chunks",
        s3_access_key="minioadmin",
        s3_secret_key="minioadmin",
        num_files=1000,
    )
    # Use chunks.output (S3 URI) in downstream steps
```
