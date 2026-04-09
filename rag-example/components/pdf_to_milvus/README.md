# PDF to Milvus Component (All-in-One)

KFP component that parses PDFs, chunks them, generates embeddings, and inserts directly into Milvus -- all in a single RayJob. Unlike the multi-step pipeline (which uses separate `parse_and_chunk` and `ingest_to_milvus` components with S3 as intermediate storage), this component does everything end-to-end without intermediate files.

> **Note**: This is an alternative to the multi-step approach. The multi-step pipeline (`parse_and_chunk` + `ingest_to_milvus`) is preferred for production because it decouples parsing from ingestion and supports both local and service embedding modes. This component only supports local embeddings.

## How It Works

1. **Submit RayJob** -- Uses CodeFlare SDK (`ManagedClusterConfig` + `RayJob`) to create a Ray cluster with the specified workers. The entrypoint script is written to a temp directory and uploaded via Ray's `working_dir` runtime env.
2. **Patch RayJob** -- Sets `num-cpus=0` on the head node so no processing actors are scheduled there, reserving it for coordination.
3. **Setup Milvus Collection** -- Creates (or recreates) a Milvus collection with fields for source file, chunk index, text, and embedding vector. Creates an IVF_FLAT index with COSINE metric.
4. **Distributed Processing** -- The Ray entrypoint:
   - Scans the PVC for PDF files.
   - Creates a Ray Dataset and repartitions across actors.
   - Each actor spawns a subprocess worker that initializes Docling, HybridChunker, SentenceTransformer, and a Milvus client.
   - For each PDF: convert with Docling, chunk with HybridChunker, embed with sentence-transformers, batch-insert into Milvus.
5. **Completion** -- Loads the Milvus collection for searching and prints processing report with throughput stats.

### Comparison with Multi-Step Pipeline

| Aspect | pdf_to_milvus (this) | parse_and_chunk + ingest_to_milvus |
|--------|---------------------|-----------------------------------|
| Intermediate storage | None | S3 (JSONL files) |
| Embedding mode | Local only | Local or deployed service |
| Failure recovery | Must reprocess all PDFs | Can re-run ingestion from S3 |
| PVC access | Read-only | Read-only |
| Resource usage | Higher per-worker (embedding model loaded on each) | Embedding runs in separate step |

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pvc_name` | `str` | *required* | Name of the PVC with input PDFs |
| `pvc_mount_path` | `str` | *required* | Mount path for the PVC inside pods |
| `input_path` | `str` | *required* | Relative path to input PDFs under the PVC |
| `ray_image` | `str` | *required* | Container image with Ray + Docling pre-installed |
| `namespace` | `str` | *required* | OpenShift namespace for the RayJob |
| `milvus_host` | `str` | *required* | Milvus service hostname |
| `milvus_port` | `int` | `19530` | Milvus gRPC port |
| `milvus_db` | `str` | `"default"` | Milvus database name |
| `collection_name` | `str` | `"rag_documents"` | Milvus collection name |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Sentence-transformers model for embeddings |
| `embedding_dim` | `int` | `384` | Dimension of the embedding vectors |
| `chunk_max_tokens` | `int` | `256` | Maximum tokens per chunk |
| `num_workers` | `int` | `2` | Number of Ray worker pods |
| `worker_cpus` | `int` | `8` | CPUs per Ray worker pod |
| `worker_memory_gb` | `int` | `16` | Memory (GB) per Ray worker pod |
| `head_cpus` | `int` | `2` | CPUs for the Ray head pod |
| `head_memory_gb` | `int` | `4` | Memory (GB) for the Ray head pod |
| `cpus_per_actor` | `int` | `4` | CPUs per processing actor |
| `min_actors` | `int` | `2` | Minimum actor pool size |
| `max_actors` | `int` | `4` | Maximum actor pool size |
| `batch_size` | `int` | `4` | Files per batch sent to each actor |
| `num_files` | `int` | `0` | Number of PDFs to process (0 = all) |
| `timeout_seconds` | `int` | `600` | Per-file processing timeout |

## Output

| Type | Description |
|------|-------------|
| `str` | Milvus collection name (e.g. `rag_documents`) |

### Milvus Collection Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | INT64 | Auto-generated primary key |
| `source_file` | VARCHAR(512) | Original PDF filename |
| `chunk_index` | INT64 | Chunk position within the source file |
| `text` | VARCHAR(8192) | Chunk text content |
| `embedding` | FLOAT_VECTOR | Vector embedding (dimension = `embedding_dim`) |

**Index**: IVF_FLAT with COSINE metric, nlist=128.

## Dependencies

- **Base image**: `registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9`
- **Python packages** (KFP pod): `codeflare-sdk==0.35.0`, `kubernetes>=28.1.0`
- **Ray runtime packages**: `opencv-python-headless`, `pypdfium2`, `pymilvus>=2.4.0`, `sentence-transformers>=2.2.0`
- **Pre-installed in `ray_image`**: `docling`, `ray`, `pandas`
- **External services**: KubeRay Operator (>= 1.1.0), Milvus (>= 2.4.0)

> **Known limitation**: This component still uses `subprocess.run(["oc", ...])` for RayJob patching and status polling. The multi-step `parse_and_chunk` component has been updated to use the Kubernetes Python client instead.

## Usage

```python
from components.pdf_to_milvus import pdf_to_milvus

@dsl.pipeline(name="My RAG Pipeline")
def my_pipeline():
    ingest = pdf_to_milvus(
        pvc_name="data-pvc",
        pvc_mount_path="/mnt/data",
        input_path="input/pdfs",
        ray_image="quay.io/rhoai-szaher/docling-ray:latest",
        namespace="ray-docling",
        milvus_host="milvus-milvus.milvus.svc.cluster.local",
    )
```
