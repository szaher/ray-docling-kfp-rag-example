# RAG Document Processing Pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline on OpenShift AI that processes PDF documents, ingests them into Milvus, deploys an LLM, and enables question-answering over the ingested content.

## Pipelines

This example provides **two pipeline variants**:

### 1. Single-Step Pipeline (`pipeline.py` / `rag_pipeline.yaml`)

All-in-one: parse, chunk, embed, and ingest into Milvus in a single RayJob. PVC is read-only. Best for simple deployments.

```
PDF (PVC) --> [RayJob: Docling + chunk + embed + Milvus insert] --> Deploy LLM
```

### 2. Multi-Step Pipeline (`pipeline_multistep.py` / `rag_multistep_pipeline.yaml`)

Four reusable components with clear separation of concerns. Best for production use and component reuse across pipelines.

```
Step 1: Deploy Embedding Model (TEI InferenceService)
                |                               |
Step 2: Parse & Chunk PDFs -----(parallel)------+
    (RayJob: Docling + HybridChunker -> JSONL on S3/MinIO)
                |                               |
Step 3: Ingest into Milvus  <----(both done)----+
    (read chunks from S3, call embedding API, insert vectors)
                |
Step 4: Deploy LLM (vLLM InferenceService)
```

Steps 1 and 2 run in parallel. Step 3 waits for both. Step 4 runs after ingestion.

## Architecture

```
                    +-------------------+
                    |   PDF Documents   |
                    |      (PVC)        |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    |  Deploy Embedding |         |  Parse & Chunk    |
    |  Model (TEI)      |         |  (RayJob/Docling) |
    |                   |         |  -> JSONL on S3   |
    +---------+---------+         +---------+---------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v----------+
                    |  Ingest to Milvus |
                    |  (embed + insert) |
                    +--------+----------+
                             |
                    +--------v----------+
                    |      Milvus       |
                    |  (vector store)   |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+         +---------v---------+
    |  Query embedding  |         |   vLLM Model      |
    |  (MiniLM / TEI)   |         |  (InferenceService)|
    +---------+---------+         +---------+---------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v----------+
                    |    RAG Answer      |
                    +-------------------+
```

## Directory Structure

```
rag-example/
+-- components/
|   +-- deploy_embedding_model/  # Deploy embedding model (TEI InferenceService)
|   +-- parse_and_chunk/         # Parse PDFs + chunk (RayJob -> JSONL on S3)
|   +-- ingest_to_milvus/       # Embed chunks + insert into Milvus
|   +-- pdf_to_milvus/          # Single-step: parse + chunk + embed + insert
|   +-- model_deployment/       # Deploy LLM (vLLM InferenceService)
+-- scripts/
|   +-- docling_chunk_process.py    # Ray entrypoint: parse + chunk -> S3
|   +-- milvus_ingest.py           # Standalone: read S3, embed, insert into Milvus
|   +-- docling_milvus_process.py  # Ray entrypoint: single-step (parse -> Milvus)
|   +-- rag_query.py               # RAG query CLI
+-- pipeline.py                    # Single-step pipeline definition
+-- pipeline_multistep.py          # Multi-step pipeline definition
+-- rag_pipeline.yaml              # Compiled single-step pipeline (import via UI)
+-- rag_multistep_pipeline.yaml    # Compiled multi-step pipeline (import via UI)
+-- rag-pipeline.ipynb             # Interactive notebook
+-- README.md
```

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OpenShift cluster | 4.14+ | With OpenShift AI (RHOAI) installed |
| KubeRay Operator | >= 1.1.0 | Manages RayJob / RayCluster CRDs |
| Milvus | >= 2.4.0 | See `../milvus-ocp/` for OCP deployment guide |
| MinIO / S3 | — | For intermediate chunk storage (multi-step pipeline) |
| KServe | Included in RHOAI | For InferenceService / vLLM + TEI runtimes |
| GPU nodes | 1+ NVIDIA GPU | For vLLM model serving |
| PVC (ReadOnlyMany or RWX) | — | Contains input PDFs at `<mount>/input/pdfs/` |
| Python | 3.11+ | For running the notebook locally |

### Python packages (notebook host)

```
codeflare-sdk>=0.25.0
kubernetes>=28.1.0
ray[default]>=2.44.1
pymilvus>=2.4.0
sentence-transformers>=2.2.0
openai>=1.0.0
```

### Container image

The Ray worker image must have Docling pre-installed:

```
quay.io/rhoai-szaher/docling-ray:latest
```

## Quick Start

### Import via KFP UI

Upload `rag_multistep_pipeline.yaml` (or `rag_pipeline.yaml` for single-step) directly to the Data Science Pipelines UI:

1. Go to **Pipelines** > **Import pipeline**
2. Upload the YAML file
3. Configure parameters and create a run

### Compile from source

```bash
# Multi-step pipeline
python pipeline_multistep.py
# Output: rag_multistep_pipeline.yaml

# Single-step pipeline
python pipeline.py
# Output: rag_pipeline.yaml
```

### Standalone scripts

```bash
# Parse & chunk (write JSONL to S3/MinIO)
export PVC_MOUNT_PATH=/mnt/data INPUT_PATH=input/pdfs
export S3_ENDPOINT=http://minio-service.default.svc.cluster.local:9000
export S3_BUCKET=rag-chunks S3_ACCESS_KEY=minioadmin S3_SECRET_KEY=minioadmin
python scripts/docling_chunk_process.py

# Ingest into Milvus (read chunks from S3)
export MILVUS_HOST=milvus-milvus.milvus.svc.cluster.local
export S3_ENDPOINT=http://minio-service.default.svc.cluster.local:9000
export S3_BUCKET=rag-chunks S3_ACCESS_KEY=minioadmin S3_SECRET_KEY=minioadmin
python scripts/milvus_ingest.py

# RAG query
export INFERENCE_URL=https://your-model-endpoint/v1 MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
python scripts/rag_query.py "What is the main topic of the documents?"
```

## Configuration

### Multi-Step Pipeline Defaults

| Parameter | Default | Description |
|---|---|---|
| `num_files` | `1000` | Number of PDFs to process (0 = all) |
| `pvc_name` | `data-pvc` | PVC name |
| `input_path` | `input/pdfs` | Input PDF path on PVC |
| `s3_endpoint` | `http://minio-service.default.svc.cluster.local:9000` | S3/MinIO endpoint |
| `s3_bucket` | `rag-chunks` | S3 bucket for intermediate chunks |
| `s3_prefix` | `chunks` | S3 key prefix for chunk files |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `embedding_dim` | `384` | Embedding vector dimension |
| `chunk_max_tokens` | `256` | Max tokens per chunk |
| `num_workers` | `2` | Ray worker pods |
| `worker_cpus` | `8` | CPUs per worker |
| `worker_memory_gb` | `16` | Memory per worker |
| `cpus_per_actor` | `4` | CPUs per Docling actor |
| `min_actors` | `2` | Min actor pool size |
| `max_actors` | `4` | Max actor pool size |
| `milvus_host` | `milvus-milvus.milvus.svc.cluster.local` | Milvus hostname |
| `collection_name` | `rag_documents` | Milvus collection |
| `llm_model_name` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM model |
| `gpu_count` | `1` | GPUs for LLM |

## Milvus Collection Schema

| Field | Type | Description |
|---|---|---|
| `id` | INT64 (auto) | Primary key |
| `source_file` | VARCHAR(512) | Source PDF filename |
| `chunk_index` | INT64 | Chunk position within the document |
| `text` | VARCHAR(8192) | Chunk text content |
| `embedding` | FLOAT_VECTOR(384) | Normalized embedding vector |

Index: `IVF_FLAT` with `COSINE` metric and `nlist=128`.

## Design Decisions

- **Multi-step separation:** Each component has a single responsibility. The embedding model, PDF parsing, Milvus ingestion, and LLM deployment are independent, reusable components that can be mixed into different pipelines.
- **Parallel execution:** Steps 1 (deploy embedding model) and 2 (parse & chunk) run in parallel since they have no dependencies on each other. Step 3 (ingest) waits for both.
- **S3 intermediate storage:** Chunks are stored as JSONL in S3 (MinIO) between the parse and ingest steps. This avoids needing a ReadWriteMany PVC (which requires NFS/CephFS), provides a checkpoint if ingestion fails, and keeps the input PVC read-only.
- **Flexible embedding:** The ingest component accepts an optional `embedding_endpoint`. When provided, it calls the deployed TEI service. When empty, it falls back to a local sentence-transformers model.
- **Subprocess isolation:** Each Ray actor runs Docling in a separate subprocess for crash isolation and timeout handling.
- **HybridChunker:** Structure-aware chunking that respects document structure (headings, paragraphs, tables) rather than splitting on arbitrary character boundaries.
- **Head node exclusion:** RayJob patches `num-cpus=0` on the head node, reserving it for coordination.
- **1000 PDF default:** The multi-step pipeline defaults to processing 1000 PDFs, a practical demo size.

## Troubleshooting

### RayJob stays in PENDING

Check that the KubeRay operator is running and that the namespace has enough CPU/memory quota for the requested workers.

### Milvus connection refused

Verify Milvus is running and reachable:
```bash
oc exec -it <pod> -n <namespace> -- python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://milvus-milvus.milvus.svc.cluster.local:19530')
print(c.list_collections())
"
```

### Embedding service not ready

Check the InferenceService status:
```bash
oc get inferenceservice -n <namespace>
oc describe inferenceservice <name> -n <namespace>
```

Ensure the embedding ServingRuntime CR exists in the namespace.

### Model deployment timeout

The LLM InferenceService waits up to 30 minutes. Ensure the ServingRuntime CR (`vllm-runtime`) exists and GPU nodes are available.

### Actors crashing with OOM

Increase `worker_memory_gb` or reduce `batch_size`. Large PDFs with many images consume significant memory during Docling parsing.
