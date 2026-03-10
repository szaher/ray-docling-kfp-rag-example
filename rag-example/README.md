# RAG Document Processing Pipeline

End-to-end Retrieval-Augmented Generation (RAG) pipeline on OpenShift AI that processes PDF documents, ingests them into Milvus, deploys an LLM, and enables question-answering over the ingested content.

## Architecture

```
                    +-------------------+
                    |   PDF Documents   |
                    |  (PVC, read-only) |
                    +--------+----------+
                             |
                    +--------v----------+
                    |   RayJob (actors)  |
                    |                   |
                    |  Docling parse    |
                    |  HybridChunker    |
                    |  Embed (MiniLM)   |
                    |  Insert -> Milvus |
                    +--------+----------+
                             |
                    +--------v----------+
                    |      Milvus       |
                    |  (vector store)   |
                    +--------+----------+
                             |
         +-------------------+-------------------+
         |                                       |
+--------v----------+                   +--------v----------+
|  Query embedding   |                   |   vLLM Model      |
|  (MiniLM)          |                   |   (InferenceService)|
+--------+-----------+                   +--------+----------+
         |                                        |
         +-------------------+--------------------+
                             |
                    +--------v----------+
                    |    RAG Answer      |
                    +-------------------+
```

## How It Works

1. **PDF Ingestion (pdf_to_milvus):** A RayJob distributes PDF processing across worker pods. Each actor runs Docling in a subprocess to parse PDFs, uses `HybridChunker` for structure-aware chunking (respects headings, paragraphs, tables), generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and inserts directly into Milvus. No intermediate files are written — the PVC is mounted read-only.

2. **Model Deployment (model_deployment):** Creates a KServe `InferenceService` with the vLLM serving runtime on OpenShift AI. The model (default: `mistralai/Mistral-7B-Instruct-v0.3`) is pulled from HuggingFace or loaded from a PVC.

3. **RAG Query:** Embeds the user question, searches Milvus for the top-k most similar chunks (cosine similarity), builds a prompt with the retrieved context, and sends it to the vLLM endpoint for answer generation.

## Directory Structure

```
rag-example/
+-- components/
|   +-- pdf_to_milvus/        # KFP component: PDF parse + chunk + embed + Milvus ingest
|   |   +-- component.py
|   |   +-- metadata.yaml
|   |   +-- __init__.py
|   |   +-- README.md
|   +-- model_deployment/      # KFP component: Deploy model via KServe InferenceService
|       +-- component.py
|       +-- metadata.yaml
|       +-- __init__.py
|       +-- README.md
+-- scripts/
|   +-- docling_milvus_process.py   # Ray Data entrypoint (runs inside RayJob)
|   +-- rag_query.py                # Standalone RAG query CLI
+-- pipeline.py                     # KFP pipeline definition (both steps)
+-- rag-pipeline.ipynb              # Interactive notebook (full walkthrough)
+-- README.md                       # This file
```

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| OpenShift cluster | 4.14+ | With OpenShift AI (RHOAI) installed |
| KubeRay Operator | >= 1.1.0 | Manages RayJob / RayCluster CRDs |
| Milvus | >= 2.4.0 | See `../milvus-ocp/` for OCP deployment guide |
| KServe | Included in RHOAI | For InferenceService / vLLM runtime |
| GPU nodes | 1+ NVIDIA GPU | For vLLM model serving |
| PVC (ReadWriteMany) | — | Contains input PDFs at `<mount>/input/pdfs/` |
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

The Ray worker image must have Docling, sentence-transformers, and pymilvus pre-installed:

```
quay.io/rhoai-szaher/docling-ray:latest
```

## Quick Start

### Option A: Interactive notebook

Open `rag-pipeline.ipynb` and follow Steps 1-10. The notebook covers everything from prerequisites verification to running a RAG query.

### Option B: KFP pipeline

```bash
# Compile the pipeline
python pipeline.py
# Output: rag_pipeline.yaml

# Upload to Kubeflow Pipelines UI or submit via SDK
```

### Option C: Standalone scripts

```bash
# 1. Run the ingestion RayJob directly (configure via env vars)
export MILVUS_HOST=milvus-milvus.milvus.svc.cluster.local
export PVC_MOUNT_PATH=/mnt/data
export INPUT_PATH=input/pdfs
python scripts/docling_milvus_process.py

# 2. Query the RAG system
export INFERENCE_URL=https://your-model-endpoint/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
python scripts/rag_query.py "What is the main topic of the documents?"
```

## Configuration

### PDF Processing Parameters

| Parameter | Default | Description |
|---|---|---|
| `pvc_name` | `data-pvc` | Name of the PVC containing input PDFs |
| `pvc_mount_path` | `/mnt/data` | Mount path inside Ray pods |
| `input_path` | `input/pdfs` | Relative path to PDFs under mount path |
| `num_files` | `0` | Number of PDFs to process (0 = all) |
| `chunk_max_tokens` | `256` | Max tokens per chunk (HybridChunker) |
| `num_workers` | `2` | Number of Ray worker pods |
| `worker_cpus` | `8` | CPUs per worker pod |
| `worker_memory_gb` | `16` | Memory (GB) per worker pod |
| `cpus_per_actor` | `4` | CPUs allocated per Docling actor |
| `min_actors` | `2` | Minimum actor pool size |
| `max_actors` | `4` | Maximum actor pool size |
| `batch_size` | `4` | Files per batch sent to each actor |
| `timeout_seconds` | `600` | Per-file processing timeout |

### Milvus Parameters

| Parameter | Default | Description |
|---|---|---|
| `milvus_host` | `milvus-milvus.milvus.svc.cluster.local` | Milvus gRPC service hostname |
| `milvus_port` | `19530` | Milvus gRPC port |
| `milvus_db` | `default` | Milvus database name |
| `collection_name` | `rag_documents` | Milvus collection name |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `embedding_dim` | `384` | Embedding vector dimension |

### Model Deployment Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace model ID |
| `gpu_count` | `1` | GPUs per replica |
| `runtime` | `vllm` | Serving runtime |

## Milvus Collection Schema

The pipeline creates a collection with the following schema:

| Field | Type | Description |
|---|---|---|
| `id` | INT64 (auto) | Primary key |
| `source_file` | VARCHAR(512) | Source PDF filename |
| `chunk_index` | INT64 | Chunk position within the document |
| `text` | VARCHAR(8192) | Chunk text content |
| `embedding` | FLOAT_VECTOR(384) | Normalized embedding vector |

Index: `IVF_FLAT` with `COSINE` metric and `nlist=128`.

## Design Decisions

- **No intermediate files:** The entire parse-chunk-embed-insert pipeline runs in memory. The PVC is mounted read-only, eliminating I/O bottleneck and storage overhead.
- **Subprocess isolation:** Each Ray actor runs Docling in a separate subprocess. If a PDF causes a crash or hang, only the subprocess is affected — the actor restarts it and continues processing.
- **HybridChunker over character splitting:** Docling's `HybridChunker` is structure-aware. It respects document structure (headings, paragraphs, tables, lists) rather than splitting on arbitrary character boundaries, producing higher-quality chunks for retrieval.
- **Actor pool strategy:** Ray Data's `ActorPoolStrategy` with min/max sizing enables elastic scaling. Each actor maintains its own Docling converter, embedding model, and Milvus client, avoiding model reload overhead between files.
- **Head node exclusion:** The RayJob patches `num-cpus=0` on the head node so that no actors are scheduled there, reserving it for coordination.

## Troubleshooting

### RayJob stays in PENDING

Check that the KubeRay operator is running and that the namespace has enough CPU/memory quota for the requested workers.

### Milvus connection refused

Verify Milvus is running and reachable from the Ray pods:
```bash
oc exec -it <ray-worker-pod> -n <namespace> -- python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://milvus-milvus.milvus.svc.cluster.local:19530')
print(c.list_collections())
"
```

### Model deployment timeout

The InferenceService waits up to 30 minutes. Check:
```bash
oc get inferenceservice -n <namespace>
oc describe inferenceservice <name> -n <namespace>
```

Ensure the ServingRuntime CR (`vllm-runtime`) exists and GPU nodes are available.

### Actors crashing with OOM

Reduce `worker_memory_gb` or increase it, depending on PDF size. Large PDFs with many images can consume significant memory during Docling parsing. Consider also reducing `batch_size` to process fewer files per actor call.
