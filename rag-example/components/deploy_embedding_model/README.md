# deploy_embedding_model Component

KFP component that deploys a text embedding model on OpenShift AI using a KServe InferenceService with a TEI-compatible ServingRuntime.

## Overview

Creates or updates an InferenceService that serves embedding requests via an OpenAI-compatible `/v1/embeddings` API. The model is pulled from HuggingFace and served using the configured embedding ServingRuntime (e.g., TEI — Text Embeddings Inference).

Uses in-cluster Kubernetes config (service account token) — no explicit API URL or token required.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | — | HuggingFace model ID (e.g., `sentence-transformers/all-MiniLM-L6-v2`) |
| `namespace` | str | — | Namespace to deploy the InferenceService |
| `serving_runtime_name` | str | `embedding-runtime` | Name of the embedding ServingRuntime CR |
| `min_replicas` | int | `1` | Minimum replicas |
| `max_replicas` | int | `1` | Maximum replicas |
| `cpu_requests` | str | `2` | CPU requests per replica |
| `cpu_limits` | str | `4` | CPU limits per replica |
| `memory_requests` | str | `4Gi` | Memory requests per replica |
| `memory_limits` | str | `8Gi` | Memory limits per replica |

## Returns

The embedding service endpoint URL (e.g., `https://all-minilm-l6-v2.rag-example.svc.cluster.local`).

## Prerequisites

- A ServingRuntime CR named `embedding-runtime` (or custom name) must exist in the target namespace. On OpenShift AI, you can create one for TEI or use a pre-installed runtime.
- The model must be accessible from HuggingFace (public or with `HF_TOKEN` configured).
- The KFP pipeline service account must have permissions to create/patch InferenceService CRs in the target namespace.

## Usage

```python
from components.deploy_embedding_model import deploy_embedding_model

@dsl.pipeline(name="My Pipeline")
def my_pipeline():
    embed = deploy_embedding_model(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        namespace="rag-example",
    )
    # Use embed.output as the embedding endpoint URL in downstream steps
```

## Dependencies

```
kubernetes>=28.1.0
```
