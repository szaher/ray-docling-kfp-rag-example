# model_deployment Component

KFP component that deploys a model on OpenShift AI using a KServe `InferenceService` with the vLLM serving runtime.

## Overview

This component creates or updates a KServe `InferenceService` CR in the specified namespace. It supports models from HuggingFace (pulled automatically) or from a PVC path. After creation, it polls the InferenceService status until the `Ready` condition is `True`, then returns the inference endpoint URL.

Uses in-cluster Kubernetes config (service account token) — no explicit API URL or token required.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | str | — | Model name/path (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) |
| `namespace` | str | — | Namespace to deploy the InferenceService |
| `runtime` | str | `vllm` | Serving runtime (used as `modelFormat.name`) |
| `model_path` | str | `""` | Optional PVC path. If empty, pulls from HuggingFace via `hf://` URI |
| `serving_runtime_name` | str | `vllm-runtime` | Name of the ServingRuntime CR |
| `min_replicas` | int | `1` | Minimum replicas |
| `max_replicas` | int | `1` | Maximum replicas |
| `gpu_count` | int | `1` | GPUs per replica (`nvidia.com/gpu` resource). Set to 0 to disable GPU requests |

## Returns

The inference endpoint URL with `/v1` suffix (e.g., `https://model-endpoint.apps.cluster.example.com/v1`), compatible with the OpenAI API format used by vLLM.

## How It Works

1. **Connect** to the Kubernetes API using in-cluster service account config.

2. **Build InferenceService spec** with:
   - `RawDeployment` mode (no Knative dependency)
   - Resource requests: 2 CPU / 8Gi memory, limits: 4 CPU / 16Gi memory
   - GPU resources if `gpu_count > 0`
   - `storageUri`: either `hf://<model_name>` or the provided `model_path`

3. **Create or update** the InferenceService (idempotent — patches if it already exists).

4. **Wait for Ready** by polling every 30 seconds for up to 30 minutes. Returns the endpoint URL from `.status.url` when ready.

## Prerequisites

- **OpenShift AI (RHOAI)** installed with KServe controller
- **ServingRuntime CR** named `vllm-runtime` (or custom name) must exist in the target namespace. RHOAI typically pre-installs this.
- **GPU nodes** available in the cluster (for vLLM model serving)
- **HuggingFace access** if pulling gated models (configure `HF_TOKEN` in the ServingRuntime or namespace secrets)
- The KFP pipeline service account must have permissions to create/patch InferenceService CRs in the target namespace.

## Usage

```python
from components.model_deployment import model_deployment

@dsl.pipeline(name="Deploy Model")
def deploy_pipeline():
    deploy = model_deployment(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        namespace="rag-example",
        gpu_count=1,
    )
```

### Standalone compilation

```bash
python component.py
# Output: component_component.yaml
```

## InferenceService Spec

The component creates an InferenceService like:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: mistral-7b-instruct-v0-3
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    model:
      modelFormat:
        name: vllm
      runtime: vllm-runtime
      storageUri: "hf://mistralai/Mistral-7B-Instruct-v0.3"
      resources:
        requests:
          cpu: "2"
          memory: 8Gi
          nvidia.com/gpu: "1"
        limits:
          cpu: "4"
          memory: 16Gi
          nvidia.com/gpu: "1"
```

## Dependencies

### Component runtime (KFP pod)

```
kubernetes>=28.1.0
```

### Cluster requirements

- KServe controller (part of OpenShift AI)
- vLLM ServingRuntime CR in the target namespace
- Sufficient GPU quota

## Notes

- The InferenceService name is derived from `model_name` by taking the last path segment, lowercasing, and replacing dots with hyphens (e.g., `mistralai/Mistral-7B-Instruct-v0.3` becomes `mistral-7b-instruct-v0-3`).
- The component uses `RawDeployment` mode, which creates a standard Kubernetes Deployment instead of a Knative Service. This avoids requiring Knative/Serverless on the cluster.
- If the InferenceService already exists, it is patched in place rather than deleted and recreated.
