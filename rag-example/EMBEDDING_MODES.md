# Embedding Modes Guide

The RAG pipeline supports **two modes** for generating embeddings. Choose based on your use case:

## Mode 1: Local Embedding (Default)

**Configuration:**
```python
EMBEDDING_MODE = "local"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```

**How it works:**
- The embedding model runs inside the pipeline component
- Uses `sentence-transformers` library
- Model is loaded each time the pipeline runs
- Model is also loaded in the notebook for RAG queries

**Pros:**
- ✅ Simpler setup - no additional services to deploy
- ✅ Good for development and testing
- ✅ No dependency on external services
- ✅ Works immediately after configuration

**Cons:**
- ❌ Model loaded multiple times (pipeline + notebook)
- ❌ Higher memory usage during pipeline runs
- ❌ Slower startup time for each run
- ❌ Cannot leverage GPU acceleration easily

**Best for:**
- Development and testing
- Small-scale deployments
- When GPU resources are limited
- Quick prototyping

---

## Mode 2: Deployed Embedding Service

**Configuration:**
```python
EMBEDDING_MODE = "service"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_SERVICE_NAME = "embedding-service"
EMBEDDING_ENDPOINT = f"http://{EMBEDDING_SERVICE_NAME}.{NAMESPACE}.svc.cluster.local:8080"
```

**How it works:**
- Embedding model deployed as a dedicated InferenceService
- Uses HuggingFace Text Embeddings Inference (TEI) runtime
- Service is shared across all pipeline runs and RAG queries
- Pipeline and notebook call the service via HTTP API

**Pros:**
- ✅ Model loaded once and reused
- ✅ Better resource utilization
- ✅ Can leverage GPU acceleration
- ✅ Centralized endpoint ensures consistency
- ✅ Production-ready architecture
- ✅ Can handle concurrent requests

**Cons:**
- ❌ Requires deploying an additional service
- ❌ Slightly more complex setup
- ❌ Additional resources needed for the service

**Best for:**
- Production deployments
- Multiple pipeline runs
- When GPUs are available
- High-throughput scenarios
- Multi-user environments

---

## Switching Between Modes

### To use Local mode:

1. Set configuration:
   ```python
   EMBEDDING_MODE = "local"
   ```

2. Run the pipeline - it will use local embeddings automatically

3. In the notebook, the RAG functions will use SentenceTransformer

### To use Service mode:

1. Set configuration:
   ```python
   EMBEDDING_MODE = "service"
   ```

2. Deploy the embedding service (section 1.2 in notebook):
   - Creates ServingRuntime with TEI image
   - Creates InferenceService
   - Waits for service to be ready

3. Verify the service is running:
   - Check pod status
   - Test the endpoint

4. Run the pipeline - it will use the deployed service

5. In the notebook, the RAG functions will call the service endpoint

---

## Performance Comparison

| Metric | Local Mode | Service Mode |
|--------|------------|--------------|
| Setup time | Instant | 2-5 minutes (initial deployment) |
| Pipeline startup | Slower (loads model) | Faster (model already loaded) |
| Memory usage | Higher per run | Shared across runs |
| GPU support | Limited | Full support via TEI |
| Concurrent queries | Limited | Excellent |
| Production ready | No | Yes |

---

## API Compatibility

Both modes use the **same API format** for embeddings:

```python
# Request
POST /v1/embeddings
{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["text to embed"]
}

# Response
{
    "data": [
        {
            "embedding": [0.123, -0.456, ...],
            "index": 0
        }
    ]
}
```

This means switching between modes requires **no code changes** in the pipeline or RAG functions - only the configuration variable.

---

## Troubleshooting

### Service mode issues:

**Service not ready:**
- Check pod logs: `kubectl logs -n <namespace> -l serving.kserve.io/inferenceservice=embedding-service`
- Verify resource requests/limits are satisfied
- Check if the TEI image is accessible

**Connection errors:**
- Verify the service endpoint URL
- Check if the service is in the same namespace
- Test from a pod in the cluster

**Model loading errors:**
- Verify the model name is correct
- Check if the model is compatible with TEI
- Ensure enough memory is allocated

### Local mode issues:

**Out of memory:**
- Reduce batch size in pipeline parameters
- Use a smaller embedding model
- Consider switching to service mode

**Slow performance:**
- Use service mode for better performance
- Reduce the number of documents
- Use GPU nodes if available

---

## Recommended Models

Both modes support any SentenceTransformers-compatible model:

**Small & Fast:**
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, 23MB)
- `sentence-transformers/paraphrase-MiniLM-L3-v2` (384 dim, 17MB)

**Better Quality:**
- `sentence-transformers/all-mpnet-base-v2` (768 dim, 420MB)
- `BAAI/bge-small-en-v1.5` (384 dim, 33MB)
- `BAAI/bge-base-en-v1.5` (768 dim, 420MB)

**Multilingual:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim)
- `sentence-transformers/LaBSE` (768 dim)

**Note:** Update `EMBEDDING_DIM` to match your model's dimension.
