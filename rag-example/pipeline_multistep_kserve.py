"""KFP Pipeline: Multi-Step RAG with KServe-Deployed Embedding Model.

Orchestrates five reusable components:
1. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
2. Deploy embedding model (KServe InferenceService with TEI)
3. Ingest into Milvus (read chunks from S3, embed via deployed endpoint, insert)
4. Download LLM to PVC (cached — skips if already present)
5. Deploy LLM for RAG inference (vLLM InferenceService from PVC)

This version uses a KServe-deployed embedding service instead of local
sentence-transformers, enabling:
- Shared embedding service across pipelines
- GPU acceleration for embeddings (optional)
- Better scalability with autoscaling
- Consistent embedding endpoint for RAG queries

Intermediate chunks are stored in S3 (MinIO), not on the PVC.
The data PVC is mounted read-only for input PDFs.
The model cache PVC stores downloaded HuggingFace models.
S3 credentials are read from a Kubernetes Secret (not pipeline parameters).
"""

from kfp import dsl
from kfp import kubernetes

from components.deploy_embedding_model.component import deploy_embedding_model
from components.download_model.component import download_model
from components.ingest_to_milvus.component import ingest_to_milvus
from components.model_deployment.component import model_deployment
from components.parse_and_chunk.component import parse_and_chunk


@dsl.pipeline(
    name="RAG Multi-Step Pipeline (KServe Embeddings)",
    description=(
        "Multi-step RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), deploy embedding model via KServe, ingest into Milvus "
        "with remote embeddings, download and deploy an LLM for inference."
    ),
)
def rag_multistep_pipeline_kserve(
    # Shared
    pvc_name: str = "data-pvc",
    pvc_mount_path: str = "/mnt/data",
    namespace: str = "ray-docling",
    # S3 (MinIO)
    s3_endpoint: str = "http://minio-service.default.svc.cluster.local:9000",
    s3_bucket: str = "rag-chunks",
    s3_prefix: str = "chunks",
    s3_secret_name: str = "minio-secret",
    # PDF parsing
    input_path: str = "input/pdfs",
    ray_image: str = "quay.io/rhoai-szaher/docling-ray:latest",
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    chunk_max_tokens: int = 256,
    num_files: int = 1000,
    # Embedding model deployment
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    embedding_serving_runtime: str = "embedding-runtime",
    embedding_min_replicas: int = 1,
    embedding_max_replicas: int = 3,
    embedding_cpu_requests: str = "2",
    embedding_cpu_limits: str = "4",
    embedding_memory_requests: str = "4Gi",
    embedding_memory_limits: str = "8Gi",
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "rag_documents",
    embed_batch_size: int = 64,
    # LLM deployment
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    model_cache_pvc: str = "model-cache-pvc",
    max_model_len: int = 4096,
    gpu_count: int = 1,
):
    # Step 1: Parse & chunk PDFs -> S3
    chunk_task = parse_and_chunk(
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        input_path=input_path,
        ray_image=ray_image,
        namespace=namespace,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        tokenizer=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        num_workers=num_workers,
        worker_cpus=worker_cpus,
        worker_memory_gb=worker_memory_gb,
        cpus_per_actor=cpus_per_actor,
        min_actors=min_actors,
        max_actors=max_actors,
        num_files=num_files,
    )
    kubernetes.use_secret_as_env(
        chunk_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )

    # Step 2: Deploy embedding model as KServe InferenceService
    embed_deploy_task = deploy_embedding_model(
        model_name=embedding_model,
        namespace=namespace,
        serving_runtime_name=embedding_serving_runtime,
        min_replicas=embedding_min_replicas,
        max_replicas=embedding_max_replicas,
        cpu_requests=embedding_cpu_requests,
        cpu_limits=embedding_cpu_limits,
        memory_requests=embedding_memory_requests,
        memory_limits=embedding_memory_limits,
    )

    # Step 3: Ingest into Milvus using deployed embedding endpoint
    ingest_task = ingest_to_milvus(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        embedding_endpoint=embed_deploy_task.output,  # Use deployed service
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embed_batch_size=embed_batch_size,
    )
    kubernetes.use_secret_as_env(
        ingest_task,
        secret_name=s3_secret_name,
        secret_key_to_env={
            "access_key": "S3_ACCESS_KEY",
            "secret_key": "S3_SECRET_KEY",
        },
    )
    ingest_task.after(chunk_task)
    ingest_task.after(embed_deploy_task)  # Wait for embedding service

    # Step 4: Download LLM to PVC (skips if already cached)
    download_task = download_model(
        model_name=llm_model_name,
        model_cache_pvc=model_cache_pvc,
    )
    kubernetes.mount_pvc(
        download_task,
        pvc_name=model_cache_pvc,
        mount_path="/mnt/models",
    )

    # Step 5: Deploy LLM for RAG inference (from PVC cache)
    deploy_task = model_deployment(
        model_name=llm_model_name,
        namespace=namespace,
        model_dir=download_task.output,
        model_cache_pvc=model_cache_pvc,
        max_model_len=max_model_len,
        gpu_count=gpu_count,
    )
    # No dependency on ingest_task — model deployment runs in parallel
    # with the data pipeline (parse → embed deploy → ingest).
    # Pipeline has two independent chains:
    #   Data chain:  parse_and_chunk → deploy_embedding_model → ingest_to_milvus
    #   Model chain: download_model  → model_deployment


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_multistep_pipeline_kserve,
        package_path="rag_multistep_pipeline_kserve.yaml",
    )
    print("Pipeline compiled to rag_multistep_pipeline_kserve.yaml")
