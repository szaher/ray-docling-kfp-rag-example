"""KFP Pipeline: Multi-Step RAG Document Processing.

Orchestrates four reusable components:
1. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
2. Ingest into Milvus (read chunks from S3, embed locally, insert)
3. Download LLM to PVC (cached — skips if already present)
4. Deploy LLM for RAG inference (vLLM InferenceService from PVC)

Intermediate chunks are stored in S3 (MinIO), not on the PVC.
The data PVC is mounted read-only for input PDFs.
The model cache PVC stores downloaded HuggingFace models.
Embedding uses a local sentence-transformers model (no TEI runtime needed).
S3 credentials are read from a Kubernetes Secret (not pipeline parameters).
"""

from kfp import dsl
from kfp import kubernetes

from components.download_model.component import download_model
from components.ingest_to_milvus.component import ingest_to_milvus
from components.model_deployment.component import model_deployment
from components.parse_and_chunk.component import parse_and_chunk


@dsl.pipeline(
    name="RAG Multi-Step Pipeline",
    description=(
        "Multi-step RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), ingest into Milvus with local embeddings, "
        "download and deploy an LLM for inference (cached on PVC)."
    ),
)
def rag_multistep_pipeline(
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
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "rag_documents",
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

    # Step 2: Ingest into Milvus (local embedding, no TEI endpoint needed)
    ingest_task = ingest_to_milvus(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
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

    # Step 3: Download LLM to PVC (skips if already cached)
    download_task = download_model(
        model_name=llm_model_name,
        model_cache_pvc=model_cache_pvc,
    )
    kubernetes.mount_pvc(
        download_task,
        pvc_name=model_cache_pvc,
        mount_path="/mnt/models",
    )

    # Step 4: Deploy LLM for RAG inference (from PVC cache)
    deploy_task = model_deployment(
        model_name=llm_model_name,
        namespace=namespace,
        model_dir=download_task.output,
        model_cache_pvc=model_cache_pvc,
        max_model_len=max_model_len,
        gpu_count=gpu_count,
    )
    deploy_task.after(ingest_task)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_multistep_pipeline,
        package_path="rag_multistep_pipeline.yaml",
    )
    print("Pipeline compiled to rag_multistep_pipeline.yaml")
