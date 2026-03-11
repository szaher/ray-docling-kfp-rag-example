"""KFP Pipeline: Multi-Step RAG Document Processing.

Orchestrates three reusable components:
1. Parse & chunk PDFs (Docling + HybridChunker via RayJob -> S3)
2. Ingest into Milvus (read chunks from S3, embed locally, insert)
3. Deploy LLM for RAG inference (vLLM InferenceService)

Intermediate chunks are stored in S3 (MinIO), not on the PVC.
The PVC is mounted read-only for input PDFs.
Embedding uses a local sentence-transformers model (no TEI runtime needed).
"""

from kfp import dsl

from components.ingest_to_milvus.component import ingest_to_milvus
from components.model_deployment.component import model_deployment
from components.parse_and_chunk.component import parse_and_chunk


@dsl.pipeline(
    name="RAG Multi-Step Pipeline",
    description=(
        "Multi-step RAG pipeline: parse & chunk PDFs with Docling "
        "(output to S3), ingest into Milvus with local embeddings, "
        "and deploy an LLM for inference."
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
    s3_access_key: str = "",
    s3_secret_key: str = "",
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
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
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

    # Step 2: Ingest into Milvus (local embedding, no TEI endpoint needed)
    ingest_task = ingest_to_milvus(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    ingest_task.after(chunk_task)

    # Step 3: Deploy LLM for RAG inference
    deploy_task = model_deployment(
        model_name=llm_model_name,
        namespace=namespace,
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
