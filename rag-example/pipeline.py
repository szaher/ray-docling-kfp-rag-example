"""KFP Pipeline: RAG Document Processing.

Orchestrates:
1. PDF -> Milvus (parse, chunk, embed, ingest -- single step)
2. Model deployment on OpenShift AI (vLLM)

All components use in-cluster Kubernetes config -- no API URL or token needed.
"""

from kfp import dsl

from components.model_deployment.component import model_deployment
from components.pdf_to_milvus.component import pdf_to_milvus


@dsl.pipeline(
    name="RAG Document Processing Pipeline",
    description=(
        "Process PDFs with Ray Data + Docling, ingest into Milvus, "
        "and deploy a model for RAG inference."
    ),
)
def rag_pipeline(
    # Shared
    pvc_name: str = "data-pvc",
    pvc_mount_path: str = "/mnt/data",
    namespace: str = "ray-docling",
    # PDF -> Milvus
    input_path: str = "input/pdfs",
    ray_image: str = "quay.io/rhoai-szaher/docling-ray:latest",
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    chunk_max_tokens: int = 256,
    num_files: int = 0,
    # Milvus
    milvus_host: str = "milvus-milvus.milvus.svc.cluster.local",
    milvus_port: int = 19530,
    collection_name: str = "rag_documents",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    # Model deployment
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    gpu_count: int = 1,
):
    # Step 1: Parse PDFs -> chunk -> embed -> insert into Milvus
    ingest_task = pdf_to_milvus(
        pvc_name=pvc_name,
        pvc_mount_path=pvc_mount_path,
        input_path=input_path,
        ray_image=ray_image,
        namespace=namespace,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_max_tokens=chunk_max_tokens,
        num_workers=num_workers,
        worker_cpus=worker_cpus,
        worker_memory_gb=worker_memory_gb,
        cpus_per_actor=cpus_per_actor,
        min_actors=min_actors,
        max_actors=max_actors,
        num_files=num_files,
    )

    # Step 2: Deploy model for RAG inference
    deploy_task = model_deployment(
        model_name=model_name,
        namespace=namespace,
        gpu_count=gpu_count,
    )
    deploy_task.after(ingest_task)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        rag_pipeline,
        package_path="rag_pipeline.yaml",
    )
    print("Pipeline compiled to rag_pipeline.yaml")
