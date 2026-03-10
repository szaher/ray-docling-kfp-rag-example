"""KFP component: Parse, chunk, embed PDFs and ingest into Milvus.

Submits a RayJob that processes PDFs using Docling's HybridChunker,
generates embeddings with sentence-transformers, and inserts directly
into Milvus. No intermediate files are written — PVC is read-only.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/modh/odh-generic-data-science-notebook:v3-20250827",
    packages_to_install=[
        "codeflare-sdk>=0.25.0",
        "kubernetes>=28.1.0",
        "ray[default]>=2.44.1",
    ],
)
def pdf_to_milvus(
    pvc_name: str,
    pvc_mount_path: str,
    input_path: str,
    ray_image: str,
    namespace: str,
    milvus_host: str,
    milvus_port: int = 19530,
    milvus_db: str = "default",
    collection_name: str = "rag_documents",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    chunk_max_tokens: int = 256,
    num_workers: int = 2,
    worker_cpus: int = 8,
    worker_memory_gb: int = 16,
    head_cpus: int = 2,
    head_memory_gb: int = 4,
    cpus_per_actor: int = 4,
    min_actors: int = 2,
    max_actors: int = 4,
    batch_size: int = 4,
    num_files: int = 0,
    timeout_seconds: int = 600,
) -> str:
    """Process PDFs and ingest directly into Milvus.

    Args:
        pvc_name: Name of the ReadWriteMany PVC with input PDFs.
        pvc_mount_path: Mount path for the PVC inside pods.
        input_path: Relative path to input PDFs under the PVC.
        ray_image: Container image with Ray + Docling pre-installed.
        namespace: OpenShift namespace for the RayJob.
        milvus_host: Milvus service hostname.
        milvus_port: Milvus gRPC port.
        milvus_db: Milvus database name.
        collection_name: Milvus collection name.
        embedding_model: Sentence-transformers model for embeddings.
        embedding_dim: Dimension of the embedding vectors.
        chunk_max_tokens: Max tokens per chunk (HybridChunker).
        num_workers: Number of Ray worker pods.
        worker_cpus: CPUs per worker pod.
        worker_memory_gb: Memory (GB) per worker pod.
        head_cpus: CPUs for the Ray head pod.
        head_memory_gb: Memory (GB) for the Ray head pod.
        cpus_per_actor: CPUs per Docling processing actor.
        min_actors: Minimum actor pool size.
        max_actors: Maximum actor pool size.
        batch_size: Files per batch sent to each actor.
        num_files: Number of PDFs to process (0 = all).
        timeout_seconds: Per-file processing timeout.

    Returns:
        The Milvus collection name and total chunks inserted.
    """
    import json
    import subprocess
    import time

    from codeflare_sdk import ManagedClusterConfig, RayJob
    from kubernetes.client import (
        V1PersistentVolumeClaimVolumeSource,
        V1Volume,
        V1VolumeMount,
    )
    from ray.runtime_env import RuntimeEnv

    rayjob_name = f"docling-milvus-{int(time.time())}"
    schedulable_cpus = worker_cpus - 2

    shared_mount = V1VolumeMount(pvc_mount_path, name="shared-data", read_only=True)
    data_volume = V1Volume(
        name="shared-data",
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name=pvc_name, read_only=True,
        ),
    )

    cluster_config = ManagedClusterConfig(
        head_cpu_requests=head_cpus,
        head_cpu_limits=head_cpus,
        head_memory_requests=head_memory_gb,
        head_memory_limits=head_memory_gb,
        num_workers=num_workers,
        worker_cpu_requests=worker_cpus,
        worker_cpu_limits=worker_cpus,
        worker_memory_requests=worker_memory_gb,
        worker_memory_limits=worker_memory_gb,
        volume_mounts=[shared_mount],
        volumes=[data_volume],
        image=ray_image,
        envs={
            "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.1",
            "RAY_USE_TLS": "0",
        },
    )

    job = RayJob(
        job_name=rayjob_name,
        entrypoint="python docling_milvus_process.py",
        cluster_config=cluster_config,
        namespace=namespace,
        runtime_env=RuntimeEnv(
            working_dir=".",
            pip=[
                "opencv-python-headless",
                "pypdfium2",
                "pymilvus>=2.4.0",
                "sentence-transformers>=2.2.0",
            ],
            env_vars={
                "PVC_MOUNT_PATH": pvc_mount_path,
                "INPUT_PATH": input_path,
                "NUM_FILES": str(num_files),
                "MIN_ACTORS": str(min_actors),
                "MAX_ACTORS": str(max_actors),
                "CPUS_PER_ACTOR": str(cpus_per_actor),
                "BATCH_SIZE": str(batch_size),
                "REPARTITION_FACTOR": "10",
                "FILE_TIMEOUT": str(timeout_seconds),
                "MAX_ERRORED_BLOCKS": "50",
                "MILVUS_HOST": milvus_host,
                "MILVUS_PORT": str(milvus_port),
                "MILVUS_DB": milvus_db,
                "MILVUS_COLLECTION": collection_name,
                "EMBEDDING_MODEL": embedding_model,
                "EMBEDDING_DIM": str(embedding_dim),
                "CHUNK_MAX_TOKENS": str(chunk_max_tokens),
            },
        ),
        ttl_seconds_after_finished=300,
    )

    job.submit()
    print(f"RayJob '{rayjob_name}' submitted.")

    # Patch head num-cpus=0 so no actors run on head
    patch = [
        {
            "op": "add",
            "path": "/spec/rayClusterSpec/headGroupSpec/rayStartParams/num-cpus",
            "value": "0",
        },
        {
            "op": "add",
            "path": "/spec/rayClusterSpec/workerGroupSpecs/0/rayStartParams/num-cpus",
            "value": str(schedulable_cpus),
        },
    ]
    subprocess.run(
        [
            "oc", "patch", "rayjob", rayjob_name,
            "-n", namespace,
            "--type", "json",
            "-p", json.dumps(patch),
        ],
        check=True,
    )

    # Wait for job completion
    print("Waiting for RayJob to complete...")
    while True:
        result = subprocess.run(
            [
                "oc", "get", "rayjob", rayjob_name,
                "-n", namespace,
                "-o", "jsonpath={.status.jobStatus}",
            ],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "SUCCEEDED":
            print("RayJob completed successfully.")
            break
        elif status == "FAILED":
            raise RuntimeError(f"RayJob '{rayjob_name}' failed.")
        time.sleep(30)

    return f"{collection_name}"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pdf_to_milvus,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
