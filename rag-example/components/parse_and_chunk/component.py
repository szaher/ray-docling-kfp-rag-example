"""KFP component: Parse PDFs with Docling and chunk with HybridChunker.

Submits a RayJob that processes PDFs using Docling's HybridChunker and
writes chunked text as JSONL files to an S3-compatible bucket (MinIO).
The PVC is mounted read-only (input PDFs only).
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/modh/odh-generic-data-science-notebook:v3-20250827",
    packages_to_install=[
        "codeflare-sdk==0.35.0",
        "kubernetes>=28.1.0",
        "ray[default]==2.53.0",
    ],
)
def parse_and_chunk(
    pvc_name: str,
    pvc_mount_path: str,
    input_path: str,
    ray_image: str,
    namespace: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_prefix: str = "chunks",
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
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
    """Parse PDFs and write chunked JSONL files to S3.

    Args:
        pvc_name: Name of the PVC with input PDFs.
        pvc_mount_path: Mount path for the PVC inside pods.
        input_path: Relative path to input PDFs under the PVC.
        ray_image: Container image with Ray + Docling pre-installed.
        namespace: OpenShift namespace for the RayJob.
        s3_endpoint: S3-compatible endpoint URL (e.g. MinIO).
        s3_bucket: S3 bucket for output chunks.
        s3_access_key: S3 access key.
        s3_secret_key: S3 secret key.
        s3_prefix: Key prefix for chunk files in S3.
        tokenizer: Tokenizer for HybridChunker (usually the embedding model name).
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
        The S3 URI where JSONL chunk files were written.
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

    rayjob_name = f"docling-chunk-{int(time.time())}"
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
        entrypoint="python docling_chunk_process.py",
        cluster_config=cluster_config,
        namespace=namespace,
        runtime_env=RuntimeEnv(
            working_dir=".",
            pip=[
                "opencv-python-headless",
                "pypdfium2",
                "boto3>=1.28.0",
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
                "CHUNK_MAX_TOKENS": str(chunk_max_tokens),
                "TOKENIZER": tokenizer,
                "S3_ENDPOINT": s3_endpoint,
                "S3_BUCKET": s3_bucket,
                "S3_PREFIX": s3_prefix,
                "S3_ACCESS_KEY": s3_access_key,
                "S3_SECRET_KEY": s3_secret_key,
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

    return f"s3://{s3_bucket}/{s3_prefix}"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        parse_and_chunk,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
