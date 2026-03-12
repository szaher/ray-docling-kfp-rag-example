"""KFP component: Parse PDFs with Docling and chunk with HybridChunker.

Submits a RayJob that processes PDFs using Docling's HybridChunker and
writes chunked text as JSONL files to an S3-compatible bucket (MinIO).
The PVC is mounted read-only (input PDFs only).
"""

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=[
        "codeflare-sdk==0.35.0",
        "kubernetes>=28.1.0",
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
    import base64
    import json
    import os
    import subprocess
    import textwrap
    import time

    # Entrypoint script that runs on the Ray cluster.
    # It parses PDFs with Docling, chunks with HybridChunker,
    # and writes JSONL files to S3.
    _DOCLING_CHUNK_PROCESS_PY = textwrap.dedent('''\
        """Ray Data + Docling: parse PDFs and chunk with HybridChunker."""

        import glob
        import io
        import json
        import multiprocessing as mp
        import os
        import queue
        import time
        from typing import Dict, List

        import pandas as pd
        import ray

        MIN_ACTORS = int(os.environ.get("MIN_ACTORS", "2"))
        MAX_ACTORS = int(os.environ.get("MAX_ACTORS", "4"))
        CPUS_PER_ACTOR = int(os.environ.get("CPUS_PER_ACTOR", "4"))
        BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
        REPARTITION_FACTOR = int(os.environ.get("REPARTITION_FACTOR", "10"))

        PVC_MOUNT_PATH = os.environ.get("PVC_MOUNT_PATH", "/mnt/data")
        INPUT_PATH = os.environ.get("INPUT_PATH", "input/pdfs")
        NUM_FILES = int(os.environ.get("NUM_FILES", "0"))

        FILE_TIMEOUT = int(os.environ.get("FILE_TIMEOUT", "600"))
        MAX_ERRORED_BLOCKS = int(os.environ.get("MAX_ERRORED_BLOCKS", "50"))

        CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "256"))
        TOKENIZER = os.environ.get("TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")

        S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "")
        S3_BUCKET = os.environ.get("S3_BUCKET", "rag-chunks")
        S3_PREFIX = os.environ.get("S3_PREFIX", "chunks")
        S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
        S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")


        def _get_s3_client():
            import boto3
            return boto3.client(
                "s3",
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
                region_name="us-east-1",
            )


        def _ensure_bucket(s3_client, bucket):
            try:
                s3_client.head_bucket(Bucket=bucket)
            except s3_client.exceptions.ClientError:
                s3_client.create_bucket(Bucket=bucket)
                print(f"Created S3 bucket '{bucket}'")


        def _converter_worker(
            req_q, res_q, cpus_per_actor, chunk_max_tokens, tokenizer_name,
            s3_endpoint, s3_bucket, s3_prefix, s3_access_key, s3_secret_key,
        ):
            os.environ["OMP_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["MKL_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import boto3
            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions, PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=cpus_per_actor, device="cpu",
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            chunker = HybridChunker(tokenizer=tokenizer_name, max_tokens=chunk_max_tokens)

            s3 = boto3.client(
                "s3", endpoint_url=s3_endpoint,
                aws_access_key_id=s3_access_key,
                aws_secret_access_key=s3_secret_key,
                region_name="us-east-1",
            )
            res_q.put(("ready",))

            while True:
                msg = req_q.get()
                if msg is None:
                    break
                file_path = msg
                try:
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    if len(file_bytes) == 0:
                        res_q.put(("error", file_path, 0, 0, "File empty"))
                        continue
                    fname = os.path.basename(file_path)
                    stem = os.path.splitext(fname)[0]
                    stream = DocumentStream(name=fname, stream=io.BytesIO(file_bytes))
                    result = converter.convert(stream)
                    doc = result.document
                    pages = getattr(doc, "pages", None)
                    page_count = len(pages) if pages is not None else 0
                    chunks = list(chunker.chunk(doc))
                    if not chunks:
                        res_q.put(("empty", file_path, page_count, 0, "No chunks"))
                        continue
                    lines = []
                    for idx, chunk in enumerate(chunks):
                        if chunk.text.strip():
                            lines.append(json.dumps({
                                "source_file": fname,
                                "chunk_index": idx,
                                "text": chunk.text,
                            }))
                    if not lines:
                        res_q.put(("empty", file_path, page_count, 0, "All chunks empty"))
                        continue
                    jsonl_body = "\\n".join(lines) + "\\n"
                    s3_key = f"{s3_prefix}/{stem}.jsonl"
                    s3.put_object(
                        Bucket=s3_bucket, Key=s3_key,
                        Body=jsonl_body.encode("utf-8"),
                        ContentType="application/jsonl",
                    )
                    res_q.put(("success", file_path, page_count, len(lines), ""))
                except Exception as e:
                    res_q.put(("error", file_path, 0, 0, str(e)[:200]))


        class DoclingChunkProcessor:
            def __init__(self):
                import socket
                self.hostname = socket.gethostname()
                self._start_worker()
                print(f"[{self.hostname}] DoclingChunkProcessor ready (pid={self._worker.pid})")

            def _start_worker(self):
                self._req_q = mp.Queue()
                self._res_q = mp.Queue()
                self._worker = mp.Process(
                    target=_converter_worker,
                    args=(
                        self._req_q, self._res_q, CPUS_PER_ACTOR,
                        CHUNK_MAX_TOKENS, TOKENIZER, S3_ENDPOINT, S3_BUCKET,
                        S3_PREFIX, S3_ACCESS_KEY, S3_SECRET_KEY,
                    ),
                    daemon=True,
                )
                self._worker.start()
                msg = self._res_q.get(timeout=600)
                assert msg[0] == "ready"

            def _restart_worker(self):
                if self._worker.is_alive():
                    self._worker.terminate()
                    self._worker.join(timeout=5)
                    if self._worker.is_alive():
                        self._worker.kill()
                        self._worker.join()
                while True:
                    try:
                        self._res_q.get_nowait()
                    except queue.Empty:
                        break
                self._start_worker()
                print(f"[{self.hostname}] Restarted converter (pid={self._worker.pid})")

            def __call__(self, batch):
                path_list = batch["path"]
                out = {k: [] for k in [
                    "source_file", "status", "page_count", "chunk_count",
                    "error", "duration_s", "actor_host",
                ]}
                for file_path in path_list:
                    fname = os.path.basename(file_path)
                    t0 = time.time()
                    self._req_q.put(str(file_path))
                    try:
                        result = self._res_q.get(timeout=FILE_TIMEOUT)
                        status, _, page_count, chunk_count, error_msg = result
                    except queue.Empty:
                        status, page_count, chunk_count = "timeout", 0, 0
                        error_msg = f"Timed out after {FILE_TIMEOUT}s"
                        self._restart_worker()
                    elapsed = round(time.time() - t0, 3)
                    out["source_file"].append(fname)
                    out["status"].append(status)
                    out["page_count"].append(page_count)
                    out["chunk_count"].append(chunk_count)
                    out["error"].append(error_msg)
                    out["duration_s"].append(elapsed)
                    out["actor_host"].append(self.hostname)
                return out


        def run():
            input_full_path = os.path.join(PVC_MOUNT_PATH, INPUT_PATH)
            pdf_paths = sorted(glob.glob(f"{input_full_path}/**/*.pdf", recursive=True))
            if NUM_FILES > 0:
                pdf_paths = pdf_paths[:NUM_FILES]
            print(f"Found {len(pdf_paths)} PDFs to process.")
            if not pdf_paths:
                print("No PDFs found. Exiting.")
                return
            s3 = _get_s3_client()
            _ensure_bucket(s3, S3_BUCKET)
            print(f"Output: s3://{S3_BUCKET}/{S3_PREFIX}/")

            target_blocks = MAX_ACTORS * REPARTITION_FACTOR
            ds = ray.data.from_pandas(pd.DataFrame({"path": pdf_paths}))
            ds = ds.repartition(target_blocks)
            print(f"Repartitioned into {target_blocks} blocks for {MAX_ACTORS} max actors.")

            results_ds = ds.map_batches(
                DoclingChunkProcessor,
                compute=ray.data.ActorPoolStrategy(
                    min_size=MIN_ACTORS, max_size=MAX_ACTORS,
                ),
                batch_size=BATCH_SIZE,
                batch_format="numpy",
                num_cpus=CPUS_PER_ACTOR,
            )

            start_time = time.time()
            success_count = error_count = timeout_count = empty_count = 0
            total_pages = total_chunks = 0
            actor_dist = {}
            errors_list = []

            for batch in results_ds.iter_batches(batch_size=200, batch_format="numpy"):
                n = len(batch["source_file"])
                for i in range(n):
                    status = str(batch["status"][i])
                    fname = str(batch["source_file"][i])
                    if status == "success":
                        success_count += 1
                    elif status == "timeout":
                        timeout_count += 1
                        errors_list.append((fname, str(batch["error"][i])))
                    elif status == "empty":
                        empty_count += 1
                    else:
                        error_count += 1
                        errors_list.append((fname, str(batch["error"][i])))
                    total_pages += int(batch["page_count"][i])
                    total_chunks += int(batch["chunk_count"][i])
                    actor = str(batch["actor_host"][i])
                    actor_dist[actor] = actor_dist.get(actor, 0) + 1

            wall_clock = time.time() - start_time
            total_files = success_count + error_count + timeout_count + empty_count

            s3_objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/")
            s3_count = s3_objects.get("KeyCount", 0)

            print("\\n" + "=" * 60)
            print("PARSE & CHUNK REPORT")
            print("=" * 60)
            print(f"Actors:            {MIN_ACTORS}..{MAX_ACTORS} (CPUs/actor: {CPUS_PER_ACTOR})")
            print(f"Output:            s3://{S3_BUCKET}/{S3_PREFIX}/")
            print(f"Total files:       {total_files}")
            print(f"Success:           {success_count}")
            print(f"Empty:             {empty_count}")
            print(f"Errors:            {error_count}")
            print(f"Timeouts:          {timeout_count}")
            print(f"Total pages:       {total_pages}")
            print(f"Total chunks:      {total_chunks}")
            print(f"S3 objects:        {s3_count}")
            print(f"Wall clock:        {wall_clock:.1f}s")
            if wall_clock > 0:
                print(f"Files/second:      {success_count / wall_clock:.2f}")
                print(f"Chunks/second:     {total_chunks / wall_clock:.2f}")
            for actor, count in sorted(actor_dist.items()):
                pct = count / total_files * 100 if total_files else 0.0
                print(f"  {actor}: {count} files ({pct:.1f}%)")
            if errors_list:
                for fname, err in errors_list[:10]:
                    print(f"  {fname}: {err[:80]}")
            print("=" * 60)


        if __name__ == "__main__":
            ray.init(ignore_reinit_error=True)
            ctx = ray.data.DataContext.get_current()
            ctx.max_errored_blocks = MAX_ERRORED_BLOCKS
            run()
    ''')

    from codeflare_sdk import ManagedClusterConfig, RayJob
    from kubernetes.client import (
        V1PersistentVolumeClaimVolumeSource,
        V1Volume,
        V1VolumeMount,
    )
    from ray.runtime_env import RuntimeEnv

    rayjob_name = f"docling-chunk-{int(time.time())}"
    schedulable_cpus = worker_cpus - 2

    # Encode entrypoint script as base64 env var to avoid creating
    # Secrets or ConfigMaps (service account lacks permissions for both).
    script_b64 = base64.b64encode(
        _DOCLING_CHUNK_PROCESS_PY.encode()
    ).decode()

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
        },
    )

    job = RayJob(
        job_name=rayjob_name,
        entrypoint="python -c \"import base64,os;exec(base64.b64decode(os.environ['ENTRYPOINT_SCRIPT']).decode())\"",
        cluster_config=cluster_config,
        namespace=namespace,
        runtime_env=RuntimeEnv(
            pip=[
                "opencv-python-headless",
                "pypdfium2",
                "boto3>=1.28.0",
            ],
            env_vars={
                "ENTRYPOINT_SCRIPT": script_b64,
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
                "S3_ACCESS_KEY": os.environ["S3_ACCESS_KEY"],
                "S3_SECRET_KEY": os.environ["S3_SECRET_KEY"],
            },
        ),
        ttl_seconds_after_finished=300,
    )

    job.submit()
    print(f"RayJob '{rayjob_name}' submitted.")

    # Remove kueue queue-name label so kueue does not manage this job
    subprocess.run(
        [
            "oc", "label", "rayjob", rayjob_name,
            "-n", namespace,
            "kueue.x-k8s.io/queue-name-",
        ],
        check=True,
    )
    print(f"Removed kueue.x-k8s.io/queue-name label from RayJob '{rayjob_name}'.")

    # Unsuspend the RayJob so it starts running
    subprocess.run(
        [
            "oc", "patch", "rayjob", rayjob_name,
            "-n", namespace,
            "--type", "merge",
            "-p", json.dumps({"spec": {"suspend": False}}),
        ],
        check=True,
    )
    print(f"Unsuspended RayJob '{rayjob_name}'.")

    # Wait for the RayCluster to be created by the RayJob controller,
    # then patch it directly (the controller ignores RayJob spec changes).
    print("Waiting for RayCluster to be created...")
    raycluster_name = ""
    for _ in range(60):
        result = subprocess.run(
            [
                "oc", "get", "rayjob", rayjob_name,
                "-n", namespace,
                "-o", "jsonpath={.status.rayClusterName}",
            ],
            capture_output=True, text=True,
        )
        raycluster_name = result.stdout.strip()
        if raycluster_name:
            break
        time.sleep(5)
    if not raycluster_name:
        raise RuntimeError(f"RayCluster was not created for RayJob '{rayjob_name}'.")
    print(f"Found RayCluster '{raycluster_name}', patching num-cpus...")

    # Patch head num-cpus=0 so no actors run on head
    patch = [
        {
            "op": "add",
            "path": "/spec/headGroupSpec/rayStartParams/num-cpus",
            "value": "0",
        },
        {
            "op": "add",
            "path": "/spec/workerGroupSpecs/0/rayStartParams/num-cpus",
            "value": str(schedulable_cpus),
        },
    ]
    subprocess.run(
        [
            "oc", "patch", "raycluster", raycluster_name,
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
