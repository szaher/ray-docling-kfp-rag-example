"""KFP component: Parse, chunk, embed PDFs and ingest into Milvus.

Submits a RayJob that processes PDFs using Docling's HybridChunker,
generates embeddings with sentence-transformers, and inserts directly
into Milvus. No intermediate files are written — PVC is read-only.
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/modh/odh-generic-data-science-notebook:v3-20250827",
    packages_to_install=[
        "codeflare-sdk==0.35.0",
        "kubernetes>=28.1.0",
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
    import tempfile
    import textwrap
    import time

    # Entrypoint script that runs on the Ray cluster.
    # Parses PDFs with Docling, chunks, embeds with sentence-transformers,
    # and inserts directly into Milvus.
    _DOCLING_MILVUS_PROCESS_PY = textwrap.dedent('''\
        """Ray Data + Docling + Milvus: parse, chunk, embed, and ingest PDFs."""

        import glob
        import io
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

        MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus-milvus.milvus.svc.cluster.local")
        MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
        MILVUS_DB = os.environ.get("MILVUS_DB", "default")
        COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "rag_documents")

        EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
        CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "256"))
        MILVUS_BATCH_SIZE = int(os.environ.get("MILVUS_BATCH_SIZE", "64"))


        def _converter_worker(
            req_q, res_q, cpus_per_actor, chunk_max_tokens, embedding_model_name,
            milvus_uri, milvus_db, collection_name, milvus_batch_size,
        ):
            os.environ["OMP_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["MKL_NUM_THREADS"] = str(cpus_per_actor)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            from docling.chunking import HybridChunker
            from docling.datamodel.base_models import DocumentStream, InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions, PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer

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
            chunker = HybridChunker(tokenizer=embedding_model_name, max_tokens=chunk_max_tokens)
            embed_model = SentenceTransformer(embedding_model_name)
            milvus_client = MilvusClient(uri=milvus_uri, db_name=milvus_db)
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
                    stream = DocumentStream(name=fname, stream=io.BytesIO(file_bytes))
                    result = converter.convert(stream)
                    doc = result.document
                    pages = getattr(doc, "pages", None)
                    page_count = len(pages) if pages is not None else 0
                    chunks = list(chunker.chunk(doc))
                    if not chunks:
                        res_q.put(("empty", file_path, page_count, 0, "No chunks"))
                        continue
                    chunk_texts = [c.text for c in chunks if c.text.strip()]
                    if not chunk_texts:
                        res_q.put(("empty", file_path, page_count, 0, "All chunks empty"))
                        continue
                    embeddings = embed_model.encode(
                        chunk_texts, normalize_embeddings=True, show_progress_bar=False,
                    ).tolist()
                    total_inserted = 0
                    for i in range(0, len(chunk_texts), milvus_batch_size):
                        batch_texts = chunk_texts[i:i + milvus_batch_size]
                        batch_embeddings = embeddings[i:i + milvus_batch_size]
                        data = [
                            {"source_file": fname, "chunk_index": i + j,
                             "text": text, "embedding": emb}
                            for j, (text, emb) in enumerate(zip(batch_texts, batch_embeddings))
                        ]
                        milvus_client.insert(collection_name=collection_name, data=data)
                        total_inserted += len(data)
                    res_q.put(("success", file_path, page_count, total_inserted, ""))
                except Exception as e:
                    res_q.put(("error", file_path, 0, 0, str(e)[:200]))


        class DoclingMilvusProcessor:
            def __init__(self):
                import socket
                self.hostname = socket.gethostname()
                self._milvus_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
                self._start_worker()
                print(f"[{self.hostname}] DoclingMilvusProcessor ready (pid={self._worker.pid})")

            def _start_worker(self):
                self._req_q = mp.Queue()
                self._res_q = mp.Queue()
                self._worker = mp.Process(
                    target=_converter_worker,
                    args=(
                        self._req_q, self._res_q, CPUS_PER_ACTOR, CHUNK_MAX_TOKENS,
                        EMBEDDING_MODEL, self._milvus_uri, MILVUS_DB,
                        COLLECTION_NAME, MILVUS_BATCH_SIZE,
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
                    "source_file", "status", "page_count", "chunks_inserted",
                    "error", "duration_s", "actor_host",
                ]}
                for file_path in path_list:
                    fname = os.path.basename(file_path)
                    t0 = time.time()
                    self._req_q.put(str(file_path))
                    try:
                        result = self._res_q.get(timeout=FILE_TIMEOUT)
                        status, _, page_count, chunks_inserted, error_msg = result
                    except queue.Empty:
                        status, page_count, chunks_inserted = "timeout", 0, 0
                        error_msg = f"Timed out after {FILE_TIMEOUT}s"
                        self._restart_worker()
                    elapsed = round(time.time() - t0, 3)
                    out["source_file"].append(fname)
                    out["status"].append(status)
                    out["page_count"].append(page_count)
                    out["chunks_inserted"].append(chunks_inserted)
                    out["error"].append(error_msg)
                    out["duration_s"].append(elapsed)
                    out["actor_host"].append(self.hostname)
                return out


        def setup_milvus_collection():
            from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
            uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
            client = MilvusClient(uri=uri, db_name=MILVUS_DB)
            if client.has_collection(COLLECTION_NAME):
                print(f"Dropping existing collection '{COLLECTION_NAME}'")
                client.drop_collection(COLLECTION_NAME)
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                ],
                description="RAG document chunks with embeddings",
            )
            client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", index_type="IVF_FLAT",
                metric_type="COSINE", params={"nlist": 128},
            )
            client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
            print(f"Collection '{COLLECTION_NAME}' created (dim={EMBEDDING_DIM}).")
            return client


        def run():
            input_full_path = os.path.join(PVC_MOUNT_PATH, INPUT_PATH)
            pdf_paths = sorted(glob.glob(f"{input_full_path}/**/*.pdf", recursive=True))
            if NUM_FILES > 0:
                pdf_paths = pdf_paths[:NUM_FILES]
            print(f"Found {len(pdf_paths)} PDFs to process.")
            if not pdf_paths:
                print("No PDFs found. Exiting.")
                return
            setup_milvus_collection()
            target_blocks = MAX_ACTORS * REPARTITION_FACTOR
            ds = ray.data.from_pandas(pd.DataFrame({"path": pdf_paths}))
            ds = ds.repartition(target_blocks)
            print(f"Repartitioned into {target_blocks} blocks for {MAX_ACTORS} max actors.")

            results_ds = ds.map_batches(
                DoclingMilvusProcessor,
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
                    total_chunks += int(batch["chunks_inserted"][i])
                    actor = str(batch["actor_host"][i])
                    actor_dist[actor] = actor_dist.get(actor, 0) + 1

            wall_clock = time.time() - start_time
            total_files = success_count + error_count + timeout_count + empty_count

            from pymilvus import MilvusClient
            client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", db_name=MILVUS_DB)
            client.load_collection(COLLECTION_NAME)
            stats = client.get_collection_stats(COLLECTION_NAME)

            print("\\n" + "=" * 60)
            print("PROCESSING REPORT")
            print("=" * 60)
            print(f"Actors:            {MIN_ACTORS}..{MAX_ACTORS} (CPUs/actor: {CPUS_PER_ACTOR})")
            print(f"Embedding model:   {EMBEDDING_MODEL}")
            print(f"Collection:        {COLLECTION_NAME}")
            print(f"Total files:       {total_files}")
            print(f"Success:           {success_count}")
            print(f"Empty:             {empty_count}")
            print(f"Errors:            {error_count}")
            print(f"Timeouts:          {timeout_count}")
            print(f"Total pages:       {total_pages}")
            print(f"Total chunks:      {total_chunks}")
            print(f"Wall clock:        {wall_clock:.1f}s")
            if wall_clock > 0:
                print(f"Files/second:      {success_count / wall_clock:.2f}")
                print(f"Chunks/second:     {total_chunks / wall_clock:.2f}")
            print(f"Collection stats:  {stats}")
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

    # Write entrypoint script to a temp directory for codeflare-sdk validation
    work_dir = tempfile.mkdtemp(prefix="rayjob-")
    with open(f"{work_dir}/docling_milvus_process.py", "w") as f:
        f.write(_DOCLING_MILVUS_PROCESS_PY)

    job = RayJob(
        job_name=rayjob_name,
        entrypoint="python docling_milvus_process.py",
        cluster_config=cluster_config,
        namespace=namespace,
        runtime_env=RuntimeEnv(
            working_dir=work_dir,
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
