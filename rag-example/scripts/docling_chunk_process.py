"""Ray Data + Docling: parse PDFs and chunk with HybridChunker.

Processes PDF files using Docling in a Ray Data actor pool, chunks them
using Docling's HybridChunker, and writes the results as JSONL files
to an S3-compatible bucket (MinIO). No embedding happens here.

All parameters are read from environment variables.
"""

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

# ---------------------------------------------------------------------------
# Parameters (environment variables)
# ---------------------------------------------------------------------------
MIN_ACTORS = int(os.environ.get("MIN_ACTORS", "2"))
MAX_ACTORS = int(os.environ.get("MAX_ACTORS", "4"))
CPUS_PER_ACTOR = int(os.environ.get("CPUS_PER_ACTOR", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
REPARTITION_FACTOR = int(os.environ.get("REPARTITION_FACTOR", "10"))

PVC_MOUNT_PATH = os.environ.get("PVC_MOUNT_PATH", "/mnt/data")
INPUT_PATH = os.environ.get("INPUT_PATH", "input/pdfs")
NUM_FILES = int(os.environ.get("NUM_FILES", "0"))  # 0 = all

FILE_TIMEOUT = int(os.environ.get("FILE_TIMEOUT", "600"))
MAX_ERRORED_BLOCKS = int(os.environ.get("MAX_ERRORED_BLOCKS", "50"))

# Chunking
CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "256"))
TOKENIZER = os.environ.get("TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")

# S3 output
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000")
S3_BUCKET = os.environ.get("S3_BUCKET", "rag-chunks")
S3_PREFIX = os.environ.get("S3_PREFIX", "chunks")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")


# ---------------------------------------------------------------------------
# S3 helper — shared client per subprocess
# ---------------------------------------------------------------------------


def _get_s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name="us-east-1",
    )


def _ensure_bucket(s3_client, bucket: str):
    try:
        s3_client.head_bucket(Bucket=bucket)
    except s3_client.exceptions.ClientError:
        s3_client.create_bucket(Bucket=bucket)
        print(f"Created S3 bucket '{bucket}'")


# ---------------------------------------------------------------------------
# Converter subprocess — owns the heavy Docling models
# ---------------------------------------------------------------------------


def _converter_worker(
    req_q: mp.Queue,
    res_q: mp.Queue,
    cpus_per_actor: int,
    chunk_max_tokens: int,
    tokenizer_name: str,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    s3_access_key: str,
    s3_secret_key: str,
):
    """Long-running subprocess: Docling parse -> HybridChunker -> write JSONL to S3."""
    os.environ["OMP_NUM_THREADS"] = str(cpus_per_actor)
    os.environ["MKL_NUM_THREADS"] = str(cpus_per_actor)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import boto3
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Initialise Docling converter
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=cpus_per_actor,
        device="cpu",
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Initialise HybridChunker
    chunker = HybridChunker(
        tokenizer=tokenizer_name,
        max_tokens=chunk_max_tokens,
    )

    # S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name="us-east-1",
    )

    # Signal parent that init is complete
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

            # Parse PDF with Docling
            stream = DocumentStream(name=fname, stream=io.BytesIO(file_bytes))
            result = converter.convert(stream)
            doc = result.document

            pages = getattr(doc, "pages", None)
            page_count = len(pages) if pages is not None else 0

            # Chunk with HybridChunker
            chunks = list(chunker.chunk(doc))

            if not chunks:
                res_q.put(("empty", file_path, page_count, 0, "No chunks produced"))
                continue

            # Build JSONL content
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

            # Upload JSONL to S3
            jsonl_body = "\n".join(lines) + "\n"
            s3_key = f"{s3_prefix}/{stem}.jsonl"
            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=jsonl_body.encode("utf-8"),
                ContentType="application/jsonl",
            )

            res_q.put(("success", file_path, page_count, len(lines), ""))

        except Exception as e:
            res_q.put(("error", file_path, 0, 0, str(e)[:200]))


# ---------------------------------------------------------------------------
# Ray Data actor
# ---------------------------------------------------------------------------


class DoclingChunkProcessor:
    """Actor that parses PDFs and chunks them, writing JSONL to S3."""

    def __init__(self):
        import socket

        self.hostname = socket.gethostname()
        self._start_worker()
        print(
            f"[{self.hostname}] DoclingChunkProcessor ready "
            f"(pid={self._worker.pid})"
        )

    def _start_worker(self):
        self._req_q = mp.Queue()
        self._res_q = mp.Queue()
        self._worker = mp.Process(
            target=_converter_worker,
            args=(
                self._req_q,
                self._res_q,
                CPUS_PER_ACTOR,
                CHUNK_MAX_TOKENS,
                TOKENIZER,
                S3_ENDPOINT,
                S3_BUCKET,
                S3_PREFIX,
                S3_ACCESS_KEY,
                S3_SECRET_KEY,
            ),
            daemon=True,
        )
        self._worker.start()
        msg = self._res_q.get(timeout=600)
        assert msg[0] == "ready"

    def _restart_worker(self):
        """Kill hung subprocess and start a fresh one."""
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

    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        path_list = batch["path"]

        out_source = []
        out_status = []
        out_pages = []
        out_chunks = []
        out_error = []
        out_duration = []
        out_host = []

        for file_path in path_list:
            fname = os.path.basename(file_path)
            t0 = time.time()

            self._req_q.put(str(file_path))

            try:
                result = self._res_q.get(timeout=FILE_TIMEOUT)
                status, _, page_count, chunk_count, error_msg = result
            except queue.Empty:
                status = "timeout"
                page_count = 0
                chunk_count = 0
                error_msg = f"Timed out after {FILE_TIMEOUT}s"
                self._restart_worker()

            elapsed = round(time.time() - t0, 3)

            out_source.append(fname)
            out_status.append(status)
            out_pages.append(page_count)
            out_chunks.append(chunk_count)
            out_error.append(error_msg)
            out_duration.append(elapsed)
            out_host.append(self.hostname)

        return {
            "source_file": out_source,
            "status": out_status,
            "page_count": out_pages,
            "chunk_count": out_chunks,
            "error": out_error,
            "duration_s": out_duration,
            "actor_host": out_host,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run():
    input_full_path = os.path.join(PVC_MOUNT_PATH, INPUT_PATH)

    pdf_paths = sorted(glob.glob(f"{input_full_path}/**/*.pdf", recursive=True))
    if NUM_FILES > 0:
        pdf_paths = pdf_paths[:NUM_FILES]

    print(f"Found {len(pdf_paths)} PDFs to process.")
    if not pdf_paths:
        print("No PDFs found. Exiting.")
        return

    # Ensure S3 bucket exists
    s3 = _get_s3_client()
    _ensure_bucket(s3, S3_BUCKET)
    print(f"Output: s3://{S3_BUCKET}/{S3_PREFIX}/")

    target_blocks = MAX_ACTORS * REPARTITION_FACTOR
    ds = ray.data.from_pandas(pd.DataFrame({"path": pdf_paths}))
    ds = ds.repartition(target_blocks)
    print(
        f"Repartitioned into {target_blocks} blocks for {MAX_ACTORS} max actors."
    )

    results_ds = ds.map_batches(
        DoclingChunkProcessor,
        compute=ray.data.ActorPoolStrategy(
            min_size=MIN_ACTORS,
            max_size=MAX_ACTORS,
        ),
        batch_size=BATCH_SIZE,
        batch_format="numpy",
        num_cpus=CPUS_PER_ACTOR,
    )

    # Collect results and print report
    start_time = time.time()
    success_count = error_count = timeout_count = empty_count = 0
    total_pages = 0
    total_chunks = 0
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

    # Count output objects in S3
    s3_objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/")
    s3_count = s3_objects.get("KeyCount", 0)

    print("\n" + "=" * 60)
    print("PARSE & CHUNK REPORT")
    print("=" * 60)
    print(f"Actors:            {MIN_ACTORS}..{MAX_ACTORS} (CPUs/actor: {CPUS_PER_ACTOR})")
    print(f"Output:            s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"\n--- Results ---")
    print(f"Total files:       {total_files}")
    print(f"Success:           {success_count}")
    print(f"Empty:             {empty_count}")
    print(f"Errors:            {error_count}")
    print(f"Timeouts:          {timeout_count}")
    print(f"Total pages:       {total_pages}")
    print(f"Total chunks:      {total_chunks}")
    print(f"S3 objects:        {s3_count}")
    print(f"\n--- Throughput ---")
    print(f"Wall clock:        {wall_clock:.1f}s")
    if wall_clock > 0:
        print(f"Files/second:      {success_count / wall_clock:.2f}")
        print(f"Chunks/second:     {total_chunks / wall_clock:.2f}")
    print(f"\n--- Actor Distribution ---")
    for actor, count in sorted(actor_dist.items()):
        pct = count / total_files * 100 if total_files else 0.0
        print(f"  {actor}: {count} files ({pct:.1f}%)")
    if errors_list:
        print(f"\n--- Errors & Timeouts (first 10) ---")
        for fname, err in errors_list[:10]:
            print(f"  {fname}: {err[:80]}")
    print("=" * 60)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    ctx = ray.data.DataContext.get_current()
    ctx.max_errored_blocks = MAX_ERRORED_BLOCKS
    run()
