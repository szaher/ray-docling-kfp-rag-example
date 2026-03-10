"""Ray Data + Docling + Milvus: parse, chunk, embed, and ingest PDFs.

Processes PDF files using Docling in a Ray Data actor pool, chunks them
using Docling's HybridChunker, generates embeddings with
sentence-transformers, and inserts directly into Milvus.

No intermediate files are written — the PVC is read-only.

All parameters are read from environment variables.
"""

import glob
import io
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

# Milvus
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus-milvus.milvus.svc.cluster.local")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_DB = os.environ.get("MILVUS_DB", "default")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "rag_documents")

# Embedding
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

# Chunking
CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "256"))

# Milvus insert batch size (within each actor)
MILVUS_BATCH_SIZE = int(os.environ.get("MILVUS_BATCH_SIZE", "64"))


# ---------------------------------------------------------------------------
# Converter subprocess — owns the heavy Docling + embedding models
# ---------------------------------------------------------------------------


def _converter_worker(
    req_q: mp.Queue,
    res_q: mp.Queue,
    cpus_per_actor: int,
    chunk_max_tokens: int,
    embedding_model_name: str,
    milvus_uri: str,
    milvus_db: str,
    collection_name: str,
    milvus_batch_size: int,
):
    """Long-running subprocess: Docling parse → HybridChunker → embed → Milvus insert.

    Keeps all heavy objects (converter, tokenizer, embedding model, Milvus client)
    alive for the lifetime of the actor to avoid repeated initialisation.
    """
    os.environ["OMP_NUM_THREADS"] = str(cpus_per_actor)
    os.environ["MKL_NUM_THREADS"] = str(cpus_per_actor)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

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

    # Initialise HybridChunker with the embedding model's tokenizer
    chunker = HybridChunker(
        tokenizer=embedding_model_name,
        max_tokens=chunk_max_tokens,
    )

    # Initialise embedding model
    embed_model = SentenceTransformer(embedding_model_name)

    # Connect to Milvus
    milvus_client = MilvusClient(uri=milvus_uri, db_name=milvus_db)

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

            # Extract text from chunks
            chunk_texts = [chunk.text for chunk in chunks if chunk.text.strip()]

            if not chunk_texts:
                res_q.put(("empty", file_path, page_count, 0, "All chunks empty"))
                continue

            # Generate embeddings
            embeddings = embed_model.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            # Insert into Milvus in batches
            total_inserted = 0
            for i in range(0, len(chunk_texts), milvus_batch_size):
                batch_texts = chunk_texts[i : i + milvus_batch_size]
                batch_embeddings = embeddings[i : i + milvus_batch_size]

                data = [
                    {
                        "source_file": fname,
                        "chunk_index": i + j,
                        "text": text,
                        "embedding": emb,
                    }
                    for j, (text, emb) in enumerate(
                        zip(batch_texts, batch_embeddings)
                    )
                ]
                milvus_client.insert(
                    collection_name=collection_name, data=data
                )
                total_inserted += len(data)

            res_q.put(("success", file_path, page_count, total_inserted, ""))

        except Exception as e:
            res_q.put(("error", file_path, 0, 0, str(e)[:200]))


# ---------------------------------------------------------------------------
# Ray Data actor
# ---------------------------------------------------------------------------


class DoclingMilvusProcessor:
    """Actor that parses PDFs, chunks, embeds, and inserts into Milvus."""

    def __init__(self):
        import socket

        self.hostname = socket.gethostname()
        self._milvus_uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
        self._start_worker()
        print(
            f"[{self.hostname}] DoclingMilvusProcessor ready "
            f"(pid={self._worker.pid}, collection={COLLECTION_NAME})"
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
                EMBEDDING_MODEL,
                self._milvus_uri,
                MILVUS_DB,
                COLLECTION_NAME,
                MILVUS_BATCH_SIZE,
            ),
            daemon=True,
        )
        self._worker.start()
        # Wait for model loading (may take several minutes)
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
        out_chunks_inserted = []
        out_error = []
        out_duration = []
        out_host = []

        for file_path in path_list:
            fname = os.path.basename(file_path)
            t0 = time.time()

            self._req_q.put(str(file_path))

            try:
                result = self._res_q.get(timeout=FILE_TIMEOUT)
                status, _, page_count, chunks_inserted, error_msg = result
            except queue.Empty:
                status = "timeout"
                page_count = 0
                chunks_inserted = 0
                error_msg = f"Timed out after {FILE_TIMEOUT}s"
                self._restart_worker()

            elapsed = round(time.time() - t0, 3)

            out_source.append(fname)
            out_status.append(status)
            out_pages.append(page_count)
            out_chunks_inserted.append(chunks_inserted)
            out_error.append(error_msg)
            out_duration.append(elapsed)
            out_host.append(self.hostname)

        return {
            "source_file": out_source,
            "status": out_status,
            "page_count": out_pages,
            "chunks_inserted": out_chunks_inserted,
            "error": out_error,
            "duration_s": out_duration,
            "actor_host": out_host,
        }


# ---------------------------------------------------------------------------
# Milvus collection setup (runs once on the driver)
# ---------------------------------------------------------------------------


def setup_milvus_collection():
    """Create or recreate the Milvus collection with an IVF_FLAT index."""
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    client = MilvusClient(uri=uri, db_name=MILVUS_DB)

    if client.has_collection(COLLECTION_NAME):
        print(f"Dropping existing collection '{COLLECTION_NAME}'")
        client.drop_collection(COLLECTION_NAME)

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(
                name="source_file", dtype=DataType.VARCHAR, max_length=512
            ),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=EMBEDDING_DIM,
            ),
        ],
        description="RAG document chunks with embeddings",
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(
        collection_name=COLLECTION_NAME, index_params=index_params
    )

    print(f"Collection '{COLLECTION_NAME}' created (dim={EMBEDDING_DIM}).")
    return client


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

    # Create the Milvus collection on the driver before launching actors
    setup_milvus_collection()

    target_blocks = MAX_ACTORS * REPARTITION_FACTOR
    ds = ray.data.from_pandas(pd.DataFrame({"path": pdf_paths}))
    ds = ds.repartition(target_blocks)
    print(
        f"Repartitioned into {target_blocks} blocks for {MAX_ACTORS} max actors."
    )

    results_ds = ds.map_batches(
        DoclingMilvusProcessor,
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
            total_chunks += int(batch["chunks_inserted"][i])

            actor = str(batch["actor_host"][i])
            actor_dist[actor] = actor_dist.get(actor, 0) + 1

    wall_clock = time.time() - start_time
    total_files = success_count + error_count + timeout_count + empty_count

    # Load collection into memory for searching
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", db_name=MILVUS_DB
    )
    client.load_collection(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)

    print("\n" + "=" * 60)
    print("PROCESSING REPORT")
    print("=" * 60)
    print(f"Actors:            {MIN_ACTORS}..{MAX_ACTORS} (CPUs/actor: {CPUS_PER_ACTOR})")
    print(f"Embedding model:   {EMBEDDING_MODEL}")
    print(f"Collection:        {COLLECTION_NAME}")
    print(f"\n--- Results ---")
    print(f"Total files:       {total_files}")
    print(f"Success:           {success_count}")
    print(f"Empty:             {empty_count}")
    print(f"Errors:            {error_count}")
    print(f"Timeouts:          {timeout_count}")
    print(f"Total pages:       {total_pages}")
    print(f"Total chunks:      {total_chunks}")
    print(f"\n--- Throughput ---")
    print(f"Wall clock:        {wall_clock:.1f}s")
    if wall_clock > 0:
        print(f"Files/second:      {success_count / wall_clock:.2f}")
        print(f"Chunks/second:     {total_chunks / wall_clock:.2f}")
    print(f"\n--- Milvus ---")
    print(f"Collection stats:  {stats}")
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
