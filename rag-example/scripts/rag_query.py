"""RAG query script.

Searches Milvus for relevant document chunks, then sends them
as context to a vLLM model endpoint for answer generation.

All parameters are read from environment variables.
"""

import os

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MILVUS_HOST = os.environ.get("MILVUS_HOST", "milvus-milvus.milvus.svc.cluster.local")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_DB = os.environ.get("MILVUS_DB", "default")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "rag_documents")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

INFERENCE_URL = os.environ.get("INFERENCE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
TOP_K = int(os.environ.get("TOP_K", "5"))


def search_milvus(query: str, client, model, top_k: int = TOP_K):
    """Search Milvus for chunks similar to the query."""
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_embedding,
        limit=top_k,
        output_fields=["source_file", "chunk_index", "text"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
    )

    contexts = []
    for hits in results:
        for hit in hits:
            contexts.append(
                {
                    "text": hit["entity"]["text"],
                    "source_file": hit["entity"]["source_file"],
                    "chunk_index": hit["entity"]["chunk_index"],
                    "score": hit["distance"],
                }
            )
    return contexts


def build_prompt(query: str, contexts: list) -> str:
    """Build a RAG prompt with retrieved context."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source_file']}, chunk {c['chunk_index']}, "
        f"score: {c['score']:.3f}]\n{c['text']}"
        for c in contexts
    )

    return (
        "You are a helpful assistant. Answer the user's question based on the "
        "provided context documents. If the answer is not in the context, say so.\n\n"
        f"## Context\n\n{context_text}\n\n"
        f"## Question\n\n{query}\n\n"
        "## Answer\n\n"
    )


def query_llm(prompt: str) -> str:
    """Send the prompt to the vLLM inference endpoint."""
    from openai import OpenAI

    client = OpenAI(
        base_url=INFERENCE_URL,
        api_key="unused",  # vLLM doesn't require a real key
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )

    return response.choices[0].message.content


def rag_query(query: str) -> dict:
    """End-to-end RAG: retrieve context from Milvus, generate answer via LLM."""
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
    milvus_client = MilvusClient(uri=uri, db_name=MILVUS_DB)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    contexts = search_milvus(query, milvus_client, embed_model)
    prompt = build_prompt(query, contexts)
    answer = query_llm(prompt)

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag_query.py 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = rag_query(question)

    print(f"\nQuestion: {result['query']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources ({len(result['contexts'])}):")
    for c in result["contexts"]:
        print(f"  - {c['source_file']} (chunk {c['chunk_index']}, score {c['score']:.3f})")
