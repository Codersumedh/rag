"""Retrieve schema context from ChromaDB for a user question."""

import chromadb

from config import CHROMA_DB_DIR, CHROMA_COLLECTION, TOP_K
from embed_utils import get_embedding_function


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embed_fn = get_embedding_function()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_schema_context(question: str, top_k: int = TOP_K) -> str:
    """Return a compact context string with top-k similar schema chunks."""
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=top_k)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    if not docs:
        return "No schema context found."

    lines = []
    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = f"{meta.get('fqn', '')}"
        lines.append(f"[{idx}] {doc} | source={src}")
    return "\n".join(lines)
