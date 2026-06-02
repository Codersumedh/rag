"""Build the ChromaDB vector store from the table schema YAML.

Run this once (or whenever schema.yaml changes):

    python embeddings.py
"""

import chromadb

from config import CHROMA_DB_DIR, CHROMA_COLLECTION
from embed_utils import get_embedding_function
from schema_loader import load_schema_chunks


def build_store():
    """Embed all chunks and persist them in ChromaDB (rebuilds from scratch)."""
    chunks = load_schema_chunks()
    print(f"Loaded {len(chunks)} schema chunks")

    if not chunks:
        raise ValueError(
            "No schema chunks found. Check schema.yaml has tables with "
            "dimensions/measures or columns."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"Stored {collection.count()} embeddings in {CHROMA_DB_DIR}")


if __name__ == "__main__":
    build_store()
