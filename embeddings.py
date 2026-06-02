"""Build the ChromaDB vector store from the table schema YAML.

Run this once (or whenever schema.yaml changes):

    python embeddings.py

It turns every table and every column into a small text chunk, embeds it
with a fast sentence-transformer, and stores it in a persistent ChromaDB.
"""

import yaml
import chromadb
from chromadb.utils import embedding_functions

from config import (
    SCHEMA_YAML_PATH,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL,
)


def load_schema_chunks():
    """Read schema.yaml and flatten it into (id, text, metadata) chunks."""
    with open(SCHEMA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chunks = []
    for table in data.get("tables", []):
        name = table["name"]
        db = table.get("database", "")
        schema = table.get("schema", "")
        fqn = ".".join(p for p in (db, schema, name) if p)
        table_desc = (table.get("description") or "").strip()

        # One chunk describing the whole table.
        chunks.append(
            {
                "id": f"table::{name}",
                "text": f"Table {fqn}: {table_desc}",
                "metadata": {"type": "table", "table": name, "fqn": fqn},
            }
        )

        # One chunk per column so similarity search can find relevant fields.
        for col in table.get("columns", []):
            col_name = col["name"]
            col_type = col.get("type", "")
            col_desc = (col.get("description") or "").strip()
            text = (
                f"Table {fqn} column {col_name} ({col_type}): {col_desc}"
            )
            chunks.append(
                {
                    "id": f"col::{name}::{col_name}",
                    "text": text,
                    "metadata": {
                        "type": "column",
                        "table": name,
                        "fqn": fqn,
                        "column": col_name,
                        "data_type": col_type,
                    },
                }
            )
    return chunks


def build_store():
    """Embed all chunks and persist them in ChromaDB (rebuilds from scratch)."""
    chunks = load_schema_chunks()
    print(f"Loaded {len(chunks)} schema chunks from {SCHEMA_YAML_PATH.name}")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    # Recreate the collection so re-runs stay clean.
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embed_fn,
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
