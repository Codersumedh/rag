"""Build the ChromaDB vector store from the table schema YAML.

Run this once (or whenever schema.yaml changes):

    python embeddings.py

It turns every table and every column into a small text chunk, embeds it
with a fast sentence-transformer, and stores it in a persistent ChromaDB.
"""

import yaml
import chromadb

from config import SCHEMA_YAML_PATH, CHROMA_DB_DIR, CHROMA_COLLECTION
from embed_utils import get_embedding_function


def load_schema_chunks():
    """Read schema.yaml and flatten it into (id, text, metadata) chunks."""
    with open(SCHEMA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chunks = []

    # Format 1: flat table list (tables -> columns)
    if (
        data.get("tables")
        and isinstance(data.get("tables"), list)
        and any(isinstance(t, dict) and t.get("columns") for t in data.get("tables", []))
    ):
        for table in data.get("tables", []):
            name = table.get("name") or table.get("table")
            if not name:
                continue
            db = table.get("database", "")
            schema = table.get("schema", "")
            fqn = ".".join(p for p in (db, schema, name) if p)
            table_desc = (table.get("description") or "").strip()

            chunks.append(
                {
                    "id": f"table::{name}",
                    "text": f"Table {fqn}: {table_desc}",
                    "metadata": {"type": "table", "table": name, "fqn": fqn},
                }
            )

            for col in table.get("columns", []):
                col_name = col.get("name")
                if not col_name:
                    continue
                col_type = col.get("type", "")
                col_desc = (col.get("description") or "").strip()
                text = f"Table {fqn} column {col_name} ({col_type}): {col_desc}"
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

    # Format 2: semantic model style (dimensions/measures and tables with table key)
    model_name = data.get("name", "")
    model_desc = (data.get("description") or "").strip()
    table_entries = data.get("tables", []) if isinstance(data.get("tables"), list) else []
    dimensions = data.get("dimensions", []) if isinstance(data.get("dimensions"), list) else []
    measures = data.get("measures", []) if isinstance(data.get("measures"), list) else []

    # Format 2b: semantic model with tables[].base_table + nested dimensions/measures
    for t in table_entries:
        if not isinstance(t, dict):
            continue
        base = t.get("base_table") or {}
        table_name = base.get("table") or t.get("table") or t.get("name")
        if not table_name:
            continue
        db = base.get("database") or t.get("database", "")
        schema = base.get("schema") or t.get("schema", "")
        fqn = ".".join(p for p in (db, schema, table_name) if p)
        t_dims = t.get("dimensions", []) if isinstance(t.get("dimensions"), list) else []
        t_measures = t.get("measures", []) if isinstance(t.get("measures"), list) else []
        if not t_dims and not t_measures:
            continue

        t_desc = (t.get("description") or "").strip()
        full_desc = " ".join(p for p in (model_desc, t_desc) if p).strip()
        chunks.append(
            {
                "id": f"table::{table_name}",
                "text": f"Semantic model {model_name}. Table {fqn}: {full_desc}",
                "metadata": {"type": "table", "table": table_name, "fqn": fqn},
            }
        )

        for d in t_dims:
            d_name = d.get("name")
            if not d_name:
                continue
            d_type = d.get("data_type", "")
            d_desc = (d.get("description") or "").strip()
            d_expr = (d.get("expr") or "").strip()
            synonyms = ", ".join(d.get("synonyms", [])) if isinstance(d.get("synonyms"), list) else ""
            text = (
                f"Table {fqn} dimension {d_name} ({d_type}). "
                f"description: {d_desc}. expr: {d_expr}. synonyms: {synonyms}"
            )
            chunks.append(
                {
                    "id": f"dim::{table_name}::{d_name}",
                    "text": text,
                    "metadata": {
                        "type": "dimension",
                        "table": table_name,
                        "fqn": fqn,
                        "column": d_expr or d_name,
                        "semantic_name": d_name,
                        "data_type": d_type,
                    },
                }
            )

        for m in t_measures:
            m_name = m.get("name")
            if not m_name:
                continue
            m_type = m.get("data_type", "")
            m_desc = (m.get("description") or "").strip()
            m_expr = (m.get("expr") or "").strip()
            synonyms = ", ".join(m.get("synonyms", [])) if isinstance(m.get("synonyms"), list) else ""
            text = (
                f"Table {fqn} measure {m_name} ({m_type}). "
                f"description: {m_desc}. expr: {m_expr}. synonyms: {synonyms}"
            )
            chunks.append(
                {
                    "id": f"measure::{table_name}::{m_name}",
                    "text": text,
                    "metadata": {
                        "type": "measure",
                        "table": table_name,
                        "fqn": fqn,
                        "column": m_expr or m_name,
                        "semantic_name": m_name,
                        "data_type": m_type,
                    },
                }
            )

    # Format 2a: root-level dimensions/measures (legacy)
    if table_entries and (dimensions or measures):
        for t in table_entries:
            table_name = t.get("table") or t.get("name")
            if not table_name:
                continue
            db = t.get("database", "")
            schema = t.get("schema", "")
            fqn = ".".join(p for p in (db, schema, table_name) if p)
            t_desc = (t.get("description") or "").strip()
            full_desc = " ".join(p for p in (model_desc, t_desc) if p).strip()
            chunks.append(
                {
                    "id": f"table::{table_name}",
                    "text": f"Semantic model {model_name}. Table {fqn}: {full_desc}",
                    "metadata": {"type": "table", "table": table_name, "fqn": fqn},
                }
            )

            for d in dimensions:
                d_name = d.get("name")
                if not d_name:
                    continue
                d_type = d.get("data_type", "")
                d_desc = (d.get("description") or "").strip()
                d_expr = (d.get("expr") or "").strip()
                synonyms = ", ".join(d.get("synonyms", [])) if isinstance(d.get("synonyms"), list) else ""
                text = (
                    f"Table {fqn} dimension {d_name} ({d_type}). "
                    f"description: {d_desc}. expr: {d_expr}. synonyms: {synonyms}"
                )
                chunks.append(
                    {
                        "id": f"dim::{table_name}::{d_name}",
                        "text": text,
                        "metadata": {
                            "type": "dimension",
                            "table": table_name,
                            "fqn": fqn,
                            "column": d_expr or d_name,
                            "semantic_name": d_name,
                            "data_type": d_type,
                        },
                    }
                )

            for m in measures:
                m_name = m.get("name")
                if not m_name:
                    continue
                m_type = m.get("data_type", "")
                m_desc = (m.get("description") or "").strip()
                m_expr = (m.get("expr") or "").strip()
                synonyms = ", ".join(m.get("synonyms", [])) if isinstance(m.get("synonyms"), list) else ""
                text = (
                    f"Table {fqn} measure {m_name} ({m_type}). "
                    f"description: {m_desc}. expr: {m_expr}. synonyms: {synonyms}"
                )
                chunks.append(
                    {
                        "id": f"measure::{table_name}::{m_name}",
                        "text": text,
                        "metadata": {
                            "type": "measure",
                            "table": table_name,
                            "fqn": fqn,
                            "column": m_expr or m_name,
                            "semantic_name": m_name,
                            "data_type": m_type,
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

    embed_fn = get_embedding_function()
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
