"""Parse schema.yaml into embedding chunks (no ChromaDB dependency)."""

import yaml

from config import SCHEMA_YAML_PATH


def _synonyms_text(field: dict) -> str:
    syns = field.get("synonyms", [])
    if isinstance(syns, list):
        return ", ".join(str(s) for s in syns)
    return ""


def load_schema_chunks():
    """Read schema.yaml and flatten it into (id, text, metadata) chunks."""
    with open(SCHEMA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chunks = []
    model_name = data.get("name", "")
    model_desc = (data.get("description") or "").strip()

    if model_name or model_desc:
        chunks.append(
            {
                "id": f"model::{model_name or 'semantic_model'}",
                "text": f"Semantic model {model_name}: {model_desc}",
                "metadata": {"type": "model", "model": model_name},
            }
        )

    table_entries = data.get("tables", []) if isinstance(data.get("tables"), list) else []

    # Format 1: tables -> columns (simple)
    if any(isinstance(t, dict) and t.get("columns") for t in table_entries):
        for table in table_entries:
            if not isinstance(table, dict) or not table.get("columns"):
                continue
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

    # Format 2: tables[].base_table + nested dimensions/measures (your schema.yaml)
    for t in table_entries:
        if not isinstance(t, dict):
            continue
        base = t.get("base_table") or {}
        physical_table = base.get("table") or t.get("table")
        logical_name = t.get("name", "")
        table_name = physical_table or logical_name
        if not table_name:
            continue

        db = base.get("database") or t.get("database", "")
        schema_name = base.get("schema") or t.get("schema", "")
        fqn = ".".join(p for p in (db, schema_name, physical_table or table_name) if p)

        t_dims = t.get("dimensions", []) if isinstance(t.get("dimensions"), list) else []
        t_measures = t.get("measures", []) if isinstance(t.get("measures"), list) else []
        if not t_dims and not t_measures and not base:
            continue

        t_desc = (t.get("description") or "").strip()
        full_desc = " ".join(p for p in (model_desc, t_desc) if p).strip()
        alias = f" (alias: {logical_name})" if logical_name and logical_name != table_name else ""
        chunks.append(
            {
                "id": f"table::{table_name}",
                "text": (
                    f"Semantic model {model_name}. Table {fqn}{alias}: {full_desc}. "
                    f"Use SQL table {fqn}."
                ),
                "metadata": {
                    "type": "table",
                    "table": table_name,
                    "fqn": fqn,
                    "logical_name": logical_name,
                },
            }
        )

        for d in t_dims:
            d_name = d.get("name")
            if not d_name:
                continue
            d_type = d.get("data_type", "")
            d_desc = (d.get("description") or "").strip()
            d_expr = (d.get("expr") or d_name).strip()
            synonyms = _synonyms_text(d)
            text = (
                f"Table {fqn} dimension {d_name} ({d_type}). "
                f"description: {d_desc}. SQL column expr: {d_expr}. synonyms: {synonyms}"
            )
            chunks.append(
                {
                    "id": f"dim::{table_name}::{d_name}",
                    "text": text,
                    "metadata": {
                        "type": "dimension",
                        "table": table_name,
                        "fqn": fqn,
                        "column": d_expr,
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
            m_expr = (m.get("expr") or m_name).strip()
            synonyms = _synonyms_text(m)
            text = (
                f"Table {fqn} measure {m_name} ({m_type}). "
                f"description: {m_desc}. SQL expr: {m_expr}. synonyms: {synonyms}"
            )
            chunks.append(
                {
                    "id": f"measure::{table_name}::{m_name}",
                    "text": text,
                    "metadata": {
                        "type": "measure",
                        "table": table_name,
                        "fqn": fqn,
                        "column": m_expr,
                        "semantic_name": m_name,
                        "data_type": m_type,
                    },
                }
            )

    return chunks
