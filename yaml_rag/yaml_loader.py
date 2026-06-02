"""Parse semantic-model YAML into chunk dicts (page_content + metadata)."""

from pathlib import Path
from typing import Any

import yaml

from config import YAML_PATH

Chunk = dict[str, Any]


def _synonyms_text(field: dict) -> str:
    syns = field.get("synonyms", [])
    if isinstance(syns, list):
        return ", ".join(str(s) for s in syns)
    return ""


def load_yaml_chunks(yaml_path: Path | None = None) -> list[Chunk]:
    path = Path(yaml_path or YAML_PATH)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    docs: list[Chunk] = []
    model_name = data.get("name", "")
    model_desc = (data.get("description") or "").strip()

    if model_name or model_desc:
        docs.append(
            {
                "page_content": f"Semantic model {model_name}: {model_desc}",
                "metadata": {"type": "model", "model": model_name},
            }
        )

    table_entries = data.get("tables", []) if isinstance(data.get("tables"), list) else []

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
        t_desc = (t.get("description") or "").strip()
        full_desc = " ".join(p for p in (model_desc, t_desc) if p).strip()
        alias = f" (alias: {logical_name})" if logical_name and logical_name != table_name else ""

        docs.append(
            {
                "page_content": (
                    f"Semantic model {model_name}. Table {fqn}{alias}: {full_desc}. "
                    f"Use SQL table {fqn}."
                ),
                "metadata": {"type": "table", "table": table_name, "fqn": fqn},
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
            docs.append(
                {
                    "page_content": (
                        f"Table {fqn} dimension {d_name} ({d_type}). "
                        f"description: {d_desc}. SQL column expr: {d_expr}. "
                        f"synonyms: {synonyms}"
                    ),
                    "metadata": {
                        "type": "dimension",
                        "table": table_name,
                        "fqn": fqn,
                        "semantic_name": d_name,
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
            docs.append(
                {
                    "page_content": (
                        f"Table {fqn} measure {m_name} ({m_type}). "
                        f"description: {m_desc}. SQL expr: {m_expr}. synonyms: {synonyms}"
                    ),
                    "metadata": {
                        "type": "measure",
                        "table": table_name,
                        "fqn": fqn,
                        "semantic_name": m_name,
                    },
                }
            )

        for col in t.get("columns", []) or []:
            col_name = col.get("name")
            if not col_name:
                continue
            col_type = col.get("type", "")
            col_desc = (col.get("description") or "").strip()
            docs.append(
                {
                    "page_content": f"Table {fqn} column {col_name} ({col_type}): {col_desc}",
                    "metadata": {"type": "column", "table": table_name, "fqn": fqn},
                }
            )

    return docs


def load_yaml_documents(yaml_path: Path | None = None):
    """LangChain Document list for Chroma.from_documents."""
    from langchain_core.documents import Document

    return [Document(**c) for c in load_yaml_chunks(yaml_path)]
