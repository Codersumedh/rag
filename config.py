"""Central configuration for the RAG -> SQL -> Snowflake pipeline.

Everything is kept simple here so you only edit one file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SCHEMA_YAML_PATH = BASE_DIR / "schema.yaml"      # your table definition
CHROMA_DB_DIR = BASE_DIR / "chroma_store"        # where embeddings are persisted
CHROMA_COLLECTION = "table_schema"

# ---------------------------------------------------------------------------
# Models (all free, run locally, no API keys)
# ---------------------------------------------------------------------------
# Fast + small sentence transformer for embeddings.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Local Hugging Face instruct model used to (1) write SQL and (2) explain results.
# Qwen2.5-Coder-1.5B is small enough to run on CPU and is good at SQL.
# You can swap this for any other instruct model from the Hub.
LLM_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# How many schema chunks to retrieve from ChromaDB for each query.
TOP_K = 8

# ---------------------------------------------------------------------------
# Snowflake connection (matches the connect method you shared)
# ---------------------------------------------------------------------------
SNOWFLAKE_CONN = {
    "host": "abc111.east-us-2.azure.snowflakecomputing.com",
    "user": "xyz@abc.com",
    "account": "DWAAS",
    "role": "AR_PRD_ABC_ROLE",
    "warehouse": "OHBI_PRD_CONSUME_FREQ_WH",
    "database": "OHBI_PRD_MART_DB",
    "schema": "CORE_ONC",
    "authenticator": "externalbrowser",
}
