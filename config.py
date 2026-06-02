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
# Fast + small sentence transformer for embeddings (Hub id).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# If Hugging Face download fails (corporate SSL), download the model once on a
# machine with access, copy the folder here, and set this path.
# Example folder layout: models/all-MiniLM-L6-v2/config.json, pytorch_model.bin, ...
LOCAL_EMBED_MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"

# Corporate SSL: point to your company root CA .pem file (optional).
# Example: r"C:\certs\company-root-ca.pem"
SSL_CERT_FILE = ""

# Set True only after the model files exist locally (skips Hub download).
HF_HUB_OFFLINE = False

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
