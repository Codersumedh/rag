"""Central configuration for the RAG -> SQL -> Snowflake pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

# Load UAIS_vars.env first (same as your notebook: load_dotenv("./Data/UAIS_vars.env"))
ENV_FILE = Path(os.getenv("UAIS_ENV_FILE", str(BASE_DIR / "UAIS_vars.env")))
load_dotenv(ENV_FILE)
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Backend: Azure (UHG gateway) vs local Hugging Face
# ---------------------------------------------------------------------------
USE_AZURE = os.getenv("USE_AZURE", "false").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCHEMA_YAML_PATH = BASE_DIR / "schema.yaml"
CHROMA_DB_DIR = BASE_DIR / "chroma_store"
CHROMA_COLLECTION = "table_schema"
TOP_K = 8

# Disable Chroma telemetry (UHG policy — same as your notebook)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ---------------------------------------------------------------------------
# Local models (used when USE_AZURE=false)
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBED_MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", "")
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "false").strip().lower() in ("1", "true", "yes")
LLM_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------
SNOWFLAKE_CONN = {
    "host": os.getenv("SNOWFLAKE_HOST", "abc111.east-us-2.azure.snowflakecomputing.com"),
    "user": os.getenv("SNOWFLAKE_USER", "xyz@abc.com"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT", "DWAAS"),
    "role": os.getenv("SNOWFLAKE_ROLE", "AR_PRD_ABC_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "OHBI_PRD_CONSUME_FREQ_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "OHBI_PRD_MART_DB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "CORE_ONC"),
    "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser"),
}
