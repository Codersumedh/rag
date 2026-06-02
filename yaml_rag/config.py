"""Paths and settings for YAML → Chroma RAG (notebook parity)."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

# Same layout as notebook: ./Data/UAIS_vars.env
UAIS_ENV_FILE = Path(os.getenv("UAIS_ENV_FILE", str(DATA_DIR / "UAIS_vars.env")))
DATABRICKS_ENV_FILE = Path(
    os.getenv("DATABRICKS_ENV_FILE", str(DATA_DIR / "databricks.env"))
)

load_dotenv(UAIS_ENV_FILE)
load_dotenv(DATABRICKS_ENV_FILE)
load_dotenv(BASE_DIR / ".env")

YAML_PATH = Path(os.getenv("YAML_PATH", str(BASE_DIR / "data" / "mart_med_uhc_mthly.yaml")))
PERSIST_DIRECTORY = Path(
    os.getenv("PERSIST_DIRECTORY", str(BASE_DIR / "chroma_store"))
)
TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))

TIKTOKEN_CACHE_DIR = BASE_DIR / "setup" / "tiktoken_cache"

# UHG policy — same as notebook
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
