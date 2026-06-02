"""Shared embedding setup (local model + corporate SSL fixes)."""

import os
from pathlib import Path

from chromadb.utils import embedding_functions

from config import (
    EMBED_MODEL,
    LOCAL_EMBED_MODEL_PATH,
    SSL_CERT_FILE,
    HF_HUB_OFFLINE,
)


def _apply_ssl_and_hub_settings():
    """Apply cert bundle / offline flags before any Hugging Face download."""
    if SSL_CERT_FILE:
        cert = str(Path(SSL_CERT_FILE).expanduser().resolve())
        os.environ["SSL_CERT_FILE"] = cert
        os.environ["REQUESTS_CA_BUNDLE"] = cert
    if HF_HUB_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"


def resolve_embed_model_path() -> str:
    """Return local folder path if configured, else Hub model id."""
    _apply_ssl_and_hub_settings()
    if LOCAL_EMBED_MODEL_PATH:
        local = Path(LOCAL_EMBED_MODEL_PATH).expanduser().resolve()
        if local.exists():
            print(f"Using local embedding model: {local}")
            return str(local)
        print(f"Warning: LOCAL_EMBED_MODEL_PATH not found: {local}")
    return EMBED_MODEL


def get_embedding_function():
    """Chroma-compatible embedding function (cosine-ready sentence-transformer)."""
    model = resolve_embed_model_path()
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model
    )
