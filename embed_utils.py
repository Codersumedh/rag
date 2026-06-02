"""Embedding backend: Azure OpenAI (corporate) or local sentence-transformer."""

import os
from pathlib import Path

from chromadb.utils import embedding_functions

from config import (
    EMBED_MODEL,
    HF_HUB_OFFLINE,
    LOCAL_EMBED_MODEL_PATH,
    SSL_CERT_FILE,
    USE_AZURE,
)


def _apply_ssl_and_hub_settings():
    if SSL_CERT_FILE:
        cert = str(Path(SSL_CERT_FILE).expanduser().resolve())
        os.environ["SSL_CERT_FILE"] = cert
        os.environ["REQUESTS_CA_BUNDLE"] = cert
    if HF_HUB_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"


def resolve_embed_model_path() -> str:
    _apply_ssl_and_hub_settings()
    if LOCAL_EMBED_MODEL_PATH:
        local = Path(LOCAL_EMBED_MODEL_PATH).expanduser().resolve()
        if local.exists():
            print(f"Using local embedding model: {local}")
            return str(local)
    return EMBED_MODEL


def get_embedding_function():
    if USE_AZURE:
        from azure_embed import AzureOpenAIEmbeddingFunction

        print("Using Azure OpenAI embeddings (UHG gateway)")
        return AzureOpenAIEmbeddingFunction()

    print("Using local sentence-transformer embeddings")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=resolve_embed_model_path()
    )
