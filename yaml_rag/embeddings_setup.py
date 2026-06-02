"""LangChain AzureOpenAIEmbeddings + tiktoken cache (notebook cells 20–21)."""

import os

from langchain_openai import AzureOpenAIEmbeddings

from azure_client import (
    AZURE_OPENAI_ENDPOINT,
    EMBEDDINGS_DEPLOYMENT_NAME,
    OPENAI_API_VERSION,
    PROJECT_ID,
    get_access_token,
)
from config import TIKTOKEN_CACHE_DIR


def setup_tiktoken_cache() -> None:
    cache_dir = os.path.abspath(str(TIKTOKEN_CACHE_DIR))
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir


def get_langchain_embeddings() -> AzureOpenAIEmbeddings:
    setup_tiktoken_cache()
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    kwargs = {
        "azure_endpoint": AZURE_OPENAI_ENDPOINT,
        "api_version": OPENAI_API_VERSION,
        "azure_deployment": EMBEDDINGS_DEPLOYMENT_NAME,
        "azure_ad_token": get_access_token(),
    }
    if PROJECT_ID:
        kwargs["default_headers"] = {"projectId": PROJECT_ID}
    return AzureOpenAIEmbeddings(**kwargs)
