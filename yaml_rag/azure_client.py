"""Azure OpenAI gateway + OAuth (matches Databricks notebook cells)."""

import os
from typing import Optional

import httpx
from openai import AzureOpenAI

from config import DATABRICKS_ENV_FILE, UAIS_ENV_FILE
from dotenv import load_dotenv

load_dotenv(UAIS_ENV_FILE)
load_dotenv(DATABRICKS_ENV_FILE)

AZURE_OPENAI_ENDPOINT = os.getenv("MODEL_ENDPOINT", "")
OPENAI_API_VERSION = os.getenv("API_VERSION", "")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_MODEL_NAME") or os.getenv("MODEL_NAME", "")
EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")

OAUTH_TOKEN_URL = os.getenv("OAUTH_TOKEN_URL", "https://api.uhg.com/oauth2/token")
OAUTH_SCOPE = os.getenv("OAUTH_SCOPE", "https://api.uhg.com/.default")


def _secret_or_env(key: str, env_name: str) -> str:
    """Notebook: dbutils.secrets.get; locally: Data/databricks.env or UAIS_vars.env."""
    try:
        from pyspark.dbutils import DBUtils  # type: ignore
        from pyspark.sql import SparkSession

        dbutils = DBUtils(SparkSession.builder.getOrCreate())
        scope = os.getenv("DATABRICKS_SECRET_SCOPE", "AIML_Training")
        return dbutils.secrets.get(scope=scope, key=key)
    except Exception:
        return os.getenv(env_name, "")


CLIENT_ID = _secret_or_env("client_id", "CLIENT_ID")
CLIENT_SECRET = _secret_or_env("client_secret", "CLIENT_SECRET")

_TOKEN: Optional[str] = None
_CHAT_CLIENT: Optional[AzureOpenAI] = None
_EMBED_CLIENT: Optional[AzureOpenAI] = None


def _default_headers() -> dict:
    if not PROJECT_ID:
        return {}
    return {"projectId": PROJECT_ID}


def get_access_token(force_refresh: bool = False) -> str:
    global _TOKEN
    if _TOKEN and not force_refresh:
        return _TOKEN

    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(
            "Set CLIENT_ID and CLIENT_SECRET in Data/databricks.env "
            "(or UAIS_vars.env). In Databricks use AIML_Training secrets."
        )

    body = {
        "grant_type": "client_credentials",
        "scope": OAUTH_SCOPE,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    with httpx.Client(timeout=120) as client:
        resp = client.post(OAUTH_TOKEN_URL, headers=headers, data=body)
        resp.raise_for_status()
        _TOKEN = resp.json()["access_token"]
    return _TOKEN


def get_chat_client() -> AzureOpenAI:
    global _CHAT_CLIENT
    if _CHAT_CLIENT is None:
        if not AZURE_OPENAI_ENDPOINT or not CHAT_DEPLOYMENT_NAME:
            raise ValueError("MODEL_ENDPOINT and CHAT_MODEL_NAME/MODEL_NAME are required.")
        _CHAT_CLIENT = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=OPENAI_API_VERSION,
            azure_ad_token=get_access_token(),
            default_headers=_default_headers(),
        )
    return _CHAT_CLIENT


def get_embeddings_client() -> AzureOpenAI:
    global _EMBED_CLIENT
    if _EMBED_CLIENT is None:
        if not AZURE_OPENAI_ENDPOINT or not EMBEDDINGS_DEPLOYMENT_NAME:
            raise ValueError("MODEL_ENDPOINT and EMBEDDINGS_MODEL_NAME are required.")
        _EMBED_CLIENT = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=OPENAI_API_VERSION,
            azure_ad_token=get_access_token(),
            default_headers=_default_headers(),
        )
    return _EMBED_CLIENT


def get_response(prompt: str) -> str:
    """Same as notebook get_response() — for generation step later."""
    client = get_chat_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=CHAT_DEPLOYMENT_NAME,
    )
    return response.choices[0].message.content or ""
