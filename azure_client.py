"""UHG / Azure OpenAI gateway client (matches your notebook import cell)."""

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import AzureOpenAI

from config import BASE_DIR, ENV_FILE

# Load UAIS_vars.env (or .env) before reading variables.
load_dotenv(ENV_FILE)
load_dotenv(BASE_DIR / ".env")

AZURE_OPENAI_ENDPOINT = os.getenv("MODEL_ENDPOINT", "")
OPENAI_API_VERSION = os.getenv("API_VERSION", "")
CHAT_DEPLOYMENT_NAME = os.getenv("CHAT_MODEL_NAME") or os.getenv("MODEL_NAME", "")
EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")

OAUTH_TOKEN_URL = os.getenv("OAUTH_TOKEN_URL", "https://api.uhg.com/oauth2/token")
OAUTH_SCOPE = os.getenv("OAUTH_SCOPE", "https://api.uhg.com/.default")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

_TOKEN: Optional[str] = None
_CHAT_CLIENT: Optional[AzureOpenAI] = None
_EMBED_CLIENT: Optional[AzureOpenAI] = None


def _default_headers() -> dict:
    if not PROJECT_ID:
        return {}
    return {"projectId": PROJECT_ID}


def get_access_token(force_refresh: bool = False) -> str:
    """OAuth2 client-credentials token (same flow as your Databricks notebook)."""
    global _TOKEN
    if _TOKEN and not force_refresh:
        return _TOKEN

    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(
            "CLIENT_ID and CLIENT_SECRET must be set in UAIS_vars.env for Azure mode."
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
    """Same pattern as your notebook get_response()."""
    client = get_chat_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=CHAT_DEPLOYMENT_NAME,
    )
    return response.choices[0].message.content or ""
