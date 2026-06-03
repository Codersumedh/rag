import os
import yaml
import json
import shutil
import warnings
from dotenv import load_dotenv
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import httpx

warnings.filterwarnings("ignore")

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(".env")

AZURE_OPENAI_ENDPOINT       = os.environ["MODEL_ENDPOINT"]
OPENAI_API_VERSION          = os.environ["API_VERSION"]
EMBEDDINGS_DEPLOYMENT_NAME  = os.environ["EMBEDDINGS_MODEL_NAME"]
PROJECT_ID                  = os.environ["PROJECT_ID"]
CLIENT_ID                   = os.environ["CLIENT_ID"]
CLIENT_SECRET               = os.environ["CLIENT_SECRET"]

# ── OAuth2 token (same pattern as notebook cell 8) ────────────────────────────
import asyncio

async def get_token():
    auth         = "https://api.uhg.com/oauth2/token"
    scope        = "https://api.uhg.com/.default"
    grant_type   = "client_credentials"

    async with httpx.AsyncClient() as client:
        body = {
            "grant_type":    grant_type,
            "scope":         scope,
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp    = await client.post(auth, headers=headers, data=body, timeout=120)
        return resp.json()["access_token"]

token = asyncio.run(get_token())

# ── Embeddings client (same pattern as notebook cell 9) ───────────────────────
embeddings_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    azure_ad_token=token,
    default_headers={
        "projectId": PROJECT_ID
    }
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=OPENAI_API_VERSION,
    azure_ad_token=token,
    default_headers={
        "projectId": PROJECT_ID
    }
)

# ── Parse YAML into chunks ────────────────────────────────────────────────────
YAML_FILE        = "uhc_mthly_medical_report.yaml"
PERSIST_DIR      = "/tmp/vector_embeddings_YAML"

def yaml_to_documents(yaml_path: str) -> list[Document]:
    """
    Flatten a semantic-model YAML into one Document per field/dimension/measure/filter.
    Each chunk contains the table context + field details so cosine similarity
    works on meaningful, self-contained text.
    """
    with open(yaml_path, "r") as f:
        model = yaml.safe_load(f)

    model_name = model.get("name", "")
    model_desc = model.get("description", "")
    docs: list[Document] = []

    for table in model.get("tables", []):
        table_name = table.get("name", "")
        table_desc = table.get("description", "")
        base        = table.get("base_table", {})
        full_table  = f"{base.get('database','')}.{base.get('schema','')}.{base.get('table','')}"

        def make_doc(field_type: str, field: dict) -> Document:
            fname    = field.get("name", "")
            fdesc    = field.get("description", "")
            fexpr    = field.get("expr", "")
            fdtype   = field.get("data_type", "")
            synonyms = ", ".join(field.get("synonyms", []))

            text = (
                f"Model: {model_name}\n"
                f"Model description: {model_desc}\n"
                f"Table: {table_name} ({full_table})\n"
                f"Table description: {table_desc}\n"
                f"Field type: {field_type}\n"
                f"Field name: {fname}\n"
                f"Description: {fdesc}\n"
                f"Column expression: {fexpr}\n"
                f"Data type: {fdtype}\n"
                f"Synonyms: {synonyms}"
            )

            metadata = {
                "model":      model_name,
                "table":      table_name,
                "full_table": full_table,
                "field_type": field_type,
                "field_name": fname,
                "expr":       fexpr,
                "data_type":  fdtype,
            }
            return Document(page_content=text, metadata=metadata)

        for section, label in [
            ("dimensions", "dimension"),
            ("measures",   "measure"),
            ("filters",    "filter"),
            ("time_dimensions", "time_dimension"),
        ]:
            for field in table.get(section, []):
                docs.append(make_doc(label, field))

    print(f"[embed_yaml] Parsed {len(docs)} chunks from '{yaml_path}'")
    return docs

# ── Build & persist ChromaDB ──────────────────────────────────────────────────
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"[embed_yaml] Cleared old vector store at {PERSIST_DIR}")

# Disable telemetry per UHG policy (same comment seen in notebook cell 21)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

docs = yaml_to_documents(YAML_FILE)

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
vectordb.persist()

print(f"[embed_yaml] Embeddings stored in ChromaDB at {PERSIST_DIR}")
print(f"[embed_yaml] Total vectors: {vectordb._collection.count()}")
