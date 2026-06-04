import os
import yaml
import shutil
import asyncio
import warnings
import httpx
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

warnings.filterwarnings("ignore")

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(".env")

AZURE_OPENAI_ENDPOINT      = os.environ["MODEL_ENDPOINT"]
OPENAI_API_VERSION         = os.environ["API_VERSION"]
EMBEDDINGS_DEPLOYMENT_NAME = os.environ["EMBEDDINGS_MODEL_NAME"]
PROJECT_ID                 = os.environ["PROJECT_ID"]
CLIENT_ID                  = os.environ["CLIENT_ID"]
CLIENT_SECRET              = os.environ["CLIENT_SECRET"]

YAML_FILE   = "schema.yaml"
PERSIST_DIR = "./vector_embeddings_YAML"

# ── OAuth2 token ──────────────────────────────────────────────────────────────
async def get_token():
    auth       = "https://api.uhg.com/oauth2/token"
    scope      = "https://api.uhg.com/.default"
    grant_type = "client_credentials"
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

# ── Embeddings client ─────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=OPENAI_API_VERSION,
    azure_ad_token=token,
    default_headers={"projectId": PROJECT_ID}
)

# ── YAML to Documents ─────────────────────────────────────────────────────────
def yaml_to_documents(yaml_path: str) -> list:
    """
    Improvements over v1:
    1. Boilerplate (model name, table name) moved to metadata only.
       Each vector stays focused on the field meaning, not repeated context.
    2. One base Document per field + one extra Document per synonym.
       BM25 directly keyword-matches the synonym text.
    3. One extra Document per sample value — queries like
       'show approved cases' now match case_status_desc via 'Approved'.
    4. Covers dimensions, measures, filters, time_dimensions.
    """
    with open(yaml_path, "r") as f:
        model = yaml.safe_load(f)

    model_name = model.get("name", "")
    docs = []

    for table in model.get("tables", []):
        table_name = table.get("name", "")
        base       = table.get("base_table", {})
        full_table = (
            f"{base.get('database','')}"
            f".{base.get('schema','')}"
            f".{base.get('table','')}"
        )

        def make_doc(field_type, field, override_text=None):
            fname    = field.get("name", "")
            fdesc    = field.get("description", "")
            fexpr    = field.get("expr", "")
            fdtype   = field.get("data_type", "")
            synonyms = field.get("synonyms", [])
            samples  = field.get("sample_values", [])

            syn_str    = ", ".join(synonyms) if synonyms else ""
            sample_str = ", ".join(str(s) for s in samples) if samples else ""

            if override_text:
                # Synonym/sample chunk: lead with the term for high BM25 score
                text = (
                    f"User may refer to this field as: {override_text}\n"
                    f"Field: {fname}\n"
                    f"Meaning: {fdesc}\n"
                    f"SQL column: {fexpr}\n"
                    f"Field type: {field_type}\n"
                    f"Data type: {fdtype}\n"
                )
            else:
                # Base chunk: full detail, no repeated model/table noise
                text = (
                    f"Field: {fname}\n"
                    f"Meaning: {fdesc}\n"
                    f"Also called: {syn_str}\n"
                    f"SQL column: {fexpr}\n"
                    f"Field type: {field_type}\n"
                    f"Data type: {fdtype}\n"
                )
                if sample_str:
                    text += f"Sample values: {sample_str}\n"

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
            ("dimensions",      "dimension"),
            ("measures",        "measure"),
            ("filters",         "filter"),
            ("time_dimensions", "time_dimension"),
        ]:
            for field in table.get(section, []):
                # 1. Base document — full field detail
                docs.append(make_doc(label, field))

                # 2. One document per synonym — BM25 direct keyword hit
                for syn in field.get("synonyms", []):
                    docs.append(make_doc(label, field, override_text=syn))

                # 3. One document per sample value — value-based query matching
                for sv in field.get("sample_values", []):
                    docs.append(make_doc(label, field, override_text=str(sv)))

    print(f"[embed] Parsed {len(docs)} chunks from '{yaml_path}'")
    return docs


# ── Build and persist ChromaDB ────────────────────────────────────────────────
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"[embed] Cleared old vector store at {PERSIST_DIR}")

os.environ["ANONYMIZED_TELEMETRY"] = "False"

docs = yaml_to_documents(YAML_FILE)

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
vectordb.persist()

print(f"[embed] Embeddings stored in ChromaDB at {PERSIST_DIR}")
print(f"[embed] Total vectors: {vectordb._collection.count()}")
