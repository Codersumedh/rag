# YAML vector RAG (Databricks notebook parity)

Self-contained folder for:

1. **Embed** semantic-model YAML with Azure OpenAI + LangChain Chroma
2. **Retrieve** similar chunks from an English question (`semantic_retrieval`)
3. **Generate** answers later via `get_response()` in `azure_client.py`

Matches your notebook: OAuth token, `UAIS_vars.env`, `Chroma.from_documents`, `semantic_retrieval`.

---

## Setup

From repo root (parent `requirements.txt` already has LangChain + OpenAI):

```bash
pip install -r requirements.txt
cd yaml_rag
```

### 1. Gateway config

```bash
copy Data\UAIS_vars.env.example Data\UAIS_vars.env
```

Edit `Data/UAIS_vars.env` with your gateway values (`MODEL_ENDPOINT`, `PROJECT_ID`, `EMBEDDINGS_MODEL_NAME`, etc.).

### 2. Databricks OAuth credentials (local)

```bash
copy Data\databricks.env.example Data\databricks.env
```

Put `CLIENT_ID` and `CLIENT_SECRET` from your Databricks secret scope `AIML_Training` in `Data/databricks.env`.

On Databricks, skip `databricks.env` — `azure_client.py` uses `dbutils.secrets.get` automatically.

### 3. YAML source

Default: `data/mart_med_uhc_mthly.yaml`. Replace with your full `mart_med_uhc_mthly.yaml` from the notebook, or set:

```bash
set YAML_PATH=C:\path\to\mart_med_uhc_mthly.yaml
```

---

## Build embeddings (once per YAML change)

```bash
python build_vectordb.py
```

Creates `chroma_store/` (local equivalent of notebook `/tmp/vector_embeddings_OPENAI`).

---

## Retrieve similar chunks

```bash
python retrieve.py "What columns describe primary cancer diagnosis?"
python retrieve.py --top-k 5 "total paid amount by month"
```

Or in Python:

```python
from vector_rag import semantic_retrieval

docs = semantic_retrieval("primary cancer type", top_k=3)
for d in docs:
    print(d.page_content)
```

---

## Files

| File | Role |
|------|------|
| `Data/UAIS_vars.env` | Gateway endpoint, project ID, model deployment names |
| `Data/databricks.env` | `CLIENT_ID` / `CLIENT_SECRET` (local only) |
| `azure_client.py` | OAuth token, Azure clients, `get_response()` |
| `embeddings_setup.py` | tiktoken cache + `AzureOpenAIEmbeddings` |
| `yaml_loader.py` | YAML → LangChain `Document` chunks |
| `build_vectordb.py` | `Chroma.from_documents` + `persist()` |
| `vector_rag.py` | `semantic_retrieval()` |
| `retrieve.py` | CLI for English queries |

---

## Next step (later)

Wire `semantic_retrieval()` + `get_response()` for full RAG Q&A over the YAML schema.
