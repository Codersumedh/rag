# Snowflake RAG Text-to-SQL

Pipeline:

1. Read table schema from `schema.yaml`
2. Embed schema chunks → ChromaDB (cosine similarity)
3. RAG: retrieve relevant schema for the user question
4. LLM generates SQL
5. Run SQL in Snowflake
6. LLM explains results

**Two backends:**

| Mode | Embeddings | LLM |
|------|------------|-----|
| **Azure (recommended at UHG)** | Azure OpenAI via gateway | Azure chat (`get_response`) |
| **Local** | sentence-transformers | Hugging Face (no API) |

Set `USE_AZURE=true` in `UAIS_vars.env` to match your Databricks notebook.

---

## Project files

- `schema.yaml` → your table metadata
- `embeddings.py` → build vector store from YAML (`build_vectordb` when Azure)
- **`yaml_rag/`** → notebook-parity folder: Chroma + `semantic_retrieval()` + `retrieve.py`
- `retrieve.py` (in `yaml_rag/`) → English question → similar YAML chunks only
- `rag.py` → `semantic_retrieval` / `retrieve_schema_context` for SQL pipeline
- `llm.py` → local text-to-SQL and explanation
- `connection.py` → Snowflake connection + query execution
- `main.py` → full end-to-end flow
- `config.py` → central config (connection and model settings)
- `azure_client.py` → OAuth token + Azure OpenAI clients (`projectId` header)
- `azure_llm.py` / `azure_embed.py` → Azure chat + embeddings
- `UAIS_vars.env.example` → copy to `UAIS_vars.env`

---

## Azure mode (UHG gateway — same as your notebook)

1. Copy `UAIS_vars.env.example` → `UAIS_vars.env`
2. Fill values from your notebook / `UAIS_vars.env`:

```env
USE_AZURE=true
MODEL_ENDPOINT=https://api.uhg.com/api/cloud/api-management/ai-gateway/1.0
API_VERSION=2025-01-01-preview
PROJECT_ID=a82c5b9d-be3d-487a-858a-96b616963921
CHAT_MODEL_NAME=gpt-4.1-mini_2025-04-14
EMBEDDINGS_MODEL_NAME=your-embeddings-deployment
CLIENT_ID=...
CLIENT_SECRET=...
```

3. Install and build embeddings (uses Azure embeddings + `projectId` header):

```bash
pip install -r requirements.txt
python embeddings.py
python main.py
```

**Important:** If you switch from local → Azure (or change embedding model), delete `chroma_store/` and run `python embeddings.py` again.

In Databricks you can keep using `dbutils.secrets` for `CLIENT_ID` / `CLIENT_SECRET`; locally put them in `UAIS_vars.env` (do not commit).

---

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2) Update schema YAML

Edit `schema.yaml` with your real table(s)/column(s).  
You can keep one table or many tables.

---

## 3) Build embeddings once

```bash
python embeddings.py
```

This creates a local `chroma_store/` folder.

---

## 4) Update Snowflake config

Edit `config.py` and set:

- `host`
- `user`
- `account`
- `role`
- `warehouse`
- `database`
- `schema`
- `authenticator` (currently `externalbrowser`)

---

## 5) Run the pipeline

```bash
python main.py
```

Then ask, for example:
- “what is total paid amount by month for last 6 months?”
- “top 10 providers by claim amount”
- “count denied claims by state”

---

## Notes

- First Snowflake connection opens browser login (because of `externalbrowser`).
- The local LLM is intentionally small for simplicity and no API usage.
- For better SQL quality later, you can swap to a larger instruct model in `config.py`.

---

## Fix: SSL error downloading embedding model (corporate network)

If you see:

`SSLError ... huggingface.co ... CERTIFICATE_VERIFY_FAILED`

do one of these:

### Option A — Use local model folder (recommended)

1. On a machine where download works (or home network), run:

```bash
python download_model.py
```

2. Copy the folder `models/all-MiniLM-L6-v2` into this project.
3. In `config.py` set:

```python
HF_HUB_OFFLINE = True
```

4. Run again:

```bash
python embeddings.py
```

### Option B — Point Python to company CA certificate

In `config.py`:

```python
SSL_CERT_FILE = r"C:\path\to\your-company-root-ca.pem"
```

Then retry `python embeddings.py`.

### Option C — Use certifi bundle (sometimes works)

```powershell
python -m pip install certifi
$env:SSL_CERT_FILE = python -m certifi
$env:REQUESTS_CA_BUNDLE = python -m certifi
python embeddings.py
```

