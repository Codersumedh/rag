# Snowflake RAG Text-to-SQL (Local + Free Models)

This project does exactly what you asked:

1. Read your table schema from a YAML file.
2. Create embeddings for schema chunks with a fast sentence-transformer.
3. Store/retrieve embeddings using ChromaDB.
4. Take a natural-English user question.
5. Run similarity search (RAG) to fetch relevant schema context.
6. Use a local Hugging Face model to generate SQL.
7. Execute SQL in Snowflake using your connection method.
8. Ask the same local model to explain the result in plain English.

No paid API keys are needed.

---

## Project files

- `schema.yaml` → your table metadata
- `embeddings.py` → build vector store from YAML
- `rag.py` → retrieve top-k relevant schema chunks
- `llm.py` → local text-to-SQL and explanation
- `connection.py` → Snowflake connection + query execution
- `main.py` → full end-to-end flow
- `config.py` → central config (connection and model settings)

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

