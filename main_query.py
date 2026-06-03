import os
import asyncio
import warnings
from dotenv import load_dotenv
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from tenacity import retry, stop_after_attempt, wait_random_exponential

warnings.filterwarnings("ignore")

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(".env")

AZURE_OPENAI_ENDPOINT      = os.environ["MODEL_ENDPOINT"]
OPENAI_API_VERSION         = os.environ["API_VERSION"]
CHAT_DEPLOYMENT_NAME       = os.environ["CHAT_MODEL_NAME"]
EMBEDDINGS_DEPLOYMENT_NAME = os.environ["EMBEDDINGS_MODEL_NAME"]
PROJECT_ID                 = os.environ["PROJECT_ID"]
CLIENT_ID                  = os.environ["CLIENT_ID"]
CLIENT_SECRET              = os.environ["CLIENT_SECRET"]

PERSIST_DIR = "/tmp/vector_embeddings_YAML"
TOP_K       = 5   # number of chunks to retrieve

# ── OAuth2 token ──────────────────────────────────────────────────────────────
import httpx

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

# ── Clients (same pattern as notebook cells 9 & 12) ───────────────────────────
chat_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    azure_deployment=CHAT_DEPLOYMENT_NAME,
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

# ── Load persisted ChromaDB ───────────────────────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
print(f"[main] Loaded ChromaDB — {vectordb._collection.count()} vectors")

# ── Semantic retrieval with cosine similarity (Method 1 from notebook cell 27) 
def semantic_retrieval(query: str, top_k: int = TOP_K):
    """
    Fetch top_k*2 candidates (cosine similarity is the default distance for
    Chroma with OpenAI embeddings), then de-duplicate on page_content.
    """
    results       = vectordb.similarity_search(query, k=top_k * 2)
    unique_results = []
    seen_contents  = set()

    for doc in results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
        if len(unique_results) >= top_k:
            break

    return unique_results

# ── Build context string from retrieved chunks ────────────────────────────────
def build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"--- Chunk {i} ---\n{doc.page_content}")
    return "\n\n".join(parts)

# ── LLM call for SQL generation (same pattern as notebook cell 12) ────────────
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_response(prompt: str) -> str:
    # this function retrieves the model's response, and returns the content of the generated message
    response = chat_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=CHAT_DEPLOYMENT_NAME,
    )
    return response.choices[0].message.content

# ── SQL-generation prompt ─────────────────────────────────────────────────────
def build_sql_prompt(user_query: str, context: str) -> str:
    return f"""You are a SQL expert. A user has asked a question in natural language.
Use ONLY the schema information provided in the context below to write a correct SQL query.
Do not invent column names — use only the 'Column expression' values from the context.

=== SCHEMA CONTEXT (retrieved from semantic model) ===
{context}

=== USER QUESTION ===
{user_query}

=== INSTRUCTIONS ===
- Write a clean, executable SQL SELECT statement.
- Use fully qualified table names (database.schema.table) from the context.
- Prefer the exact column expressions listed in the context.
- Add meaningful aliases for readability.
- Do not add any explanation — return SQL only.

SQL:
"""

# ── Main pipeline ─────────────────────────────────────────────────────────────
def query_to_sql(user_query: str, verbose: bool = True) -> str:
    if verbose:
        print(f"\n[query] User query: {user_query}")

    # Step 1 — cosine similarity retrieval
    retrieved_docs = semantic_retrieval(user_query, top_k=TOP_K)

    if verbose:
        print(f"[query] Retrieved {len(retrieved_docs)} chunks:")
        for i, doc in enumerate(retrieved_docs, 1):
            field = doc.metadata.get("field_name", "?")
            ftype = doc.metadata.get("field_type", "?")
            print(f"  {i}. [{ftype}] {field}")

    # Step 2 — build context + prompt
    context    = build_context(retrieved_docs)
    prompt     = build_sql_prompt(user_query, context)

    # Step 3 — send to LLM
    sql_output = get_response(prompt)

    if verbose:
        print("\n[query] Generated SQL:\n")
        print(sql_output)

    return sql_output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example queries — replace or extend as needed
    test_queries = [
        "Show me total authorizations grouped by primary cancer type for the last 3 months",
        "What are the distinct ICD diagnosis codes used in UHC medical oncology?",
        "List patients with secondary cancer descriptions along with their treatment regimen",
    ]

    for q in test_queries:
        print("=" * 70)
        sql = query_to_sql(q, verbose=True)
        print("=" * 70)
