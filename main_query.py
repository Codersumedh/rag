import os
import asyncio
import warnings
from dotenv import load_dotenv
import openai
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from tenacity import retry, stop_after_attempt, wait_random_exponential
import httpx

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

PERSIST_DIR = "./vector_embeddings_YAML"
TOP_K       = 5

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

# ── Clients ───────────────────────────────────────────────────────────────────
chat_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    azure_deployment=CHAT_DEPLOYMENT_NAME,
    azure_ad_token=token,
    default_headers={"projectId": PROJECT_ID}
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=OPENAI_API_VERSION,
    azure_ad_token=token,
    default_headers={"projectId": PROJECT_ID}
)

# ── Load ChromaDB ─────────────────────────────────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
print(f"[main] Loaded ChromaDB — {vectordb._collection.count()} vectors")

# ── Load all docs for BM25 (keyword retriever) ───────────────────────────────
# BM25 does exact keyword + synonym matching which fixes the problem you saw —
# "treatment regimen" will now directly keyword-match against synonyms in the chunk text
all_docs = vectordb.get()   # returns dict with 'documents' and 'metadatas'

from langchain_core.documents import Document as LCDocument

bm25_docs = [
    LCDocument(page_content=text, metadata=meta)
    for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = TOP_K

# ── Semantic retriever (cosine similarity via Chroma) ─────────────────────────
semantic_retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# ── Hybrid retriever: BM25 (0.4) + Semantic (0.6) ────────────────────────────
# Weight semantic slightly higher but BM25 ensures keyword/synonym hits always surface
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],
)

# ── Hybrid retrieval with deduplication ───────────────────────────────────────
def hybrid_retrieval(query: str, top_k: int = TOP_K):
    """
    Method 2: BM25 + Semantic hybrid (EnsembleRetriever).
    BM25 catches exact keyword & synonym matches (e.g. 'treatment regimen').
    Semantic catches conceptual similarity.
    Results are RRF-fused then deduplicated.
    """
    results        = hybrid_retriever.invoke(query)
    unique_results = []
    seen_contents  = set()

    for doc in results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
        if len(unique_results) >= top_k:
            break

    return unique_results

# ── Build context ─────────────────────────────────────────────────────────────
def build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"--- Chunk {i} ---\n{doc.page_content}")
    return "\n\n".join(parts)

# ── LLM call ──────────────────────────────────────────────────────────────────
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_response(prompt: str) -> str:
    # this function retrieves the model's response, and returns the content of the generated message
    response = chat_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=CHAT_DEPLOYMENT_NAME,
    )
    return response.choices[0].message.content

# ── SQL prompt ────────────────────────────────────────────────────────────────
def build_sql_prompt(user_query: str, context: str) -> str:
    return f"""You are a SQL expert. A user has asked a question in natural language.
Use ONLY the schema information provided in the context below to write a correct SQL query.
Do not invent column names — use only the 'Column expression' values from the context.

=== SCHEMA CONTEXT (retrieved from semantic model via hybrid search) ===
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

    # Hybrid retrieval (BM25 + Semantic)
    retrieved_docs = hybrid_retrieval(user_query, top_k=TOP_K)

    if verbose:
        print(f"[query] Retrieved {len(retrieved_docs)} chunks (hybrid BM25 + semantic):")
        for i, doc in enumerate(retrieved_docs, 1):
            field = doc.metadata.get("field_name", "?")
            ftype = doc.metadata.get("field_type", "?")
            print(f"  {i}. [{ftype}] {field}")

    context    = build_context(retrieved_docs)
    prompt     = build_sql_prompt(user_query, context)
    sql_output = get_response(prompt)

    if verbose:
        print("\n[query] Generated SQL:\n")
        print(sql_output)

    return sql_output


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "List patients with secondary cancer descriptions and treatment regimen description",
        "Show me total authorizations grouped by primary cancer type",
        "What are the distinct ICD diagnosis codes used?",
    ]

    for q in test_queries:
        print("=" * 70)
        sql = query_to_sql(q, verbose=True)
        print("=" * 70)
