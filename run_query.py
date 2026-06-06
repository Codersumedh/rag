import json
import pandas as pd
import snowflake.connector as snow
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from main_query import query_to_sql, get_response

MAX_ROWS_FOR_LLM = 20

# ── Snowflake connection ───────────────────────────────────────────────────────
def get_snowflake_connection():
    conn = snow.connect(
        host="abc111.east-us-2.azure.snowflakecomputing.com",
        user="xyz@abc.com",
        account="DWAAS",
        role="AR_PRD_ABC_ROLE",
        warehouse="OHBI_PRD_CONSUME_FREQ_WH",
        database="OHBI_PRD_MART_DB",
        schema="CORE_ONC",
        authenticator='externalbrowser'
    )
    return conn

# ── Execute SQL ───────────────────────────────────────────────────────────────
def execute_sql(sql: str) -> pd.DataFrame:
    conn = get_snowflake_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()

# ── Clean SQL ─────────────────────────────────────────────────────────────────
def clean_sql(sql: str) -> str:
    s = sql.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
    if s.endswith("```"):
        s = "\n".join(s.split("\n")[:-1])
    return s.strip()

# ── Build compact chat context (low token) ────────────────────────────────────
def build_chat_context(history: list) -> str:
    """
    From previous turns keep only: user query + generated SQL.
    Skips insights/results to save tokens.
    Last 5 turns max.
    """
    if not history:
        return ""
    recent = history[-5:]
    lines = []
    for i, turn in enumerate(recent, 1):
        lines.append(f"Turn {i}:")
        lines.append(f"  Q: {turn['query']}")
        lines.append(f"  SQL: {turn['sql']}")
    return "\n".join(lines)

# ── Explain results ───────────────────────────────────────────────────────────
def explain_results(user_query: str, sql: str, df: pd.DataFrame) -> str:
    total  = len(df)
    sample = df.head(MAX_ROWS_FOR_LLM).to_string(index=False)

    prompt = f"""You are a data analyst. Explain the query results clearly for a business user.

QUESTION: {user_query}
SQL: {sql}
RESULTS SAMPLE ({min(total, MAX_ROWS_FOR_LLM)} of {total} rows):
{sample}

Write 3-5 sentences highlighting key trends, top values, and patterns.
Do not repeat column names verbatim. Be concise and business-friendly.
{"Note: only a sample was analyzed; full result has " + str(total) + " rows." if total > MAX_ROWS_FOR_LLM else ""}

Insights:"""
    return get_response(prompt)

# ── Generate 3 follow-up suggestions ─────────────────────────────────────────
def generate_suggestions(user_query: str, sql: str, df: pd.DataFrame) -> list:
    """
    Returns a list of 3 dicts: [{"question": "...", "reason": "..."}, ...]
    Parsed from strict JSON output of the LLM.
    """
    cols   = ", ".join(df.columns.tolist())
    sample = df.head(5).to_string(index=False)

    prompt = f"""You are a data analyst assistant. A user just asked this question about medical oncology data:

QUESTION: {user_query}
COLUMNS RETURNED: {cols}
SAMPLE RESULTS:
{sample}

Suggest exactly 3 follow-up questions the user might want to ask next.
Each suggestion must be a natural language question about the data.

Respond ONLY with a valid JSON array, no explanation, no markdown:
[
  {{"question": "...", "reason": "one sentence why this is useful"}},
  {{"question": "...", "reason": "one sentence why this is useful"}},
  {{"question": "...", "reason": "one sentence why this is useful"}}
]"""

    raw = get_response(prompt)
    try:
        # strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        return json.loads(clean.strip())
    except Exception:
        # fallback if LLM doesn't return valid JSON
        return [
            {"question": "Show the top 10 records by count", "reason": "Quick overview of volume"},
            {"question": "Break this down by cancer type", "reason": "Understand distribution"},
            {"question": "Compare approval vs denial rates", "reason": "Measure outcomes"},
        ]

# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(user_query: str, chat_history: list) -> dict:
    """
    Returns dict with: sql, df, insights, suggestions
    chat_history: list of {query, sql} dicts from previous turns
    """
    context  = build_chat_context(chat_history)
    sql_raw  = query_to_sql(user_query, verbose=False, context=context)
    sql      = clean_sql(sql_raw)
    df       = execute_sql(sql)
    insights = explain_results(user_query, sql, df)
    suggestions = generate_suggestions(user_query, sql, df)

    return {
        "query":       user_query,
        "sql":         sql,
        "df":          df,
        "insights":    insights,
        "suggestions": suggestions,
    }


if __name__ == "__main__":
    q = input("Enter your question: ").strip()
    result = run_pipeline(q, [])
    print("\nSQL:\n", result["sql"])
    print("\nResults:\n", result["df"].to_string(index=False))
    print("\nInsights:\n", result["insights"])
    print("\nSuggestions:")
    for s in result["suggestions"]:
        print(f"  - {s['question']} ({s['reason']})")
