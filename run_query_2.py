import pandas as pd
import snowflake.connector as snow
from snowflake.connector.pandas_tools import write_pandas
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# ── import SQL generation from main_query.py ──────────────────────────────────
from main_query import query_to_sql, get_response

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

# ── Execute SQL and return DataFrame ─────────────────────────────────────────
def execute_sql(sql: str) -> pd.DataFrame:
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(sql, conn)
        return df
    finally:
        conn.close()

# ── Explain results using LLM ─────────────────────────────────────────────────
MAX_ROWS_FOR_LLM = 20   # send at most this many rows to LLM prompt

def explain_results(user_query: str, sql: str, df: pd.DataFrame) -> str:
    """
    Send the original question + SQL + a sample of results to the LLM
    and get a plain-English explanation / insights back.
    Full df is shown in UI separately.
    """
    total_rows = len(df)
    sample     = df.head(MAX_ROWS_FOR_LLM).to_string(index=False)

    prompt = f"""You are a data analyst. A user asked a question in natural language.
A SQL query was generated and executed against a Snowflake database.
Your job is to explain the results clearly and provide meaningful insights.

=== ORIGINAL QUESTION ===
{user_query}

=== SQL EXECUTED ===
{sql}

=== RESULTS SAMPLE (first {min(total_rows, MAX_ROWS_FOR_LLM)} of {total_rows} rows) ===
{sample}

=== INSTRUCTIONS ===
- Summarize what the data shows in plain English.
- Highlight key trends, top values, anomalies, or patterns.
- Be concise but insightful (3-6 sentences).
- Do NOT repeat the SQL or column names verbatim — explain like talking to a business user.
- If total rows > {MAX_ROWS_FOR_LLM}, mention that only a sample was shown for analysis.

Insights:
"""
    return get_response(prompt)

def clean_sql(sql: str) -> str:
    """Strip markdown fences if LLM adds them."""
    s = sql.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
    if s.endswith("```"):
        s = "\n".join(s.split("\n")[:-1])
    return s.strip()


if __name__ == "__main__":
    user_query = input("\nEnter your question: ").strip()

    print("\n[run_query] Generating SQL...")
    sql       = query_to_sql(user_query, verbose=True)
    sql_clean = clean_sql(sql)
    print(f"\n[run_query] Executing SQL:\n{sql_clean}\n")

    try:
        df = execute_sql(sql_clean)
        print("\n[run_query] Results:")
        print(df.to_string(index=False))
        print(f"\n[run_query] {len(df)} row(s) returned.")

        print("\n[run_query] Generating insights...")
        insights = explain_results(user_query, sql_clean, df)
        print(f"\n[run_query] Insights:\n{insights}")

    except Exception as e:
        print(f"\n[run_query] Error: {e}")
