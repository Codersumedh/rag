import pandas as pd
import snowflake.connector as snow
from snowflake.connector.pandas_tools import write_pandas
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# ── import SQL generation from main_query.py ──────────────────────────────────
from main_query import query_to_sql

# ── Snowflake connection (same pattern as your app.py) ────────────────────────
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

print("Connected to Snowflake")

# ── Ask user for natural language query ───────────────────────────────────────
user_query = input("\nEnter your question: ").strip()

# ── Step 1: get SQL from LLM via cosine similarity retrieval ──────────────────
print("\n[run_query] Generating SQL...")
sql = query_to_sql(user_query, verbose=True)

# ── Step 2: clean up SQL (strip markdown fences if LLM adds them) ─────────────
sql_clean = sql.strip()
if sql_clean.startswith("```"):
    sql_clean = "\n".join(sql_clean.split("\n")[1:])   # remove first ```sql line
if sql_clean.endswith("```"):
    sql_clean = "\n".join(sql_clean.split("\n")[:-1])  # remove last ```
sql_clean = sql_clean.strip()

print(f"\n[run_query] Executing SQL:\n{sql_clean}\n")

# ── Step 3: execute and print results ─────────────────────────────────────────
try:
    results = pd.read_sql(sql_clean, conn)
    print("\n[run_query] Results:")
    print(results.to_string(index=False))
    print(f"\n[run_query] {len(results)} row(s) returned.")
except Exception as e:
    print(f"\n[run_query] Error executing SQL: {e}")
finally:
    conn.close()
    print("\n[run_query] Snowflake connection closed.")
