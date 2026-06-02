"""SQL generation + explanation via Azure chat (UHG gateway)."""

import re

import pandas as pd

from azure_client import get_response


def _extract_sql(text: str) -> str:
    block = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if block:
        return block.group(1).strip()
    semicolon = text.find(";")
    if semicolon != -1:
        return text[: semicolon + 1].strip()
    return text.strip()


def generate_sql(user_question: str, retrieved_schema_context: str) -> str:
    prompt = f"""You are an expert Snowflake SQL assistant.
Given schema context and user question, return ONLY ONE executable SELECT SQL query.

Rules:
- Use only tables/columns from context.
- Prefer fully qualified names when present (database.schema.table).
- Never use INSERT/UPDATE/DELETE/DROP.
- Output SQL in a ```sql fenced block.

Schema context:
{retrieved_schema_context}

User question:
{user_question}
"""
    return _extract_sql(get_response(prompt))


def explain_result(user_question: str, sql: str, result_df: pd.DataFrame) -> str:
    if result_df.empty:
        preview = "Result is empty."
    else:
        preview = result_df.head(20).to_markdown(index=False)

    prompt = f"""You are a data analyst.
Explain query output in plain English for a business user.
Be concise, factual, and mention if output is empty.

User question:
{user_question}

SQL used:
{sql}

Result preview:
{preview}
"""
    return get_response(prompt).strip() or "No explanation generated."
