"""End-to-end pipeline:
1) embed user query
2) retrieve relevant schema chunks from ChromaDB
3) generate SQL with local HF model
4) execute SQL in Snowflake
5) explain result with local HF model
"""

from connection import run_query
from llm import explain_result, generate_sql
from rag import retrieve_schema_context


def ask(question: str):
    print("\n=== User Question ===")
    print(question)

    schema_context = retrieve_schema_context(question)
    print("\n=== Retrieved Schema Context ===")
    print(schema_context)

    sql = generate_sql(question, schema_context)
    print("\n=== Generated SQL ===")
    print(sql)

    df = run_query(sql)
    print("\n=== Query Result (top rows) ===")
    print(df.head(20))

    explanation = explain_result(question, sql, df)
    print("\n=== LLM Explanation ===")
    print(explanation)

    return {
        "question": question,
        "schema_context": schema_context,
        "sql": sql,
        "dataframe": df,
        "explanation": explanation,
    }


if __name__ == "__main__":
    user_question = input("Ask your business question in English: ").strip()
    ask(user_question)

