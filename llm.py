"""LLM facade: Azure chat (default when USE_AZURE=true) or local Hugging Face."""

from config import USE_AZURE

if USE_AZURE:
    from azure_llm import explain_result, generate_sql
else:
    import re

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from config import LLM_MODEL

    _GEN = None

    def get_generator():
        global _GEN
        if _GEN is None:
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            _GEN = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=280,
                do_sample=False,
            )
        return _GEN

    def _extract_sql(text: str) -> str:
        block = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
        if block:
            return block.group(1).strip()
        semicolon = text.find(";")
        if semicolon != -1:
            return text[: semicolon + 1].strip()
        return text.strip()

    def generate_sql(user_question: str, retrieved_schema_context: str) -> str:
        generator = get_generator()
        prompt = f"""You are an expert Snowflake SQL assistant.
Given schema context and user question, return ONLY ONE executable SELECT SQL query.
Rules:
- Use only tables/columns from context.
- Prefer fully qualified names when present.
- Never use INSERT/UPDATE/DELETE/DROP.
- Output SQL in ```sql fenced block.

Schema context:
{retrieved_schema_context}

User question:
{user_question}
"""
        out = generator(prompt)[0]["generated_text"]
        return _extract_sql(out)

    def explain_result(user_question: str, sql: str, result_df: pd.DataFrame) -> str:
        generator = get_generator()
        preview = (
            "Result is empty."
            if result_df.empty
            else result_df.head(20).to_markdown(index=False)
        )
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
        out = generator(prompt)[0]["generated_text"]
        return out.replace(prompt, "").strip() or "No explanation generated."
