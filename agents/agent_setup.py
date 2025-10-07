from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd
import io
import json

def get_llm(api_key: str):
    """Retorna uma instância do LLM Gemini Flash com timeout."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.0,
            request_timeout=30  # Timeout de 30 segundos
        )
    except Exception as e:
        print(f"Erro ao criar LLM: {e}")
        raise e

def get_dataset_preview(df: pd.DataFrame) -> str:
    """Preview compacto para reduzir tokens."""
    MAX_COLS = 30
    MAX_ROWS_SAMPLE = 3
    cols = df.columns.tolist()[:MAX_COLS]
    dtypes = {c: str(df.dtypes[c]) for c in cols}
    sample = df[cols].head(MAX_ROWS_SAMPLE).to_dict(orient="records")

    preview = (
        f"Shape: {df.shape}\n"
        f"Columns (limited to {MAX_COLS}): {cols}\n"
        f"Dtypes: {dtypes}\n"
        f"Sample first {MAX_ROWS_SAMPLE} rows (dict): {sample}\n"
    )
    return preview