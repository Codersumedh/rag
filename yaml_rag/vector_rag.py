"""Semantic retrieval from Chroma (notebook cell 27)."""

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import PERSIST_DIRECTORY
from embeddings_setup import get_langchain_embeddings

_vectordb: Optional[Chroma] = None


def get_vectordb(persist_directory: Path | None = None) -> Chroma:
    """Load existing Chroma store (run build_vectordb.py first)."""
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    persist_directory = Path(persist_directory or PERSIST_DIRECTORY)
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"No vector store at {persist_directory}. Run: python build_vectordb.py"
        )

    _vectordb = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_langchain_embeddings(),
    )
    return _vectordb


def semantic_retrieval(query: str, top_k: int = 3) -> list[Document]:
    """
    Retrieve top_k semantically relevant documents from ChromaDB (notebook parity).
    """
    vectordb = get_vectordb()
    results = vectordb.similarity_search(query, k=top_k * 2)
    unique_results: list[Document] = []
    seen_contents: set[str] = set()

    for doc in results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
        if len(unique_results) >= top_k:
            break

    return unique_results
