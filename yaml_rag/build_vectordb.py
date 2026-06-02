"""Build Chroma vector store from YAML (notebook cell 21)."""

import os
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma

from config import PERSIST_DIRECTORY, YAML_PATH
from embeddings_setup import get_langchain_embeddings
from yaml_loader import load_yaml_documents


def build_vectordb(
    yaml_path: Path | None = None,
    persist_directory: Path | None = None,
    *,
    rebuild: bool = True,
) -> Chroma:
    """
    Store schema chunks in Chroma (same as notebook):

        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, ...)
        vectordb.persist()
    """
    persist_directory = Path(persist_directory or PERSIST_DIRECTORY)
    texts = load_yaml_documents(yaml_path)

    if not texts:
        raise ValueError(f"No documents loaded from {yaml_path or YAML_PATH}")

    embeddings = get_langchain_embeddings()

    if rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    try:
        vectordb.persist()
    except Exception:
        pass

    print(f"Embeddings stored in ChromaDB ({len(texts)} chunks → {persist_directory}).")
    return vectordb


if __name__ == "__main__":
    build_vectordb()
