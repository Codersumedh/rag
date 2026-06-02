"""ChromaDB embedding function backed by Azure OpenAI embeddings."""

from typing import List

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from azure_client import EMBEDDINGS_DEPLOYMENT_NAME, get_embeddings_client


class AzureOpenAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        client = get_embeddings_client()
        texts: List[str] = [doc if isinstance(doc, str) else str(doc) for doc in input]
        response = client.embeddings.create(
            model=EMBEDDINGS_DEPLOYMENT_NAME,
            input=texts,
        )
        return [item.embedding for item in response.data]
