from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings

from vectorstore import get_vectorstore


def get_index() -> VectorStoreIndex:
    """
    Retrieves a VectorStoreIndex instance.
    This function initializes a vector store using the `get_vectorstore` function
    and then creates a `VectorStoreIndex` from the vector store using the specified
    embedding model from the settings.

    Returns
    -------
        VectorStoreIndex: An instance of VectorStoreIndex created from the vector store.

    """
    vectorstore = get_vectorstore()
    return VectorStoreIndex.from_vector_store(
        vectorstore,
        embed_model=Settings.embed_model,
    )
