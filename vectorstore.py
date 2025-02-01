import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from constants import Directory
# from rag_services.ingestion.splitter_registry import SplitterId, splitter_registry
from logger import logger


def get_vectorstore() -> ChromaVectorStore:
    """
    Initializes and returns a ChromaVectorStore instance.
    This function creates a persistent client for the Chroma database using the
    directory specified by Directory.VECTORSTORE_DIRECTORY. It then retrieves or
    creates a collection named "transcription_project" within the database and
    returns a ChromaVectorStore instance associated with this collection.

    Returns
    -------
        ChromaVectorStore: An instance of ChromaVectorStore associated with the
        "transcription_project" collection.

    """
    db = chromadb.PersistentClient(path=str(Directory.VECTORSTORE_DIRECTORY.value))
    chroma_collection = db.get_or_create_collection("capelin_ai")
    return ChromaVectorStore(chroma_collection=chroma_collection)


# def add_documents(documents: list, splitter: SplitterId) -> None:
#     splitter_function = splitter_registry.get(splitter)
#     vector_store = get_vectorstore()
#     if splitter_function: # although this check is not required, but it is still added for more safety
#         pipeline = splitter_function(vector_store=vector_store)
#         issue_documents = pipeline.run(documents=documents)

#         parsed_documents = issue_documents
#         logger.info(f"[INFO] Added {len(parsed_documents)} documents to the vectorstore using splitter `{splitter}`")

#     else:
#         logger.error("[ERROR] Invalid splitter function for splitter '{splitter}'")

