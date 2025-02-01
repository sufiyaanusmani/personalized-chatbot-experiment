from typing import Any

from llama_index.core.retrievers import QueryFusionRetriever

from constants import NUM_QUERIES
from index import get_index
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator


def get_retriever(user_id: str, top_n: int) -> QueryFusionRetriever:
    """
    Creates and returns a QueryFusionRetriever instance configured with the specified number of top results.

    Args:
    ----
        top_n (int): The number of top results to retrieve.

    Returns:
    -------
        QueryFusionRetriever: An instance of QueryFusionRetriever configured with the specified parameters.

    """
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="user_id", operator=FilterOperator.EQ, value=user_id)
        ]
    )
    index_retriever = get_index().as_retriever(similarity_top_k=top_n, filters=filters)
    return QueryFusionRetriever(
        [index_retriever],
        similarity_top_k=top_n,
        num_queries=NUM_QUERIES,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )

def retrieve_documents(question: str, user_id:str, top_n: int) -> tuple[None, Any]:
    """
    Retrieve documents based on a given question.

    Args:
    ----
        question (str): The question to retrieve documents for.
        top_n (int): The number of top documents to retrieve.

    Returns:
    -------
        tuple[None, Any]: A tuple containing None and the retrieved documents.

    """
    retriever = get_retriever(user_id, top_n)
    docs = retriever.retrieve(question)
    return docs
