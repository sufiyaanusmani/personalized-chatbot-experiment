
from llama_index.core.settings import Settings
from config import Config
from pathlib import Path
from index import get_index
from logger import logger

def init_rag() -> None:
    """
    Initializes the Retrieval-Augmented Generation (RAG) system.
    This function sets up the RAG system by initializing the model provider
    (e.g., OpenAI) and configuring the necessary settings such as chunk size
    and chunk overlap. It also initializes the vector store and index.
    Steps:
    1. Determines the model provider and initializes it accordingly.
    2. Sets the chunk size and chunk overlap in the Settings.
    3. Initializes the vector store and index.
    Logs:
    - Logs an info message when the OpenAI models are initialized.
    - Logs an error message if the model provider is invalid.
    - Logs an info message when the vector store and index are initialized.
    """
    init_groq()

    # Settings.chunk_size = CHUNK_SIZE
    # Settings.chunk_overlap = CHUNK_OVERLAP
    get_index() # Initialize vectorstore and index

    logger.info("[SUCCESS] RAG system initialized successfully.")


def init_groq() -> None:
    """
    Initializes the OpenAI settings for the application.
    This function sets up the OpenAI language model (LLM) and embedding model
    using the API key and model configurations specified in the Config and Model
    classes. The temperature for the LLM is set to the default value defined in
    the llama_index.core.constants module.
    Dependencies:
        - llama_index.core.constants.DEFAULT_TEMPERATURE
        - llama_index.embeddings.openai.OpenAIEmbedding
        - llama_index.llms.openai.OpenAI
        - Config.OPENAI_API_KEY
        - Model.LLM.value
        - Model.EMBEDDING_MODEL.value
    Sets:
        - Settings.llm: An instance of OpenAI initialized with the specified API key,
          model, and temperature.
        - Settings.embed_model: An instance of OpenAIEmbedding initialized with the
          specified API key and model.
    """
    from llama_index.core.constants import DEFAULT_TEMPERATURE
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(
        base_url=Config.GROQ_BASE_URL,
        api_key=Config.GROQ_API_KEY,
        model=Config.GROQ_MODEL,
        temperature=DEFAULT_TEMPERATURE,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder=Path("model/bge-large-en-v1.5"),
    )
