from pathlib import Path
from enum import Enum

# Get the current file's directory
current_dir = Path(__file__).resolve()

NUM_QUERIES = 5

BASE_DIR = current_dir.parent

VECTORSTORE_DIRECTORY = BASE_DIR / "db"
VECTORSTORE_DIRECTORY.mkdir(parents=True, exist_ok=True)

class Directory(Enum):
    """
    Enum class representing various directory constants.

    Attributes
    ----------
        BASE_DIR (str): The base directory path.
        VECTORSTORE_DIRECTORY (str): The directory path for vector store.

    """

    BASE_DIR = BASE_DIR
    VECTORSTORE_DIRECTORY = VECTORSTORE_DIRECTORY