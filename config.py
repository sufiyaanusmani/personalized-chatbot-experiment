import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL=os.getenv("GROQ_BASE_URL")
    GROQ_MODEL=os.getenv("GROQ_MODEL")