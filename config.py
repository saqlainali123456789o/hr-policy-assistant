import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your environment variables."
    )
