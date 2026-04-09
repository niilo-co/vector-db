import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_THRESHOLD = int(os.getenv("CHUNK_THRESHOLD", "1000"))
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# LLM-based PDF extraction (uses vision model to extract text from PDF pages)
# Set to "true" to enable — uses OPENAI_API_KEY for the API call
LLM_EXTRACTION_ENABLED = os.getenv("LLM_EXTRACTION_ENABLED", "false").lower() == "true"
LLM_EXTRACTION_MODEL = os.getenv("LLM_EXTRACTION_MODEL", "gpt-5.4-mini")
LLM_EXTRACTION_MAX_PAGES = int(os.getenv("LLM_EXTRACTION_MAX_PAGES", "50"))