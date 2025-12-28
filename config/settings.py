import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Centralized configuration for the RAG system.
    Changes here apply across the entire application.
    """
    
    # --- API CONFIGURATIONS ---
    # We use os.getenv to ensure secrets aren't hardcoded
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Selection (Llama 3 via Groq for speed & free tier)
    LLM_MODEL = "llama3-8b-8192"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Runs locally on CPU
    
    # --- PATH CONFIGURATIONS ---
    VECTOR_DB_PATH = "chroma_db_data"
    COLLECTION_NAME = "enterprise_rag_docs"
    
    # --- RAG PARAMETERS (The "Secret Sauce") ---
    # Parent Chunk: Large context (2000 chars) for the LLM to read
    PARENT_CHUNK_SIZE = 2000
    PARENT_CHUNK_OVERLAP = 200
    
    # Child Chunk: Small context (400 chars) for accurate search
    CHILD_CHUNK_SIZE = 400
    CHILD_CHUNK_OVERLAP = 50
    
    # Retrieval Settings
    RETRIEVAL_K = 10         # Fetch top 10 chunks initially
    RERANK_TOP_N = 5         # Keep top 5 most relevant after reranking