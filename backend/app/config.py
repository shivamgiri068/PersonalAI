import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "PersonalAI — Personalized RAG Assistant"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'personal_ai.db'))}")
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploads")))
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_store")))

    # Chunking & NLP Settings
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 4

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
