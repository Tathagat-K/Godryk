# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Milvus configuration
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_COLLECTION_NAME: str

    # Embeddings
    EMBEDDING_MODEL_NAME: str
    SPARSE_EMBEDDING_MODEL_NAME: str
    SPARSE_EMBEDDING_DEVICE: str
    EMBEDDING_DIM: int

    # Ingestion settings
    INGEST_CHUNK_SIZE: int
    INGEST_CHUNK_OVERLAP: int
    ALLOWED_FILE_EXTENSIONS: List[str]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
