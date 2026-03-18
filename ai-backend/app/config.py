# FILE: ai-backend/app/config.py

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS Bedrock
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"

    # Bedrock model IDs
    bedrock_llm_model: str = "mistral.mistral-7b-instruct-v0:2"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "whatsapp_ai"
    messages_collection: str = "messages"

    # ChromaDB
    chroma_db_path: str = "./chroma_data"
    chroma_collection_name: str = "whatsapp_messages"

    # Retrieval config
    top_k_results: int = 10
    retrieval_max_distance: float = 0.6

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
