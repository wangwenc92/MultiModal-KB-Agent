from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.anthropic.com"
    LLM_MODEL: str = "claude-sonnet-4-20250514"

    # Embedding
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # MySQL
    MYSQL_URL: str = "mysql+pymysql://root:password@localhost:3306/multimodal_kb"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # Upload
    UPLOAD_DIR: str = "./data/uploads"

    # Log
    LOG_LEVEL: str = "INFO"

    # API Auth
    API_KEY: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
