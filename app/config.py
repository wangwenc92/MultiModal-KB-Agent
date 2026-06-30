from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM（阿里云DashScope，OpenAI兼容接口）
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"

    # Embedding（阿里云DashScope）
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "text-embedding-v3"

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

    # 千问视觉模型（图片识别）
    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-vl-max"
    QWEN_MAX_TOKENS: int = 2048
    QWEN_TEMPERATURE: float = 0.1

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
