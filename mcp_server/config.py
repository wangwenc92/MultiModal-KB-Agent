"""MCP Server 配置加载"""
import os
from pathlib import Path


def load_env() -> None:
    """从 .env 文件加载环境变量（可选）"""
    try:
        from dotenv import load_dotenv
        # 从项目根目录开始查找 .env
        for parent in Path(__file__).resolve().parents:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)
                return
    except ImportError:
        pass


def load_qwen_config() -> dict:
    """加载千问视觉模型配置"""
    load_env()
    return {
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": os.getenv("QWEN_MODEL", "qwen-vl-max"),
        "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", "2048")),
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.1")),
    }


def load_db_config() -> dict:
    """加载数据库配置"""
    load_env()
    return {
        "mysql_url": os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/multimodal_kb"),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    }
