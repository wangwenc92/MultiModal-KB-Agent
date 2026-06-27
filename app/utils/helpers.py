import os
import uuid
from pathlib import Path


def generate_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:16]


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
