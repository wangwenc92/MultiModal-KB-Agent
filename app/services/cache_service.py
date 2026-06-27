from __future__ import annotations

import json
import hashlib
from typing import Optional
from app.config import get_settings
from app.utils.logger import log

settings = get_settings()


class CacheService:
    """Redis缓存服务"""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import redis
            self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._client

    def get(self, key: str) -> Optional[str]:
        try:
            client = self._get_client()
            return client.get(key)
        except Exception as e:
            log.warning(f"Cache get failed: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = 300):
        try:
            client = self._get_client()
            client.setex(key, ttl, value)
        except Exception as e:
            log.warning(f"Cache set failed: {e}")

    def delete(self, key: str):
        try:
            client = self._get_client()
            client.delete(key)
        except Exception as e:
            log.warning(f"Cache delete failed: {e}")

    def exists(self, key: str) -> bool:
        try:
            client = self._get_client()
            return client.exists(key) > 0
        except Exception:
            return False

    def get_json(self, key: str) -> Optional[dict | list]:
        val = self.get(key)
        if val:
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return None
        return None

    def set_json(self, key: str, value: dict | list, ttl: int = 300):
        self.set(key, json.dumps(value, ensure_ascii=False), ttl)

    def hash_key(self, prefix: str, *args) -> str:
        raw = "|".join(str(a) for a in args)
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{prefix}:{h}"

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            return client.ping()
        except Exception:
            return False


_cache: CacheService | None = None


def get_cache() -> CacheService:
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
