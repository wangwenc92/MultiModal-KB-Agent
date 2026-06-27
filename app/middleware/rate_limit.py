import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.cache_service import get_cache
from app.utils.logger import log


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于Redis的限流中间件"""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        # 跳过健康检查和文档
        if request.url.path in ("/health", "/docs", "/openapi.json", "/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        cache = get_cache()
        key = f"rate_limit:{client_ip}"

        try:
            current = cache.get(key)
            if current is None:
                cache.set(key, "1", ttl=self.window_seconds)
            elif int(current) >= self.max_requests:
                log.warning(f"Rate limit exceeded for {client_ip}")
                return Response(
                    content='{"code": 429, "message": "Too many requests", "detail": "Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                )
            else:
                client = cache._get_client()
                client.incr(key)
        except Exception:
            pass  # Redis不可用时跳过限流

        return await call_next(request)
