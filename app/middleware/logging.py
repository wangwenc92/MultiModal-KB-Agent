import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logger import log


class LoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        method = request.method
        path = request.url.path

        response = await call_next(request)

        duration = int((time.time() - start) * 1000)
        status = response.status_code

        if path not in ("/health", "/docs", "/openapi.json"):
            log.info(f"{method} {path} → {status} ({duration}ms)")

        return response
