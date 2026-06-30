from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.utils.logger import log
from app.utils.exceptions import generic_exception_handler
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.logging import LoggingMiddleware
from app.api import chat, knowledge, upload, admin, analytics

settings = get_settings()

app = FastAPI(
    title="MultiModal Knowledge Base Agent",
    description="多模态智能知识库Agent系统",
    version="0.1.0",
)

# 中间件（按注册的逆序执行）
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 异常处理
app.add_exception_handler(Exception, generic_exception_handler)

# 注册路由
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(upload.router)
app.include_router(admin.router)
app.include_router(analytics.router)


@app.on_event("startup")
async def startup():
    log.info("Starting MultiModal KB Agent...")
    from app.utils.helpers import ensure_dir
    ensure_dir(settings.UPLOAD_DIR)
    ensure_dir(settings.CHROMA_PERSIST_DIR)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "name": "MultiModal Knowledge Base Agent",
        "version": "0.1.0",
        "docs": "/docs",
    }
