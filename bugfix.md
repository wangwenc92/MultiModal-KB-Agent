# MultiModal-KB-Agent 项目问题清单

## 1. `datetime.utcnow` 已弃用 → `app/models/database.py`

Python 3.12+ 中 `datetime.utcnow()` 已标记为弃用，多处使用会触发警告。应改为 `datetime.now(timezone.utc)`：

- 第27行：`KnowledgeBaseModel.created_at`
- 第45行：`DocumentModel.created_at`
- 第68行：`SessionModel.created_at`
- 第82行：`MessageModel.created_at`

## 2. FastAPI `on_event("startup")` 已弃用 → `app/main.py:41`

`@app.on_event("startup")` 在 FastAPI 新版本中已弃用，应改用 `lifespan` 上下文管理器：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup logic
    yield
    # shutdown logic (if needed)
```

## 3. 限流中间件 Race Condition → `app/middleware/rate_limit.py:26-38`

`get` + `incr` 两步操作不是原子操作，高并发下存在竞态条件。应使用 Redis Lua 脚本或管道实现原子递增+过期：

```python
# 方案：使用 INCR + EXPIRE 原子操作
pipe = client.pipeline()
pipe.incr(key)
pipe.expire(key, self.window_seconds)
result = pipe.execute()
```

## 4. 缓存哈希使用不安全的 MD5 → `app/services/cache_service.py:67`

`hashlib.md5()` 被认为不安全，应使用 `hashlib.sha256()`：

```python
h = hashlib.sha256(raw.encode()).hexdigest()[:16]
```

## 5. LLMService 硬编码参数 → `app/services/llm_service.py:27-29`

`max_tokens=4096`、`timeout=60`、`max_retries=2` 硬编码在代码中，无法通过配置调整。应加入 `Settings` 或构造函数参数：

```python
self._llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    max_tokens=getattr(settings, 'LLM_MAX_TOKENS', 4096),
    timeout=getattr(settings, 'LLM_TIMEOUT', 60),
    max_retries=getattr(settings, 'LLM_MAX_RETRIES', 2),
    ...
)
```

## 6. Agent Action 解析容错性差 → `app/core/agent/executor.py:36-49`

JSON 解析失败后回退到简单的 `key=value` 格式解析，对嵌套 JSON、数组参数等复杂输入无法处理。建议使用更健壮的解析策略：

- 提取代码块中的 JSON（` ```json ... ``` `）
- 支持多行 Action Input
- 对无法解析的输入给出更明确的错误提示

## 7. 短期记忆 Token 计数硬编码编码器 → `app/core/memory/short_term.py:46`

`tiktoken.get_encoding("cl100k_base")` 是 OpenAI 的编码器，当使用 Anthropic Claude 或其他模型时 token 计数不准确。应从配置读取编码器名称或根据 LLM 模型自动选择。

## 8. 长期记忆仅存内存不持久化 → `app/core/memory/long_term.py:11`

`_long_term_store: dict[str, list[dict]]` 是进程内存字典，服务重启后所有长期记忆丢失。应持久化到数据库表中。

## 9. Embedding 回退使用确定性哈希 → `app/services/embedding_service.py:17-39`

当 API 和本地模型都不可用时，回退到 `_hash_embedding()` 基于 SHA-512 的确定性哈希。这意味着不同文本可能产生相似向量，检索质量严重下降。至少应记录警告提示用户配置正确的 Embedding 服务。

## 10. MCP Server 不安全路径注入 → `app/core/mcp/server.py:8`

`sys.path.insert(0, ".")` 依赖当前工作目录，在不同部署环境下可能导致 ImportError。应使用绝对路径或 `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))`。

## 11. RAG 流式方法中同步检索阻塞 → `app/core/rag/chain.py:64`

`ask_stream()` 中使用同步的 `self.retriever.retrieve()`，在异步上下文中会阻塞事件循环。应使用线程池异步化：

```python
import asyncio
loop = asyncio.get_event_loop()
results = await loop.run_in_executor(None, self.retriever.retrieve, question, kb_id)
```

## 12. VectorStore 相似度转换不准确 → `app/core/rag/embedder.py:62`

`score = 1 - dist` 假定 ChromaDB 返回的是 cosine distance，但如果 `hnsw:space` 配置为其他度量（如 L2），转换结果将完全错误。应在创建 collection 时统一度量并在搜索时正确处理。

## 13. 前端 API 异常静默处理 → `frontend/app.py:374-399`

`api_get`、`api_post`、`api_delete` 在发生异常时静默返回 `None`，用户无法区分网络错误、超时、服务端错误。建议至少记录错误日志并根据异常类型给出不同提示。

## 14. 文件上传缺少大小限制 → `app/api/upload.py:17`

没有校验上传文件大小，攻击者可以上传超大文件耗尽服务器资源。应添加 `max_upload_size` 配置和校验。

## 15. 限流中间件异常完全静默吞掉 → `app/middleware/rate_limit.py:39`

`except Exception: pass` 完全忽略 Redis 连接异常，包括认证失败等不应忽略的错误。应至少记录 warning 日志，并通过 `str(e)` 区分不同的异常类型。

## 16. Pydantic `model_config` 旧式写法 → `app/config.py:34`

`model_config = {"env_file": ".env", ...}` 是 Pydantic v2 的旧写法，推荐使用：

```python
model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
```
