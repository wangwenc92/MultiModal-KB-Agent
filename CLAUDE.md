# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。
## 项目概述

MultiModal-KB-Agent — 基于 FastAPI 的多模态智能知识库 Agent 系统，支持文档/图片/音频/视频索引与智能问答，具备 RAG 检索问答和 Agent 智能体两种模式。代码库使用中文（注释、界面文案、文档均为中文）。

## 语言规范

- **所有对话和文档都使用中文**：与用户的所有交互、生成的文档、回复内容均使用中文。
- **注释使用中文**：代码中的行内注释、块注释、函数/类文档字符串均使用中文编写。
- **错误提示使用中文**：API 返回的错误信息、日志中的错误描述、前端提示信息均使用中文。
- **文档使用中文 Markdown 格式**：所有 `.md` 文件使用中文编写，遵循 Markdown 格式规范。

## 工作约束

- **默认使用基础工具**（Read、Edit、Write、Grep、Glob、Bash）完成任务，保持简洁高效。
- **仅在以下场景使用 Agent 子代理**：项目初始规划、重大架构重构、需要并行探索多个独立方向时。
- 不要过度工程化：能直接改的文件不要抽象，能一步到位的不要拆分步骤。
- 不主动添加注释、文档文件或 README，除非用户明确要求。

## 常用命令

### 安装依赖
```bash
pip install -r requirements.txt
# 开发/测试环境：
pip install -e ".[dev]"
```

### 启动后端
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端（Streamlit）
```bash
streamlit run frontend/app.py
```

### 运行测试
```bash
pytest tests/ -v
# 单个测试：
pytest tests/test_rag.py::TestDocumentLoader::test_load_text -v
```

### 初始化数据库表
```bash
python scripts/init_db.py
```

### Docker 部署
```bash
cd docker && docker-compose up -d
```

### MCP Server（stdio 模式）
```bash
python -m app.core.mcp.server
```

## 配置说明

所有配置通过 `.env` 文件管理（参见 `.env.example`），关键变量：
- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` — LLM 服务配置（默认：qwen-plus，DashScope 阿里云）
- `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` — Embedding 服务（默认：text-embedding-v3，DashScope）
- `MYSQL_URL` — MySQL 连接字符串，用于元数据存储
- `REDIS_URL` — Redis，用于缓存和限流
- `CHROMA_PERSIST_DIR` — ChromaDB 向量存储路径
- `API_KEY` — 可选的 API 认证密钥

配置通过 `app.config.get_settings()` 一次性加载（使用 `@lru_cache` 缓存）。

## 架构

### 分层结构

```
API 层 (app/api/)           → FastAPI 路由：chat、knowledge、upload、admin、analytics
    ↓
核心引擎 (app/core/)        → 四个并行引擎：
    ├── agent/              → LangChain 风格 Agent（ReAct 模式：Thought→Action→Observation 循环）
    ├── rag/                → RAG 流水线：loader → splitter → embedder → retriever → chain
    ├── multimodal/         → 图片 (OCR)、音频 (Whisper)、视频 (FFmpeg) 处理
    └── memory/             → 短期记忆（会话）、长期记忆（跨会话）、上下文窗口
    ↓
服务层 (app/services/)      → LLM、embedding、缓存、微调模型服务
    ↓
存储层                       → MySQL（元数据）、ChromaDB（向量）、Redis（缓存/会话）、文件系统（上传）
```

### 两种对话模式

- **RAG 模式**（`app/core/rag/chain.py`）：从 ChromaDB 检索相关文本块，构建上下文，调用 LLM 生成带来源引用的回答
- **Agent 模式**（`app/core/agent/executor.py`）：ReAct 循环 — LLM 决定工具调用（web_search 使用 duckduckgo_search、calculator、code_exec、file_ops、kb_search），执行后迭代，最多 `max_iterations=5` 轮

通过 `POST /api/chat/send` 的 `mode` 字段选择模式。

### 关键设计模式

- **单例服务**：所有服务使用模块级 `_instance` 全局变量 + `get_*()` 工厂函数（如 `get_retriever()`、`get_cache()`、`get_llm_service()`）
- **工具注册表**：`app/core/agent/tools/__init__.py` 定义 `TOOL_REGISTRY` 字典，映射工具名称到实例。每个工具具有 `name`、`description` 和 `invoke(input: dict)` 方法
- **数据库模型**：SQLAlchemy 模型位于 `app/models/database.py` — KnowledgeBase、Document、Chunk、Session、Message，所有 ID 为 32 位字符串（由 `helpers.generate_id()` 生成）
- **中间件栈**：CORS → 限流（基于 Redis，60 请求/分钟）→ 日志（在 `app/main.py` 中逆序注册）
- **MCP Server**：`app/core/mcp/server.py` 通过 stdio JSON-RPC 将知识库暴露为 MCP 工具/资源

### 前端

Streamlit 应用（`frontend/app.py`）与后端 API（`http://localhost:8000`）通信。侧栏管理知识库和文件上传；主区域为对话界面，支持 RAG/Agent 模式切换。

## 外部依赖服务

- MySQL 8.0 — 元数据存储
- Redis 7.x — 缓存和限流
- LLM API（阿里云 DashScope，OpenAI 兼容接口）— 对话和 Agent 推理（默认 qwen-plus）
- Embedding API（DashScope text-embedding-v3）— 文档向量化（备选：本地模型 / 哈希回退）
- 可选：PaddleOCR、Whisper、FFmpeg — 多模态功能

## MCP 工具配置

### 千问视觉模型图片识别 (`qwen-vision-mcp`)

独立的 MCP Server（`scripts/mcp_qwen_vision.py`），使用千问 (Qwen) 视觉模型识别图片内容，通过 `image_recognize` 工具暴露给 Claude Code。

**前置条件：**
1. 获取阿里云 DashScope API Key：https://dashscope.aliyuncs.com/
2. 在 `.env` 或 MCP server 的 `env` 中配置 `QWEN_API_KEY`

**配置方式**（`.env` 中添加）：
```bash
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-vl-max
QWEN_MAX_TOKENS=2048
QWEN_TEMPERATURE=0.1
```

**可用模型**：`qwen-vl-max`（最强）、`qwen-vl-plus`（性价比）、`qwen2.5-vl-72b-instruct`（开源版）

**Claude Code MCP 注册方式**（在 `.claude/settings.local.json` 的 `mcpServers` 中加入）：
```json
{
  "qwen-vision": {
    "command": "python",
    "args": ["scripts/mcp_qwen_vision.py"],
    "env": {
      "QWEN_API_KEY": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
      "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "QWEN_MODEL": "qwen-vl-max"
    }
  }
}
```

**手动测试**：
```bash
# 测试工具列表
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/mcp_qwen_vision.py

# 测试图片识别
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"image_recognize","arguments":{"image_path":"D:/path/to/image.png"}}}' | python scripts/mcp_qwen_vision.py
```

**使用场景**：当需要识别/分析图片内容时，直接调用 `image_recognize` 工具，传入图片绝对路径即可。

