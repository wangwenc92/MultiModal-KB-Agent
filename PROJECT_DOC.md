# Multi-Modal Intelligent Knowledge Base Agent - 项目技术文档

## 一、项目概述

### 1.1 项目名称
**MultiModal-KB-Agent** — 多模态智能知识库Agent系统

### 1.2 项目定位
基于大语言模型的多模态智能知识库Agent，支持文档/图片/音频/视频的自动索引与智能问答，
具备Agent任务规划、工具调用、上下文工程等能力，面向企业级AI应用落地场景。

### 1.3 核心能力矩阵

| 能力模块 | 技术栈 | JD覆盖 |
|---------|--------|--------|
| RAG系统 | LangChain + ChromaDB + 混合检索 | 大模型应用落地、RAG技能 |
| AI Agent | LangChain Agent (ReAct) + 工具注册表 | AI Agent实践 |
| 多模态处理 | Whisper + Qwen-VL + PaddleOCR + FFmpeg | 多模态数据处理 |
| 模型工程化 | FastAPI + Redis + Docker + MCP协议 | 模型工程化与服务支撑 |
| 数据工程 | Pandas + NumPy + MySQL + SQLAlchemy | 数据与存储协同 |
| 图片识别 | Qwen-VL视觉模型 + PaddleOCR + Windows OCR | 多模态图片理解 |

### 1.4 最终交付物
- 完整可运行的AI Agent后端服务（FastAPI）
- Streamlit Web UI 前端（含数据分析看板）
- MCP Server（暴露知识库检索为工具/资源）
- 千问视觉模型 MCP 工具（图片识别）
- API接口文档（FastAPI自动生成）
- Docker一键部署方案（MySQL + Redis + 后端 + 前端）

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│     (对话界面 / 文件上传 / 知识库管理 / 数据分析看板)         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI API Layer                         │
│   (路由 / 限流 / 日志 / 异常处理 / CORS / 中间件栈)           │
└──────┬───────────┬───────────┬───────────┬──────────────────┘
       │           │           │           │
┌──────▼───┐ ┌─────▼────┐ ┌───▼────┐ ┌───▼─────────┐
│  Agent   │ │   RAG    │ │MultiModal│ │  Memory     │
│  Engine  │ │  Engine  │ │ Engine  │ │  Manager    │
│          │ │          │ │         │ │             │
│ -ReAct   │ │ -文档解析│ │ -图片OCR│ │ -短期记忆   │
│  循环    │ │ -分块策略│ │ -音频转写│ │ -长期记忆   │
│ -5个工具 │ │ -向量检索│ │ -视频分析│ │ -上下文窗口 │
│ -工具注册│ │ -链式回答│ │ -AI描述  │ │             │
└──────┬───┘ └─────┬────┘ └───┬────┘ └──────┬──────┘
       │           │          │              │
┌──────▼───────────▼──────────▼──────────────▼──────────────┐
│                    Storage Layer                           │
│  ChromaDB(向量)  MySQL(元数据)  Redis(缓存/会话)  本地文件  │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│                    MCP 协议扩展                            │
│  app/core/mcp/server — 知识库检索 MCP 工具/资源            │
│  scripts/mcp_qwen_vision — 千问视觉模型 MCP 图片识别       │
└───────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
multimodal-kb-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI入口（启动/路由/中间件注册）
│   ├── config.py                  # 配置管理（pydantic-settings + @lru_cache）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                # 对话API（RAG + Agent双模式）
│   │   ├── knowledge.py           # 知识库管理API（CRUD）
│   │   ├── upload.py              # 文件上传API（自动解析→分块→向量化）
│   │   ├── admin.py               # 管理API（系统统计/健康检查）
│   │   └── analytics.py           # 数据分析API（使用统计/热力图）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── executor.py        # Agent执行器（ReAct循环：Think→Act→Observe）
│   │   │   ├── prompt.py          # Agent提示词模板
│   │   │   └── tools/             # 工具注册表（5个内置工具）
│   │   │       ├── __init__.py    # TOOL_REGISTRY 字典
│   │   │       ├── search.py      # Web搜索（duckduckgo_search）
│   │   │       ├── calculator.py  # 数学计算
│   │   │       ├── code_exec.py   # 代码执行（受限沙箱）
│   │   │       ├── file_ops.py    # 文件操作
│   │   │       └── kb_search.py   # 知识库检索
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py          # 文档加载器（PDF/Word/MD/TXT/CSV）
│   │   │   ├── splitter.py        # 文本分块（Recursive/Sentence/Markdown）
│   │   │   ├── embedder.py        # VectorStore（ChromaDB封装）
│   │   │   ├── retriever.py       # 检索器（向量检索 + threshold过滤）
│   │   │   └── chain.py           # RAG完整链路（检索→构建Prompt→LLM→来源引用）
│   │   ├── multimodal/
│   │   │   ├── __init__.py
│   │   │   ├── image.py           # 图片处理（OCR + Qwen-VL AI描述）
│   │   │   ├── audio.py           # 音频处理（Whisper转写）
│   │   │   ├── video.py           # 视频处理（FFmpeg关键帧+分析）
│   │   │   ├── ocr.py             # PaddleOCR引擎封装
│   │   │   ├── windows_ocr.py     # Windows OCR引擎（备选方案）
│   │   │   └── pipeline.py        # 多模态统一处理管线
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── short_term.py      # 短期记忆（会话内滑动窗口，10轮）
│   │   │   ├── long_term.py       # 长期记忆（跨会话，用户偏好/摘要）
│   │   │   └── context.py         # 上下文窗口管理（Token预算分配）
│   │   └── mcp/
│   │       ├── __init__.py
│   │       └── server.py          # MCP Server（JSON-RPC over stdio）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic数据模型（请求/响应）
│   │   └── database.py            # SQLAlchemy模型（ID为16位字符串）
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py         # LLM调用（langchain-openai → DashScope）
│   │   ├── embedding_service.py   # Embedding（API / 本地 / 哈希回退三级降级）
│   │   ├── cache_service.py       # Redis缓存（会话/查询/Embedding三级缓存）
│   │   └── finetuned_service.py   # 微调模型推理接口
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limit.py          # 限流中间件（Redis，60请求/分钟）
│   │   └── logging.py             # 请求日志中间件
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 日志工具（loguru）
│       ├── helpers.py           # 通用工具（ID生成/目录创建/文件处理）
│       └── exceptions.py          # 自定义异常 + 全局异常处理器
├── frontend/
│   ├── app.py                     # Streamlit主应用（对话界面）
│   └── pages/
│       └── analytics.py           # 数据分析看板（Plotly图表）
├── scripts/
│   ├── init_db.py                 # 数据库初始化（建表）
│   ├── benchmark.py               # 性能测试脚本
│   ├── finetune_data.py           # 微调数据准备（QA对提取）
│   ├── finetune.py                # LoRA微调训练
│   ├── test_tools.py              # Agent工具独立测试
│   └── mcp_qwen_vision.py         # 千问视觉模型 MCP Server（图片识别）
├── docker/
│   ├── Dockerfile                 # 后端服务镜像
│   ├── Dockerfile.frontend        # 前端服务镜像
│   └── docker-compose.yml         # 一键部署（mysql+redis+backend+frontend）
├── tests/
│   ├── __init__.py
│   ├── test_rag.py                # RAG单元测试
│   └── test_api.py                # API集成测试
├── data/
│   ├── chroma/                    # ChromaDB持久化目录（.gitignore）
│   └── uploads/                   # 上传文件存储（.gitignore）
├── .streamlit/
│   └── config.toml               # Streamlit主题配置
├── .env.example                   # 环境变量模板（含QWEN视觉模型配置）
├── CLAUDE.md                      # Claude Code 项目指引
├── PROJECT_DOC.md                 # 项目技术文档
├── requirements.txt
├── pyproject.toml
├── README.md
└── resume_project.txt             # 项目简历描述
```

---

## 三、技术选型

### 3.1 核心框架

| 组件 | 技术选型 | 版本要求 | 说明 |
|------|---------|---------|------|
| Web框架 | FastAPI | >=0.100 | 异步高性能，自动生成API文档 |
| Agent框架 | LangChain | >=0.2 | 主流Agent开发框架（ReAct模式） |
| 前端 | Streamlit | >=1.30 | 快速搭建AI应用UI |
| LLM | 阿里云 DashScope（Qwen） | - | 通过OpenAI兼容接口调用，默认 qwen-plus |
| Embedding | DashScope text-embedding-v3 | - | 三级降级：API → 本地模型 → 哈希回退 |

### 3.2 存储层

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 向量数据库 | ChromaDB | 文档向量存储与检索（本地持久化） |
| 关系数据库 | MySQL 8.0 | 元数据、用户、会话管理 |
| 缓存 | Redis 7.x | 会话缓存、热点数据、限流 |
| 对象存储 | 本地文件系统 | 上传文件存储 |

### 3.3 多模态处理

| 能力 | 技术选型 | 说明 |
|------|---------|------|
| 音频转写 | OpenAI Whisper（本地模型） | 音频→文本，支持中英文 |
| 图片OCR | PaddleOCR / Windows OCR（备选） | 图片文字提取 |
| 图片理解 | 千问视觉模型（Qwen-VL） | 图片内容描述（通过DashScope API） |
| 视频处理 | FFmpeg + 关键帧提取 | 视频→图片序列→分析 |
| 文档解析 | PyMuPDF / python-docx / unstructured | 多格式文档解析 |

### 3.4 Python依赖清单

```
# === Web框架 ===
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
streamlit>=1.30.0

# === LangChain生态 ===
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0

# === 向量数据库 ===
chromadb>=0.4.0

# === 数据库 ===
pymysql>=1.1.0
sqlalchemy>=2.0.0
redis>=5.0.0

# === 多模态处理 ===
openai-whisper>=20231117
Pillow>=10.0.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
ffmpeg-python>=0.2.0
pymupdf>=1.23.0
python-docx>=1.0.0
unstructured>=0.12.0

# === 数据处理 ===
pandas>=2.1.0
numpy>=1.25.0

# === 工具库 ===
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
loguru>=0.7.0
httpx>=0.25.0
tiktoken>=0.5.0
aiofiles>=23.0.0
duckduckgo_search>=8.0.0
```

---

## 四、实施完成状态

> 以下为项目各阶段的实施记录。✓ 标记表示该步骤已完成。

---

### Phase 1: 基础RAG系统 ✅（已全部完成）

**目标**: 搭建项目骨架，实现文档上传→分块→向量化→检索→问答的完整链路。

#### Step 1.1: 项目初始化 ✓

```
执行内容（已完成）:
1. ✅ 创建项目目录结构
2. ✅ 创建 pyproject.toml / requirements.txt
3. ✅ 创建 .env.example（含 DashScope/MySQL/Redis/Qwen-VL 等配置模板）
4. ✅ 创建 app/config.py（pydantic-settings + @lru_cache）
5. ✅ 创建 app/main.py（FastAPI入口，含健康检查 + 中间件栈 + 路由注册）
6. ✅ 初始化 git 仓库

当前配置项（config.py）:
- LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（默认 qwen-plus，DashScope 接口）
- EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL（默认 text-embedding-v3）
- MYSQL_URL / REDIS_URL / CHROMA_PERSIST_DIR / UPLOAD_DIR
- API_KEY（可选认证） / LOG_LEVEL
```

#### Step 1.2: 数据模型定义 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/models/schemas.py — Pydantic请求/响应模型
2. ✅ 创建 app/models/database.py — SQLAlchemy ORM模型
3. ✅ 创建 scripts/init_db.py — 数据库初始化脚本

核心数据模型:
- ChatRequest / ChatResponse（含 sources + session_id）
- DocumentUpload（含 doc_id, filename, chunk_count, status）
- KnowledgeBase / Document / Chunk / Session / Message / ToolCall
- 所有 ID 为 16 位字符串（uuid4 截取）
```

#### Step 1.3: 文档加载器 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/rag/loader.py — DocumentLoader

支持格式:
- PDF（PyMuPDF，保留页码）
- DOCX（python-docx，段落提取）
- Markdown / TXT / CSV
- 统一输出: [{ content, metadata: { source, page, type } }]

2. ✅ 创建 app/core/rag/splitter.py — TextSplitter

分块策略:
- RecursiveCharacterSplitter（默认，chunk_size=500, overlap=50）
- SentenceSplitter / MarkdownSplitter
- 保留 metadata（来源、页码）
```

#### Step 1.4: Embedding与向量存储 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/services/embedding_service.py — EmbeddingService

三级降级策略:
  - API 模式：DashScope text-embedding-v3（OpenAI兼容接口）
  - 本地模式：shibing624/text2vec-base-chinese（备选，numpy兼容性问题暂禁用）
  - 哈希回退：基于 SHA-512 的确定性向量（保底方案）
- 内存级缓存（相同文本不重复调用 API）
- 批量 Embedding（每次最多 100 条）

2. ✅ 创建 app/core/rag/embedder.py — VectorStore（ChromaDB封装）

接口:
- add_documents / search / delete_by_doc_id / get_stats
- 检索后按相似度阈值过滤（threshold=0.7）

3. ✅ 创建 app/core/rag/retriever.py — Retriever

流程: Query Embedding → 向量检索 top_k → 阈值过滤 → 返回（含相似度分数）
```

#### Step 1.5: RAG问答链路 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/services/llm_service.py — LLMService
   - 使用 langchain-openai（兼容 DashScope 接口）
   - Stream + 非 Stream 双模式
   - 模型: qwen-plus

2. ✅ 创建 app/core/rag/chain.py — RAGChain
   - 完整链路：向量检索 → 构建 Prompt → LLM → 来源引用
   - System Prompt: "你是知识库助手，基于以下参考资料回答用户问题..."

3. ✅ 创建 app/api/chat.py — 对话API
   - POST /api/chat/send（mode: "rag" | "agent"）
   - GET /api/chat/history/{session_id}
```

#### Step 1.6: 文件上传与知识库管理API ✓

```
执行内容（已完成）:
1. ✅ 创建 app/api/upload.py — 文件上传API
   - POST /api/upload（自动解析→分块→Embedding→存入向量库）
   - DELETE /api/upload/{doc_id}

2. ✅ 创建 app/api/knowledge.py — 知识库管理API
   - POST /api/knowledge/create / GET /api/knowledge/list
   - GET /api/knowledge/{kb_id} / DELETE /api/knowledge/{kb_id}

3. ✅ 创建 app/api/admin.py — 管理API
   - GET /api/admin/stats / GET /api/admin/health
```

#### Step 1.7: Streamlit基础前端 ✓

```
执行内容（已完成）:
1. ✅ 创建 frontend/app.py
   - 左侧栏: 知识库列表 + 创建 + 上传文件
   - 主区域: 对话界面（消息气泡 + 输入框）
   - RAG/Agent 模式切换
   - 引用来源可展开查看
```

---

### Phase 2: AI Agent框架 ✅（已全部完成）

**目标**: 实现Agent任务规划与执行，支持多工具调用，具备多轮对话记忆。

#### Step 2.1: 工具集开发 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/agent/tools/ 目录下 5 个工具

工具清单:
- web_search: Web搜索（duckduckgo_search 库）
- calculator: 数学计算（安全 eval）
- code_execute: 代码执行（subprocess + 超时控制 10 秒）
- file_operations: 文件读写/目录列表（限制工作目录）
- knowledge_base_search: 知识库检索（封装 RAG 为 Agent 工具）

2. ✅ 创建 app/core/agent/tools/__init__.py — TOOL_REGISTRY 字典
   - 统一注册：name → 工具实例
   - 接口: get_tools() / get_tool_descriptions()

3. ✅ 创建 scripts/test_tools.py — 工具独立测试脚本
```

#### Step 2.2: Agent执行器 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/agent/executor.py — AgentExecutor

实现方式:
- 使用 LangChain create_react_agent + AgentExecutor
- ReAct 循环: Thought → Action → Observation
- 最大迭代次数: 5 轮
- 返回: 最终回答 + 执行轨迹（每步的工具调用和结果）

2. ✅ 创建 app/core/agent/prompt.py

提示词模板:
- SYSTEM_PROMPT: 角色定义 + 工具使用规范
- 要求先思考再行动(Think → Act → Observe)
- 注意: 原 planner.py 为重型规划方案，实际使用 LangChain 内置 ReAct 框架
```

#### Step 2.3: 记忆管理 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/memory/short_term.py — ShortTermMemory
   - 滑动窗口: 保留最近 N 轮对话（默认 10 轮）
   - Token 预算控制：超出时自动裁剪

2. ✅ 创建 app/core/memory/long_term.py — LongTermMemory
   - MySQL 存储：用户偏好 / 重要事实 / 对话摘要

3. ✅ 创建 app/core/memory/context.py — ContextManager
   - 构建完整上下文窗口（System + 长期记忆 + 会话历史 + RAG结果 + 用户问题）
   - 总 Token 预算: ~7500 tokens
```

#### Step 2.4: Agent集成到API ✓

```
执行内容（已完成）:
1. ✅ 更新 app/api/chat.py
   - POST /api/chat/send 新增 mode 参数: "rag"（默认）/ "agent"
   - Agent 模式返回值含 agent_trace（执行轨迹）
   - Agent 模式请求超时 300 秒

2. ✅ 更新前端
   - RAG/Agent 模式切换
   - Agent 模式下工具调用过程折叠展示
```

---

### Phase 3: 多模态能力 ✅（已全部完成）

**目标**: 支持图片/音频/视频的处理与索引，实现多模态统一检索与问答。

#### Step 3.1: 音频处理模块 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/multimodal/audio.py — AudioProcessor
   - 使用 OpenAI Whisper 本地模型进行转写
   - 支持格式: mp3, wav, m4a, flac, ogg
   - 支持带时间戳的分段转写
   - 转写结果自动进入 RAG 分块流程
```

#### Step 3.2: 图片处理模块 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/multimodal/ocr.py — OCREngine（PaddleOCR）
   - extract_text / extract_text_with_boxes（含位置坐标 + 置信度）
   - 支持中英文混合识别

2. ✅ 创建 app/core/multimodal/windows_ocr.py — Windows OCR（备选方案）
   - 使用 Windows.Media.Ocr 原生 API（Windows-only）
   - 当 PaddleOCR 不可用时自动降级

3. ✅ 创建 app/core/multimodal/image.py — ImageProcessor
   - OCR 提取 + AI 描述（Qwen-VL 视觉模型）
   - 合并结果统一索引
```

#### Step 3.3: 视频处理模块 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/multimodal/video.py — VideoProcessor
   - extract_frames: FFmpeg 按时间间隔提取关键帧（默认每 30 秒）
   - extract_audio: FFmpeg 提取音频轨道
   - process_video: 完整流程（关键帧→图片处理 + 音频→Whisper转写）
```

#### Step 3.4: 多模态统一处理管线 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/multimodal/pipeline.py — MultiModalPipeline
   - 根据文件类型自动路由到对应处理器
   - 统一输出: { content: str, metadata: Dict }
   - 结果自动进入 RAG 分块和索引流程

2. ✅ 更新 app/api/upload.py — 支持所有多模态格式上传
```

---

### Phase 4: 工程化与优化 ✅（已完成）

**目标**: API规范化、缓存优化、并发控制、Docker部署。

#### Step 4.1: 缓存与性能优化 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/services/cache_service.py — CacheService（Redis封装）
   - 会话缓存 / 查询缓存（5 分钟 TTL）/ Embedding 缓存
   - 接口: get / set / delete / exists / ttl

2. ✅ Embedding 批处理（每次最多 100 条）
```

#### Step 4.2: API规范化 ✓

```
执行内容（已完成）:
1. ✅ 中间件栈：
   - app/middleware/rate_limit.py — 限流（Redis，60 请求/分钟）
   - app/middleware/logging.py — 请求日志（请求/响应/耗时）
   - CORS 中间件（app/main.py 中注册）
   - 注意: 认证中间件（auth.py）未实现，可通过 API_KEY 可选配置

2. ✅ app/utils/exceptions.py — 自定义异常 + 全局异常处理器
3. ✅ 统一响应格式（通过异常处理器保证一致性）
```

#### Step 4.3: Docker部署 ✓

```
执行内容（已完成）:
1. ✅ docker/Dockerfile — 后端镜像（python:3.11-slim）
2. ✅ docker/Dockerfile.frontend — 前端镜像
3. ✅ docker/docker-compose.yml
   - 服务: backend(8000) + frontend(8501) + mysql(3306) + redis(6379)
   - 数据卷持久化 + 健康检查 + 环境变量注入
4. ✅ 注意: 原计划中独立的 ChromaDB 服务未部署（ChromaDB 以内嵌模式运行）
```

#### Step 4.4: 测试 ✓

```
执行内容（已完成）:
1. ✅ tests/test_rag.py — RAG 单元测试
2. ✅ tests/test_api.py — API 集成测试
3. ✅ scripts/benchmark.py — 性能测试脚本（含并发压力测试）

注意: tests/test_agent.py 未独立创建（Agent 测试集成在 test_api.py 中）
```

---

### Phase 5: 进阶功能 ✅（已全部完成）

#### Step 5.1: 模型微调(Fine-tuning) ✓

```
执行内容（已完成）:
1. ✅ 创建 scripts/finetune_data.py — 微调数据准备（从知识库提取 QA 对）
2. ✅ 创建 scripts/finetune.py — LoRA 微调（transformers + peft）
   - 支持 Qwen/LLaMA 等开源模型
   - epochs=3, lr=2e-4, lora_rank=8
3. ✅ 创建 app/services/finetuned_service.py — 微调模型推理接口
```

#### Step 5.2: MCP协议集成 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/core/mcp/server.py — MCP Server（JSON-RPC over stdio）
   - 将知识库检索能力暴露为 MCP 工具
   - 文档管理能力暴露为 MCP 资源

2. ✅ 创建 scripts/mcp_qwen_vision.py — 千问视觉模型 MCP Server
   - 独立的图片识别 MCP 工具
   - 通过 image_recognize 工具暴露给 Claude Code
```

#### Step 5.3: 数据分析看板 ✓

```
执行内容（已完成）:
1. ✅ 创建 app/api/analytics.py — 数据分析API
   - 知识库使用统计（查询次数、热门问题）
   - 系统性能统计（响应时间、成功率）

2. ✅ 创建 frontend/pages/analytics.py
   - 数据分析页面（侧栏可访问）
   - Plotly 图表展示
```

---

## 五、关键技术设计细节

### 5.1 RAG检索策略

```
采用混合检索(Hybrid Search)策略:

1. 向量检索: 语义相似度匹配
   - 使用 ChromaDB/Milvus ANN检索
   - 返回 top_k=10 候选

2. 关键词检索: BM25精确匹配
   - 对query分词后进行关键词匹配
   - 补充向量检索可能遗漏的精确匹配

3. 重排序(Rerank):
   - 使用交叉编码器对候选结果重排序
   - 或使用LLM进行相关性评分
   - 最终返回 top_k=5

4. 引用追溯:
   - 每个chunk携带完整metadata
   - 回答时标注来源(文件名+页码+chunk位置)
```

### 5.2 上下文工程方案

```
上下文构建优先级(从高到低):

1. System Prompt (~500 tokens)
   - 角色定义
   - 回答规范
   - 工具使用说明

2. RAG检索结果 (~2000 tokens)
   - 最相关的5个文档片段
   - 每个片段标注来源

3. 会话历史 (~2000 tokens)
   - 最近10轮对话
   - 超出时自动摘要早期对话

4. 长期记忆 (~500 tokens)
   - 用户偏好
   - 历史关键信息

5. 用户当前问题 (~500 tokens)

6. 预留给回答 (~2000 tokens)

总Token预算: ~7500 tokens(适配大多数模型上下文窗口)
```

### 5.3 Agent执行流程

```
用户问题输入
    │
    ▼
┌─────────────┐
│ 意图识别     │ ← LLM判断: 直接回答 / 单工具 / 多步推理
└──────┬──────┘
       │
  ┌────┼────────────────┐
  ▼    ▼                 ▼
直接  单工具            多步推理
回答  调用              (ReAct循环)
  │    │                 │
  │    ▼                 ▼
  │  工具执行    ┌──────────────┐
  │    │        │ Think(思考)  │
  │    ▼        │ Act(执行工具) │
  │  返回结果    │ Observe(观察) │
  │    │        └──────┬───────┘
  │    │               │ (循环直到完成)
  ▼    ▼               ▼
┌─────────────────────────┐
│    生成最终回答           │
│    (含引用来源+执行轨迹)  │
└─────────────────────────┘
```

### 5.4 多模态处理流程

```
文件上传
    │
    ▼
┌──────────────┐
│  文件类型检测  │
└──────┬───────┘
       │
  ┌────┼────────┬──────────┐
  ▼    ▼        ▼          ▼
文档  图片      音频       视频
  │    │        │          │
  ▼    ▼        ▼          ▼
解析  OCR+     Whisper    ffmpeg
文本  AI描述   转写       提取
  │    │        │       关键帧+音频
  │    ▼        │          │
  │  合并结果    │     ┌────┼────┐
  │    │        │     ▼         ▼
  │    │        │  图片处理   音频转写
  │    │        │     │         │
  │    │        │     └────┬────┘
  │    │        │          │
  ▼    ▼        ▼          ▼
┌─────────────────────────────┐
│      统一文本输出             │
│      (含来源metadata)        │
└──────────────┬──────────────┘
               │
               ▼
         RAG分块索引
```

---

## 六、环境搭建指南

### 6.1 本地开发环境

```bash
# 1. Python环境(建议3.10+)
conda create -n kb-agent python=3.10
conda activate kb-agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 系统依赖
# FFmpeg(视频/音频处理)
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: apt install ffmpeg

# 4. 环境变量
copy .env.example .env  # Windows
# 编辑 .env 填入 DashScope API Key 等配置

# 5. 初始化数据库
python scripts/init_db.py

# 6. 启动后端（使用 Anaconda 环境的 uvicorn）
uvicorn app.main:app --reload --port 8000

# 7. 启动前端
streamlit run frontend/app.py --server.port 8501
```

### 6.2 Docker环境

```bash
# 一键启动
cd docker
docker-compose up -d

# 查看日志
docker-compose logs -f backend

# 停止
docker-compose down
```

---

## 七、简历撰写参考

### 项目描述

**多模态智能知识库Agent** | 独立开发 | 202X.XX - 202X.XX

基于LangChain + RAG + Agent架构，开发支持文档/图片/音频/视频多模态数据的智能知识库系统。

**核心工作:**
1. 设计并实现RAG系统，采用向量检索+阈值过滤策略，支持文档上传→自动解析→分块→向量化→问答的完整链路
2. 基于LangChain构建ReAct Agent框架，集成Web搜索/代码执行/文件操作/知识库检索5个工具，支持多步推理与工具调用
3. 实现多模态处理管线（Whisper音频转写/PaddleOCR图片文字提取/Qwen-VL视觉模型描述/FFmpeg视频分析），支持统一索引与检索
4. 设计上下文工程方案（短期记忆+长期记忆+Token预算分配），优化多轮对话质量
5. 基于FastAPI封装AI服务API，集成Redis缓存中间件与限流，Docker容器化部署
6. 实现MCP协议集成，将知识库检索和图片识别能力暴露为标准工具供外部调用

**技术栈:** Python, LangChain, FastAPI, ChromaDB, MySQL, Redis, Qwen/DashScope, Whisper, PaddleOCR, Docker

---

## 八、验收Checklist

每个Phase完成后，逐项检查:

### Phase 1 验收 ✅
- [x] 项目能正常启动(uvicorn + streamlit)
- [x] 能上传PDF/DOCX/MD/TXT文件
- [x] 文件自动分块并索引到向量库
- [x] 基于知识库的问答正常工作
- [x] 回答包含引用来源
- [x] 知识库CRUD正常

### Phase 2 验收 ✅
- [x] Agent模式能正确选择和调用工具
- [x] 多步推理任务正确执行
- [x] 多轮对话上下文连贯
- [x] 跨会话记忆正常
- [x] 工具调用轨迹可查看

### Phase 3 验收 ✅
- [x] 音频文件自动转写并索引
- [x] 图片OCR和AI描述正常
- [x] 视频关键帧提取和分析正常
- [x] 多模态内容可被检索和问答

### Phase 4 验收 ✅
- [x] API响应格式统一（异常处理器保证）
- [x] 限流生效（Redis，60请求/分钟）
- [x] Redis缓存正常工作
- [x] Docker一键部署成功
- [x] 性能测试脚本可用（scripts/benchmark.py）
- [x] 测试用例全部通过

### Phase 5 验收 ✅
- [x] LoRA微调流程完整（scripts/finetune.py）
- [x] MCP协议集成可用（app/core/mcp/server.py）
- [x] 数据分析看板展示正确（frontend/pages/analytics.py）
