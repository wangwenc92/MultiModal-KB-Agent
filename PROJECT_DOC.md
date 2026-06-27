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
| RAG系统 | LangChain + Milvus/ChromaDB | 大模型应用落地、RAG技能 |
| AI Agent | LangChain Agent + 工具链 | AI Agent实践 |
| 多模态处理 | Whisper + 视觉模型 + FFmpeg | 多模态数据处理 |
| 模型工程化 | FastAPI + Redis + Docker | 模型工程化与服务支撑 |
| 数据工程 | Pandas + NumPy + MySQL | 数据与存储协同 |

### 1.4 最终交付物
- 完整可运行的AI Agent后端服务
- Streamlit Web UI 前端
- API接口文档（自动生成）
- Docker一键部署方案
- 性能测试报告

---

## 二、系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│              (对话界面 / 文件上传 / 知识库管理)               │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI API Layer                         │
│         (路由 / 认证 / 限流 / 日志 / 异常处理)               │
└──────┬───────────┬───────────┬───────────┬──────────────────┘
       │           │           │           │
┌──────▼───┐ ┌─────▼────┐ ┌───▼────┐ ┌───▼─────────┐
│  Agent   │ │   RAG    │ │MultiModal│ │  Memory     │
│  Engine  │ │  Engine  │ │ Engine  │ │  Manager    │
│          │ │          │ │         │ │             │
│ -规划器  │ │ -文档解析│ │ -图片OCR│ │ -短期记忆   │
│ -工具链  │ │ -分块策略│ │ -音频转写│ │ -长期记忆   │
│ -执行器  │ │ -向量检索│ │ -视频分析│ │ -上下文窗口 │
└──────┬───┘ └─────┬────┘ └───┬────┘ └──────┬──────┘
       │           │          │              │
┌──────▼───────────▼──────────▼──────────────▼──────────────┐
│                    Storage Layer                           │
│  Milvus(向量)  MySQL(元数据)  Redis(缓存/会话)  MinIO(文件) │
└───────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
multimodal-kb-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI入口
│   ├── config.py                  # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py                # 对话API
│   │   ├── knowledge.py           # 知识库管理API
│   │   ├── upload.py              # 文件上传API
│   │   └── admin.py               # 管理API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── planner.py         # Agent任务规划器
│   │   │   ├── executor.py        # Agent执行器
│   │   │   ├── tools/             # 工具集
│   │   │   │   ├── __init__.py
│   │   │   │   ├── search.py      # Web搜索工具
│   │   │   │   ├── calculator.py  # 计算器
│   │   │   │   ├── code_exec.py   # 代码执行
│   │   │   │   ├── file_ops.py    # 文件操作
│   │   │   │   └── kb_search.py   # 知识库检索
│   │   │   └── prompt.py          # Agent提示词模板
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py          # 文档加载器(PDF/Word/MD/TXT)
│   │   │   ├── splitter.py        # 文本分块策略
│   │   │   ├── embedder.py        # Embedding封装
│   │   │   ├── retriever.py       # 检索器(混合检索)
│   │   │   └── reranker.py        # 重排序
│   │   ├── multimodal/
│   │   │   ├── __init__.py
│   │   │   ├── image.py           # 图片处理(OCR/描述)
│   │   │   ├── audio.py           # 音频处理(Whisper转写)
│   │   │   ├── video.py           # 视频处理(关键帧+分析)
│   │   │   └── ocr.py             # OCR引擎封装
│   │   └── memory/
│   │       ├── __init__.py
│   │       ├── short_term.py      # 短期记忆(会话内)
│   │       ├── long_term.py       # 长期记忆(跨会话)
│   │       └── context.py         # 上下文窗口管理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic数据模型
│   │   └── database.py            # 数据库连接
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py         # LLM调用封装
│   │   ├── embedding_service.py   # Embedding服务
│   │   └── cache_service.py       # Redis缓存服务
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 日志工具
│       └── helpers.py             # 通用工具函数
├── frontend/
│   └── app.py                     # Streamlit前端
├── scripts/
│   ├── init_db.py                 # 数据库初始化
│   └── benchmark.py               # 性能测试脚本
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_agent.py
│   └── test_api.py
├── docs/
│   └── api.md                     # API文档
├── .env.example                   # 环境变量模板
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 三、技术选型

### 3.1 核心框架

| 组件 | 技术选型 | 版本要求 | 说明 |
|------|---------|---------|------|
| Web框架 | FastAPI | >=0.100 | 异步高性能，自动生成API文档 |
| Agent框架 | LangChain | >=0.2 | 主流Agent开发框架 |
| 前端 | Streamlit | >=1.30 | 快速搭建AI应用UI |
| LLM | Claude API / OpenAI | - | 大模型调用 |
| Embedding | text-embedding-3-small | - | 文本向量化 |

### 3.2 存储层

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 向量数据库 | ChromaDB(开发) / Milvus(生产) | 文档向量存储与检索 |
| 关系数据库 | MySQL 8.0 | 元数据、用户、会话管理 |
| 缓存 | Redis 7.x | 会话缓存、热点数据、限流 |
| 对象存储 | 本地文件系统(开发) / MinIO(生产) | 上传文件存储 |

### 3.3 多模态处理

| 能力 | 技术选型 | 说明 |
|------|---------|------|
| 音频转写 | OpenAI Whisper API | 音频→文本 |
| 图片OCR | PaddleOCR / Tesseract | 图片文字提取 |
| 图片理解 | Claude Vision / GPT-4V | 图片内容描述 |
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
# milvus-lite>=2.4.0  # 生产环境

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
```

---

## 四、分阶段实施计划

---

### Phase 1: 基础RAG系统（第1-2周）

**目标**: 搭建项目骨架，实现文档上传→分块→向量化→检索→问答的完整链路。

#### Step 1.1: 项目初始化

```
执行内容:
1. 创建项目目录结构(按2.2节目录结构)
2. 创建 pyproject.toml / requirements.txt
3. 创建 .env.example (API密钥、数据库连接等配置模板)
4. 创建 app/config.py (使用 pydantic-settings 管理配置)
5. 创建 app/main.py (FastAPI基础入口，含健康检查接口)
6. 初始化 git 仓库，创建 .gitignore

配置项(config.py)需要包含:
- LLM_API_KEY: 大模型API密钥
- LLM_BASE_URL: API基础URL
- LLM_MODEL: 模型名称(默认claude-sonnet-4-20250514)
- EMBEDDING_MODEL: Embedding模型名称
- MYSQL_URL: MySQL连接串
- REDIS_URL: Redis连接串
- CHROMA_PERSIST_DIR: ChromaDB持久化目录
- UPLOAD_DIR: 上传文件存储目录
- LOG_LEVEL: 日志级别

验证标准:
- uvicorn app.main:app 能正常启动
- 访问 /docs 能看到Swagger文档
- 访问 /health 返回 {"status": "ok"}
```

#### Step 1.2: 数据模型定义

```
执行内容:
1. 创建 app/models/schemas.py — 定义所有Pydantic数据模型

核心数据模型:
- ChatRequest: 对话请求(question, session_id, knowledge_base_id)
- ChatResponse: 对话响应(answer, sources, session_id)
- DocumentUpload: 文档上传响应(doc_id, filename, status, chunk_count)
- KnowledgeBase: 知识库模型(id, name, description, doc_count, created_at)
- Document: 文档模型(id, kb_id, filename, file_type, file_size, chunk_count, status)
- Chunk: 文本块模型(id, doc_id, content, metadata, embedding_status)
- Session: 会话模型(id, title, created_at, message_count)
- Message: 消息模型(id, session_id, role, content, created_at, sources)
- ToolCall: 工具调用模型(tool_name, input, output, duration_ms)

2. 创建 app/models/database.py — SQLAlchemy模型定义(对应上述Pydantic模型)
3. 创建 scripts/init_db.py — 数据库初始化脚本(建表)

验证标准:
- Python import无报错
- init_db.py 能成功连接MySQL并创建所有表
```

#### Step 1.3: 文档加载器

```
执行内容:
1. 创建 app/core/rag/loader.py

实现 DocumentLoader 类，支持以下格式:
- PDF: 使用 PyMuPDF(fitz) 提取文本，保留页码信息
- DOCX: 使用 python-docx 提取段落文本
- Markdown: 直接读取，解析标题层级
- TXT: 直接读取
- CSV: 使用 pandas 读取为结构化文本

每个加载器输出统一格式:
[
    {
        "content": "文本内容",
        "metadata": {
            "source": "文件名",
            "page": 页码(如有),
            "type": "文件类型"
        }
    }
]

2. 创建 app/core/rag/splitter.py

实现 TextSplitter 类，支持以下分块策略:
- RecursiveCharacterSplitter: 递归字符分割(默认)
  - chunk_size: 500 tokens
  - chunk_overlap: 50 tokens
  - separators: ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
- SentenceSplitter: 按句子分割
- MarkdownSplitter: 按Markdown标题分割

关键实现细节:
- 分块时保留metadata(来源、页码等)
- 每个chunk计算token数量
- 支持中文分词优化

验证标准:
- 能成功解析PDF/DOCX/MD/TXT文件
- 分块结果符合预期(chunk_size和overlap正确)
- metadata信息完整传递
```

#### Step 1.4: Embedding与向量存储

```
执行内容:
1. 创建 app/services/embedding_service.py

实现 EmbeddingService 类:
- 封装 OpenAI text-embedding-3-small 调用
- 支持批量Embedding(每次最多100条)
- 实现本地缓存(相同文本不重复调用API)
- 接口: embed_text(text) -> List[float]
- 接口: embed_texts(texts: List[str]) -> List[List[float]]

2. 创建 app/core/rag/embedder.py

实现 VectorStore 类，封装 ChromaDB:
- 初始化: 创建/获取 Collection
- add_documents(chunks, embeddings, metadatas) — 批量写入
- search(query_embedding, top_k=5) — 向量检索
- delete_by_doc_id(doc_id) — 按文档ID删除
- get_stats() — 返回统计信息(文档数、chunk数等)

3. 创建 app/core/rag/retriever.py

实现 Retriever 类:
- retrieve(query, kb_id, top_k=5) — 完整检索流程
  1. Query Embedding
  2. 向量检索 top_k
  3. 按相似度阈值过滤(threshold=0.7)
  4. 返回结果(含相似度分数)

验证标准:
- Embedding调用成功，返回正确维度的向量
- ChromaDB能正确存储和检索
- 检索结果按相似度降序排列
```

#### Step 1.5: RAG问答链路

```
执行内容:
1. 创建 app/services/llm_service.py

实现 LLMService 类:
- 封装 Claude API 调用(使用 langchain_anthropic)
- 支持流式输出(stream=True)
- 统一消息格式处理
- 接口: chat(messages, tools=None) -> response
- 接口: chat_stream(messages) -> AsyncGenerator

2. 创建 app/core/rag/retriever.py — 补充完整RAG链路

实现 RAGChain 类(完整RAG流程):
- 输入: 用户问题 + 知识库ID
- 流程:
  1. 问题向量化
  2. 向量检索相关文档(top_k=5)
  3. 构建Prompt:
     System: "你是知识库助手，基于以下参考资料回答用户问题。
             如果参考资料中没有相关信息，请如实告知。
             回答时请标注引用来源。"
     Context: 检索到的文档片段(含来源标注)
     User: 用户问题
  4. 调用LLM生成回答
  5. 返回(answer, sources)

3. 创建 app/api/chat.py — 对话API

API接口:
- POST /api/chat/send — 发送消息，获取回答
  Request: { question: str, session_id?: str, knowledge_base_id: str }
  Response: { answer: str, sources: List[Source], session_id: str }

- GET /api/chat/history/{session_id} — 获取会话历史

验证标准:
- 上传文档后，能基于文档内容回答问题
- 回答中包含引用来源(文件名+页码)
- 流式输出正常工作
```

#### Step 1.6: 文件上传与知识库管理API

```
执行内容:
1. 创建 app/api/upload.py — 文件上传API

API接口:
- POST /api/upload — 上传文件
  Request: multipart/form-data(file) + knowledge_base_id
  流程: 保存文件 → 加载解析 → 分块 → Embedding → 存入向量库
  Response: { doc_id, filename, chunk_count, status }

- DELETE /api/upload/{doc_id} — 删除文档及其chunks

2. 创建 app/api/knowledge.py — 知识库管理API

API接口:
- POST /api/knowledge/create — 创建知识库
- GET /api/knowledge/list — 获取知识库列表
- GET /api/knowledge/{kb_id} — 获取知识库详情(含文档列表)
- DELETE /api/knowledge/{kb_id} — 删除知识库(级联删除文档和chunks)

3. 创建 app/api/admin.py — 管理API

API接口:
- GET /api/admin/stats — 系统统计(文档数、chunk数、会话数)
- GET /api/admin/health — 健康检查(各组件连通性)

验证标准:
- 上传文件后自动完成解析→分块→向量化流程
- 知识库CRUD正常
- 文件删除时向量库数据同步清除
```

#### Step 1.7: Streamlit基础前端

```
执行内容:
1. 创建 frontend/app.py

实现基础UI:
- 左侧栏: 知识库列表 + 创建知识库 + 上传文件
- 主区域: 对话界面(消息气泡 + 输入框)
- 支持选择知识库进行问答
- 支持查看回答的引用来源
- 显示会话历史

关键组件:
- st.sidebar: 知识库管理
- st.chat_message: 对话消息展示
- st.file_uploader: 文件上传
- st.expander: 引用来源展开

验证标准:
- 可以在UI上创建知识库、上传文件
- 上传后自动索引完成
- 对话正常，回答基于知识库内容
- 引用来源可展开查看
```

---

### Phase 2: AI Agent框架（第3-4周）

**目标**: 实现Agent任务规划与执行，支持多工具调用，具备多轮对话记忆。

#### Step 2.1: 工具集开发

```
执行内容:
1. 创建 app/core/agent/tools/ 目录下各工具文件

工具1: search.py — Web搜索工具
- 使用 duckduckgo_search 库调用 DuckDuckGo Lite 接口
- 输入: query(str)
- 输出: 搜索结果列表(title, body, url)
- 接口: class WebSearchTool(BaseTool)
- 注意: 原 httpx 直调 DuckDuckGo API 方案在国内网络环境下不可用，已替换

工具2: calculator.py — 数学计算工具
- 使用 Python eval 安全执行数学表达式
- 输入: expression(str)
- 输出: 计算结果
- 安全: 使用 ast.literal_eval 或 sympy，禁止危险操作

工具3: code_exec.py — 代码执行工具
- 在受限环境中执行Python代码
- 输入: code(str)
- 输出: 执行结果(stdout)或错误信息
- 安全: 使用 subprocess + 超时控制(10秒) + 资源限制

工具4: file_ops.py — 文件操作工具
- 支持: 读取文件、写入文件、列出目录
- 输入: operation(str), path(str), content?(str)
- 安全: 限制在指定工作目录内

工具5: kb_search.py — 知识库检索工具
- 封装RAG检索为Agent工具
- 输入: query(str), kb_id(str)
- 输出: 检索结果(含来源)

2. 创建 app/core/agent/tools/__init__.py
- 注册所有工具，导出 TOOL_REGISTRY

每个工具需要:
- 继承 langchain.tools.BaseTool
- 定义 name, description, args_schema
- 实现 _run() 方法
- 实现错误处理和超时控制

验证标准:
- 每个工具独立测试可用
- 工具描述准确，LLM能正确选择工具
- 错误场景处理得当(超时、权限、格式错误)
```

#### Step 2.2: Agent规划器与执行器

```
执行内容:
1. 创建 app/core/agent/planner.py

实现 AgentPlanner 类:
- 使用 LangChain 的 create_react_agent 或自定义规划逻辑
- 输入: 用户问题 + 可用工具列表 + 上下文
- 输出: 执行计划(步骤列表)
- 支持:
  - 简单问题直接回答(不调用工具)
  - 单工具调用
  - 多步推理(链式工具调用)
  - 并行工具调用(多工具同时执行)

2. 创建 app/core/agent/executor.py

实现 AgentExecutor 类:
- 接收规划器输出的执行计划
- 逐步执行，收集中间结果
- 处理异常和重试
- 支持最大迭代次数限制(默认5次)
- 返回: 最终回答 + 执行轨迹(每步的工具调用和结果)

3. 创建 app/core/agent/prompt.py

定义Agent提示词模板:
- SYSTEM_PROMPT: 角色定义 + 工具使用规范 + 输出格式
- PLANNING_PROMPT: 任务规划提示
- TOOL_SELECTION_PROMPT: 工具选择提示
- FINAL_ANSWER_PROMPT: 最终回答生成提示

Prompt设计要点:
- 明确告知可用工具及其用途
- 要求先思考再行动(Think → Act → Observe)
- 限制推理步数，避免无限循环
- 要求输出结构化的思考过程

验证标准:
- "帮我搜索最新的AI新闻" → 调用WebSearchTool → 返回结果
- "计算 (15*23+45)/3" → 调用CalculatorTool → 返回结果
- "在知识库中搜索XXX并总结" → 调用KBSearchTool → 总结返回
- "搜索XXX，然后用代码分析数据" → 多步推理正确执行
```

#### Step 2.3: 记忆管理

```
执行内容:
1. 创建 app/core/memory/short_term.py

实现 ShortTermMemory 类:
- 基于会话的消息历史管理
- 滑动窗口策略: 保留最近N轮对话(默认10轮)
- Token预算控制: 超出token限制时自动裁剪
- 接口: add_message(role, content) / get_history() / clear()

2. 创建 app/core/memory/long_term.py

实现 LongTermMemory 类:
- 基于MySQL的跨会话记忆存储
- 记忆类型:
  - 用户偏好(从对话中提取)
  - 重要事实(关键信息摘要)
  - 对话摘要(会话结束时自动生成)
- 接口: save_memory(memory_type, content) / search_memories(query, top_k)

3. 创建 app/core/memory/context.py

实现 ContextManager 类:
- 组合短期记忆和长期记忆
- 构建完整上下文窗口:
  1. System Prompt
  2. 长期记忆(相关用户偏好/历史摘要)
  3. 当前会话历史(短期记忆)
  4. 当前RAG检索结果(如有)
  5. 用户当前问题
- Token预算分配:
  - System Prompt: ~500 tokens
  - 长期记忆: ~500 tokens
  - 会话历史: ~2000 tokens
  - RAG上下文: ~2000 tokens
  - 预留给回答: ~2000 tokens
- 接口: build_context(query, session_id, kb_id) -> messages

验证标准:
- 多轮对话中能正确引用前文
- 跨会话能记住用户偏好
- Token超限时自动裁剪，不报错
```

#### Step 2.4: Agent集成到API

```
执行内容:
1. 更新 app/api/chat.py — 集成Agent能力

改造 POST /api/chat/send 接口:
- 新增 mode 参数: "rag"(默认) 或 "agent"
- agent模式下调用AgentExecutor处理问题
- 返回值新增 agent_trace 字段(执行轨迹)

新增 API接口:
- POST /api/chat/agent — Agent模式专用接口
  Request: { question, session_id, tools?: List[str] }
  Response: { answer, agent_trace, session_id }

- GET /api/chat/tools — 获取可用工具列表

2. 更新前端 app.py
- 新增"模式切换"(RAG模式 / Agent模式)
- Agent模式下展示工具调用过程(折叠显示)
- Agent模式请求超时设为300秒（多轮推理链路较长，60秒默认超时不足）

验证标准:
- RAG模式和Agent模式都能正常工作
- Agent模式下工具调用轨迹可查看
- 模式切换不影响会话连续性
```

---

### Phase 3: 多模态能力（第5-6周）

**目标**: 支持图片/音频/视频的处理与索引，实现多模态统一检索与问答。

#### Step 3.1: 音频处理模块

```
执行内容:
1. 创建 app/core/multimodal/audio.py

实现 AudioProcessor 类:
- transcribe(audio_path) -> str
  - 使用 OpenAI Whisper API 进行语音转写
  - 支持格式: mp3, wav, m4a, flac, ogg
  - 支持中文和英文
  - 返回: 转写文本 + 时间戳
- transcribe_with_timestamps(audio_path) -> List[Segment]
  - 返回带时间戳的分段转写结果
  - 每个Segment包含: start_time, end_time, text

实现细节:
- 大文件自动分片处理(每片25MB)
- 支持异步处理(上传后异步转写)
- 转写结果自动进入RAG分块流程

验证标准:
- 上传音频文件后自动转写为文本
- 转写结果正确索引到向量库
- 能基于音频内容回答问题
```

#### Step 3.2: 图片处理模块

```
执行内容:
1. 创建 app/core/multimodal/ocr.py

实现 OCREngine 类:
- 使用 PaddleOCR 进行文字识别
- extract_text(image_path) -> str
- extract_text_with_boxes(image_path) -> List[TextBlock]
  - 返回: 文字 + 位置坐标 + 置信度
- 支持中英文混合识别

2. 创建 app/core/multimodal/image.py

实现 ImageProcessor 类:
- ocr_extract(image_path) -> str
  - 调用OCREngine提取文字
- describe_image(image_path) -> str
  - 调用Claude Vision / GPT-4V生成图片描述
  - 用于图片内容的语义索引
- process_image(image_path) -> Dict
  - 完整处理流程: OCR提取 + AI描述 + 合并结果
  - 输出统一格式供RAG索引

支持格式: jpg, png, bmp, tiff, webp

实现细节:
- OCR和AI描述的结果合并，提高检索覆盖度
- 图片描述包含: 内容概述、文字内容、关键信息
- 处理结果自动进入RAG分块流程

验证标准:
- 上传图片后能提取文字内容
- 图片描述准确，包含关键信息
- 能基于图片内容回答问题(如"图片中的表格数据是什么")
```

#### Step 3.3: 视频处理模块

```
执行内容:
1. 创建 app/core/multimodal/video.py

实现 VideoProcessor 类:
- extract_frames(video_path, interval=30) -> List[str]
  - 使用 ffmpeg 按时间间隔提取关键帧
  - 默认每30秒一帧
  - 返回: 关键帧图片路径列表
- extract_audio(video_path) -> str
  - 使用 ffmpeg 提取音频轨道
  - 返回: 音频文件路径
- process_video(video_path) -> Dict
  - 完整处理流程:
    1. 提取关键帧
    2. 对每帧进行图片处理(OCR+描述)
    3. 提取音频并转写
    4. 合并所有结果
  - 输出: { frames: [...], transcript: str, summary: str }

实现细节:
- 视频处理为异步任务(可能耗时较长)
- 提供处理进度回调
- 大文件支持断点续处理

验证标准:
- 上传视频后能提取关键帧和音频
- 视频内容(画面+音频)都能被索引
- 能基于视频内容回答问题
```

#### Step 3.4: 多模态统一处理管线

```
执行内容:
1. 创建 app/core/multimodal/pipeline.py

实现 MultiModalPipeline 类:
- 统一入口，根据文件类型自动路由到对应处理器
- 支持的文件类型映射:
  - .pdf/.docx/.md/.txt → DocumentLoader
  - .jpg/.png/.bmp/.tiff/.webp → ImageProcessor
  - .mp3/.wav/.m4a/.flac/.ogg → AudioProcessor
  - .mp4/.avi/.mkv/.mov → VideoProcessor
- process_file(file_path) -> ProcessingResult
  - 自动检测文件类型
  - 调用对应处理器
  - 统一输出格式: { content: str, metadata: Dict }
  - 结果自动进入RAG分块和索引流程

2. 更新 app/api/upload.py
- 文件上传接口支持所有多模态格式
- 返回处理状态(处理中/完成/失败)
- 支持批量上传

3. 更新前端
- 文件上传组件支持所有格式
- 显示处理进度和状态
- 处理完成后通知用户

验证标准:
- 上传任意支持格式的文件，系统自动处理并索引
- 处理结果可在对话中被检索和引用
- 混合检索正常工作(同时检索文档和多模态内容)
```

---

### Phase 4: 工程化与优化（第7周）

**目标**: API规范化、缓存优化、并发控制、Docker部署。

#### Step 4.1: 缓存与性能优化

```
执行内容:
1. 创建 app/services/cache_service.py

实现 CacheService 类(Redis封装):
- 会话缓存: 缓存活跃会话的上下文(减少重复构建)
- 查询缓存: 相同问题+相同知识库 → 缓存回答(5分钟TTL)
- Embedding缓存: 相同文本不重复调用Embedding API
- 接口: get/set/delete/exists/ttl

2. 优化 RAG 检索性能:
- Embedding批处理(攒批后一次调用)
- 向量检索结果缓存
- 热点查询预计算

3. 优化 API 响应:
- FastAPI依赖注入优化
- 数据库连接池配置(SQLAlchemy pool_size=10)
- Redis连接池配置

验证标准:
- 相同问题重复查询，第二次响应时间<100ms
- 并发10个请求不报错
- Redis缓存命中率>30%
```

#### Step 4.2: API规范化

```
执行内容:
1. 统一API响应格式:

class APIResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Any = None

class APIError(BaseModel):
    code: int
    message: str
    detail: str = ""

2. 创建中间件:
- app/middleware/auth.py — API Key认证(简单Bearer Token)
- app/middleware/rate_limit.py — 限流中间件(基于Redis，每分钟60次)
- app/middleware/logging.py — 请求日志中间件(请求/响应/耗时)
- app/middleware/cors.py — CORS配置

3. 异常处理:
- app/utils/exceptions.py — 自定义异常类
- FastAPI全局异常处理器

4. 更新API文档:
- 每个接口添加详细docstring
- 添加请求/响应示例
- 标注错误码说明

验证标准:
- 所有API返回统一格式
- 限流生效(超过限制返回429)
- /docs 文档完整可用
```

#### Step 4.3: Docker部署

```
执行内容:
1. 创建 docker/Dockerfile — 后端服务镜像
   - 基于 python:3.11-slim
   - 安装系统依赖(ffmpeg, tesseract等)
   - 安装Python依赖
   - 暴露8000端口

2. 创建 docker/Dockerfile.frontend — 前端服务镜像
   - 基于 python:3.11-slim
   - 安装streamlit依赖
   - 暴露8501端口

3. 创建 docker/docker-compose.yml
   服务列表:
   - backend: FastAPI后端(8000)
   - frontend: Streamlit前端(8501)
   - mysql: MySQL 8.0(3306)
   - redis: Redis 7(6379)
   - chromadb: ChromaDB服务(8000)

   配置:
   - 网络: 所有服务在同一网络
   - 持久化: mysql/redis/chromadb 数据卷挂载
   - 环境变量: 通过.env文件注入
   - 健康检查: 每个服务配置healthcheck

4. 创建 docker/.env.example — Docker环境变量模板

验证标准:
- docker-compose up -d 一键启动所有服务
- 所有服务健康检查通过
- 端到端功能正常(上传→索引→问答)
```

#### Step 4.4: 测试

```
执行内容:
1. 创建 tests/test_rag.py
   - 测试文档加载各格式
   - 测试分块策略正确性
   - 测试Embedding生成
   - 测试向量检索准确性
   - 测试完整RAG问答链路

2. 创建 tests/test_agent.py
   - 测试各工具独立功能
   - 测试Agent单步推理
   - 测试Agent多步推理
   - 测试Agent错误处理

3. 创建 tests/test_api.py
   - 测试所有API接口
   - 测试认证和限流
   - 测试错误处理
   - 测试并发场景

4. 创建 scripts/benchmark.py
   - 单次问答延迟测试
   - 并发压力测试(10/50/100并发)
   - 向量检索性能测试
   - 生成性能测试报告

验证标准:
- 所有测试用例通过
- 单次问答延迟<3秒(P95)
- 并发50请求无错误
- 生成完整测试报告
```

---

### Phase 5: 进阶功能（第8周，可选加分项）

#### Step 5.1: 模型微调(Fine-tuning)

```
执行内容:
1. 创建 scripts/finetune_data.py
   - 从知识库中提取QA对(使用LLM自动生成)
   - 格式化为训练数据格式(JSONL)
   - 数据清洗和去重

2. 创建 scripts/finetune.py
   - 使用LoRA进行轻量微调
   - 基于 transformers + peft 库
   - 支持 Qwen/LLaMA 等开源模型
   - 训练参数: epochs=3, lr=2e-4, lora_rank=8

3. 创建 app/services/finetuned_service.py
   - 加载微调后的模型
   - 提供推理接口
   - 与基础模型A/B对比

验证标准:
- 微调数据集格式正确
- 微调训练正常完成
- 微调模型在领域问题上优于基础模型
```

#### Step 5.2: MCP协议集成

```
执行内容:
1. 创建 app/core/mcp/
   - server.py — MCP Server实现
   - tools.py — 暴露给MCP的工具定义
   - resources.py — 暴露给MCP的资源定义

2. 实现功能:
   - 将知识库检索能力暴露为MCP工具
   - 将文档管理能力暴露为MCP资源
   - 支持其他MCP客户端连接使用

验证标准:
- MCP客户端能连接到服务
- 能通过MCP协议调用知识库检索
- 文档资源可被MCP客户端发现和读取
```

#### Step 5.3: 数据分析看板

```
执行内容:
1. 创建 app/api/analytics.py
   - 知识库使用统计(查询次数、热门问题)
   - 系统性能统计(响应时间、成功率)
   - 用户行为分析(活跃会话、常用工具)

2. 更新前端
   - 添加数据分析页面
   - 图表展示(使用plotly)
   - 支持时间范围筛选

验证标准:
- 统计数据准确
- 图表正确展示
- 性能开销可接受
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
# 1. Python环境(建议3.11+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 系统依赖
# FFmpeg(视频/音频处理)
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: apt install ffmpeg

# 4. 环境变量
cp .env.example .env
# 编辑 .env 填入API密钥等配置

# 5. 初始化数据库
python scripts/init_db.py

# 6. 启动后端
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
1. 设计并实现RAG系统，采用混合检索(向量+BM25)+重排序策略，检索召回率达XX%
2. 基于LangChain构建Agent框架，集成Web搜索、代码执行等10+工具，支持多步推理与任务规划
3. 实现多模态处理管线，支持PDF/图片OCR/音频转写/视频分析的统一索引与检索
4. 设计上下文工程方案，结合短期记忆与长期记忆，优化多轮对话质量
5. 基于FastAPI封装AI服务API，集成Redis缓存与异步处理，P95延迟<XXms
6. 使用Docker容器化部署，支持一键启动和水平扩展

**技术栈:** Python, LangChain, FastAPI, ChromaDB/Milvus, MySQL, Redis, Whisper, PaddleOCR, Docker

---

## 八、验收Checklist

每个Phase完成后，逐项检查:

### Phase 1 验收
- [ ] 项目能正常启动(uvicorn + streamlit)
- [ ] 能上传PDF/DOCX/MD/TXT文件
- [ ] 文件自动分块并索引到向量库
- [ ] 基于知识库的问答正常工作
- [ ] 回答包含引用来源
- [ ] 知识库CRUD正常

### Phase 2 验收
- [ ] Agent模式能正确选择和调用工具
- [ ] 多步推理任务正确执行
- [ ] 多轮对话上下文连贯
- [ ] 跨会话记忆正常
- [ ] 工具调用轨迹可查看

### Phase 3 验收
- [ ] 音频文件自动转写并索引
- [ ] 图片OCR和AI描述正常
- [ ] 视频关键帧提取和分析正常
- [ ] 多模态内容可被检索和问答

### Phase 4 验收
- [ ] API响应格式统一
- [ ] 认证和限流生效
- [ ] Redis缓存正常工作
- [ ] Docker一键部署成功
- [ ] 性能测试通过(延迟、并发)
- [ ] 测试用例全部通过

### Phase 5 验收(可选)
- [ ] LoRA微调流程完整
- [ ] MCP协议集成可用
- [ ] 数据分析看板展示正确
