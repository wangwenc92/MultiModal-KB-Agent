# MultiModal-KB-Agent - 项目技术文档

## 一、项目概述

### 1.1 项目名称
**MultiModal-KB-Agent** — 多模态智能知识库 Agent 系统

### 1.2 项目定位
基于大语言模型的多模态智能知识库 Agent，支持文档/图片/音频/视频的自动索引与智能问答，
具备 Agent 任务规划、工具调用、上下文工程等能力，面向企业级 AI 应用落地场景。

### 1.3 核心能力矩阵

| 能力模块 | 技术栈 | JD 覆盖 |
|---------|--------|--------|
| RAG 系统 | LangChain + ChromaDB + 向量检索 | 大模型应用落地、RAG 技能 |
| AI Agent | LangChain ReAct + 工具注册表（5 个工具） | AI Agent 实践 |
| 多模态处理 | Whisper + Qwen-VL + PaddleOCR + FFmpeg | 多模态数据处理 |
| 模型工程化 | FastAPI + Redis + Docker + MCP 协议 | 模型工程化与服务支撑 |
| 数据工程 | Pandas + NumPy + MySQL + SQLAlchemy | 数据与存储协同 |
| 图片识别 | Qwen-VL 视觉模型 + PaddleOCR + Windows OCR | 多模态图片理解 |

### 1.4 最终交付物
- FastAPI 后端服务（RAG + Agent 双模式）
- Streamlit Web UI 前端（含数据分析看板）
- 统一 MCP Server（知识库检索 + 图片识别）
- Docker 一键部署方案（MySQL + Redis + 后端 + 前端）
- 3 个工作树独立维护，可单独开发部署

---

## 二、系统架构

### 2.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (frontend)               │
│     (对话界面 / 文件上传 / 知识库管理 / 数据分析看板)         │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend (backend)                 │
│       (API 路由 / Agent 引擎 / RAG 引擎 / 多模态处理)        │
└──────┬───────────┬───────────┬───────────┬──────────────────┘
       │           │           │           │
┌──────▼───┐ ┌─────▼────┐ ┌───▼────┐ ┌───▼─────────┐
│  Agent   │ │   RAG    │ │MultiModal│ │  Memory     │
│  Engine  │ │  Engine  │ │ Engine  │ │  Manager    │
│          │ │          │ │         │ │             │
│ -ReAct   │ │ -文档解析│ │ -图片OCR│ │ -短期记忆   │
│  循环    │ │ -分块策略│ │ -音频转写│ │ -长期记忆   │
│ -5个工具 │ │ -向量检索│ │ -视频分析│ │ (DB 持久化) │
│ -工具注册│ │ -链式回答│ │ -AI描述  │ │             │
└──────┬───┘ └─────┬────┘ └───┬────┘ └──────┬──────┘
       │           │          │              │
┌──────▼───────────▼──────────▼──────────────▼──────────────┐
│                    Storage Layer                            │
│  ChromaDB(向量)  MySQL(元数据+记忆)  Redis(缓存/会话)  文件  │
└────────────┬──────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────────────────────────────┐
│              MCP Server (mcp-server 独立工作树)             │
│   kb_tools: 知识库搜索/列表   vision_tools: 图片识别       │
│   └── 通过 --tools kb|vision|all 参数选择                   │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Git 工作树架构

项目重构为 **4 个独立的 git 工作树**，每个在同一个仓库的不同分支上维护：

```
MultiModal-KB-Agent/
│
├── [master 分支 - D:/test2]               ← 共享基础设施
│   ├── .env.example / .gitignore
│   ├── CLAUDE.md / PROJECT_DOC.md
│   ├── docker/docker-compose.yml          ← 完整编排 4 个服务
│   └── pyproject.toml / requirements.txt  ← 整合依赖
│
├── [backend 分支 - D:/kb-backend]         ← FastAPI 后端
│   ├── app/                               ← 全部后端代码
│   │   ├── api/          (6 个路由)
│   │   ├── core/agent/   (ReAct Agent + 5 工具)
│   │   ├── core/rag/     (RAG 流水线)
│   │   ├── core/multimodal/(图片/音频/视频处理)
│   │   ├── core/memory/  (短期+长期记忆，DB 持久化)
│   │   ├── services/     (LLM/Embedding/Cache/微调)
│   │   ├── models/       (Pydantic + SQLAlchemy)
│   │   └── middleware/   (限流/日志)
│   ├── tests/            (19 个测试全部通过)
│   ├── Dockerfile
│   └── requirements.txt
│
├── [mcp-server 分支 - D:/kb-mcp]          ← 统一 MCP 服务
│   └── mcp_server/
│       ├── server.py      ← JSON-RPC stdio 主循环
│       ├── kb_tools.py    ← 知识库工具
│       ├── vision_tools.py← 图片识别工具
│       └── config.py      ← 配置加载
│
└── [frontend 分支 - D:/kb-frontend]       ← Streamlit 前端
    ├── frontend/          ← Streamlit 应用
    └── requirements.txt   ← 仅 streamlit + requests + plotly
```

---

## 三、技术选型

### 3.1 核心框架

| 组件 | 技术选型 | 版本要求 | 说明 |
|------|---------|---------|------|
| Web 框架 | FastAPI | >=0.100 | 异步高性能，自动生成 API 文档 |
| Agent 框架 | LangChain | >=0.2 | ReAct 模式（手写循环，不使用 create_react_agent） |
| 前端 | Streamlit | >=1.30 | 快速搭建 AI 应用 UI |
| LLM | 阿里云 DashScope（Qwen） | - | OpenAI 兼容接口，默认 qwen-plus |
| Embedding | DashScope text-embedding-v3 | - | 三级降级：API → 本地模型 → 哈希回退 |

### 3.2 存储层

| 组件 | 技术选型 | 用途 |
|------|---------|------|
| 向量数据库 | ChromaDB | 文档向量存储与检索（本地持久化） |
| 关系数据库 | MySQL 8.0 | 元数据、会话、长期记忆管理 |
| 缓存 | Redis 7.x | 会话缓存、查询缓存、Embedding 缓存、限流 |
| 对象存储 | 本地文件系统 | 上传文件存储 |

### 3.3 多模态处理

| 能力 | 技术选型 | 说明 |
|------|---------|------|
| 音频转写 | OpenAI Whisper（本地模型） | 音频→文本，支持中英文 |
| 图片 OCR | PaddleOCR / Windows OCR（备选） | 图片文字提取 |
| 图片理解 | 千问视觉模型（Qwen-VL） | 图片内容描述（通过 DashScope API） |
| 视频处理 | FFmpeg + 关键帧提取 | 视频→图片序列→分析 |
| 文档解析 | PyMuPDF / python-docx | 多格式文档解析 |

### 3.4 Agent 工具

| 工具名 | 类名 | 说明 |
|--------|------|------|
| web_search | WebSearchTool | Web 搜索（duckduckgo_search） |
| calculator | CalculatorTool | 数学计算（安全 AST eval） |
| code_execute | CodeExecTool | Python 代码执行（subprocess + 10s 超时） |
| file_operations | FileOpsTool | 文件读写/目录列表（沙箱路径） |
| knowledge_base_search | KBSearchTool | 知识库向量检索 |

### 3.5 外部依赖

- MySQL 8.0 — 元数据存储 + 长期记忆持久化
- Redis 7.x — 缓存和限流
- LLM API（阿里云 DashScope，OpenAI 兼容接口）— 对话和 Agent 推理（默认 qwen-plus）
- Embedding API（DashScope text-embedding-v3）— 文档向量化
- 可选：PaddleOCR、Whisper、FFmpeg — 多模态功能

---

## 四、项目工作树详解

### 4.1 master 分支（共享基础设施）

**路径**: `D:/test2`
**目的**: 只存放共享配置和编排，不做开发。

```
master/
├── .env.example          ← 环境变量模板
├── .gitignore            ← 全局 gitignore
├── CLAUDE.md             ← Claude Code 项目指引
├── PROJECT_DOC.md        ← 本文档
├── docker/
│   └── docker-compose.yml ← 完整编排（mysql+redis+backend+frontend）
├── pyproject.toml        ← 整合的 Python 项目配置
└── requirements.txt      ← 整合依赖
```

**启动方式**:
```bash
cd docker && docker-compose up -d
```

### 4.2 backend 工作树（FastAPI 后端）

**路径**: `D:/kb-backend`
**分支**: `backend`
**测试**: 19 个测试全部通过

**目录结构**:
```
app/
├── main.py               ← FastAPI 入口（中间件栈 + 路由注册）
├── config.py             ← pydantic-settings 配置管理（@lru_cache）
├── api/                  ← 6 个路由模块
│   ├── chat.py           ← 对话 API（RAG + Agent 双模式，支持 max_iterations）
│   ├── knowledge.py      ← 知识库 CRUD
│   ├── upload.py         ← 文件上传（自动解析→分块→向量化）
│   ├── admin.py          ← 系统统计/健康检查
│   └── analytics.py      ← 数据分析 API
├── core/
│   ├── agent/            ← ReAct Agent（手写循环，非 LangChain 标准 Agent）
│   │   ├── executor.py   ← AgentExecutor（含 sync run + async arun）
│   │   ├── prompt.py     ← Agent 提示词模板
│   │   └── tools/        ← 工具注册表（5 个 BaseTool）
│   ├── rag/              ← RAG 流水线
│   │   ├── loader.py     ← 文档加载器（PDF/Word/MD/TXT/CSV）
│   │   ├── splitter.py   ← 文本分块（Recursive/Sentence/Markdown）
│   │   ├── embedder.py   ← VectorStore（ChromaDB 封装）
│   │   ├── retriever.py  ← 检索器（向量检索 + 阈值过滤）
│   │   └── chain.py      ← RAG 完整链路
│   ├── multimodal/       ← 多模态处理
│   │   ├── image.py      ← 图片处理（PaddleOCR + Qwen-VL 描述）
│   │   ├── audio.py      ← 音频转写（Whisper）
│   │   ├── video.py      ← 视频处理（FFmpeg 关键帧）
│   │   ├── ocr.py        ← PaddleOCR 引擎
│   │   ├── windows_ocr.py← Windows OCR 备选
│   │   └── pipeline.py   ← 统一处理管线
│   └── memory/           ← 记忆管理
│       ├── short_term.py ← 短期记忆（会话内滑动窗口 10 轮）
│       ├── long_term.py  ← 长期记忆（MySQL 持久化 + 内存 L1 缓存）
│       └── context.py    ← 上下文窗口管理（Token 预算分配）
├── models/
│   ├── schemas.py        ← Pydantic 请求/响应模型
│   └── database.py       ← SQLAlchemy ORM（含 MemoryEntry 表）
├── services/
│   ├── llm_service.py    ← LLM 调用（langchain-openai → DashScope）
│   ├── embedding_service.py ← Embedding（API/本地/哈希 三级降级）
│   ├── cache_service.py  ← Redis 缓存
│   └── finetuned_service.py ← 微调模型推理
├── middleware/
│   ├── rate_limit.py     ← 限流（Redis，60 请求/分钟）
│   └── logging.py        ← 请求日志
└── utils/
    ├── logger.py         ← loguru 日志
    ├── helpers.py        ← 工具函数（ID 生成/目录创建）
    └── exceptions.py     ← 全局异常处理（仅保留通用兜底）
```

#### 后端改进点（重构中修复）

| 问题 | 修复方式 |
|------|----------|
| Agent 同步阻塞 | 新增 `arun()` 方法，通过 `asyncio.to_thread()` 异步执行 |
| 正则解析脆弱 | `Action:\s*(\S+)` 只匹配工具名，不再依赖换行符 |
| max_iterations 写死 | 从 `ChatRequest` 传入，默认 5，范围 1-20 |
| 长期记忆丢失 | MySQL 持久化（`MemoryEntryModel`）+ 内存 L1 缓存（5 分钟 TTL） |
| 无用代码/依赖 | 移除未使用的异常类、3 个无用依赖包 |
| 全局单例 | 保持兼容性的同时，构造函数可接受可选参数（增量改造中） |

**启动方式**:
```bash
uvicorn app.main:app --reload --port 8000
```

### 4.3 mcp-server 工作树（统一 MCP 服务）

**路径**: `D:/kb-mcp`
**分支**: `mcp-server`
**目的**: 合并了原有的 `app/core/mcp/server.py`（知识库工具）和 `scripts/mcp_qwen_vision.py`（图片识别）为统一 MCP 包。

```
mcp_server/
├── __init__.py           ← 统一入口（导出 run_server, main）
├── server.py             ← JSON-RPC stdio 主循环
├── kb_tools.py           ← 知识库工具（3 个）
├── vision_tools.py       ← 图片识别工具（零外部依赖，仅 httpx）
└── config.py             ← 配置加载（.env → DashScope/MySQL）
```

**可用工具**:

| 工具名 | 来源 | 说明 |
|--------|------|------|
| knowledge_base_search | kb_tools.py | 向量检索知识库 |
| list_knowledge_bases | kb_tools.py | 列出知识库列表 |
| list_documents | kb_tools.py | 列出知识库文档 |
| image_recognize | vision_tools.py | 千问视觉模型图片识别 |

**使用方式**:
```bash
# 仅注册知识库工具
python -m mcp_server --tools kb

# 仅注册图片识别工具
python -m mcp_server --tools vision

# 注册全部工具（默认）
python -m mcp_server --tools all
```

**Claude Code MCP 配置**（`.claude/settings.local.json`）:
```json
{
  "kb-mcp": {
    "command": "python",
    "args": ["-m", "mcp_server", "--tools", "kb"],
    "env": { "MYSQL_URL": "mysql+pymysql://root:password@localhost:3306/multimodal_kb" }
  },
  "qwen-vision": {
    "command": "python",
    "args": ["-m", "mcp_server", "--tools", "vision"],
    "env": { "QWEN_API_KEY": "sk-..." }
  }
}
```

**设计要点**:
- `kb_tools.py` 使用懒加载（`import` 放在函数内部），避免启动时加载 ChromaDB
- `vision_tools.py` 保持零依赖（仅 `httpx`），可独立运行
- 两份 JSON-RPC 主循环合并到 `server.py`，消除 ~90% 代码重复

### 4.4 frontend 工作树（Streamlit 前端）

**路径**: `D:/kb-frontend`
**分支**: `frontend`
**目的**: 精简的 Streamlit 前端，移除后端依赖。

```
frontend/
├── app.py                ← 主应用（对话界面 + 知识库管理）
└── pages/
    └── analytics.py      ← 数据分析看板（Plotly 图表）
```

**启动方式**:
```bash
streamlit run frontend/app.py --server.port 8501
```

---

## 五、工作树开发工作流

### 5.1 启动所有服务

```bash
# 方式一：Docker 一键启动（所有服务）
cd /d/test2 && docker-compose up -d

# 方式二：本地开发
# 终端 1：启动后端
cd /d/kb-backend && uvicorn app.main:app --reload --port 8000

# 终端 2：启动前端
cd /d/kb-frontend && streamlit run frontend/app.py --server.port 8501
```

### 5.2 在各工作树中开发

```bash
# 进入后端工作树
cd /d/kb-backend
git status          # backend 分支
# ... 修改代码 ...
git add -A && git commit -m "xxx"
git push origin backend

# 进入 MCP 工作树
cd /d/kb-mcp
git status          # mcp-server 分支

# 进入前端工作树
cd /d/kb-frontend
git status          # frontend 分支
```

### 5.3 一键回滚

```bash
# 全部回滚（回到原始单仓库状态）
git checkout master
git branch -D backend mcp-server frontend
git worktree remove /d/kb-backend /d/kb-mcp /d/kb-frontend
git reset --hard <重构前的提交哈希>
```

### 5.4 工作树间关系

```
master (共享编排)
  ├── backend     ─── 依赖: MySQL, Redis, Chroma, LLM API
  │                   └── 启动: uvicorn app.main:app --port 8000
  │
  ├── mcp-server  ─── 依赖: 同 backend（共享同一 MySQL/ChromaDB）
  │                   └── 启动: python -m mcp_server --tools all
  │
  └── frontend    ─── 依赖: backend API（通过 API_BASE 环境变量连接）
                      └── 启动: streamlit run frontend/app.py
```

**启动顺序**: MySQL+Redis → backend → frontend → mcp-server（按需）

---

## 六、环境搭建指南

### 6.1 前置条件

- Python 3.10+
- MySQL 8.0
- Redis 7.x
- FFmpeg（视频/音频处理）
- 阿里云 DashScope API Key

### 6.2 初始化

```bash
# 1. 克隆仓库
git clone https://github.com/wangwenc92/MultiModal-KB-Agent.git
cd MultiModal-KB-Agent

# 2. 创建工作树
git branch backend mcp-server frontend
git worktree add ../kb-backend backend
git worktree add ../kb-mcp mcp-server
git worktree add ../kb-frontend frontend

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 和数据库连接

# 4. 安装依赖（backend）
cd ../kb-backend
pip install -r requirements.txt

# 5. 初始化数据库表
python scripts/init_db.py

# 6. 启动后端
uvicorn app.main:app --reload --port 8000

# 7. 启动前端
cd ../kb-frontend
pip install -r requirements.txt
streamlit run frontend/app.py --server.port 8501
```

### 6.3 Docker 部署

```bash
cd MultiModal-KB-Agent/docker
docker-compose up -d
# 服务: backend(8000) + frontend(8501) + mysql(3306) + redis(6379)
```

---

## 七、Agent 执行流程

### 7.1 ReAct 循环

```
用户问题输入
    │
    ▼
┌─────────────────┐
│  构建上下文      │ ← 长期记忆 + 会话历史 + RAG 检索
└────────┬────────┘
         ▼
┌─────────────────┐
│  ReAct 循环     │ ← 最多 max_iterations 步
│ (for step=1..N) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 有 Action   无 Action
    │         │
    ▼         ▼
 执行工具    返回最终回答
  (同步)     (final_answer)
    │
    ▼
 Observation
 反馈给 LLM
    │
    └──── 回到循环
```

### 7.2 Agent 输入/输出契约

**输入** (`AgentExecutor.run()`):

| 参数 | 类型 | 说明 |
|------|------|------|
| question | str | 用户问题 |
| tool_names | list[str] \| None | 限定使用的工具子集 |
| context | str | 上下文（记忆 + RAG 检索） |

**输出** (dict):

| 字段 | 类型 | 说明 |
|------|------|------|
| answer | str | 最终回答文本 |
| trace | list[dict] | 执行轨迹，每步含 step/thought/action/observation/type |

---

## 八、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/chat/send | 发送消息（mode: rag \| agent，可选 max_iterations） |
| GET | /api/chat/history/{session_id} | 获取会话历史 |
| GET | /api/chat/sessions | 列出会话 |
| DELETE | /api/chat/sessions/{session_id} | 删除会话 |
| GET | /api/chat/tools | 列出可用工具 |
| POST | /api/knowledge/create | 创建知识库 |
| GET | /api/knowledge/list | 知识库列表 |
| GET | /api/knowledge/{kb_id} | 知识库详情 |
| DELETE | /api/knowledge/{kb_id} | 删除知识库 |
| POST | /api/upload | 上传文件 |
| DELETE | /api/upload/{doc_id} | 删除文档 |
| GET | /api/admin/stats | 系统统计 |
| GET | /api/admin/health | 健康检查 |
| GET | /api/analytics/overview | 使用统计概览 |

---

## 九、验收清单

### 基础功能 ✅
- [x] 项目启动（uvicorn + streamlit）
- [x] 上传 PDF/DOCX/MD/TXT/图片/音频/视频
- [x] 文件自动分块并索引到向量库
- [x] 基于知识库的 RAG 问答
- [x] 回答包含引用来源
- [x] 知识库 CRUD

### Agent 功能 ✅
- [x] Agent 模式正确选择和调用工具
- [x] 多步推理正确执行
- [x] 异步 Agent 不阻塞事件循环
- [x] max_iterations 可配置
- [x] 正则解析健壮
- [x] 工具调用轨迹可查看

### 多模态功能 ✅
- [x] 音频转写并索引
- [x] 图片 OCR 和 AI 描述
- [x] 视频关键帧提取
- [x] 多模态内容可检索

### 工程化 ✅
- [x] 限流生效（Redis，60 请求/分钟）
- [x] Redis 缓存正常
- [x] Docker 一键部署
- [x] 长期记忆持久化（MySQL）
- [x] 19 个测试全部通过

### 代码架构 ✅
- [x] 3 个独立 git 工作树
- [x] MCP 服务器统一合并
- [x] 清理无用依赖和死代码
- [x] 各工作树可独立开发/部署
