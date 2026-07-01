# 项目演进对比 — 从单仓库到多工作树

## 概述

本文档记录 MultiModal-KB-Agent 项目从最初的单仓库单体架构，经过重构演变为多工作树架构的完整过程，对比各阶段的差异与改进。

---

## 第一阶段：初始状态（单仓库单体架构）

**对应 commit**: `d63446b` — `eb70a8c`

### 目录结构

```
D:/test2/
├── app/                          ← 所有后端代码混在一起
│   ├── main.py / config.py
│   ├── api/           (5 个路由)
│   ├── core/agent/    (ReAct Agent)
│   ├── core/rag/      (RAG 流水线)
│   ├── core/multimodal/(图片/音频/视频)
│   ├── core/memory/   (记忆管理)
│   ├── core/mcp/      (MCP 服务器)
│   ├── services/      (LLM/Embedding/Cache)
│   ├── models/        (Pydantic + SQLAlchemy)
│   ├── middleware/     (限流/日志)
│   └── utils/         (工具函数)
├── frontend/                     ← 前端
│   └── app.py
├── scripts/                      ← 脚本散落
│   └── mcp_qwen_vision.py       (独立的 MCP)
├── docker/                       ← Docker 配置
├── tests/
└── ...
```

### 存在的问题

| 问题 | 严重程度 | 说明 |
|------|:--------:|------|
| 单分支开发 | 🔴 | 所有代码绑在 master，改一行前端也要全量推送 |
| 全局单例耦合 | 🔴 | 18 个 `get_*()` 工厂函数，模块间强耦合 |
| Agent 阻塞事件循环 | 🔴 | `agent.run()` 同步调用，卡住 FastAPI 事件循环 |
| 正则解析脆弱 | 🟡 | `Action:\s*(.+?)(?:\n\|$)` 依赖换行符格式 |
| 迭代次数写死 | 🟡 | `max_iterations=5` 无法外部配置 |
| 长期记忆丢失 | 🟡 | `_long_term_store` 是内存字典，重启丢失 |
| MCP 代码重复 | 🟡 | 两份 JSON-RPC 主循环，~90% 重复代码 |
| 无用依赖 | 🟡 | `langchain-community` `langchain-anthropic` `unstructured` |
| 无用代码 | 🟡 | `AppException` 定义了但从未被 raise |
| 无流式输出 | 🟢 | 用户必须等全部生成完才看到结果 |

---

## 第二阶段：重构后（多工作树架构）

**对应 commit**: `48fc1a2` — 最新

### 目录结构

```
D:/test2/                         ← master：共享基础设施
├── .env.example / .gitignore
├── CLAUDE.md / PROJECT_DOC.md
├── docker/docker-compose.yml    ← 完整编排 4 个服务
├── pyproject.toml / requirements.txt
│
├── backend/                      ← backend 分支：FastAPI 后端
│   ├── app/                     ← 全部后端代码（不含 mcp/）
│   │   ├── main.py / config.py
│   │   ├── api/         (6 个路由 + 流式 SSE)
│   │   ├── core/agent/  (异步 Agent + 流式事件)
│   │   ├── core/rag/    (RAG + 流式输出)
│   │   ├── core/multimodal/
│   │   ├── core/memory/ (MySQL 持久化)
│   │   ├── services/    (LLM/Embedding/Cache)
│   │   ├── models/      (+ MemoryEntry 表)
│   │   ├── middleware/  (限流/日志)
│   │   └── utils/       (仅保留通用兜底)
│   ├── tests/          (19 个测试全部通过)
│   ├── scripts/        (+ init_db.py)
│   ├── Dockerfile / requirements.txt
│   └── .env
│
├── mcp-server/                   ← mcp-server 分支：统一 MCP 包
│   └── mcp_server/
│       ├── server.py            ← 合并后的 JSON-RPC 主循环
│       ├── kb_tools.py          ← 知识库工具（3 个）
│       ├── vision_tools.py      ← 图片识别工具（1 个）
│       └── config.py            ← 统一配置加载
│
└── frontend/                     ← frontend 分支：Streamlit 前端
    ├── frontend/
    │   ├── app.py               ← 对接 SSE 流式输出
    │   └── pages/analytics.py
    ├── Dockerfile.frontend
    └── requirements.txt         ← 精简依赖
```

### 已解决的问题

| 问题 | 解决方案 | commit |
|------|---------|--------|
| 单分支开发 | 拆为 4 个独立分支（master/backend/mcp-server/frontend） | `7fd0849` `c65b760` `50b6987` `e8cacab` |
| Agent 阻塞事件循环 | 新增 `arun()` 方法，通过 `asyncio.to_thread()` 异步执行 | `e28d83f` |
| 正则解析脆弱 | `Action:\s*(\S+)` 只匹配工具名，跨行前瞻匹配 Action Input | `e28d83f` |
| 迭代次数写死 | `ChatRequest` 新增 `max_iterations` 字段（默认5，范围1-20） | `e28d83f` |
| 长期记忆丢失 | `MemoryEntryModel` 表 + 内存 L1 缓存(5分钟) + MySQL L2(30天 TTL) | `e28d83f` |
| MCP 代码重复 | 合并为统一 `mcp_server/` 包，`--tools kb\|vision\|all` 参数切换 | `e8cacab` |
| 无用依赖 | 移除 `langchain-community`、`langchain-anthropic`、`unstructured` | `48fc1a2` |
| 无用代码 | 删除 `AppException/NotFoundException/BadRequestException` | `48fc1a2` |
| 无用方法 | 删除 `context.extract_memory()`（从未被调用） | `48fc1a2` |
| 无流式输出 | 新增 `POST /api/chat/stream` SSE 端点 + Agent 流式事件 | `3d9b1d5` `762890a` |
| 测试断言错误 | 修复 `test_list_dir` 文件路径安全检查问题 | `ba02142` |
| 前端端口 | 前端默认 API 地址从 8001 修正为 8000 | `a5843a1` |

### 新增功能

| 功能 | 说明 | commit |
|------|------|--------|
| **SSE 流式输出** | RAG 和 Agent 模式都支持逐字显示 | `3d9b1d5` |
| **Agent 流式事件** | Agent 逐步产出 thought/tool_call/tool_result/chunk/done 事件 | `3d9b1d5` |
| **统一 MCP 包** | 原来的两份 MCP 合并为一个可插拔包 | `e8cacab` |
| **长期记忆持久化** | MySQL 存储，重启不丢失 | `e28d83f` |
| **GitHub 多分支** | 4 个分支各自管理，可独立部署 | `7fd0849` |

---

## 关键指标对比

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| **分支数** | 1 (master) | 4 (master + 3 工作树) |
| **全局单例** | 18 个 | 18 个（待 P2 增量改造） |
| **测试通过数** | 14/19（5 个因无 MySQL 失败） | 19/19 ✅ |
| **MCP 代码** | 2 份重复实现 ≈ 450 行 | 1 份统一包 ≈ 300 行 |
| **依赖数量** | 43 个 | 39 个（移除了 4 个无用包） |
| **LLM 调用** | Claude/OpenAI | 阿里云 DashScope Qwen |
| **Embedding** | OpenAI text-embedding-3-small | DashScope text-embedding-v3 + 三级降级 |
| **对话模式** | 需等待完整回答 | 流式逐字输出 ✅ |
| **Agent 执行** | 同步阻塞 | 异步非阻塞 ✅ |
| **长期记忆** | 内存字典（易失） | MySQL 持久化 ✅ |

---

## 回滚方式

如需要恢复到初始状态：

```bash
# 删除工作树
cd /d/test2
rm -rf backend mcp-server frontend
git worktree prune
git checkout master
git reset --hard d63446b
git push origin master --force
```

## 时间线

```
2026-07-01  d63446b  初始提交：单仓库单体架构
2026-07-01  48efd69  切换LLM为阿里云DashScope
2026-07-01  eb70a8c  更新技术文档
─── 重构开始 ───
2026-07-01  48fc1a2  P0修复：清理无用代码和依赖
2026-07-01  56e6b8a  清理临时文件
2026-07-01  7fd0849  master分支清理，拆分工作树
2026-07-01  c65b760  backend分支建立
2026-07-01  50b6987  frontend分支建立
2026-07-01  e8cacab  mcp-server分支建立 + MCP合并
2026-07-01  e28d83f  P1修复：Agent异步化/正则/配置化/记忆持久化
2026-07-01  ba02142  测试修复，19/19全部通过
2026-07-01  3d9b1d5  SSE流式输出
2026-07-01  762890a  前端对接流式
2026-07-01  a5843a1  最终调整
```
