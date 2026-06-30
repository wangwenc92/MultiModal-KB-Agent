"""知识库工具 - 需要 MySQL + ChromaDB 环境"""
import json
import sys
from pathlib import Path

# 将项目根目录加入路径（需在 kb-mcp 工作树中运行）
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

TOOL_DEFINITIONS = [
    {
        "name": "knowledge_base_search",
        "description": "从指定知识库中检索相关文档片段。输入查询关键词和知识库ID。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题"},
                "kb_id": {"type": "string", "description": "知识库ID"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
            },
            "required": ["query", "kb_id"],
        },
    },
    {
        "name": "list_knowledge_bases",
        "description": "列出所有可用的知识库。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_documents",
        "description": "列出指定知识库中的所有文档。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "知识库ID"},
            },
            "required": ["kb_id"],
        },
    },
]


def _get_retriever():
    """懒加载 retriever，避免模块导入时触发 ChromaDB 初始化"""
    from app.core.rag.retriever import get_retriever
    return get_retriever()


def _get_db():
    """懒加载数据库会话"""
    from app.models.database import SessionLocal
    return SessionLocal()


def handle_tool_call(tool_name: str, arguments: dict) -> str:
    """处理 KB 工具调用"""
    if tool_name == "knowledge_base_search":
        return _search_kb(arguments)
    elif tool_name == "list_knowledge_bases":
        return _list_kbs()
    elif tool_name == "list_documents":
        return _list_docs(arguments)
    return json.dumps({"error": f"Unknown KB tool: {tool_name}"})


def _search_kb(args: dict) -> str:
    query = args.get("query", "")
    kb_id = args.get("kb_id", "")
    top_k = args.get("top_k", 5)
    retriever = _get_retriever()
    results = retriever.retrieve(query, kb_id=kb_id)
    output = []
    for i, r in enumerate(results[:top_k], 1):
        meta = r.get("metadata", {})
        output.append({
            "index": i,
            "content": r["content"],
            "source": meta.get("source", ""),
            "page": meta.get("page"),
            "score": round(r["score"], 3),
        })
    return json.dumps(output, ensure_ascii=False)


def _list_kbs() -> str:
    from app.models.database import KnowledgeBaseModel
    db = _get_db()
    try:
        kbs = db.query(KnowledgeBaseModel).all()
        result = [{"id": kb.id, "name": kb.name, "description": kb.description} for kb in kbs]
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()


def _list_docs(args: dict) -> str:
    from app.models.database import DocumentModel
    kb_id = args.get("kb_id", "")
    db = _get_db()
    try:
        docs = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).all()
        result = [
            {"id": d.id, "filename": d.filename, "type": d.file_type,
             "chunks": d.chunk_count, "status": d.status}
            for d in docs
        ]
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()
