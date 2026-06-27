"""
MCP Server - 将知识库系统暴露为MCP服务
支持的工具: knowledge_base_search, list_knowledge_bases
支持的资源: knowledge_bases, documents
"""
import json
import sys
sys.path.insert(0, ".")

from app.core.rag.retriever import get_retriever
from app.models.database import SessionLocal, KnowledgeBaseModel, DocumentModel
from app.utils.logger import log

# MCP协议消息格式
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
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
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

RESOURCE_DEFINITIONS = [
    {
        "uri": "kb://knowledge_bases",
        "name": "知识库列表",
        "description": "所有知识库的元数据信息",
        "mimeType": "application/json",
    },
]


def handle_tool_call(tool_name: str, arguments: dict) -> str:
    """处理MCP工具调用"""
    if tool_name == "knowledge_base_search":
        return _search_kb(arguments)
    elif tool_name == "list_knowledge_bases":
        return _list_kbs()
    elif tool_name == "list_documents":
        return _list_docs(arguments)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _search_kb(args: dict) -> str:
    query = args.get("query", "")
    kb_id = args.get("kb_id", "")
    top_k = args.get("top_k", 5)

    retriever = get_retriever()
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
    db = SessionLocal()
    try:
        kbs = db.query(KnowledgeBaseModel).all()
        result = [{"id": kb.id, "name": kb.name, "description": kb.description} for kb in kbs]
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()


def _list_docs(args: dict) -> str:
    kb_id = args.get("kb_id", "")
    db = SessionLocal()
    try:
        docs = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).all()
        result = [
            {"id": d.id, "filename": d.filename, "type": d.file_type, "chunks": d.chunk_count, "status": d.status}
            for d in docs
        ]
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()


def handle_resource_read(uri: str) -> str:
    """处理MCP资源读取"""
    if uri == "kb://knowledge_bases":
        return _list_kbs()
    return json.dumps({"error": f"Unknown resource: {uri}"})


# === MCP Server 入口（stdio协议） ===

def run_mcp_server():
    """运行MCP Server（stdio模式）"""
    log.info("MCP Server starting on stdio...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            req_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}, "resources": {}},
                        "serverInfo": {"name": "multimodal-kb-agent", "version": "0.1.0"},
                    },
                }
            elif method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result_text = handle_tool_call(tool_name, arguments)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]},
                }
            elif method == "resources/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"resources": RESOURCE_DEFINITIONS}}
            elif method == "resources/read":
                uri = params.get("uri", "")
                result_text = handle_resource_read(uri)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": result_text}]},
                }
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            log.error(f"MCP error: {e}")
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
