"""
统一 MCP Server - 合并知识库工具和图片识别工具
通过 --tools 参数控制注册哪些工具：
  python -m mcp_server --tools kb        # 仅注册知识库工具
  python -m mcp_server --tools vision    # 仅注册图片识别工具
  python -m mcp_server --tools kb vision # 同时注册两者
  python -m mcp_server --tools all       # 注册全部
"""
import json
import sys
import argparse


def build_tool_list(enable_kb: bool = False, enable_vision: bool = False) -> list:
    """根据开关组装工具定义列表"""
    tools = []
    if enable_kb:
        from . import kb_tools
        tools.extend(kb_tools.TOOL_DEFINITIONS)
    if enable_vision:
        from . import vision_tools
        tools.extend(vision_tools.TOOL_DEFINITIONS)
    return tools


def handle_tool_call(tool_name: str, arguments: dict,
                     enable_kb: bool, enable_vision: bool) -> str:
    """路由工具调用到对应的处理模块"""
    if enable_kb:
        from . import kb_tools
        for td in kb_tools.TOOL_DEFINITIONS:
            if td["name"] == tool_name:
                return kb_tools.handle_tool_call(tool_name, arguments)
    if enable_vision:
        from . import vision_tools
        for td in vision_tools.TOOL_DEFINITIONS:
            if td["name"] == tool_name:
                return vision_tools.handle_tool_call(arguments)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_server(enable_kb: bool = False, enable_vision: bool = False):
    """运行 MCP Server（stdio JSON-RPC 模式）"""
    tools = build_tool_list(enable_kb, enable_vision)
    server_info = {"name": "multimodal-kb-mcp", "version": "0.1.0"}
    capabilities = {"tools": {}}
    if enable_kb:
        capabilities["resources"] = {}

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
                        "capabilities": capabilities,
                        "serverInfo": server_info,
                    },
                }
            elif method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                args = params.get("arguments", {})
                result_text = handle_tool_call(tool_name, args, enable_kb, enable_vision)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]},
                }
            elif method == "resources/list" and enable_kb:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "kb://knowledge_bases",
                                "name": "知识库列表",
                                "description": "所有知识库的元数据信息",
                                "mimeType": "application/json",
                            }
                        ]
                    },
                }
            elif method == "resources/read" and enable_kb:
                uri = params.get("uri", "")
                from . import kb_tools
                if uri == "kb://knowledge_bases":
                    result_text = kb_tools.handle_tool_call("list_knowledge_bases", {})
                else:
                    result_text = json.dumps({"error": f"Unknown resource: {uri}"})
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": result_text}]},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="MultiModal-KB MCP Server")
    parser.add_argument("--tools", nargs="+", default=["all"],
                        help="注册的工具: kb, vision, all（默认 all）")
    args = parser.parse_args()

    tools = args.tools
    enable_kb = "all" in tools or "kb" in tools
    enable_vision = "all" in tools or "vision" in tools

    run_server(enable_kb=enable_kb, enable_vision=enable_vision)


if __name__ == "__main__":
    main()
