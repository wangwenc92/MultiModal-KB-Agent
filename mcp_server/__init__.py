"""
MultiModal-KB MCP Server

统一的 MCP Server，合并知识库工具和图片识别工具。

使用方式：
  python -m mcp_server --tools kb        # 仅知识库工具
  python -m mcp_server --tools vision    # 仅图片识别
  python -m mcp_server --tools all       # 全部工具（默认）
"""

from .server import run_server, main

__all__ = ["run_server", "main"]
