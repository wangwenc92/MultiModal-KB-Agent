from __future__ import annotations

from app.core.agent.tools.search import WebSearchTool
from app.core.agent.tools.calculator import CalculatorTool
from app.core.agent.tools.code_exec import CodeExecTool
from app.core.agent.tools.file_ops import FileOpsTool
from app.core.agent.tools.kb_search import KBSearchTool

TOOL_REGISTRY = {
    "web_search": WebSearchTool(),
    "calculator": CalculatorTool(),
    "code_execute": CodeExecTool(),
    "file_operations": FileOpsTool(),
    "knowledge_base_search": KBSearchTool(),
}


def get_tools(tool_names: list[str] | None = None) -> list:
    if tool_names is None:
        return list(TOOL_REGISTRY.values())
    return [TOOL_REGISTRY[name] for name in tool_names if name in TOOL_REGISTRY]


def get_tool_descriptions() -> list[dict]:
    return [
        {"name": t.name, "description": t.description}
        for t in TOOL_REGISTRY.values()
    ]
