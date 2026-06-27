from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索互联网获取最新信息。适用于查询实时新闻、技术文档、事实性问题等。输入搜索关键词。"
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            return self._search_duckduckgo(query)
        except Exception as e:
            return f"搜索失败: {str(e)}"

    def _search_duckduckgo(self, query: str) -> str:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"未找到关于 '{query}' 的相关结果"

        lines = []
        for r in results:
            lines.append(f"- {r.get('title', '')}")
            lines.append(f"  摘要: {r.get('body', '')}")
            lines.append(f"  链接: {r.get('href', '')}")
        return "\n".join(lines)
