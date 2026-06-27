from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class KBSearchInput(BaseModel):
    query: str = Field(description="检索关键词或问题")
    kb_id: str = Field(description="知识库ID")


class KBSearchTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = "从指定知识库中检索相关文档片段。适用于查找已索引的文档信息。输入检索关键词和知识库ID。"
    args_schema: type[BaseModel] = KBSearchInput

    def _run(self, query: str, kb_id: str) -> str:
        try:
            from app.core.rag.retriever import get_retriever
            retriever = get_retriever()
            results = retriever.retrieve(query, kb_id=kb_id)

            if not results:
                return f"在知识库中未找到关于 '{query}' 的相关信息"

            parts = []
            for i, r in enumerate(results, 1):
                meta = r.get("metadata", {})
                source = meta.get("source", "未知来源")
                page = meta.get("page", "")
                page_info = f" 第{page}页" if page else ""
                parts.append(f"[{i}] (来源: {source}{page_info}, 相似度: {r['score']:.2f})\n{r['content']}")

            return "\n\n".join(parts)
        except Exception as e:
            return f"知识库检索失败: {str(e)}"
