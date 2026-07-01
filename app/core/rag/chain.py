from __future__ import annotations

from typing import AsyncGenerator
from app.core.rag.retriever import get_retriever
from app.services.llm_service import get_llm_service
from app.services.cache_service import get_cache
from app.utils.logger import log

RAG_SYSTEM_PROMPT = """你是多模态智能知识库助手。请基于以下参考资料回答用户问题。

规则：
1. 仅基于提供的参考资料回答，如果资料中没有相关信息，请如实告知"参考资料中未找到相关信息"
2. 回答要准确、简洁、有条理
3. 回答时请标注引用来源，格式为 [来源: 文件名]
4. 如果有多个相关来源，请综合分析后回答"""


class RAGChain:
    """RAG问答链路：检索 -> 构建上下文 -> 生成回答"""

    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm_service()
        self.cache = get_cache()
        self.last_sources: list[dict] = []

    def _build_context(self, results: list[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            source = meta.get("source", "未知来源")
            page = meta.get("page", "")
            page_info = f" 第{page}页" if page else ""
            parts.append(f"[{i}] (来源: {source}{page_info}, 相似度: {r['score']:.2f})\n{r['content']}")
        return "\n\n".join(parts)

    def ask(self, question: str, kb_id: str) -> dict:
        # 查询缓存
        cache_key = self.cache.hash_key("rag", question, kb_id)
        cached = self.cache.get_json(cache_key)
        if cached:
            log.info(f"RAG cache hit for question: {question[:30]}...")
            return cached

        results = self.retriever.retrieve(question, kb_id=kb_id)
        context = self._build_context(results)
        answer = self.llm.chat(RAG_SYSTEM_PROMPT, question, context)

        sources = self._build_sources(results)

        log.info(f"RAG answer generated with {len(sources)} sources")
        result = {"answer": answer, "sources": sources}
        self.cache.set_json(cache_key, result, ttl=300)
        return result

    def _build_sources(self, results: list[dict]) -> list[dict]:
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append({
                "content": r["content"][:200],
                "filename": meta.get("source", ""),
                "page": meta.get("page"),
                "score": round(r["score"], 3),
            })
        return sources

    async def ask_stream(self, question: str, kb_id: str) -> AsyncGenerator[str, None]:
        results = self.retriever.retrieve(question, kb_id=kb_id)
        context = self._build_context(results)
        self.last_sources = self._build_sources(results)
        async for chunk in self.llm.chat_stream(RAG_SYSTEM_PROMPT, question, context):
            yield chunk


_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
