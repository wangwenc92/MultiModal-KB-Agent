from __future__ import annotations

from app.core.memory.short_term import get_short_term_memory
from app.core.memory.long_term import get_long_term_memory
from app.core.rag.retriever import get_retriever
from app.utils.logger import log


class ContextManager:
    """上下文窗口管理器：组合短期记忆、长期记忆、RAG检索结果"""

    def __init__(self):
        self.short_term = get_short_term_memory()
        self.long_term = get_long_term_memory()
        self.retriever = get_retriever()

    def add_message(self, session_id: str, role: str, content: str):
        self.short_term.add_message(session_id, role, content)

    def build_context(self, query: str, session_id: str, kb_id: str = "") -> str:
        parts = []

        # 1. 长期记忆
        long_term = self.long_term.get_summary("default")
        if long_term:
            parts.append(f"[用户记忆]\n{long_term}")

        # 2. 会话历史
        history = self.short_term.get_history_text(session_id)
        if history:
            parts.append(f"[对话历史]\n{history}")

        # 3. RAG检索结果
        if kb_id:
            results = self.retriever.retrieve(query, kb_id=kb_id)
            if results:
                rag_parts = []
                for i, r in enumerate(results, 1):
                    meta = r.get("metadata", {})
                    source = meta.get("source", "")
                    page = meta.get("page", "")
                    page_info = f" 第{page}页" if page else ""
                    rag_parts.append(f"[{i}] (来源: {source}{page_info}) {r['content']}")
                parts.append(f"[参考资料]\n" + "\n".join(rag_parts))

        context = "\n\n".join(parts)
        log.info(f"Built context: {len(context)} chars")
        return context

_context_manager: ContextManager | None = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
