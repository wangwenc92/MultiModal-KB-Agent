from __future__ import annotations

from app.core.rag.embedder import get_vector_store
from app.utils.logger import log


class Retriever:
    """检索器：向量检索 + 相似度过滤"""

    def __init__(self, top_k: int = 5, score_threshold: float = 0.3):
        self.vector_store = get_vector_store()
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, kb_id: str | None = None) -> list[dict]:
        results = self.vector_store.search(query, top_k=self.top_k, kb_id=kb_id)
        filtered = [r for r in results if r["score"] >= self.score_threshold]
        log.info(f"Retrieved {len(filtered)}/{len(results)} chunks above threshold {self.score_threshold}")
        return filtered


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
