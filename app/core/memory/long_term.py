from __future__ import annotations

from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.utils.helpers import generate_id
from app.utils.logger import log

# 长期记忆使用JSON字段存储在专门的memory表中
# 为简化实现，这里用内存字典 + 数据库会话历史的方式实现

_long_term_store: dict[str, list[dict]] = {}


class LongTermMemory:
    """长期记忆：跨会话的用户偏好和关键信息"""

    def save_memory(self, user_id: str, memory_type: str, content: str):
        if user_id not in _long_term_store:
            _long_term_store[user_id] = []
        _long_term_store[user_id].append({
            "type": memory_type,
            "content": content,
        })
        log.info(f"Saved long-term memory for {user_id}: {memory_type}")

    def search_memories(self, user_id: str, query: str = "", top_k: int = 5) -> list[dict]:
        memories = _long_term_store.get(user_id, [])
        if not query:
            return memories[:top_k]
        # 简单关键词匹配
        scored = []
        query_lower = query.lower()
        for m in memories:
            score = sum(1 for word in query_lower.split() if word in m["content"].lower())
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]

    def get_summary(self, user_id: str) -> str:
        memories = _long_term_store.get(user_id, [])
        if not memories:
            return ""
        parts = [f"- [{m['type']}] {m['content']}" for m in memories[-10:]]
        return "用户记忆:\n" + "\n".join(parts)

    def clear(self, user_id: str):
        _long_term_store.pop(user_id, None)


_long_term: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _long_term
    if _long_term is None:
        _long_term = LongTermMemory()
    return _long_term
