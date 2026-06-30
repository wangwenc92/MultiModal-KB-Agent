from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, MemoryEntryModel
from app.utils.helpers import generate_id
from app.utils.logger import log

# 内存 L1 缓存：user_id -> list[dict]，避免每次读取都查 DB
_cache: dict[str, list[dict]] = {}
_cache_ttl: dict[str, datetime] = {}
CACHE_TTL = timedelta(minutes=5)
MEMORY_TTL = timedelta(days=30)


class LongTermMemory:
    """长期记忆：跨会话的用户偏好和关键信息

    分层存储：
    - L1：内存缓存（5 分钟过期）
    - L2：MySQL（30 天 TTL）
    """

    def _load_from_db(self, user_id: str) -> list[dict]:
        """从 DB 加载用户记忆到缓存"""
        db: Session = SessionLocal()
        try:
            records = (
                db.query(MemoryEntryModel)
                .filter(MemoryEntryModel.user_id == user_id)
                .order_by(MemoryEntryModel.created_at.desc())
                .limit(50)
                .all()
            )
            result = [
                {"type": r.type, "content": r.content, "created_at": r.created_at.isoformat()}
                for r in records
            ]
            _cache[user_id] = result
            _cache_ttl[user_id] = datetime.utcnow() + CACHE_TTL
            return result
        finally:
            db.close()

    def _get_cached(self, user_id: str) -> list[dict]:
        """获取缓存的记忆，缓存过期则重新加载"""
        if user_id in _cache:
            if datetime.utcnow() < _cache_ttl.get(user_id, datetime.min):
                return _cache[user_id]
        return self._load_from_db(user_id)

    def save_memory(self, user_id: str, memory_type: str, content: str):
        """保存一条记忆（写入 DB + 更新缓存）"""
        db: Session = SessionLocal()
        try:
            record = MemoryEntryModel(
                id=generate_id(),
                user_id=user_id,
                type=memory_type,
                content=content,
            )
            db.add(record)
            db.commit()
        finally:
            db.close()
        # 清除缓存，下次读取会重新加载
        _cache.pop(user_id, None)
        _cache_ttl.pop(user_id, None)
        log.info(f"Saved long-term memory for {user_id}: {memory_type}")

    def search_memories(self, user_id: str, query: str = "", top_k: int = 5) -> list[dict]:
        """搜索用户记忆（关键词匹配）"""
        memories = self._get_cached(user_id)
        if not query:
            return memories[:top_k]
        query_lower = query.lower()
        scored = []
        for m in memories:
            score = sum(1 for word in query_lower.split() if word in m["content"].lower())
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:top_k]]

    def get_summary(self, user_id: str) -> str:
        """获取用户记忆摘要（供上下文构建使用）"""
        memories = self._get_cached(user_id)
        if not memories:
            return ""
        parts = [f"- [{m['type']}] {m['content']}" for m in memories[-10:]]
        return "用户记忆:\n" + "\n".join(parts)

    def clear(self, user_id: str):
        """清除用户的所有记忆"""
        db: Session = SessionLocal()
        try:
            db.query(MemoryEntryModel).filter(MemoryEntryModel.user_id == user_id).delete()
            db.commit()
        finally:
            db.close()
        _cache.pop(user_id, None)
        _cache_ttl.pop(user_id, None)

    @staticmethod
    def clean_expired():
        """清理过期记忆（30 天 TTL）—— 可定时调用"""
        cutoff = datetime.utcnow() - MEMORY_TTL
        db: Session = SessionLocal()
        try:
            deleted = db.query(MemoryEntryModel).filter(
                MemoryEntryModel.created_at < cutoff
            ).delete()
            db.commit()
            if deleted:
                log.info(f"Cleaned {deleted} expired memory entries")
        finally:
            db.close()


_long_term: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _long_term
    if _long_term is None:
        _long_term = LongTermMemory()
    return _long_term
