from __future__ import annotations

import tiktoken
from app.utils.logger import log


class ShortTermMemory:
    """短期记忆：基于会话的消息历史，滑动窗口 + Token预算控制"""

    def __init__(self, max_turns: int = 10, max_tokens: int = 2000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._sessions: dict[str, list[dict]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        self._trim(session_id)

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def get_history_text(self, session_id: str) -> str:
        messages = self.get_history(session_id)
        if not messages:
            return ""
        parts = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts)

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def _trim(self, session_id: str):
        messages = self._sessions.get(session_id, [])
        # 按轮次裁剪
        if len(messages) > self.max_turns * 2:
            self._sessions[session_id] = messages[-(self.max_turns * 2):]
            messages = self._sessions[session_id]

        # 按Token裁剪
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            total = sum(len(enc.encode(m["content"])) for m in messages)
            while total > self.max_tokens and len(messages) > 2:
                removed = messages.pop(0)
                total -= len(enc.encode(removed["content"]))
        except Exception:
            # tiktoken不可用时仅按轮次裁剪
            pass


_short_term: ShortTermMemory | None = None


def get_short_term_memory() -> ShortTermMemory:
    global _short_term
    if _short_term is None:
        _short_term = ShortTermMemory()
    return _short_term
