from __future__ import annotations

from typing import AsyncGenerator, TYPE_CHECKING
from app.config import get_settings
from app.utils.logger import log

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

settings = get_settings()


class LLMService:
    """LLM调用封装，使用OpenAI兼容接口"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self) -> "ChatOpenAI":
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                openai_api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                max_tokens=4096,
                timeout=60,
                max_retries=2,
            )
        return self._llm

    def chat(self, system_prompt: str, user_message: str, context: str = "") -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system_prompt)]
        if context:
            messages.append(HumanMessage(content=f"参考资料:\n{context}\n\n用户问题: {user_message}"))
        else:
            messages.append(HumanMessage(content=user_message))

        resp = self.llm.invoke(messages)
        return resp.content

    async def chat_stream(self, system_prompt: str, user_message: str, context: str = "") -> AsyncGenerator[str, None]:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system_prompt)]
        if context:
            messages.append(HumanMessage(content=f"参考资料:\n{context}\n\n用户问题: {user_message}"))
        else:
            messages.append(HumanMessage(content=user_message))

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
