from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncGenerator, Optional
from app.core.agent.tools import get_tools, TOOL_REGISTRY
from app.core.agent.prompt import SYSTEM_PROMPT, TOOL_LIST_TEMPLATE, FINAL_ANSWER_TEMPLATE
from app.services.llm_service import get_llm_service
from app.utils.logger import log


class AgentExecutor:
    """Agent执行器：解析LLM输出 → 调用工具 → 循环直到得到最终回答"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.llm = get_llm_service()

    def _build_system_prompt(self, tool_names: list[str] | None = None) -> str:
        tools = get_tools(tool_names)
        tool_lines = [TOOL_LIST_TEMPLATE.format(name=t.name, description=t.description) for t in tools]
        return SYSTEM_PROMPT.format(tools="\n".join(tool_lines))

    def _parse_action(self, text: str) -> tuple[str, str | dict] | None:
        """解析LLM输出中的 Action 和 Action Input

        Action 只匹配工具名（单个非空白词），避免跨行误匹配。
        Action Input 支持 JSON、key=value 和纯文本三种格式。
        """
        # 只匹配第一个非空白的工具名（不依赖换行符）
        action_match = re.search(r"Action:\s*(\S+)", text)
        if not action_match:
            return None

        action = action_match.group(1).strip()

        # 匹配 Action Input —— 捕获到文本末尾或下一个 "Action:" / "Thought:"
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\n\s*(?:Action:|Thought:|Final\s*Answer:)|\Z)",
            text, re.DOTALL
        )

        action_input_raw = input_match.group(1).strip() if input_match else ""

        # 尝试解析 JSON
        if action_input_raw.startswith("{"):
            try:
                return action, json.loads(action_input_raw)
            except json.JSONDecodeError:
                pass

        # 尝试 key=value 格式
        if "=" in action_input_raw:
            pairs = [p.strip() for p in action_input_raw.split(",") if "=" in p]
            if pairs:
                parsed = {}
                for pair in pairs:
                    k, _, v = pair.partition("=")
                    parsed[k.strip()] = v.strip()
                return action, parsed

        # 纯文本作为 query 参数
        return action, {"query": action_input_raw} if action_input_raw else {}

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        tool = TOOL_REGISTRY.get(tool_name)
        if not tool:
            return f"错误: 未知工具 '{tool_name}'，可用工具: {list(TOOL_REGISTRY.keys())}"

        try:
            start = time.time()
            result = tool.invoke(tool_input)
            duration = int((time.time() - start) * 1000)
            log.info(f"Tool [{tool_name}] executed in {duration}ms")
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def run(self, question: str, tool_names: list[str] | None = None, context: str = "") -> dict:
        """同步执行 Agent 推理循环"""
        system_prompt = self._build_system_prompt(tool_names)
        trace = []
        messages = []

        if context:
            user_msg = f"参考资料:\n{context}\n\n用户问题: {question}"
        else:
            user_msg = question

        for step in range(self.max_iterations):
            log.info(f"Agent step {step + 1}/{self.max_iterations}")

            # 调用LLM
            response = self.llm.chat(system_prompt, user_msg)
            thought_match = re.search(r"Thought:\s*(.+?)(?=\n|$)", response)
            thought = thought_match.group(1).strip() if thought_match else response[:200]

            trace.append({
                "step": step + 1,
                "thought": thought,
                "response": response,
            })

            # 检查是否有Action
            parsed = self._parse_action(response)
            if not parsed:
                # 没有Action，视为最终回答
                trace[-1]["type"] = "final_answer"
                return {"answer": response, "trace": trace}

            action, action_input = parsed
            trace[-1]["type"] = "tool_call"
            trace[-1]["action"] = action
            trace[-1]["action_input"] = action_input

            # 执行工具
            observation = self._execute_tool(action, action_input)
            trace[-1]["observation"] = observation

            # 构建下一步的输入
            user_msg = f"{response}\n\nObservation: {observation}\n\n请继续思考或给出最终回答。"

        # 超过最大迭代次数，强制生成最终回答
        final_prompt = FINAL_ANSWER_TEMPLATE.format(
            question=question,
            observations="\n".join(
                f"步骤{i+1}: {t.get('action', '')} -> {t.get('observation', '')}"
                for i, t in enumerate(trace) if t.get("observation")
            ),
        )
        final_answer = self.llm.chat("请直接给出最终回答。", final_prompt)
        trace.append({"step": len(trace) + 1, "type": "forced_final", "thought": "达到最大迭代次数"})
        return {"answer": final_answer, "trace": trace}

    async def arun(self, question: str, tool_names: list[str] | None = None, context: str = "") -> dict:
        """异步执行 Agent 推理循环（通过线程池避免阻塞事件循环）"""
        return await asyncio.to_thread(self.run, question, tool_names, context)

    async def stream_run(self, question: str, tool_names: list[str] | None = None,
                         context: str = "") -> AsyncGenerator[dict, None]:
        """流式 Agent 执行：逐步产出事件，最终流式输出回答"""
        system_prompt = self._build_system_prompt(tool_names)
        trace = []

        user_msg = f"参考资料:\n{context}\n\n用户问题: {question}" if context else question

        for step in range(self.max_iterations):
            log.info(f"Agent stream step {step + 1}/{self.max_iterations}")

            # 调用LLM
            response = await asyncio.to_thread(self.llm.chat, system_prompt, user_msg)
            thought_match = re.search(r"Thought:\s*(.+?)(?=\n|$)", response)
            thought = thought_match.group(1).strip() if thought_match else response[:200]

            yield {"type": "thought", "content": thought, "step": step + 1}

            # 检查是否有Action
            parsed = self._parse_action(response)
            if not parsed:
                trace.append({"step": step + 1, "type": "final_answer", "thought": thought})
                # 分块输出最终回答
                async for chunk in _chunk_text(response):
                    yield {"type": "chunk", "content": chunk}
                yield {"type": "done", "trace": trace}
                return

            action, action_input = parsed
            yield {"type": "tool_call", "name": action, "input": action_input, "step": step + 1}
            trace.append({"step": step + 1, "type": "tool_call", "action": action, "action_input": action_input, "thought": thought})

            # 执行工具
            observation = await asyncio.to_thread(self._execute_tool, action, action_input)
            yield {"type": "tool_result", "name": action, "content": observation[:500]}
            trace[-1]["observation"] = observation

            user_msg = f"{response}\n\nObservation: {observation}\n\n请继续思考或给出最终回答。"

        # 超过最大迭代次数
        trace.append({"step": len(trace) + 1, "type": "forced_final", "thought": "达到最大迭代次数"})
        yield {"type": "thought", "content": "达到最大迭代次数，正在生成最终回答...", "step": self.max_iterations + 1}

        final_prompt = FINAL_ANSWER_TEMPLATE.format(
            question=question,
            observations="\n".join(
                f"步骤{i+1}: {t.get('action', '')} -> {t.get('observation', '')}"
                for i, t in enumerate(trace) if t.get("observation")
            ),
        )
        final_answer = await asyncio.to_thread(self.llm.chat, "请直接给出最终回答。", final_prompt)
        async for chunk in _chunk_text(final_answer):
            yield {"type": "chunk", "content": chunk}
        yield {"type": "done", "trace": trace}


_agent_executor: AgentExecutor | None = None


def get_agent_executor(max_iterations: int = 5) -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None or _agent_executor.max_iterations != max_iterations:
        _agent_executor = AgentExecutor(max_iterations=max_iterations)
    return _agent_executor


async def _chunk_text(text: str, chunk_size: int = 5) -> AsyncGenerator[str, None]:
    """将文本分块输出，模拟流式效果。每块约 chunk_size 个字。"""
    import unicodedata

    i = 0
    while i < len(text):
        # 按标点符号或长度切分
        end = min(i + chunk_size, len(text))
        # 尽量在标点处切分
        for sep in "。！？，、；：.!?,\n":
            pos = text.find(sep, i + 1)
            if 0 < pos < end:
                end = pos + 1
                break
        yield text[i:end]
        i = end
        await asyncio.sleep(0.01)  # 稍微减速，让前端有逐字效果
