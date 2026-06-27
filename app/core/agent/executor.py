from __future__ import annotations

import json
import re
import time
from typing import Optional
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

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        # 解析 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

        if not action_match:
            return None

        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""

        # 尝试解析JSON输入
        try:
            action_input = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            # 如果不是JSON，尝试简单的key=value格式
            if "=" in action_input:
                pairs = action_input.split(",")
                action_input = {}
                for pair in pairs:
                    k, _, v = pair.partition("=")
                    action_input[k.strip()] = v.strip()
            else:
                # 作为单个参数
                action_input = {"query": action_input} if action_input else {}

        return action, action_input

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
            thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response)
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


_agent_executor: AgentExecutor | None = None


def get_agent_executor() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = AgentExecutor()
    return _agent_executor
