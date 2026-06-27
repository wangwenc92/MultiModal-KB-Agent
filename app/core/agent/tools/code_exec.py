import subprocess
import sys
import tempfile
import os
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CodeExecInput(BaseModel):
    code: str = Field(description="要执行的Python代码")


class CodeExecTool(BaseTool):
    name: str = "code_execute"
    description: str = "在安全沙箱中执行Python代码并返回输出。适用于数据分析、文件处理、算法验证等。输入完整的Python代码。"
    args_schema: type[BaseModel] = CodeExecInput

    def _run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tempfile.gettempdir(),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[退出码: {result.returncode}]"
            return output.strip() if output.strip() else "代码执行完成，无输出"
        except subprocess.TimeoutExpired:
            return "错误: 代码执行超时（限制10秒）"
        except Exception as e:
            return f"执行错误: {str(e)}"
        finally:
            os.unlink(tmp_path)
