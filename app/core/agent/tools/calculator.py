import ast
import operator
import math
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 (15*23+45)/3 或 sqrt(144)")


# 安全的操作符映射
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"不支持的常量类型: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op(safe_eval(node.left), safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if not op:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op(safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            func = SAFE_FUNCTIONS[node.func.id]
            args = [safe_eval(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"不支持的函数调用")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"未定义的变量: {node.id}")
    else:
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "执行数学计算。输入数学表达式，支持加减乘除、幂运算、三角函数、对数等。示例: (15*23+45)/3, sqrt(144), sin(pi/2)"
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree)
            if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
                result = int(result)
            return f"计算结果: {expression} = {result}"
        except ZeroDivisionError:
            return "错误: 除以零"
        except Exception as e:
            return f"计算错误: {str(e)}"
