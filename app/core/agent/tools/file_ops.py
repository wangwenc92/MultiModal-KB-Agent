import os
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


WORK_DIR = Path("./workspace")
WORK_DIR.mkdir(exist_ok=True)


class FileOpsInput(BaseModel):
    operation: str = Field(description="操作类型: read(读取), write(写入), list(列目录)")
    path: str = Field(description="文件或目录的相对路径")
    content: str = Field(default="", description="写入操作时的文件内容")


class FileOpsTool(BaseTool):
    name: str = "file_operations"
    description: str = "文件操作工具。支持读取文件(read)、写入文件(write)、列目录(list)。路径为相对于工作目录的相对路径。"
    args_schema: type[BaseModel] = FileOpsInput

    def _run(self, operation: str, path: str, content: str = "") -> str:
        target = (WORK_DIR / path).resolve()
        if not str(target).startswith(str(WORK_DIR.resolve())):
            return "错误: 路径超出工作目录范围"

        try:
            if operation == "read":
                return self._read(target)
            elif operation == "write":
                return self._write(target, content)
            elif operation == "list":
                return self._list(target)
            else:
                return f"不支持的操作: {operation}，可选: read, write, list"
        except Exception as e:
            return f"操作失败: {str(e)}"

    def _read(self, path: Path) -> str:
        if not path.exists():
            return f"文件不存在: {path.name}"
        if path.stat().st_size > 100_000:
            return "文件过大（>100KB），请指定读取范围"
        return path.read_text(encoding="utf-8", errors="replace")

    def _write(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"已写入 {path.name}（{len(content)} 字符）"

    def _list(self, path: Path) -> str:
        if not path.exists():
            return f"目录不存在: {path.name}"
        if path.is_file():
            return f"{path.name} (文件, {path.stat().st_size} 字节)"
        items = []
        for item in sorted(path.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            size = item.stat().st_size if item.is_file() else ""
            items.append(f"{prefix} {item.name}" + (f" ({size} 字节)" if size else ""))
        return "\n".join(items) if items else "空目录"
