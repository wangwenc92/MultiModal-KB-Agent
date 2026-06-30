#!/usr/bin/env python3
"""
千问视觉模型图片识别 MCP Server
独立运行，不依赖项目其他模块（无需 MySQL/ChromaDB/Redis/langchain）
仅需 httpx 和 python-dotenv（均在 requirements.txt 中）
启动方式: python scripts/mcp_qwen_vision.py
"""
import json
import sys
import base64
import os
from pathlib import Path

# ============================================================
# 配置
# ============================================================

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

DEFAULT_PROMPT = """请详细描述这张图片的内容。要求：
1. 如果包含文字，请完整提取所有文字内容
2. 如果是图表/表格，请描述数据和结构
3. 如果是照片/场景，请描述主要元素和布局
4. 提取所有关键信息：数字、日期、名称、标识等
5. 如有二维码、条形码、logo等特殊元素，请特别指出"""

TOOL_DEFINITIONS = [
    {
        "name": "image_recognize",
        "description": (
            "使用千问视觉模型识别图片内容。传入图片文件的绝对路径，"
            "返回图片的详细文字描述，包括：场景描述、文字内容提取（OCR）、"
            "图表/表格数据解读、关键信息提取（数字、日期、名称等）。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图片文件的绝对路径。支持格式：jpg、jpeg、png、bmp、webp、tiff、tif",
                },
                "prompt": {
                    "type": "string",
                    "description": "自定义提示词，用于指定识别重点。不传则使用默认提示词",
                    "default": "",
                },
            },
            "required": ["image_path"],
        },
    }
]


# ============================================================
# 配置加载
# ============================================================

def load_qwen_config() -> dict:
    """从 .env 和环境变量加载千问配置"""
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).resolve().parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass

    return {
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": os.getenv("QWEN_MODEL", "qwen-vl-max"),
        "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", "2048")),
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.1")),
    }


# ============================================================
# 图片编码
# ============================================================

def encode_image(image_path: str) -> tuple:
    """验证文件存在和格式，编码为 base64"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {image_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式: {ext}，支持: {', '.join(sorted(SUPPORTED_FORMATS))}")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp", "webp": "webp", "tiff": "tiff", "tif": "tiff"}
    mime_ext = mime_map.get(ext.lstrip("."), "jpeg")

    return data, f"image/{mime_ext}"


# ============================================================
# 千问视觉 API 调用
# ============================================================

def call_qwen_vision(image_path: str, prompt: str = "") -> str:
    """调用千问视觉模型识别图片"""
    import httpx

    config = load_qwen_config()
    if not config["api_key"]:
        return json.dumps({"error": "QWEN_API_KEY 未配置，请在 .env 中设置，或在 MCP server 的 env 中提供"})

    try:
        image_data, mime_type = encode_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    user_prompt = prompt.strip() if prompt.strip() else DEFAULT_PROMPT

    request_body = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                ],
            }
        ],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
    }

    url = f"{config['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0)) as client:
            resp = client.post(url, json=request_body, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return content
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"千问API返回错误 (HTTP {e.response.status_code}): {e.response.text[:500]}"})
    except httpx.TimeoutException:
        return json.dumps({"error": "千问API请求超时（60秒），请重试"})
    except Exception as e:
        return json.dumps({"error": f"调用千问API失败: {str(e)}"})


# ============================================================
# MCP Server（stdio JSON-RPC）
# ============================================================

def run_mcp_server():
    """stdio JSON-RPC 主循环"""
    server_name = "qwen-vision-mcp"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            req_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": server_name, "version": "0.1.0"},
                    },
                }
            elif method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                if tool_name == "image_recognize":
                    image_path = arguments.get("image_path", "")
                    user_prompt = arguments.get("prompt", "")
                    result_text = call_qwen_vision(image_path, user_prompt)
                else:
                    result_text = json.dumps({"error": f"Unknown tool: {tool_name}"})

                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": result_text}]},
                }
            elif method == "tools/call/notification":
                # 某些 MCP 客户端发送 notification 形式，忽略即可
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
