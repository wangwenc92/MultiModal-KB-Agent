"""千问视觉模型图片识别工具 - 零外部依赖（仅 httpx）"""
import json
import base64
from pathlib import Path

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

    mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp",
                 "webp": "webp", "tiff": "tiff", "tif": "tiff"}
    mime_ext = mime_map.get(ext.lstrip("."), "jpeg")
    return data, f"image/{mime_ext}"


def handle_tool_call(arguments: dict) -> str:
    """处理 image_recognize 工具调用"""
    import httpx
    from .config import load_qwen_config

    image_path = arguments.get("image_path", "")
    user_prompt = arguments.get("prompt", "")

    config = load_qwen_config()
    if not config["api_key"]:
        return json.dumps({"error": "QWEN_API_KEY 未配置，请在 .env 中设置"})

    try:
        image_data, mime_type = encode_image(image_path)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    prompt = user_prompt.strip() if user_prompt.strip() else DEFAULT_PROMPT

    request_body = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"千问API返回错误 (HTTP {e.response.status_code}): {e.response.text[:500]}"})
    except httpx.TimeoutException:
        return json.dumps({"error": "千问API请求超时（60秒），请重试"})
    except Exception as e:
        return json.dumps({"error": f"调用千问API失败: {str(e)}"})
