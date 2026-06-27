from __future__ import annotations

from pathlib import Path
from app.core.multimodal.ocr import get_ocr_engine
from app.services.llm_service import get_llm_service
from app.utils.logger import log

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

IMAGE_DESCRIBE_PROMPT = """请详细描述这张图片的内容。要求：
1. 如果包含文字，请完整提取所有文字内容
2. 如果是图表/表格，请描述数据和结构
3. 如果是照片/场景，请描述主要元素
4. 提取所有关键信息（数字、日期、名称等）"""


class ImageProcessor:
    """图片处理：OCR文字提取 + AI内容描述"""

    def __init__(self):
        self.ocr = get_ocr_engine()
        self.llm = get_llm_service()

    def process(self, image_path: str) -> dict:
        """
        完整图片处理流程
        返回: {"ocr_text": str, "description": str, "combined": str}
        """
        path = Path(image_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        # OCR提取文字
        ocr_text = ""
        try:
            ocr_text = self.ocr.extract_text(image_path)
            # 检查是否是OCR不可用的提示信息
            if "[OCR" in ocr_text and "不可用" in ocr_text:
                ocr_text = ""  # 清空，避免存储错误提示
        except Exception as e:
            log.warning(f"OCR failed for {path.name}: {e}")

        # AI描述
        description = ""
        try:
            description = self._describe_with_llm(image_path)
        except Exception as e:
            log.warning(f"LLM description failed for {path.name}: {e}")

        # 合并结果 - 即使OCR和LLM都失败，也存储基本图片信息
        parts = []
        if description:
            parts.append(f"[图片描述]\n{description}")
        if ocr_text:
            parts.append(f"[OCR文字]\n{ocr_text}")

        # 如果没有任何内容，添加基本图片信息
        if not parts:
            file_info = f"[图片文件]\n文件名: {path.name}\n类型: {path.suffix.lower()}\n说明: 这是一个图片文件，包含视觉内容"
            parts.append(file_info)

        combined = "\n\n".join(parts)

        log.info(f"Processed image {path.name}: OCR={len(ocr_text)} chars, desc={len(description)} chars")
        return {
            "ocr_text": ocr_text,
            "description": description,
            "combined": combined,
        }

    def _describe_with_llm(self, image_path: str) -> str:
        """使用视觉模型描述图片"""
        import base64
        from langchain_core.messages import HumanMessage, SystemMessage

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower().replace(".", "")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp", "webp": "webp"}.get(ext, "jpeg")

        message = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_DESCRIBE_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{image_data}"},
                },
            ]
        )

        llm = self.llm.llm
        resp = llm.invoke([message])
        return resp.content


_image_processor: ImageProcessor | None = None


def get_image_processor() -> ImageProcessor:
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor
