"""Windows OCR 引擎 - 使用 Windows.Media.Ocr API"""
from __future__ import annotations

import os
import json
from pathlib import Path
from app.utils.logger import log

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class WindowsOCREngine:
    """使用 Windows 10/11 内置 OCR 引擎"""

    def __init__(self):
        self._ocr_engine = None
        self._available = None

    def _check_available(self) -> bool:
        """检查 Windows OCR 是否可用"""
        if self._available is not None:
            return self._available

        try:
            import winrt
            self._available = True
            log.info("Windows OCR engine available")
        except ImportError:
            self._available = False
            log.warning("Windows OCR not available (winrt not installed)")
        return self._available

    def extract_text(self, image_path: str) -> str:
        """提取图片中的文字"""
        if not self._check_available():
            return "[Windows OCR不可用：请安装 winrt 包]"

        try:
            import asyncio
            return asyncio.run(self._extract_async(image_path))
        except Exception as e:
            log.error(f"Windows OCR failed: {e}")
            return f"[OCR失败: {str(e)}]"

    async def _extract_async(self, image_path: str) -> str:
        """异步提取文字"""
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import BitmapDecoder, SoftwareBitmap
        from winrt.windows.storage import StorageFile, FileAccessMode
        from winrt.windows.globalization import Language

        # 检查并缩放大图片
        from PIL import Image
        img = Image.open(image_path)
        max_dim = 4096  # Windows OCR 限制
        if img.width > max_dim or img.height > max_dim:
            ratio = min(max_dim / img.width, max_dim / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            # 保存临时文件
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), "ocr_temp.png")
            img.save(temp_path)
            image_path = temp_path

        # 加载图片
        file = await StorageFile.get_file_from_path_async(os.path.abspath(image_path))
        stream = await file.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()

        # 创建 OCR 引擎（中文+英文）
        engine = OcrEngine.try_create_from_language(Language("zh-CN"))
        if not engine:
            engine = OcrEngine.try_create_from_language(Language("en-US"))

        if not engine:
            return "[无法创建OCR引擎]"

        # 识别文字
        result = await engine.recognize_async(bitmap)
        lines = [line.text for line in result.lines]
        text = "\n".join(lines)

        log.info(f"Windows OCR extracted {len(text)} chars from {Path(image_path).name}")
        return text


_windows_ocr = None


def get_windows_ocr() -> WindowsOCREngine:
    global _windows_ocr
    if _windows_ocr is None:
        _windows_ocr = WindowsOCREngine()
    return _windows_ocr
