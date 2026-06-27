from __future__ import annotations

import os
from pathlib import Path
from app.utils.logger import log

# 设置 PaddleOCR 模型目录（避免中文路径问题）
PADDLE_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "paddleocr_models")
os.environ["PADDLEOCR_MODEL_DIR"] = PADDLE_MODEL_DIR

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class OCREngine:
    """OCR文字识别引擎（PaddleOCR / Tesseract回退）"""

    def __init__(self):
        self._engine = None
        self._engine_type = None

    def _init_engine(self):
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR
            # 使用纯英文路径避免中文路径问题
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                det_model_dir=os.path.join(PADDLE_MODEL_DIR, "det", "ch", "ch_PP-OCRv4_det_infer"),
                rec_model_dir=os.path.join(PADDLE_MODEL_DIR, "rec", "ch", "ch_PP-OCRv4_rec_infer"),
                cls_model_dir=os.path.join(PADDLE_MODEL_DIR, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
            )
            self._engine_type = "paddle"
            log.info("OCR engine initialized: PaddleOCR")
        except Exception as e:
            log.warning(f"PaddleOCR init failed: {e}")
            try:
                from app.core.multimodal.windows_ocr import get_windows_ocr
                self._engine = get_windows_ocr()
                if self._engine._check_available():
                    self._engine_type = "windows"
                    log.info("OCR engine initialized: Windows OCR")
                else:
                    raise Exception("Windows OCR not available")
            except Exception:
                try:
                    import pytesseract
                    # 检查 tesseract 可执行文件是否存在
                    import shutil
                    if shutil.which('tesseract'):
                        self._engine = pytesseract
                        self._engine_type = "tesseract"
                        log.info("OCR engine initialized: Tesseract")
                    else:
                        raise Exception("Tesseract not in PATH")
                except Exception:
                    self._engine_type = "none"
                    log.warning("No OCR engine available")

    def extract_text(self, image_path: str) -> str:
        """提取图片中的文字"""
        self._init_engine()
        if self._engine_type == "paddle":
            return self._extract_paddle(image_path)
        elif self._engine_type == "tesseract":
            return self._extract_tesseract(image_path)
        elif self._engine_type == "windows":
            return self._extract_windows(image_path)
        else:
            return "[OCR不可用：请安装 paddleocr 或 pytesseract]"

    def extract_text_with_boxes(self, image_path: str) -> list[dict]:
        """提取文字及其位置坐标"""
        self._init_engine()
        if self._engine_type == "paddle":
            return self._extract_paddle_detailed(image_path)
        return [{"text": self.extract_text(image_path), "confidence": 0.0, "box": []}]

    def _extract_paddle(self, image_path: str) -> str:
        result = self._engine.ocr(image_path, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
        text = "\n".join(texts)
        log.info(f"PaddleOCR extracted {len(text)} chars from {Path(image_path).name}")
        return text

    def _extract_paddle_detailed(self, image_path: str) -> list[dict]:
        result = self._engine.ocr(image_path, cls=True)
        items = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    items.append({
                        "text": line[1][0],
                        "confidence": line[1][1],
                        "box": line[0],
                    })
        return items

    def _extract_tesseract(self, image_path: str) -> str:
        from PIL import Image
        img = Image.open(image_path)
        text = self._engine.image_to_string(img, lang="chi_sim+eng")
        log.info(f"Tesseract extracted {len(text)} chars from {Path(image_path).name}")
        return text.strip()

    def _extract_windows(self, image_path: str) -> str:
        """使用 Windows OCR 提取文字"""
        try:
            text = self._engine.extract_text(image_path)
            log.info(f"Windows OCR extracted {len(text)} chars from {Path(image_path).name}")
            return text
        except Exception as e:
            log.error(f"Windows OCR extraction failed: {e}")
            return f"[Windows OCR失败: {str(e)}]"


_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine
