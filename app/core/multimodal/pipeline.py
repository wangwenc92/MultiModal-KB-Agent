from __future__ import annotations

from pathlib import Path
from app.core.rag.loader import DocumentLoader
from app.core.rag.splitter import TextSplitter
from app.core.rag.embedder import get_vector_store
from app.utils.logger import log

# 文件类型分类
DOCUMENT_TYPES = {".pdf", ".docx", ".doc", ".md", ".txt", ".csv"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_TYPES = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}

ALL_SUPPORTED = DOCUMENT_TYPES | IMAGE_TYPES | AUDIO_TYPES | VIDEO_TYPES


class MultiModalPipeline:
    """多模态统一处理管线：根据文件类型自动路由到对应处理器"""

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

    def process_file(self, file_path: str) -> dict:
        """
        处理文件，返回统一格式
        返回: {"content": str, "metadata": dict, "file_type": str}
        """
        ext = Path(file_path).suffix.lower()
        if ext not in ALL_SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext in DOCUMENT_TYPES:
            return self._process_document(file_path)
        elif ext in IMAGE_TYPES:
            return self._process_image(file_path)
        elif ext in AUDIO_TYPES:
            return self._process_audio(file_path)
        elif ext in VIDEO_TYPES:
            return self._process_video(file_path)

    def process_and_index(self, file_path: str, doc_id: str, kb_id: str) -> int:
        """处理文件并索引到向量库，返回chunk数量"""
        result = self.process_file(file_path)
        content = result["content"]
        metadata = result["metadata"]
        metadata["kb_id"] = kb_id  # 添加kb_id到metadata

        if not content.strip():
            log.warning(f"No content extracted from {Path(file_path).name}")
            return 0

        # 分块
        chunks = self.splitter.split([{"content": content, "metadata": metadata}])

        # 存入向量库
        vs = get_vector_store()
        count = vs.add_documents(chunks, doc_id)

        log.info(f"Processed and indexed {Path(file_path).name}: {count} chunks")
        return count

    def _process_document(self, file_path: str) -> dict:
        """处理文档类型"""
        pages = self.loader.load(file_path)
        content = "\n\n".join(p["content"] for p in pages)
        metadata = pages[0]["metadata"] if pages else {"source": Path(file_path).name, "type": "document"}
        return {"content": content, "metadata": metadata, "file_type": "document"}

    def _process_image(self, file_path: str) -> dict:
        """处理图片类型"""
        from app.core.multimodal.image import get_image_processor
        processor = get_image_processor()
        result = processor.process(file_path)
        return {
            "content": result["combined"],
            "metadata": {"source": Path(file_path).name, "type": "image"},
            "file_type": "image",
        }

    def _process_audio(self, file_path: str) -> dict:
        """处理音频类型"""
        from app.core.multimodal.audio import get_audio_processor
        processor = get_audio_processor()
        result = processor.transcribe(file_path)
        return {
            "content": result["text"],
            "metadata": {"source": Path(file_path).name, "type": "audio", "language": result.get("language", "")},
            "file_type": "audio",
        }

    def _process_video(self, file_path: str) -> dict:
        """处理视频类型"""
        from app.core.multimodal.video import get_video_processor
        processor = get_video_processor()
        result = processor.process(file_path)
        return {
            "content": result["combined"],
            "metadata": {"source": Path(file_path).name, "type": "video"},
            "file_type": "video",
        }


_pipeline: MultiModalPipeline | None = None


def get_pipeline() -> MultiModalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiModalPipeline()
    return _pipeline
