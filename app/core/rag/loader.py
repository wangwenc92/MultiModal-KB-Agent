from pathlib import Path
from typing import Optional
from app.utils.logger import log


class DocumentLoader:
    """统一文档加载器，支持 PDF/DOCX/MD/TXT/CSV 格式"""

    def load(self, file_path: str) -> list[dict]:
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".md": self._load_markdown,
            ".txt": self._load_text,
            ".csv": self._load_csv,
        }
        loader = loaders.get(ext)
        if not loader:
            raise ValueError(f"Unsupported file type: {ext}")
        docs = loader(file_path)
        log.info(f"Loaded {len(docs)} pages from {Path(file_path).name}")
        return docs

    def _load_pdf(self, file_path: str) -> list[dict]:
        import fitz
        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "content": text,
                    "metadata": {"source": Path(file_path).name, "page": i + 1, "type": "pdf"},
                })
        doc.close()
        return pages

    def _load_docx(self, file_path: str) -> list[dict]:
        from docx import Document
        doc = Document(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not full_text:
            return []
        return [{"content": full_text, "metadata": {"source": Path(file_path).name, "page": 1, "type": "docx"}}]

    def _load_markdown(self, file_path: str) -> list[dict]:
        text = Path(file_path).read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [{"content": text, "metadata": {"source": Path(file_path).name, "page": 1, "type": "markdown"}}]

    def _load_text(self, file_path: str) -> list[dict]:
        text = Path(file_path).read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [{"content": text, "metadata": {"source": Path(file_path).name, "page": 1, "type": "txt"}}]

    def _load_csv(self, file_path: str) -> list[dict]:
        import pandas as pd
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        return [{"content": text, "metadata": {"source": Path(file_path).name, "page": 1, "type": "csv"}}]
