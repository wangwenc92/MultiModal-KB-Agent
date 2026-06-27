import re
from typing import Optional
from app.utils.logger import log


class TextSplitter:
    """文本分块器，支持递归字符分割、句子分割、Markdown分割"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[dict], strategy: str = "recursive") -> list[dict]:
        strategies = {
            "recursive": self._split_recursive,
            "sentence": self._split_sentence,
            "markdown": self._split_markdown,
        }
        splitter = strategies.get(strategy, self._split_recursive)
        chunks = []
        for doc in documents:
            chunks.extend(splitter(doc))
        log.info(f"Split {len(documents)} documents into {len(chunks)} chunks (strategy={strategy})")
        return chunks

    def _split_recursive(self, doc: dict) -> list[dict]:
        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""]
        text = doc["content"]
        return self._recursive_split(text, separators, doc["metadata"])

    def _recursive_split(self, text: str, separators: list[str], metadata: dict) -> list[dict]:
        if len(text) <= self.chunk_size:
            return [{"content": text.strip(), "metadata": metadata}] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            return self._char_split(text, metadata)

        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    if len(current) <= self.chunk_size:
                        chunks.append({"content": current.strip(), "metadata": metadata})
                    else:
                        chunks.extend(self._recursive_split(current, remaining_seps, metadata))
                current = part
        if current.strip():
            if len(current) <= self.chunk_size:
                chunks.append({"content": current.strip(), "metadata": metadata})
            else:
                chunks.extend(self._recursive_split(current, remaining_seps, metadata))

        return self._apply_overlap(chunks)

    def _char_split(self, text: str, metadata: dict) -> list[dict]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"content": chunk, "metadata": metadata})
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: list[dict]) -> list[dict]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]["content"]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            current = chunks[i]["content"]
            if not current.startswith(overlap_text.strip()):
                chunks[i]["content"] = overlap_text + current
            result.append(chunks[i])
        return result

    def _split_sentence(self, doc: dict) -> list[dict]:
        text = doc["content"]
        metadata = doc["metadata"]
        sentences = re.split(r'(?<=[。！？.!?])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        for sent in sentences:
            candidate = current + sent if current else sent
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append({"content": current, "metadata": metadata})
                current = sent
        if current:
            chunks.append({"content": current, "metadata": metadata})
        return self._apply_overlap(chunks)

    def _split_markdown(self, doc: dict) -> list[dict]:
        text = doc["content"]
        metadata = doc["metadata"]
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        sections = [s.strip() for s in sections if s.strip()]

        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append({"content": section, "metadata": metadata})
            else:
                chunks.extend(self._recursive_split(
                    section,
                    ["\n\n", "\n", "。", ".", " ", ""],
                    metadata,
                ))
        return chunks
