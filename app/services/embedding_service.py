from __future__ import annotations

import hashlib
import struct
import os
from app.config import get_settings
from app.utils.logger import log

os.environ.setdefault("HF_HUB_OFFLINE", "1")

settings = get_settings()

# 内存缓存: text -> embedding
_cache: dict[str, list[float]] = {}


def _hash_embedding(text: str, dim: int = 768) -> list[float]:
    """基于哈希的确定性embedding回退方案"""
    import math
    h = hashlib.sha512(text.encode("utf-8")).digest()
    # 循环填充到目标维度
    raw = []
    while len(raw) < dim:
        for i in range(0, len(h) - 3, 4):
            val = struct.unpack("f", h[i:i+4])[0]
            # 过滤掉 NaN 和 Infinity
            if math.isfinite(val):
                raw.append(val)
            else:
                raw.append(0.0)
            if len(raw) >= dim:
                break
        h = hashlib.sha512(h).digest()
    # 归一化
    vec = raw[:dim]
    norm = sum(x*x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class EmbeddingService:
    """Embedding服务，支持API、本地模型和哈希回退"""

    def __init__(self):
        self._local_model = None
        self._openai_client = None
        self._mode = None  # 'api', 'local', 'hash'

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )
        return self._openai_client

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading local embedding model: shibing624/text2vec-base-chinese")
            self._local_model = SentenceTransformer("shibing624/text2vec-base-chinese")
            log.info("Local embedding model loaded")
        return self._local_model

    def _detect_mode(self):
        if self._mode is not None:
            return
        # 尝试API
        try:
            client = self._get_openai_client()
            resp = client.embeddings.create(input=["test"], model=settings.EMBEDDING_MODEL)
            _ = resp.data[0].embedding
            self._mode = "api"
            log.info("Embedding mode: API")
            return
        except Exception as e:
            log.warning(f"Embedding API unavailable: {e}")
        # 本地模型存在兼容性问题(numpy 2.x segfault)，先用哈希回退
        # 如需启用本地模型，取消下方注释
        # try:
        #     model = self._get_local_model()
        #     emb = model.encode(["test"], normalize_embeddings=True)
        #     _ = emb[0].tolist()
        #     self._mode = "local"
        #     log.info("Embedding mode: local model")
        #     return
        # except Exception as e:
        #     log.warning(f"Local embedding model unavailable: {e}")
        self._mode = "hash"
        log.info("Embedding mode: hash fallback (configure EMBEDDING_API for better results)")

    def embed_text(self, text: str) -> list[float]:
        if text in _cache:
            return _cache[text]
        embedding = self._embed([text])[0]
        _cache[text] = embedding
        return embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        to_embed = []
        indices = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            if t in _cache:
                results[i] = _cache[t]
            else:
                to_embed.append(t)
                indices.append(i)

        if to_embed:
            batch_size = 100
            for start in range(0, len(to_embed), batch_size):
                batch = to_embed[start : start + batch_size]
                batch_indices = indices[start : start + batch_size]
                embeddings = self._embed(batch)
                for j, emb in enumerate(embeddings):
                    _cache[batch[j]] = emb
                    results[batch_indices[j]] = emb

        return results

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._detect_mode()
        if self._mode == "api":
            return self._embed_api(texts)
        elif self._mode == "local":
            return self._embed_local(texts)
        return self._embed_hash(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._get_local_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai_client()
        resp = client.embeddings.create(input=texts, model=settings.EMBEDDING_MODEL)
        return [data.embedding for data in resp.data]

    def _embed_hash(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(t) for t in texts]


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
