from __future__ import annotations

import chromadb
from app.config import get_settings
from app.services.embedding_service import get_embedding_service
from app.utils.logger import log
from app.utils.helpers import generate_id

settings = get_settings()


class VectorStore:
    """ChromaDB向量存储封装"""

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_service = get_embedding_service()

    def add_documents(self, chunks: list[dict], doc_id: str) -> int:
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        ids = [f"{doc_id}_{generate_id()}" for _ in chunks]
        metadatas = []
        for c in chunks:
            m = dict(c.get("metadata", {}))
            m["doc_id"] = doc_id
            metadatas.append(m)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        log.info(f"Added {len(chunks)} chunks to vector store for doc {doc_id}")
        return len(chunks)

    def search(self, query: str, top_k: int = 5, kb_id: str | None = None) -> list[dict]:
        query_embedding = self.embedding_service.embed_text(query)
        where = {"kb_id": kb_id} if kb_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                hits.append({
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # cosine distance -> similarity
                })
        return hits

    def delete_by_doc_id(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})
        log.info(f"Deleted chunks for doc {doc_id} from vector store")

    def delete_by_kb_id(self, kb_id: str):
        self.collection.delete(where={"kb_id": kb_id})
        log.info(f"Deleted all chunks for kb {kb_id} from vector store")

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
