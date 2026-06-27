from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database import get_db, KnowledgeBaseModel, DocumentModel, SessionModel, ChunkModel
from app.core.rag.embedder import get_vector_store

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    kb_count = db.query(KnowledgeBaseModel).count()
    doc_count = db.query(DocumentModel).count()
    chunk_count = db.query(func.sum(DocumentModel.chunk_count)).scalar() or 0
    session_count = db.query(SessionModel).count()

    vs_stats = {}
    try:
        vs = get_vector_store()
        vs_stats = vs.get_stats()
    except Exception:
        pass

    return {
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "chunks": chunk_count,
        "sessions": session_count,
        "vector_store": vs_stats,
    }


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {"api": True, "database": False, "vector_store": False}

    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass

    try:
        vs = get_vector_store()
        vs.get_stats()
        checks["vector_store"] = True
    except Exception:
        pass

    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}
