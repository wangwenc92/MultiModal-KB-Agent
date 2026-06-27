from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database import get_db, KnowledgeBaseModel, DocumentModel, ChunkModel
from app.models.schemas import KnowledgeBaseCreate, KnowledgeBaseOut
from app.core.rag.embedder import get_vector_store
from app.utils.helpers import generate_id
from app.utils.logger import log

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/create", response_model=KnowledgeBaseOut)
async def create_knowledge_base(req: KnowledgeBaseCreate, db: Session = Depends(get_db)):
    kb = KnowledgeBaseModel(
        id=generate_id(),
        name=req.name,
        description=req.description,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    log.info(f"Created knowledge base: {kb.id} - {kb.name}")
    return KnowledgeBaseOut(id=kb.id, name=kb.name, description=kb.description, doc_count=0, chunk_count=0, created_at=kb.created_at)


@router.get("/list", response_model=list[KnowledgeBaseOut])
async def list_knowledge_bases(db: Session = Depends(get_db)):
    kbs = db.query(KnowledgeBaseModel).order_by(KnowledgeBaseModel.created_at.desc()).all()
    result = []
    for kb in kbs:
        doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb.id).count()
        chunk_count = (
            db.query(func.sum(DocumentModel.chunk_count))
            .filter(DocumentModel.kb_id == kb.id)
            .scalar()
            or 0
        )
        result.append(KnowledgeBaseOut(
            id=kb.id, name=kb.name, description=kb.description,
            doc_count=doc_count, chunk_count=chunk_count, created_at=kb.created_at,
        ))
    return result


@router.get("/{kb_id}", response_model=KnowledgeBaseOut)
async def get_knowledge_base(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb.id).count()
    chunk_count = (
        db.query(func.sum(DocumentModel.chunk_count))
        .filter(DocumentModel.kb_id == kb.id)
        .scalar()
        or 0
    )
    return KnowledgeBaseOut(
        id=kb.id, name=kb.name, description=kb.description,
        doc_count=doc_count, chunk_count=chunk_count, created_at=kb.created_at,
    )


@router.delete("/{kb_id}")
async def delete_knowledge_base(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 删除向量库中的数据
    try:
        vs = get_vector_store()
        vs.delete_by_kb_id(kb_id)
    except Exception as e:
        log.warning(f"Failed to delete vector data for kb {kb_id}: {e}")

    db.delete(kb)
    db.commit()
    log.info(f"Deleted knowledge base: {kb_id}")
    return {"code": 200, "message": "Knowledge base deleted"}
