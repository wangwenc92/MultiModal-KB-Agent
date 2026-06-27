from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from app.models.database import (
    get_db, KnowledgeBaseModel, DocumentModel,
    SessionModel, MessageModel,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
async def get_overview(db: Session = Depends(get_db)):
    """系统总览统计"""
    kb_count = db.query(KnowledgeBaseModel).count()
    doc_count = db.query(DocumentModel).count()
    chunk_count = db.query(func.sum(DocumentModel.chunk_count)).scalar() or 0
    session_count = db.query(SessionModel).count()
    message_count = db.query(MessageModel).count()

    return {
        "knowledge_bases": kb_count,
        "documents": doc_count,
        "chunks": chunk_count,
        "sessions": session_count,
        "messages": message_count,
    }


@router.get("/activity")
async def get_activity(days: int = 7, db: Session = Depends(get_db)):
    """最近N天的活动趋势"""
    since = datetime.utcnow() - timedelta(days=days)

    # 按天统计会话数
    sessions_by_day = (
        db.query(
            func.date(SessionModel.created_at).label("date"),
            func.count().label("count"),
        )
        .filter(SessionModel.created_at >= since)
        .group_by(func.date(SessionModel.created_at))
        .all()
    )

    # 按天统计消息数
    messages_by_day = (
        db.query(
            func.date(MessageModel.created_at).label("date"),
            func.count().label("count"),
        )
        .filter(MessageModel.created_at >= since)
        .group_by(func.date(MessageModel.created_at))
        .all()
    )

    # 按天统计文档上传数
    docs_by_day = (
        db.query(
            func.date(DocumentModel.created_at).label("date"),
            func.count().label("count"),
        )
        .filter(DocumentModel.created_at >= since)
        .group_by(func.date(DocumentModel.created_at))
        .all()
    )

    return {
        "period_days": days,
        "sessions": [{"date": str(r.date), "count": r.count} for r in sessions_by_day],
        "messages": [{"date": str(r.date), "count": r.count} for r in messages_by_day],
        "documents": [{"date": str(r.date), "count": r.count} for r in docs_by_day],
    }


@router.get("/knowledge_bases")
async def get_kb_stats(db: Session = Depends(get_db)):
    """各知识库使用统计"""
    kbs = db.query(KnowledgeBaseModel).all()
    result = []
    for kb in kbs:
        doc_count = db.query(DocumentModel).filter(DocumentModel.kb_id == kb.id).count()
        chunk_count = (
            db.query(func.sum(DocumentModel.chunk_count))
            .filter(DocumentModel.kb_id == kb.id)
            .scalar()
            or 0
        )
        result.append({
            "id": kb.id,
            "name": kb.name,
            "documents": doc_count,
            "chunks": chunk_count,
            "created_at": kb.created_at.isoformat(),
        })

    # 按chunk数降序
    result.sort(key=lambda x: x["chunks"], reverse=True)
    return result


@router.get("/file_types")
async def get_file_type_stats(db: Session = Depends(get_db)):
    """文件类型分布"""
    type_stats = (
        db.query(
            DocumentModel.file_type,
            func.count().label("count"),
            func.sum(DocumentModel.file_size).label("total_size"),
        )
        .group_by(DocumentModel.file_type)
        .all()
    )
    return [
        {"file_type": r.file_type, "count": r.count, "total_size": r.total_size or 0}
        for r in type_stats
    ]
