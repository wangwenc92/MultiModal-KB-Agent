import os
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.models.database import get_db, DocumentModel, ChunkModel
from app.models.schemas import DocumentUploadResponse
from app.core.multimodal.pipeline import get_pipeline, ALL_SUPPORTED
from app.core.rag.embedder import get_vector_store
from app.config import get_settings
from app.utils.helpers import generate_id, ensure_dir
from app.utils.logger import log

settings = get_settings()
router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post("", response_model=DocumentUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALL_SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALL_SUPPORTED))}")

    # 保存文件
    upload_dir = ensure_dir(os.path.join(settings.UPLOAD_DIR, knowledge_base_id))
    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    doc_id = generate_id()

    # 创建文档记录
    doc = DocumentModel(
        id=doc_id,
        kb_id=knowledge_base_id,
        filename=file.filename,
        file_type=ext,
        file_size=len(content),
        status="processing",
    )
    db.add(doc)
    db.commit()

    # 使用统一管线处理
    try:
        pipeline = get_pipeline()
        chunk_count = pipeline.process_and_index(file_path, doc_id, knowledge_base_id)

        # 记录chunks到数据库（简化：只记录数量，向量库已有完整数据）
        doc.chunk_count = chunk_count
        doc.status = "completed"
        db.commit()

        log.info(f"Uploaded and indexed: {file.filename}, {chunk_count} chunks")
        return DocumentUploadResponse(
            doc_id=doc_id, filename=file.filename, chunk_count=chunk_count, status="completed",
        )

    except Exception as e:
        doc.status = "failed"
        db.commit()
        log.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        vs = get_vector_store()
        vs.delete_by_doc_id(doc_id)
    except Exception as e:
        log.warning(f"Failed to delete vector data for doc {doc_id}: {e}")

    db.delete(doc)
    db.commit()
    log.info(f"Deleted document: {doc_id}")
    return {"code": 200, "message": "Document deleted"}
