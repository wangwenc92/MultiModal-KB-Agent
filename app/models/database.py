from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, Float, ForeignKey, create_engine, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.MYSQL_URL, pool_size=10, max_overflow=20, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class KnowledgeBaseModel(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(32), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("DocumentModel", back_populates="knowledge_base", cascade="all, delete-orphan")


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String(32), primary_key=True)
    kb_id = Column(String(32), ForeignKey("knowledge_bases.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBaseModel", back_populates="documents")
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    __tablename__ = "chunks"

    id = Column(String(32), primary_key=True)
    doc_id = Column(String(32), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, default=dict)
    embedding_status = Column(String(20), default="pending")

    document = relationship("DocumentModel", back_populates="chunks")


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True)
    title = Column(String(200), default="新对话")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("MessageModel", back_populates="session", cascade="all, delete-orphan")


class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(String(32), primary_key=True)
    session_id = Column(String(32), ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    agent_trace = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("SessionModel", back_populates="messages")
