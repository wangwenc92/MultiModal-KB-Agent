from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, Field


# === 通用 ===

class APIResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[Union[dict, list]] = None


class APIError(BaseModel):
    code: int
    message: str
    detail: str = ""


# === 知识库 ===

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field(default="", max_length=500, description="知识库描述")


class KnowledgeBaseOut(BaseModel):
    id: str
    name: str
    description: str
    doc_count: int = 0
    chunk_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


# === 文档 ===

class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int = 0
    status: str = "processing"
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str


# === 对话 ===

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(default=None, description="会话ID，为空则创建新会话")
    knowledge_base_id: str = Field(..., description="知识库ID")
    mode: str = Field(default="rag", description="模式: rag 或 agent")
    max_iterations: int = Field(default=5, ge=1, le=20, description="Agent最大推理步数")


class SourceInfo(BaseModel):
    content: str
    filename: str
    page: Optional[int] = None
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    session_id: str
    agent_trace: Optional[list[dict]] = None


# === 会话 ===

class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: Optional[list[dict]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# === Agent ===

class ToolCallInfo(BaseModel):
    tool_name: str
    input: dict
    output: str
    duration_ms: int = 0


class AgentTrace(BaseModel):
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
