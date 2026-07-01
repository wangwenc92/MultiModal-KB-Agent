from __future__ import annotations

import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.models.database import get_db, SessionModel, MessageModel
from app.models.schemas import ChatRequest, ChatResponse, SourceInfo, SessionOut, MessageOut
from app.core.rag.chain import get_rag_chain
from app.core.agent.executor import get_agent_executor
from app.core.memory.context import get_context_manager
from app.core.agent.tools import get_tool_descriptions
from app.utils.helpers import generate_id
from app.utils.logger import log

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_or_create_session(db: Session, session_id: str | None, question: str) -> SessionModel:
    if session_id:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if session:
            return session
    session = SessionModel(id=generate_id(), title=question[:50])
    db.add(session)
    db.flush()
    return session


@router.post("/send", response_model=ChatResponse)
async def send_message(req: ChatRequest, db: Session = Depends(get_db)):
    session = _get_or_create_session(db, req.session_id, req.question)

    # 保存用户消息
    user_msg = MessageModel(
        id=generate_id(),
        session_id=session.id,
        role="user",
        content=req.question,
    )
    db.add(user_msg)

    # 更新记忆
    ctx = get_context_manager()
    ctx.add_message(session.id, "user", req.question)

    agent_trace = None
    sources = []

    if req.mode == "agent":
        # Agent模式 —— 异步执行，不阻塞事件循环
        context = ctx.build_context(req.question, session.id, req.knowledge_base_id)
        agent = get_agent_executor(max_iterations=req.max_iterations)
        result = await agent.arun(req.question, context=context)
        answer = result["answer"]
        agent_trace = result.get("trace", [])
    else:
        # RAG模式
        rag = get_rag_chain()
        result = rag.ask(req.question, req.knowledge_base_id)
        answer = result["answer"]
        sources = [SourceInfo(**s) for s in result.get("sources", [])]

    # 更新记忆
    ctx.add_message(session.id, "assistant", answer)

    # 保存AI回答
    ai_msg = MessageModel(
        id=generate_id(),
        session_id=session.id,
        role="assistant",
        content=answer,
        sources=[s.model_dump() for s in sources] if sources else None,
        agent_trace=agent_trace,
    )
    db.add(ai_msg)
    db.commit()

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session.id,
        agent_trace=agent_trace,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """流式对话接口（SSE）。逐块返回完整回答，用户逐字看到内容。"""
    session = _get_or_create_session(db, req.session_id, req.question)

    # 保存用户消息
    user_msg = MessageModel(
        id=generate_id(), session_id=session.id, role="user", content=req.question,
    )
    db.add(user_msg)
    db.commit()

    ctx = get_context_manager()
    ctx.add_message(session.id, "user", req.question)

    async def event_generator():
        full_answer = ""
        sources_data = []
        agent_trace_data = None

        try:
            if req.mode == "agent":
                # Agent模式：先出中间状态事件，再流式输出最终回答
                context = ctx.build_context(req.question, session.id, req.knowledge_base_id)
                agent = get_agent_executor(max_iterations=req.max_iterations)

                async for event in agent.stream_run(req.question, context=context):
                    if event["type"] == "thought":
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    elif event["type"] == "tool_call":
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    elif event["type"] == "tool_result":
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    elif event["type"] == "chunk":
                        full_answer += event["content"]
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    elif event["type"] == "done":
                        agent_trace_data = event.get("trace", [])
            else:
                # RAG模式：直接流式输出
                rag = get_rag_chain()
                async for chunk in rag.ask_stream(req.question, req.knowledge_base_id):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
                sources_data = rag.last_sources if hasattr(rag, "last_sources") else []

        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            return

        # 保存AI回答
        ctx.add_message(session.id, "assistant", full_answer)
        try:
            ai_msg = MessageModel(
                id=generate_id(),
                session_id=session.id,
                role="assistant",
                content=full_answer,
                sources=sources_data if sources_data else None,
                agent_trace=agent_trace_data,
            )
            db.add(ai_msg)
            db.commit()
        except Exception as e:
            log.error(f"Failed to save streaming message: {e}")

        # 结束事件
        done_event = {
            "type": "done",
            "session_id": session.id,
            "sources": sources_data,
            "agent_trace": agent_trace_data,
        }
        yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/tools")
async def list_tools():
    return get_tool_descriptions()


@router.get("/history/{session_id}", response_model=list[MessageOut])
async def get_history(session_id: str, db: Session = Depends(get_db)):
    messages = (
        db.query(MessageModel)
        .filter(MessageModel.session_id == session_id)
        .order_by(MessageModel.created_at)
        .all()
    )
    return messages


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(db: Session = Depends(get_db)):
    sessions = db.query(SessionModel).order_by(SessionModel.created_at.desc()).all()
    result = []
    for s in sessions:
        msg_count = db.query(MessageModel).filter(MessageModel.session_id == s.id).count()
        result.append(SessionOut(id=s.id, title=s.title, created_at=s.created_at, message_count=msg_count))
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        return {"code": 404, "message": "Session not found"}
    db.delete(session)
    db.commit()
    return {"code": 200, "message": "Session deleted"}

