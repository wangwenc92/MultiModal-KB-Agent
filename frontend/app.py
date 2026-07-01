import streamlit as st
import requests
import json
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="MultiModal KB Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Modern CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent: #6C63FF;
    --accent-light: #8B85FF;
    --accent-dark: #4F46E5;
    --surface: #161B22;
    --surface-hover: #1C2333;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --success: #3FB950;
    --warning: #D29922;
    --danger: #F85149;
    --gradient-1: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    --gradient-2: linear-gradient(135deg, #6C63FF 0%, #A855F7 100%);
    --glass-bg: rgba(22, 27, 34, 0.7);
    --glass-border: rgba(108, 99, 255, 0.15);
}

/* ── Global ── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 50%, #1A1040 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.5rem !important;
    letter-spacing: -0.02em;
}

/* ── Sidebar divider ── */
section[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 0.8rem 0;
}

/* ── KB card buttons ── */
.kb-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.kb-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(108, 99, 255, 0.15);
    transform: translateY(-1px);
}
.kb-card.active {
    border-color: var(--accent);
    background: rgba(108, 99, 255, 0.1);
    box-shadow: 0 0 24px rgba(108, 99, 255, 0.2);
}
.kb-card .kb-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 2px;
}
.kb-card .kb-meta {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

/* ── Main title ── */
.main-title {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2rem;
    letter-spacing: -0.03em;
    margin-bottom: 0;
    line-height: 1.2;
}
.main-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 4px;
    font-weight: 400;
}

/* ── Mode badge ── */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.mode-rag {
    background: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.3);
}
.mode-agent {
    background: rgba(168, 85, 247, 0.15);
    color: #C084FC;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

/* ── Welcome screen ── */
.welcome-container {
    text-align: center;
    padding: 4rem 2rem;
}
.welcome-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    display: block;
}
.welcome-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.welcome-desc {
    color: var(--text-secondary);
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    max-width: 700px;
    margin: 0 auto;
}
.feature-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.25s ease;
}
.feature-card:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}
.feature-card .feat-icon { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.feature-card .feat-title { font-weight: 600; color: var(--text-primary); font-size: 0.9rem; }
.feature-card .feat-desc { color: var(--text-secondary); font-size: 0.78rem; margin-top: 4px; line-height: 1.4; }

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    border: 1px solid transparent;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08), rgba(59, 130, 246, 0.06));
    border-color: rgba(108, 99, 255, 0.12);
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: var(--surface);
    border-color: var(--border);
}

/* ── Sources section ── */
.source-card {
    background: rgba(108, 99, 255, 0.06);
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.source-card .source-file {
    font-weight: 600;
    color: var(--accent-light);
    font-size: 0.85rem;
    margin-bottom: 4px;
}
.source-card .source-text {
    color: var(--text-secondary);
    font-size: 0.8rem;
    line-height: 1.5;
}

/* ── Agent trace ── */
.trace-step {
    background: rgba(168, 85, 247, 0.06);
    border: 1px solid rgba(168, 85, 247, 0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.trace-step .trace-header {
    font-weight: 600;
    color: #C084FC;
    font-size: 0.85rem;
    margin-bottom: 6px;
}
.trace-step .trace-thought {
    color: var(--text-secondary);
    font-size: 0.82rem;
    font-style: italic;
    padding-left: 0.75rem;
    border-left: 2px solid var(--accent);
    margin: 6px 0;
}
.trace-step .trace-action {
    background: rgba(0,0,0,0.25);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.8rem;
    color: #7DD3FC;
    margin: 6px 0;
}
.trace-step .trace-observation {
    color: var(--text-secondary);
    font-size: 0.8rem;
    margin-top: 4px;
}

/* ── Empty state ── */
.empty-upload {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: var(--text-secondary);
    transition: border-color 0.2s;
}
.empty-upload:hover {
    border-color: var(--accent);
}

/* ── Sidebar KB info card ── */
.kb-info-card {
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
}
.kb-info-card .info-label {
    color: var(--text-secondary);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.kb-info-card .info-value {
    color: var(--text-primary);
    font-size: 1.1rem;
    font-weight: 600;
}

/* ── Upload progress ── */
.upload-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.85rem;
}
.upload-item .upload-ok { color: var(--success); }
.upload-item .upload-fail { color: var(--danger); }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Expander styling ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
}

/* ── File uploader area ── */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* ── Button polish ── */
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: var(--gradient-2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
    transform: translateY(-1px);
}

/* ── Sidebar info/warning/success ── */
section[data-testid="stSidebar"] div[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 0.85rem;
}

/* ── Pulse animation for active state ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(108, 99, 255, 0.2); }
    50% { box-shadow: 0 0 20px rgba(108, 99, 255, 0.4); }
}
.kb-card.active {
    animation: pulse-glow 2s ease-in-out infinite;
}

/* ── Remove default Streamlit padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────
def api_get(path: str):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, data: dict = None, files=None, form_data=None, timeout=60):
    try:
        if files:
            resp = requests.post(f"{API_BASE}{path}", files=files, data=form_data, timeout=timeout)
        else:
            resp = requests.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"请求失败: {e}")
        return None


def api_delete(path: str):
    try:
        resp = requests.delete(f"{API_BASE}{path}", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "current_kb" not in st.session_state:
    st.session_state.current_kb = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "rag"


def render_sources(sources: list):
    """Render source citations."""
    for src in sources:
        page_info = f" - 第{src['page']}页" if src.get("page") else ""
        st.markdown(
            f"""<div class="source-card">
                <div class="source-file">📄 {src['filename']}{page_info}</div>
                <div class="source-text">{src['content'][:150]}...</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_agent_trace(trace: list):
    """Render agent execution trace."""
    for t in trace:
        html = f'<div class="trace-step"><div class="trace-header">Step {t.get("step", "?")} &mdash; {t.get("type", "")}</div>'
        if t.get("thought"):
            html += f'<div class="trace-thought">💭 {t["thought"]}</div>'
        if t.get("action"):
            html += f'<div class="trace-action">🔧 {t["action"]}: {t.get("action_input", "")}</div>'
        if t.get("observation"):
            html += f'<div class="trace-observation">📤 {str(t["observation"])[:300]}</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 智能知识库")
    st.caption("Multimodal Knowledge Base Agent")

    # Create KB
    with st.expander("✨ 创建新知识库", expanded=False):
        kb_name = st.text_input("名称", key="new_kb_name", placeholder="例：产品文档库")
        kb_desc = st.text_area("描述（可选）", key="new_kb_desc", height=68, placeholder="简要描述知识库用途...")
        if st.button("🚀 创建知识库", use_container_width=True, type="primary"):
            if kb_name:
                result = api_post("/api/knowledge/create", {"name": kb_name, "description": kb_desc})
                if result:
                    st.success(f"「{result['name']}」创建成功")
                    st.rerun()
            else:
                st.warning("请输入知识库名称")

    st.divider()

    # KB list
    kbs = api_get("/api/knowledge/list") or []
    if kbs:
        st.caption(f"共 {len(kbs)} 个知识库")
        for kb in kbs:
            is_active = st.session_state.current_kb == kb["id"]
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(
                    f"{'▸' if is_active else ''} {kb['name']}",
                    key=f"kb_{kb['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_kb = kb["id"]
                    st.session_state.session_id = None
                    st.session_state.messages = []
                    st.rerun()
            with c2:
                if st.button("🗑", key=f"del_kb_{kb['id']}", help="删除此知识库"):
                    api_delete(f"/api/knowledge/{kb['id']}")
                    if st.session_state.current_kb == kb["id"]:
                        st.session_state.current_kb = None
                    st.rerun()

            # Show meta under active KB
            if is_active:
                st.markdown(
                    f"""<div style="padding:0 0.5rem 0.3rem;">
                        <span style="font-size:0.75rem;color:#8B949E;">
                            📄 {kb.get('doc_count', 0)} 文档
                        </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """<div style="text-align:center;padding:1.5rem;color:#8B949E;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">📭</div>
                <div style="font-size:0.9rem;">暂无知识库</div>
                <div style="font-size:0.78rem;margin-top:4px;">点击上方「创建新知识库」开始</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # File upload section
    if st.session_state.current_kb:
        st.divider()

        # KB info
        kb_detail = api_get(f"/api/knowledge/{st.session_state.current_kb}")
        if kb_detail:
            st.markdown(
                f"""<div class="kb-info-card">
                    <div style="display:flex;justify-content:space-between;">
                        <div>
                            <div class="info-label">文档数</div>
                            <div class="info-value">{kb_detail['doc_count']}</div>
                        </div>
                        <div>
                            <div class="info-label">文本块</div>
                            <div class="info-value">{kb_detail['chunk_count']}</div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📤 上传文件")
        st.caption("支持 PDF / DOCX / MD / TXT / CSV / 图片 / 音频 / 视频")
        uploaded_files = st.file_uploader(
            "拖放或选择文件",
            type=["pdf", "docx", "md", "txt", "csv", "jpg", "png", "mp3", "wav", "mp4"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("⬆️ 开始上传处理", use_container_width=True, type="primary"):
            results = []
            with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):
                for f in uploaded_files:
                    result = api_post(
                        "/api/upload",
                        files={"file": (f.name, f.getvalue())},
                        form_data={"knowledge_base_id": st.session_state.current_kb},
                    )
                    results.append((f.name, result))
            ok = sum(1 for _, r in results if r)
            fail = len(results) - ok
            if ok:
                st.success(f"成功处理 {ok} 个文件" + (f"，{fail} 个失败" if fail else ""))
            else:
                st.error("所有文件上传失败")
            st.rerun()


# ─────────────────────────────────────────────
#  Main content area
# ─────────────────────────────────────────────

# Header
col_title, col_mode = st.columns([4, 1])
with col_title:
    st.markdown(
        '<div class="main-title">💬 智能知识库问答</div>'
        '<div class="main-subtitle">基于 RAG 和 Agent 的多模态智能检索问答系统</div>',
        unsafe_allow_html=True,
    )
with col_mode:
    mode = st.selectbox(
        "问答模式",
        ["rag", "agent"],
        format_func=lambda x: "📚 RAG 检索问答" if x == "rag" else "🤖 Agent 智能体",
        label_visibility="collapsed",
    )
    st.session_state.mode = mode
    badge_class = "mode-rag" if mode == "rag" else "mode-agent"
    badge_icon = "📚" if mode == "rag" else "🤖"
    badge_text = "RAG 检索问答" if mode == "rag" else "Agent 智能体"
    st.markdown(
        f'<div style="text-align:right;"><span class="mode-badge {badge_class}">{badge_icon} {badge_text}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── No KB selected: welcome screen ──
if not st.session_state.current_kb:
    st.markdown(
        """<div class="welcome-container">
            <span class="welcome-icon">🧠</span>
            <div class="welcome-title">欢迎使用 MultiModal KB Agent</div>
            <div class="welcome-desc">
                一个支持文档、图片、音频、视频的多模态智能知识库。<br>
                请在左侧选择或创建一个知识库开始使用。
            </div>
            <div class="feature-grid">
                <div class="feature-card">
                    <span class="feat-icon">📄</span>
                    <div class="feat-title">多格式支持</div>
                    <div class="feat-desc">PDF、Word、Markdown、CSV 等文档格式</div>
                </div>
                <div class="feature-card">
                    <span class="feat-icon">🖼️</span>
                    <div class="feat-title">多模态理解</div>
                    <div class="feat-desc">图片 OCR、音频转写、视频分析</div>
                </div>
                <div class="feature-card">
                    <span class="feat-icon">🤖</span>
                    <div class="feat-title">智能 Agent</div>
                    <div class="feat-desc">推理、工具调用、多步问题求解</div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    # ── Chat messages ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 引用来源 ({len(msg['sources'])})"):
                    render_sources(msg["sources"])
            if msg.get("agent_trace"):
                with st.expander(f"🔍 Agent 执行轨迹 ({len(msg['agent_trace'])} 步)"):
                    render_agent_trace(msg["agent_trace"])

    # ── Chat input ──
    placeholder_text = "输入你的问题，按 Enter 发送..."
    if question := st.chat_input(placeholder_text):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            status_placeholder = st.empty()
            answer_text = ""
            sources_data = []
            agent_trace_data = []
            current_step_text = ""

            try:
                resp = requests.post(
                    f"{API_BASE}/api/chat/stream",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "knowledge_base_id": st.session_state.current_kb,
                        "mode": st.session_state.mode,
                    },
                    stream=True,
                    timeout=300,
                )

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    event = json.loads(line[6:])
                    etype = event.get("type")

                    if etype == "chunk":
                        answer_text += event["content"]
                        text_placeholder.markdown(answer_text + "▌")

                    elif etype == "thought":
                        current_step_text = f"💭 {event['content']}"
                        status_placeholder.caption(current_step_text)

                    elif etype == "tool_call":
                        current_step_text = f"🔧 调用: {event['name']}"
                        status_placeholder.caption(current_step_text)

                    elif etype == "tool_result":
                        content_preview = event.get("content", "")[:80]
                        current_step_text = f"📤 {event['name']} → {content_preview}..."
                        status_placeholder.caption(current_step_text)

                    elif etype == "done":
                        text_placeholder.markdown(answer_text)
                        status_placeholder.empty()
                        st.session_state.session_id = event.get("session_id", st.session_state.session_id)
                        sources_data = event.get("sources", [])
                        agent_trace_data = event.get("agent_trace", [])
                        break

                    elif etype == "error":
                        text_placeholder.markdown(f"❌ 错误: {event['content']}")
                        break

            except Exception as e:
                text_placeholder.markdown(f"❌ 请求失败: {e}")

            # 显示引用来源
            if sources_data:
                with st.expander(f"📎 引用来源 ({len(sources_data)})"):
                    render_sources(sources_data)
            # 显示 Agent 轨迹
            if agent_trace_data:
                with st.expander(f"🔍 Agent 执行轨迹 ({len(agent_trace_data)} 步)"):
                    render_agent_trace(agent_trace_data)

            # 存到会话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources_data,
                "agent_trace": agent_trace_data,
            })
                    st.markdown(
                        """<div style="background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,0.3);
                            border-radius:10px;padding:1rem;color:#F85149;">
                            ⚠️ 获取回答失败，请检查后端服务是否正常运行
                        </div>""",
                        unsafe_allow_html=True,
                    )
