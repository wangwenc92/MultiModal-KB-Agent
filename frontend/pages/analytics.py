import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="数据分析看板", page_icon="📊", layout="wide")

# ─────────────────────────────────────────────
#  Analytics CSS (shared palette with app.py)
# ─────────────────────────────────────────────
ANALYTICS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent: #6C63FF;
    --accent-light: #8B85FF;
    --surface: #161B22;
    --border: #30363D;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --success: #3FB950;
    --warning: #D29922;
    --danger: #F85149;
    --gradient-1: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
}

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Page title ── */
.analytics-title {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2rem;
    letter-spacing: -0.03em;
    margin-bottom: 0;
}
.analytics-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--accent);
    box-shadow: 0 4px 24px rgba(108, 99, 255, 0.12);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ── Section headers ── */
.analytics-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.analytics-section .section-icon {
    font-size: 1.4rem;
}
.analytics-section .section-title {
    font-weight: 700;
    font-size: 1.15rem;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.analytics-section .section-desc {
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-left: auto;
}

/* ── Chart containers ── */
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem;
    transition: border-color 0.2s;
}
.chart-card:hover {
    border-color: rgba(108, 99, 255, 0.3);
}
.chart-card .chart-label {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

/* ── Dataframe styling ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ── Divider ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}

/* ── Spacing ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
}

/* ── Info cards (empty state) ── */
.empty-card {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
}

/* ── Stat mini-card inside sections ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108, 99, 255, 0.1);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--accent-light);
    font-weight: 600;
}
</style>
"""
st.markdown(ANALYTICS_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  API helper
# ─────────────────────────────────────────────
def api_get(path: str):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown(
    '<div class="analytics-title">📊 数据分析看板</div>'
    '<div class="analytics-subtitle">知识库使用统计与数据洞察</div>',
    unsafe_allow_html=True,
)
st.markdown("")


# ─────────────────────────────────────────────
#  Overview metrics
# ─────────────────────────────────────────────
overview = api_get("/api/analytics/overview")
if overview:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("知识库", overview.get("knowledge_bases", 0))
    m2.metric("文档", overview.get("documents", 0))
    m3.metric("文本块", overview.get("chunks", 0))
    m4.metric("会话", overview.get("sessions", 0))
    m5.metric("消息", overview.get("messages", 0))
else:
    st.markdown(
        '<div class="empty-card">⚠️ 无法获取统计数据，请确认后端服务已启动</div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ─────────────────────────────────────────────
#  Activity trends (7 days)
# ─────────────────────────────────────────────
st.markdown(
    '<div class="analytics-section">'
    '  <span class="section-icon">📈</span>'
    '  <span class="section-title">最近 7 天活动趋势</span>'
    '  <span class="section-desc">会话 / 消息 / 文档上传</span>'
    '</div>',
    unsafe_allow_html=True,
)

activity = api_get("/api/analytics/activity?days=7")
if activity:
    try:
        import pandas as pd

        c1, c2, c3 = st.columns(3)

        chart_configs = [
            ("💬 会话数", activity.get("sessions", []), c1),
            ("📝 消息数", activity.get("messages", []), c2),
            ("📤 文档上传", activity.get("documents", []), c3),
        ]

        for label, data, col in chart_configs:
            with col:
                if data:
                    df = pd.DataFrame(data)
                    st.markdown(f'<div class="chart-card"><div class="chart-label">{label}</div>', unsafe_allow_html=True)
                    st.bar_chart(df.set_index("date"), height=200)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="chart-card"><div class="chart-label">{label}</div>'
                        '<div style="text-align:center;padding:2rem;color:#8B949E;">暂无数据</div></div>',
                        unsafe_allow_html=True,
                    )
    except ImportError:
        st.info("需要安装 pandas: pip install pandas")
else:
    st.markdown(
        '<div class="empty-card">📈 暂无活动数据</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Knowledge base details
# ─────────────────────────────────────────────
st.markdown(
    '<div class="analytics-section">'
    '  <span class="section-icon">📁</span>'
    '  <span class="section-title">知识库详情</span>'
    '  <span class="section-desc">各知识库文档与文本块统计</span>'
    '</div>',
    unsafe_allow_html=True,
)

kb_stats = api_get("/api/analytics/knowledge_bases")
if kb_stats:
    try:
        import pandas as pd

        df = pd.DataFrame(kb_stats)
        if not df.empty:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.dataframe(
                df[["name", "documents", "chunks", "created_at"]],
                use_container_width=True,
                column_config={
                    "name": st.column_config.TextColumn("知识库名称", width="medium"),
                    "documents": st.column_config.NumberColumn("文档数", format="%d"),
                    "chunks": st.column_config.NumberColumn("文本块", format="%d"),
                    "created_at": st.column_config.TextColumn("创建时间", width="medium"),
                },
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Chunk distribution chart (if multiple KBs)
            if len(kb_stats) > 1:
                chart_data = {kb["name"]: kb["chunks"] for kb in kb_stats if kb["chunks"] > 0}
                if chart_data:
                    st.markdown(
                        '<div class="analytics-section">'
                        '  <span class="section-icon">📊</span>'
                        '  <span class="section-title">文本块分布</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.bar_chart(chart_data, height=280)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-card">📭 暂无知识库数据</div>',
                unsafe_allow_html=True,
            )
    except ImportError:
        st.info("需要安装 pandas: pip install pandas")
else:
    st.markdown(
        '<div class="empty-card">📭 暂无知识库数据</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  File type distribution
# ─────────────────────────────────────────────
st.markdown(
    '<div class="analytics-section">'
    '  <span class="section-icon">📄</span>'
    '  <span class="section-title">文件类型分布</span>'
    '  <span class="section-desc">各格式文件数量统计</span>'
    '</div>',
    unsafe_allow_html=True,
)

file_types = api_get("/api/analytics/file_types")
if file_types:
    try:
        import pandas as pd

        df = pd.DataFrame(file_types)
        if not df.empty:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown('<div class="chart-card"><div class="chart-label">📋 文件类型明细</div>', unsafe_allow_html=True)
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "file_type": st.column_config.TextColumn("文件类型", width="medium"),
                        "count": st.column_config.NumberColumn("数量", format="%d"),
                    },
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                chart_data = {ft["file_type"]: ft["count"] for ft in file_types}
                st.markdown('<div class="chart-card"><div class="chart-label">📊 类型分布</div>', unsafe_allow_html=True)
                st.bar_chart(chart_data, height=300)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-card">📄 暂无文件数据</div>',
                unsafe_allow_html=True,
            )
    except ImportError:
        st.info("需要安装 pandas: pip install pandas")
else:
    st.markdown(
        '<div class="empty-card">📄 暂无文件数据</div>',
        unsafe_allow_html=True,
    )
