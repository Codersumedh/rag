import time
import json
import uuid
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from run_query import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedOnc Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL STORAGE  (JSON file, simple, no DB needed)
# ─────────────────────────────────────────────────────────────────────────────
STORAGE_FILE = "./chat_storage.json"

def load_storage() -> dict:
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            return json.load(f)
    return {"chats": {}, "order": []}   # order = list of chat ids newest first

def save_storage(data: dict):
    with open(STORAGE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def new_chat(storage: dict) -> str:
    cid   = str(uuid.uuid4())[:8]
    title = f"Chat {len(storage['order']) + 1}"
    storage["chats"][cid] = {"title": title, "messages": []}
    storage["order"].insert(0, cid)
    save_storage(storage)
    return cid

def delete_chat(storage: dict, cid: str):
    storage["chats"].pop(cid, None)
    if cid in storage["order"]:
        storage["order"].remove(cid)
    save_storage(storage)

def append_message(storage: dict, cid: str, msg: dict):
    storage["chats"][cid]["messages"].append(msg)
    # Auto-title: first 40 chars of first user query
    if len(storage["chats"][cid]["messages"]) == 1:
        storage["chats"][cid]["title"] = msg["query"][:40] + ("…" if len(msg["query"]) > 40 else "")
    save_storage(storage)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0d1117 !important;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}

/* ── Sidebar new chat button ── */
div[data-testid="stSidebar"] .stButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.5rem 0.8rem !important;
    margin-bottom: 2px !important;
    transition: background 0.15s !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #30363d !important;
    transform: none !important;
}

/* ── Active chat in sidebar ── */
.chat-item-active > div > button {
    background: #1f6feb22 !important;
    border-color: #1f6feb88 !important;
    color: #58a6ff !important;
}

/* ── Main chat area ── */
.chat-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 1rem 1rem 6rem 1rem;
}

/* ── Welcome screen ── */
.welcome {
    text-align: center;
    padding: 5rem 2rem 2rem 2rem;
}
.welcome h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 20%, #58a6ff 60%, #00d4aa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.welcome p { color: #8b949e; font-size: 1rem; }
.welcome-chips {
    display: flex; flex-wrap: wrap; gap: 0.6rem;
    justify-content: center; margin-top: 2rem;
}
.chip {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    color: #8b949e;
    cursor: pointer;
}

/* ── Chat bubbles ── */
.msg-user {
    display: flex; justify-content: flex-end;
    margin: 1rem 0 0.3rem 0;
}
.msg-user .bubble {
    background: #1f6feb;
    color: #ffffff;
    padding: 0.7rem 1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.5;
}
.msg-assistant { margin: 0.3rem 0 1rem 0; }
.msg-assistant .avatar {
    width: 28px; height: 28px;
    background: linear-gradient(135deg, #00d4aa, #1f6feb);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.75rem; color: #0d1117; font-weight: 700;
    margin-right: 0.5rem; vertical-align: top; margin-top: 2px;
    flex-shrink: 0;
}
.msg-header { display: flex; align-items: flex-start; margin-bottom: 0.6rem; }
.msg-header span { font-size: 0.78rem; color: #8b949e; padding-top: 6px; }

/* ── Insights card ── */
.insights-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #00d4aa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin-bottom: 1rem;
}

/* ── SQL block ── */
.sql-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #79c0ff;
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 1rem;
}

/* ── Metric pills ── */
.metric-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-pill {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    color: #8b949e;
}
.metric-pill b { color: #00d4aa; }

/* ── Suggestion chips ── */
.suggestions-row {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    margin-top: 1rem;
}
.suggestion-chip {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    font-size: 0.8rem;
    color: #8b949e;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    max-width: 280px;
}
.suggestion-chip:hover { border-color: #58a6ff; color: #58a6ff; }
.suggestion-chip .sq { font-size: 0.72rem; color: #58a6ff; display: block; margin-bottom: 2px; }

/* ── Input bar ── */
.input-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, #0d1117 30%);
    padding: 1rem;
    z-index: 100;
}
.input-inner {
    max-width: 860px; margin: 0 auto;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    display: flex; align-items: flex-end; gap: 0.5rem;
    padding: 0.5rem 0.6rem;
}

/* ── Streamlit input override ── */
.stTextInput input {
    background: transparent !important;
    border: none !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    box-shadow: none !important;
    padding: 0.4rem 0.2rem !important;
}
.stTextInput input:focus { box-shadow: none !important; border: none !important; }

/* ── Send button ── */
.send-btn > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.2rem !important;
    width: auto !important;
    transition: background 0.15s !important;
}
.send-btn > button:hover { background: #388bfd !important; transform: none !important; }

/* ── Section label ── */
.sec-label {
    font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: #8b949e; margin: 0.8rem 0 0.4rem 0; font-weight: 600;
}

/* ── Divider ── */
.msg-divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }

/* ── Spinner ── */
.thinking {
    color: #8b949e; font-size: 0.85rem; font-style: italic;
    padding: 0.5rem 0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8b949e", size=11),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d"),
    margin=dict(l=10, r=10, t=36, b=10),
)

def auto_chart(df: pd.DataFrame):
    if df is None or df.empty or len(df.columns) < 2:
        return []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    charts   = []

    if cat_cols and num_cols:
        cat, num = cat_cols[0], num_cols[0]
        top = df.nlargest(20, num) if len(df) > 20 else df
        fig = px.bar(top, x=cat, y=num,
                     color=num, color_continuous_scale=["#1f6feb", "#00d4aa"],
                     title=f"{num} by {cat}")
        fig.update_layout(**PLOTLY_THEME)
        fig.update_coloraxes(showscale=False)
        charts.append(fig)

        if df[cat].nunique() <= 12:
            pie = px.pie(top, names=cat, values=num,
                         color_discrete_sequence=["#1f6feb","#00d4aa","#f78166","#d2a8ff","#ffa657"],
                         title=f"Share of {num}", hole=0.45)
            pie.update_layout(**PLOTLY_THEME)
            charts.append(pie)

    elif len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                         color_discrete_sequence=["#00d4aa"],
                         title=f"{num_cols[1]} vs {num_cols[0]}")
        fig.update_layout(**PLOTLY_THEME)
        charts.append(fig)

    elif len(num_cols) == 1:
        fig = px.histogram(df, x=num_cols[0],
                           color_discrete_sequence=["#1f6feb"],
                           title=f"Distribution of {num_cols[0]}")
        fig.update_layout(**PLOTLY_THEME)
        charts.append(fig)

    return charts

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "storage" not in st.session_state:
    st.session_state.storage = load_storage()

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

storage = st.session_state.storage

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.6rem 0 1rem 0;">'
        '<span style="font-size:1.1rem;font-weight:700;color:#c9d1d9;">🧬 MedOnc</span>'
        '</div>',
        unsafe_allow_html=True
    )

    if st.button("＋  New Chat", key="new_chat_btn"):
        cid = new_chat(storage)
        st.session_state.active_chat = cid
        st.session_state.pending_query = ""
        st.rerun()

    st.markdown(
        '<p style="font-size:0.7rem;color:#8b949e;letter-spacing:0.08em;'
        'text-transform:uppercase;margin:1rem 0 0.4rem 0;">Recent</p>',
        unsafe_allow_html=True
    )

    # List chats
    for cid in storage["order"]:
        chat = storage["chats"].get(cid)
        if not chat:
            continue
        is_active = cid == st.session_state.active_chat
        col1, col2 = st.columns([5, 1])
        with col1:
            label = ("▶ " if is_active else "") + chat["title"]
            if st.button(label, key=f"chat_{cid}"):
                st.session_state.active_chat = cid
                st.session_state.pending_query = ""
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{cid}"):
                delete_chat(storage, cid)
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = None
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
active_cid = st.session_state.active_chat

# ── No chat selected → welcome screen ────────────────────────────────────────
if not active_cid or active_cid not in storage["chats"]:
    st.markdown("""
    <div class="welcome">
        <h1>MedOnc Analytics</h1>
        <p>Ask questions about UHC medical oncology authorizations, treatments, and clinical data</p>
        <div class="welcome-chips">
            <div class="chip">Total authorizations by cancer type</div>
            <div class="chip">Approval rate by treatment regimen</div>
            <div class="chip">Distinct ICD codes used</div>
            <div class="chip">Patients on 2nd line therapy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    chat     = storage["chats"][active_cid]
    messages = chat["messages"]

    # ── Render chat history ───────────────────────────────────────────────────
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for idx, msg in enumerate(messages):

        # User bubble
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{msg["query"]}</div></div>',
            unsafe_allow_html=True
        )

        # Assistant response
        st.markdown(
            '<div class="msg-assistant">'
            '<div class="msg-header">'
            '<div class="avatar">M</div>'
            '<span>MedOnc Analytics</span>'
            '</div></div>',
            unsafe_allow_html=True
        )

        # Metrics
        df = pd.DataFrame(msg["df_records"], columns=msg["df_columns"]) if msg.get("df_records") else None

        if df is not None and not df.empty:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            pills = f'<div class="metric-row">'
            pills += f'<div class="metric-pill"><b>{len(df)}</b> rows</div>'
            pills += f'<div class="metric-pill"><b>{len(df.columns)}</b> columns</div>'
            for col in num_cols[:3]:
                val = df[col].sum()
                fmt = f"{val:,.0f}" if val > 100 else f"{val:,.2f}"
                pills += f'<div class="metric-pill"><b>{fmt}</b> {col[:12]}</div>'
            pills += '</div>'
            st.markdown(pills, unsafe_allow_html=True)

        # Insights
        st.markdown('<p class="sec-label">Insights</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="insights-card">{msg["insights"]}</div>', unsafe_allow_html=True)

        # Charts (only for last message to avoid overload)
        if df is not None and idx == len(messages) - 1:
            charts = auto_chart(df)
            if charts:
                st.markdown('<p class="sec-label">Visualizations</p>', unsafe_allow_html=True)
                if len(charts) == 2:
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(charts[0], use_container_width=True)
                    c2.plotly_chart(charts[1], use_container_width=True)
                else:
                    st.plotly_chart(charts[0], use_container_width=True)

        # SQL (collapsed by default)
        with st.expander("View SQL", expanded=False):
            st.markdown(f'<div class="sql-block">{msg["sql"]}</div>', unsafe_allow_html=True)

        # Full results table
        if df is not None and not df.empty:
            st.markdown('<p class="sec-label">Full Results</p>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=min(320, 50 + len(df) * 35))

        # Suggestions (only on last message)
        if idx == len(messages) - 1 and msg.get("suggestions"):
            st.markdown('<p class="sec-label">Suggested follow-ups</p>', unsafe_allow_html=True)
            sug_cols = st.columns(min(3, len(msg["suggestions"])))
            for si, sug in enumerate(msg["suggestions"][:3]):
                with sug_cols[si]:
                    if st.button(
                        f"💬 {sug['question']}",
                        key=f"sug_{active_cid}_{idx}_{si}",
                        help=sug.get("reason", ""),
                    ):
                        st.session_state.pending_query = sug["question"]
                        st.rerun()

        if idx < len(messages) - 1:
            st.markdown('<hr class="msg-divider">', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT BAR (fixed bottom) — only show when a chat is active
# ─────────────────────────────────────────────────────────────────────────────
if active_cid and active_cid in storage["chats"]:
    st.markdown('<div style="height:80px"></div>', unsafe_allow_html=True)  # spacer

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        default_val = st.session_state.get("pending_query", "")
        user_input  = st.text_input(
            label="",
            value=default_val,
            placeholder="Ask anything about your medical oncology data…",
            key=f"input_{active_cid}",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Send →", key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    # Trigger on send or if pending_query was set by suggestion click
    trigger_query = ""
    if send and user_input.strip():
        trigger_query = user_input.strip()
    elif st.session_state.pending_query:
        trigger_query = st.session_state.pending_query
        st.session_state.pending_query = ""

    if trigger_query:
        # Build compact history context (query + sql only, last 5 turns)
        messages    = storage["chats"][active_cid]["messages"]
        chat_history = [
            {"query": m["query"], "sql": m["sql"]}
            for m in messages[-5:]
        ]

        with st.spinner("Thinking…"):
            try:
                result = run_pipeline(trigger_query, chat_history)

                # Store df as records (JSON-serializable)
                df = result["df"]
                msg_record = {
                    "query":       trigger_query,
                    "sql":         result["sql"],
                    "insights":    result["insights"],
                    "suggestions": result["suggestions"],
                    "df_columns":  df.columns.tolist(),
                    "df_records":  df.values.tolist(),
                }
                append_message(storage, active_cid, msg_record)
                st.session_state.storage = storage
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")
