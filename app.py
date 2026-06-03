import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from main_query import query_to_sql
from run_query import execute_sql, explain_results, clean_sql

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedOnc Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e17;
    color: #e2e8f0;
    font-family: 'Syne', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hero Header ── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    text-align: center;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00d4aa22, #3b82f622);
    border: 1px solid #00d4aa44;
    color: #00d4aa;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #00d4aa 70%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Input area ── */
.stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 3px #00d4aa18 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #3b82f6) !important;
    color: #0a0e17 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Step cards (pipeline) ── */
.step-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.9rem;
    transition: border-color 0.3s;
}
.step-card.active  { border-color: #f59e0b; }
.step-card.done    { border-color: #00d4aa; }
.step-card.pending { opacity: 0.4; }
.step-icon { font-size: 1.2rem; width: 1.5rem; text-align: center; }
.step-label { font-weight: 600; color: #94a3b8; }
.step-card.done .step-label { color: #00d4aa; }
.step-card.active .step-label { color: #f59e0b; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem 0;
}
.section-header h3 {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e293b, transparent);
}

/* ── SQL block ── */
.sql-block {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #7dd3fc;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Insights card ── */
.insights-card {
    background: linear-gradient(135deg, #0d1f1a, #0d1525);
    border: 1px solid #00d4aa33;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.97rem;
    line-height: 1.75;
    color: #cbd5e1;
    position: relative;
}
.insights-card::before {
    content: '"';
    position: absolute;
    top: -0.4rem;
    left: 1rem;
    font-size: 4rem;
    color: #00d4aa22;
    font-family: Georgia, serif;
    line-height: 1;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 120px;
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00d4aa;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper: auto chart ────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def auto_chart(df: pd.DataFrame):
    """Pick the most appropriate Plotly chart based on column types."""
    if df is None or df.empty or len(df.columns) < 2:
        return None

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(exclude="number").columns.tolist()

    # ── 1 category + 1 numeric → bar ──────────────────────────────────────────
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        cat, num = cat_cols[0], num_cols[0]
        top = df.nlargest(20, num) if len(df) > 20 else df
        fig = px.bar(
            top, x=cat, y=num,
            color=num,
            color_continuous_scale=["#3b82f6", "#00d4aa"],
            title=f"{num} by {cat}",
        )
        fig.update_layout(**PLOTLY_THEME)
        fig.update_coloraxes(showscale=False)

        # if few categories add pie side by side
        if df[cat].nunique() <= 12:
            pie = px.pie(
                top, names=cat, values=num,
                color_discrete_sequence=px.colors.sequential.Teal,
                title=f"{num} distribution",
                hole=0.45,
            )
            pie.update_layout(**PLOTLY_THEME)
            return [fig, pie]
        return [fig]

    # ── 2+ numerics → line / scatter ──────────────────────────────────────────
    if len(num_cols) >= 2:
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            color_discrete_sequence=["#00d4aa"],
            title=f"{num_cols[1]} vs {num_cols[0]}",
        )
        fig.update_layout(**PLOTLY_THEME)
        return [fig]

    # ── single numeric → histogram ────────────────────────────────────────────
    if len(num_cols) == 1:
        fig = px.histogram(
            df, x=num_cols[0],
            color_discrete_sequence=["#3b82f6"],
            title=f"Distribution of {num_cols[0]}",
        )
        fig.update_layout(**PLOTLY_THEME)
        return [fig]

    return None


# ── Step tracker state ────────────────────────────────────────────────────────
STEPS = [
    ("🔍", "Semantic search on YAML"),
    ("🤖", "Generating SQL"),
    ("❄️",  "Executing on Snowflake"),
    ("💡", "Generating insights"),
]

def render_steps(current: int):
    for i, (icon, label) in enumerate(STEPS):
        if i < current:
            cls = "done";    icon_disp = "✅"
        elif i == current:
            cls = "active";  icon_disp = "⏳"
        else:
            cls = "pending"; icon_disp = icon
        st.markdown(
            f'<div class="step-card {cls}">'
            f'  <span class="step-icon">{icon_disp}</span>'
            f'  <span class="step-label">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def section(title: str):
    st.markdown(
        f'<div class="section-header">'
        f'  <h3>{title}</h3>'
        f'  <div class="section-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">UHC · Medical Oncology · NL → SQL</div>
    <h1>MedOnc Analytics</h1>
    <p>Ask anything about authorizations, treatments & clinical reviews in plain English</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("#### Ask a question")
    user_query = st.text_area(
        label="",
        placeholder="e.g. Show total authorizations by primary cancer type for the last 3 months",
        height=130,
        label_visibility="collapsed",
    )
    run_btn = st.button("Run Analysis →")

    st.markdown("<br>", unsafe_allow_html=True)
    steps_placeholder = st.empty()

with right:
    results_area = st.empty()

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and user_query.strip():

    t_start = time.time()

    with left:
        with steps_placeholder.container():
            render_steps(0)   # step 0 active

    # ── Step 0 + 1: SQL generation (retrieval + LLM) ──────────────────────────
    with left:
        with steps_placeholder.container():
            render_steps(1)

    sql_raw   = query_to_sql(user_query.strip(), verbose=False)
    sql_clean = clean_sql(sql_raw)

    with left:
        with steps_placeholder.container():
            render_steps(2)   # Snowflake

    # ── Step 2: Execute on Snowflake ──────────────────────────────────────────
    try:
        df = execute_sql(sql_clean)
    except Exception as e:
        with right:
            st.error(f"❌ Snowflake error: {e}")
        with left:
            with steps_placeholder.container():
                render_steps(-1)
        st.stop()

    with left:
        with steps_placeholder.container():
            render_steps(3)   # insights

    # ── Step 3: Explain results ───────────────────────────────────────────────
    insights = explain_results(user_query.strip(), sql_clean, df)

    elapsed = round(time.time() - t_start, 1)

    with left:
        with steps_placeholder.container():
            render_steps(4)   # all done
        st.markdown(
            f'<p style="color:#00d4aa;font-size:0.8rem;margin-top:0.5rem">'
            f'✓ Completed in {elapsed}s</p>',
            unsafe_allow_html=True
        )

    # ── Right panel output ────────────────────────────────────────────────────
    with results_area.container():

        # ── Metrics row ───────────────────────────────────────────────────────
        num_cols = df.select_dtypes(include="number").columns.tolist()
        metric_html = '<div class="metric-row">'
        metric_html += f'<div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Rows</div></div>'
        metric_html += f'<div class="metric-card"><div class="val">{len(df.columns)}</div><div class="lbl">Columns</div></div>'
        for col in num_cols[:3]:
            val = df[col].sum()
            fmt = f"{val:,.0f}" if val > 100 else f"{val:,.2f}"
            metric_html += f'<div class="metric-card"><div class="val">{fmt}</div><div class="lbl">{col[:14]}</div></div>'
        metric_html += '</div>'
        st.markdown(metric_html, unsafe_allow_html=True)

        # ── Insights ──────────────────────────────────────────────────────────
        section("AI Insights")
        st.markdown(
            f'<div class="insights-card">{insights}</div>',
            unsafe_allow_html=True,
        )

        # ── Charts ────────────────────────────────────────────────────────────
        charts = auto_chart(df)
        if charts:
            section("Visualizations")
            if len(charts) == 2:
                c1, c2 = st.columns(2)
                c1.plotly_chart(charts[0], use_container_width=True)
                c2.plotly_chart(charts[1], use_container_width=True)
            else:
                st.plotly_chart(charts[0], use_container_width=True)

        # ── SQL ───────────────────────────────────────────────────────────────
        section("Generated SQL")
        st.markdown(f'<div class="sql-block">{sql_clean}</div>', unsafe_allow_html=True)

        # ── Full results table ────────────────────────────────────────────────
        section(f"Full Results  ({len(df)} rows)")
        st.dataframe(df, use_container_width=True, height=min(400, 60 + len(df) * 35))

elif run_btn:
    st.warning("Please enter a question first.")
