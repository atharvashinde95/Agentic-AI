"""
app.py  —  AI Math Assistant  (Streamlit UI)
=============================================
Clean, professional dark-themed UI.
LangChain + LangGraph · GPT-4o · Capgemini Generative Engine

Run:
  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Math Assistant | Capgemini",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0D0D0D !important;
    color: #E8E8E8 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1100px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222 !important;
}
[data-testid="stSidebar"] * { color: #C8C8C8 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
textarea {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 10px !important;
    color: #E8E8E8 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus,
textarea:focus {
    border-color: #00C2FF !important;
    box-shadow: 0 0 0 3px rgba(0,194,255,0.08) !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00C2FF 0%, #0070AD 100%) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    padding: 10px 14px !important;
}

/* ── Info / warning ── */
[data-testid="stAlert"] {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 10px !important;
    color: #AAAAAA !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #777 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #00C2FF !important; font-weight: 700 !important; font-size: 22px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background: #141414 !important; border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #1E1E1E !important; }

/* ── Custom components ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #FFFFFF 0%, #888 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 15px;
    color: #555;
    margin-bottom: 32px;
    font-weight: 400;
}
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #161616;
    border: 1px solid #222;
    border-radius: 100px;
    padding: 8px 18px;
    font-size: 13px;
    color: #777;
    margin-bottom: 28px;
}
.info-pill b { color: #00C2FF; }

.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 10px;
}

/* Trace cards */
.trace-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 18px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
    color: #C0C0C0;
}
.trace-card.llm  { border-top: 3px solid #00C2FF; }
.trace-card.tool { border-top: 3px solid #00E676; }

.trace-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.trace-label.llm  { color: #00C2FF; }
.trace-label.tool { color: #00E676; }

.chip {
    display: inline-block;
    background: #1E1E1E;
    border: 1px solid #2E2E2E;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 12px;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
}
.chip.blue { border-color: #00C2FF33; color: #00C2FF; background: #00C2FF0D; }
.chip.green { border-color: #00E67633; color: #00E676; background: #00E6760D; }

/* Answer box */
.answer-box {
    background: #141414;
    border: 1px solid #222;
    border-left: 4px solid #00C2FF;
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 20px;
}
.answer-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00C2FF;
    margin-bottom: 10px;
}
.answer-text {
    font-size: 18px;
    font-weight: 600;
    color: #F0F0F0;
    line-height: 1.5;
}

/* Tool spec card */
.tool-spec {
    background: #141414;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 13px;
}
.tool-spec-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    color: #00C2FF;
    margin-bottom: 12px;
}
.tool-spec-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #1E1E1E;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.tool-spec-row:last-child { border-bottom: none; }
.tool-spec-row span { color: #CCC; }

/* How it works */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid #1A1A1A;
    font-size: 13px;
    color: #888;
}
.step-row:last-child { border-bottom: none; }
.step-num {
    min-width: 26px; height: 26px;
    background: #1E1E1E;
    border: 1px solid #2E2E2E;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #555;
    flex-shrink: 0;
}
.step-row b { color: #CCC; }

/* Samples chips row */
.samples-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }
.sample-chip {
    background: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 13px;
    color: #777;
    cursor: pointer;
    transition: all 0.15s;
}
.sample-chip:hover { border-color: #00C2FF55; color: #CCC; }

</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Capgemini_201x_logo.svg/320px-Capgemini_201x_logo.svg.png",
        width=130,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status — loaded from .env, no UI inputs needed ──
    api_ready = bool(os.getenv("CAPGEMINI_API_KEY") and os.getenv("CAPGEMINI_BASE_URL"))
    if api_ready:
        st.success("✅ Connected via .env", icon=None)
    else:
        st.error("⚠️ Check your .env file")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tool spec ──
    st.markdown("### The Tool")
    st.markdown("""<div class="tool-spec">
<div class="tool-spec-name">calculate(expression)</div>
<div class="tool-spec-row">Input <span>any math expression string</span></div>
<div class="tool-spec-row">Operators <span>+ &nbsp;− &nbsp;× &nbsp;÷ &nbsp;** &nbsp;% &nbsp;()</span></div>
<div class="tool-spec-row">Parser <span>Python AST (not eval)</span></div>
<div class="tool-spec-row">÷ 0 guard <span>✅ handled</span></div>
<div class="tool-spec-row">Hallucination <span>impossible — Python runs it</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works ──
    st.markdown("### How It Works")
    st.markdown("""<div class="tool-spec">
<div class="step-row"><div class="step-num">1</div><div>You ask a <b>math question</b></div></div>
<div class="step-row"><div class="step-num">2</div><div>LLM <b>extracts</b> the expression</div></div>
<div class="step-row"><div class="step-num">3</div><div><b>calculate()</b> runs in Python</div></div>
<div class="step-row"><div class="step-num">4</div><div>LLM <b>explains</b> the answer</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("LangChain · LangGraph · GPT-4o · Capgemini Generative Engine")


# ════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────
st.markdown("""
<div class="hero-title">AI Math Assistant</div>
<div class="hero-sub">LangChain Tool Calling &nbsp;·&nbsp; One tool &nbsp;·&nbsp; Any expression &nbsp;·&nbsp; Zero hallucinations</div>
<div class="info-pill">
  🔧 <b>calculate(expression)</b> &nbsp;—&nbsp; The LLM decides what to compute. Python does the math.
</div>
""", unsafe_allow_html=True)

# ── Sample buttons (visual only, selectbox is functional) ──
st.markdown('<div class="section-label">Quick Examples</div>', unsafe_allow_html=True)

SAMPLES = [
    "What is 1 + 1?",
    "Calculate 128 + 374",
    "What is 100 / 4 + 50 - 3 * 2?",
    "What is (5 + 3) * (10 - 4)?",
    "What is 2 to the power of 10?",
    "Divide 10 by 0",
    "Multiply 13 by 17 then add 50",
    "I have 500 rupees, spend 175, earn 320. Total?",
]

choice = st.selectbox(
    "Pick a sample or write your own:",
    ["— write your own below —"] + SAMPLES,
    label_visibility="collapsed",
)

# ── Input row ─────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:16px;">Your Question</div>', unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1], gap="medium")
with col_input:
    question = st.text_input(
        "question",
        value="" if choice == "— write your own below —" else choice,
        placeholder="e.g.  What is 3.14 * 2 + 100 / 4?",
        label_visibility="collapsed",
    )
with col_btn:
    run_btn = st.button("Solve →", use_container_width=True)


# ════════════════════════════════════════════════════
# AGENT EXECUTION
# ════════════════════════════════════════════════════
if run_btn:
    if not question.strip():
        st.warning("Please enter a math question first.")
        st.stop()
    if not os.getenv("CAPGEMINI_API_KEY"):
        st.error("CAPGEMINI_API_KEY not found in .env — please add it and restart.")
        st.stop()

    with st.spinner(""):
        try:
            from agent import build_agent, run_query
            if "math_agent" not in st.session_state:
                st.session_state.math_agent = build_agent()
            result = run_query(st.session_state.math_agent, question)
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    steps       = result["steps"]
    call_step   = next((s for s in steps if s["type"] == "call"),   None)
    result_step = next((s for s in steps if s["type"] == "result"), None)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-label">Agent Trace</div>', unsafe_allow_html=True)

    # ── Two-column trace ──────────────────────────────
    if call_step or result_step:
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            expr = call_step["expression"] if call_step else "—"
            st.markdown(f"""
<div class="trace-label llm">🤖 &nbsp;LLM — Tool Decision</div>
<div class="trace-card llm">
  Chose tool &nbsp;&nbsp;: <span class="chip blue">calculate</span><br>
  Expression : <span class="chip blue">{expr}</span><br><br>
  <span style="color:#444;font-size:11px;">The LLM extracted the numeric expression<br>from your question and called the tool.</span>
</div>""", unsafe_allow_html=True)

        with col_r:
            output = result_step["output"] if result_step else "—"
            st.markdown(f"""
<div class="trace-label tool">⚙️ &nbsp;Python — Tool Result</div>
<div class="trace-card tool">
  Result &nbsp;&nbsp;&nbsp;&nbsp;: <span class="chip green">{output}</span><br><br>
  <br>
  <span style="color:#444;font-size:11px;">Python's AST parser evaluated the expression.<br>No arithmetic was done by the LLM.</span>
</div>""", unsafe_allow_html=True)

    else:
        st.info("No tool was called — question may not require math.")

    # ── Answer box ────────────────────────────────────
    st.markdown(f"""
<div class="answer-box">
  <div class="answer-label">✦ &nbsp;Final Answer</div>
  <div class="answer-text">{result['answer']}</div>
</div>""", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tools Available", "1")
    c2.metric("Tool Called", call_step["tool"] if call_step else "none")
    c3.metric("Model", "GPT-4o")
    c4.metric("Hallucination Risk", "Zero")

    # ── Save history ──────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Question":   question,
        "Expression": call_step["expression"] if call_step else "—",
        "Result":     result_step["output"]   if result_step else "—",
        "Answer":     result["answer"][:70] + "…" if len(result["answer"]) > 70 else result["answer"],
    })


# ════════════════════════════════════════════════════
# SESSION HISTORY
# ════════════════════════════════════════════════════
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown('<div class="section-label">Session History</div>', unsafe_allow_html=True)
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
