"""
app.py  —  AI Math Assistant  (Streamlit UI)
=============================================
Single-tool LangChain + LangGraph math agent.
Powered by GPT-4o via Capgemini Generative Engine.

Run:
  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title = "AI Math Assistant | Capgemini",
    page_icon  = "🧮",
    layout     = "wide",
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  .step-call {
    background: #EBF5FB; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #0070AD; margin-bottom: 10px;
    font-family: 'DM Mono', monospace; font-size: 14px;
  }
  .step-result {
    background: #EAFAF1; border-radius: 10px; padding: 14px 18px;
    border-left: 4px solid #2DC653; margin-bottom: 10px;
    font-family: 'DM Mono', monospace; font-size: 14px;
  }
  .answer-box {
    background: linear-gradient(135deg, #0070AD, #004E7C);
    color: white; border-radius: 12px; padding: 22px 26px;
    font-size: 20px; font-weight: 700; margin-top: 16px;
    font-family: 'Syne', sans-serif;
  }
  .tool-info {
    background: #F8F9FA; border-radius: 10px; padding: 16px;
    border: 1px solid #E0E0E0; font-size: 13px;
  }
  .tag {
    font-size: 10px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .sidebar-box {
    background: #EBF5FB; border-radius: 8px;
    padding: 14px; font-size: 13px; color: #2c5f7a;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Capgemini_201x_logo.svg/320px-Capgemini_201x_logo.svg.png", width=150)
    st.markdown("## ⚙️ Configuration")

    api_key  = st.text_input("Capgemini API Key", type="password", placeholder="Paste your key")
    base_url = st.text_input("Generative Engine URL", placeholder="https://your-engine/v1")

    if api_key:  os.environ["CAPGEMINI_API_KEY"]  = api_key
    if base_url: os.environ["CAPGEMINI_BASE_URL"] = base_url

    st.divider()
    st.markdown("### 🛠️ The Tool")
    st.markdown("""<div class="tool-info">
<b style="font-size:15px">🔢 calculate(expression)</b><br><br>
Evaluates <b>any</b> math expression:<br><br>
<code>1 + 1</code><br>
<code>128 + 374</code><br>
<code>3.14 * 2</code><br>
<code>100 / 4 + 50 - 3 * 2</code><br>
<code>(5 + 3) * (10 - 4)</code><br>
<code>2 ** 10</code><br><br>
Supports: <b>+ − × ÷ ** % ()</b><br><br>
✅ Safe — uses Python AST parser<br>
✅ No eval(), no code injection<br>
✅ Division by zero protected
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔄 How It Works")
    st.markdown("""<div class="sidebar-box">
1️⃣ You ask a math question<br>
2️⃣ LLM extracts the expression<br>
3️⃣ <b>calculate()</b> runs in Python<br>
4️⃣ LLM explains the answer<br><br>
<b>LLM never does arithmetic itself</b> — zero hallucination risk.
</div>""", unsafe_allow_html=True)

    st.caption("LangChain + LangGraph · GPT-4o · Capgemini")

# ── Header ────────────────────────────────────────────────
st.markdown("<h1 style='font-size:2rem;margin-bottom:4px;'>🧮 AI Math Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#666;font-size:15px;margin-bottom:8px;'>One tool · Any expression · LangChain + LangGraph · No hallucinations</p>", unsafe_allow_html=True)

st.info("💡 **One tool handles everything.** The LLM reads your question, extracts the math expression, and passes it to `calculate()`. Python does the arithmetic — not the LLM.")

# ── Samples ───────────────────────────────────────────────
SAMPLES = [
    "What is 1 + 1?",
    "Calculate 128 + 374",
    "What is 100 / 4 + 50 - 3 * 2?",
    "Multiply 13 by 17 then add 50",
    "What is (5 + 3) * (10 - 4)?",
    "Divide 10 by 0",
    "What is 2 to the power of 10?",
    "I have 500 rupees, spend 175, then earn 320. How much do I have?",
]

st.markdown("#### 💬 Ask Any Math Question")
col_a, col_b = st.columns([3, 1])
with col_a:
    choice   = st.selectbox("Sample questions:", ["— write your own —"] + SAMPLES)
    question = st.text_input(
        "Your question:",
        value       = "" if choice == "— write your own —" else choice,
        placeholder = "e.g. What is 3.14 * 2 + 100 / 4?",
    )
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Solve", use_container_width=True)

# ── Run ───────────────────────────────────────────────────
if run_btn:
    if not question.strip():
        st.warning("Please enter a math question.")
        st.stop()
    if not api_key:
        st.warning("⚠️ Enter your Capgemini API Key in the sidebar.")
        st.stop()

    with st.spinner("🤖 Agent thinking..."):
        try:
            from agent import build_agent, run_query

            if "math_agent" not in st.session_state:
                st.session_state.math_agent = build_agent()

            result = run_query(st.session_state.math_agent, question)

        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    st.divider()
    st.markdown("### 🔍 Agent Trace")

    # ── Show steps ────────────────────────────────────────
    steps = result["steps"]
    call_step   = next((s for s in steps if s["type"] == "call"),   None)
    result_step = next((s for s in steps if s["type"] == "result"), None)

    if call_step or result_step:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="tag" style="color:#0070AD;">🤖 LLM Decision — Tool Call</p>', unsafe_allow_html=True)
            if call_step:
                st.markdown(f"""<div class="step-call">
Tool &nbsp;&nbsp;&nbsp;&nbsp;: <b>calculate</b><br>
Expression: <b>{call_step['expression']}</b>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="step-call">No tool call recorded.</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="tag" style="color:#2DC653;">⚙️ Python Result</p>', unsafe_allow_html=True)
            if result_step:
                st.markdown(f"""<div class="step-result">
<b>{result_step['output']}</b>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="step-result">No result recorded.</div>', unsafe_allow_html=True)
    else:
        st.info("No tool was called — the question may not be mathematical.")

    # ── Final answer ──────────────────────────────────────
    st.markdown(f"""<div class="answer-box">
  <div style="font-size:11px;opacity:.7;letter-spacing:.1em;margin-bottom:6px;">FINAL ANSWER</div>
  {result['answer']}
</div>""", unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("🛠️ Tools Available", "1  (calculate)")
    c2.metric("🧠 Model",           "GPT-4o")
    c3.metric("✅ Accuracy",        "Python — not LLM")

    # ── Save to history ───────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    expr = call_step["expression"] if call_step else "—"
    res  = result_step["output"]   if result_step else "—"
    st.session_state.history.append({
        "Question":   question,
        "Expression": expr,
        "Result":     res,
        "Answer":     result["answer"][:60] + "..." if len(result["answer"]) > 60 else result["answer"],
    })

# ── History ───────────────────────────────────────────────
if st.session_state.get("history"):
    st.divider()
    st.markdown("### 📋 Session History")
    st.dataframe(st.session_state.history, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
