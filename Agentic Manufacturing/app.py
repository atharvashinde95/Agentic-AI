"""
app.py
-------
Streamlit UI for the Predictive Maintenance Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st
from agent import run_agent


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Agent",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .react-box {
    padding: 14px 18px;
    margin-bottom: 10px;
    border-radius: 8px;
    border-left: 5px solid #ccc;
    font-size: 14px;
    line-height: 1.6;
  }
  .box-reasoning    { border-left-color: #6C63FF; background: #f3f2ff; }
  .box-tool_call    { border-left-color: #F5A623; background: #fffbf0; }
  .box-observation  { border-left-color: #4CAF50; background: #f0fff4; }
  .box-final        { border-left-color: #00BCD4; background: #e8fdff; }
  .box-emergency    { border-left-color: #E53935; background: #fff5f5; }
  .step-label {
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🤖 Predictive Maintenance — Agentic AI (LangChain + LangGraph)")
st.markdown(
    "**ReAct Loop powered by LangGraph** — the agent autonomously "
    "Reasons → calls Tools → reads Observations → Reasons again, "
    "until all maintenance actions are complete."
)
st.divider()


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Quick presets + thresholds
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ Quick Test Cases")

    if st.button("🟢 Case 1 — Healthy (Low Risk)", use_container_width=True):
        st.session_state.temp    = 70
        st.session_state.vib     = 3.0
        st.session_state.load    = 50
        st.session_state.runtime = 40

    if st.button("🟡 Case 2 — Warning (Medium Risk)", use_container_width=True):
        st.session_state.temp    = 85
        st.session_state.vib     = 6.0
        st.session_state.load    = 75
        st.session_state.runtime = 90

    if st.button("🔴 Case 3 — Critical (High Risk)", use_container_width=True):
        st.session_state.temp    = 95
        st.session_state.vib     = 8.5
        st.session_state.load    = 90
        st.session_state.runtime = 120

    if st.button("🚨 Case 4 — Emergency Shutdown", use_container_width=True):
        st.session_state.temp    = 110
        st.session_state.vib     = 9.5
        st.session_state.load    = 95
        st.session_state.runtime = 180

    st.divider()
    st.subheader("📊 Anomaly Thresholds")
    st.markdown("""
| Sensor | Safe | Anomaly |
|---|---|---|
| Temperature | ≤ 85°C | > 85°C |
| Vibration | ≤ 7.0 | > 7.0 |
| Load | ≤ 85% | > 85% |
| Runtime | ≤ 100h | > 100h |
    """)

    st.divider()
    st.subheader("🔧 LangChain Stack")
    st.markdown("""
- `langchain` 0.3.x  
- `langchain-openai` 0.3.x  
- `langgraph` 0.4.x  
- `@tool` decorator on all tools  
- `create_react_agent` loop  
    """)


# ─────────────────────────────────────────────────────────────
# SENSOR INPUT SLIDERS
# ─────────────────────────────────────────────────────────────
st.subheader("📡 Machine Sensor Input")

# Machine ID input
machine_id = st.text_input(
    "🏭 Machine ID",
    value=st.session_state.get("machine_id", "M-101"),
    help="Unique identifier for the machine being analyzed"
)

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "🌡️ Temperature (°C)", 50, 120,
        value=int(st.session_state.get("temp", 70)), step=1,
        help="Anomaly if > 85°C"
    )
    vibration = st.slider(
        "📳 Vibration Level (0–10)", 0.0, 10.0,
        value=float(st.session_state.get("vib", 3.0)), step=0.5,
        help="Anomaly if > 7.0"
    )

with col2:
    load = st.slider(
        "⚙️ Machine Load (%)", 0, 100,
        value=int(st.session_state.get("load", 50)), step=1,
        help="Anomaly if > 85%"
    )
    runtime_hours = st.slider(
        "⏱️ Runtime Hours", 0, 200,
        value=int(st.session_state.get("runtime", 40)), step=5,
        help="Anomaly if > 100 hrs"
    )

# Live status metrics
st.markdown("**Live Sensor Status:**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Temperature", f"{temperature}°C",
          "⚠️ ANOMALY" if temperature > 85 else "✅ Normal",
          delta_color="inverse" if temperature > 85 else "normal")
m2.metric("Vibration", str(vibration),
          "⚠️ ANOMALY" if vibration > 7 else "✅ Normal",
          delta_color="inverse" if vibration > 7 else "normal")
m3.metric("Load", f"{load}%",
          "⚠️ ANOMALY" if load > 85 else "✅ Normal",
          delta_color="inverse" if load > 85 else "normal")
m4.metric("Runtime", f"{runtime_hours}h",
          "⚠️ ANOMALY" if runtime_hours > 100 else "✅ Normal",
          delta_color="inverse" if runtime_hours > 100 else "normal")

st.divider()


# ─────────────────────────────────────────────────────────────
# RUN AGENT BUTTON
# ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Maintenance Agent", type="primary", use_container_width=True):

    with st.spinner("🤖 LangGraph ReAct agent running..."):
        result = run_agent(machine_id, temperature, vibration, load, runtime_hours)

    # ── ERROR ─────────────────────────────────────────────────
    if result["error"]:
        st.error(f"**Agent Error:** {result['error']}")
        st.stop()

    steps         = result["steps"]
    final_summary = result["final_summary"]
    final_json    = result["final_json"]

    tools_executed = [s for s in steps if s["type"] == "tool_call"]
    st.success(f"✅ Agent completed — {len(steps)} messages, {len(tools_executed)} tool(s) called")

    # ── REACT LOOP TRACE ──────────────────────────────────────
    st.subheader("🔁 LangGraph ReAct Loop — Full Trace")
    st.caption(
        "Every message in the agent's message graph: "
        "AIMessage (tool calls) → ToolMessage (observations) → final AIMessage (JSON)"
    )

    for step in steps:
        stype   = step["type"]
        content = step["content"]
        tool_nm = step["tool_name"]
        snum    = step["step_number"]

        is_emergency = tool_nm == "immediate_shutdown"
        box_class = "box-emergency" if is_emergency else f"box-{stype}"

        label_map = {
            "tool_call":   f"⚡ Tool Call → {tool_nm}",
            "observation": f"👁️ Observation ← {tool_nm}",
            "final":       "✅ Final JSON Assessment",
        }
        label = label_map.get(stype, stype.upper())
        if is_emergency:
            label = f"🚨 EMERGENCY → {tool_nm}"

        with st.expander(f"Step {snum} — {label}", expanded=True):
            st.markdown(
                f'<div class="react-box {box_class}">'
                f'<div class="step-label">{label}</div>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── FINAL JSON ASSESSMENT ────────────────────────────────
    st.subheader("📋 Final Assessment (Strict JSON Output)")

    if final_json:
        health = final_json.get("health", "Unknown")
        risk   = final_json.get("risk",   "Unknown")
        anomalies = final_json.get("anomalies", [])
        actions   = final_json.get("actions",   [])
        reason    = final_json.get("reason",    "")

        health_emoji = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}.get(health, "⚪")
        risk_emoji   = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

        j1, j2, j3 = st.columns(3)
        j1.metric("Machine ID", final_json.get("machine_id", machine_id))
        j2.metric("Health",     f"{health_emoji} {health}")
        j3.metric("Risk Level", f"{risk_emoji} {risk}")

        if anomalies:
            st.markdown("**🔍 Anomalies Detected:**")
            for a in anomalies:
                st.warning(f"⚠️ {a}")
        else:
            st.success("✅ No anomalies detected.")

        if actions:
            st.markdown("**⚙️ Actions Taken:**")
            for a in actions:
                st.markdown(f"- `{a}`")

        if reason:
            st.info(f"**💬 Agent Reasoning:** {reason}")

        with st.expander("📄 Raw JSON Output"):
            st.json(final_json)
    else:
        # Fallback: show raw final text if JSON extraction failed
        if final_summary:
            st.info(final_summary)
        else:
            st.warning("No final assessment returned.")

    st.divider()

    # ── TOOL EXECUTION SUMMARY ────────────────────────────────
    st.subheader("⚙️ Tool Execution Log")

    if not tools_executed:
        st.info("No tools were called.")
    else:
        for s in tools_executed:
            tn = s["tool_name"]
            obs_step = next(
                (o for o in steps if o["type"] == "observation" and o["tool_name"] == tn),
                None
            )
            obs_text = obs_step["content"] if obs_step else "No observation captured."

            if tn == "immediate_shutdown":
                st.error(f"**🚨 `{tn}`**\n\n{obs_text}")
            elif tn in ("reduce_machine_load", "schedule_maintenance", "shift_to_backup"):
                st.warning(f"**⚠️ `{tn}`**\n\n{obs_text}")
            else:
                st.success(f"**✅ `{tn}`**\n\n{obs_text}")

    st.caption("Run complete. Adjust sliders or use a Quick Test Case to try another scenario.")
