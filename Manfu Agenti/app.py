# ─────────────────────────────────────────────
#  app.py  —  Streamlit live dashboard
# ─────────────────────────────────────────────

import time
import threading
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from simulator import simulator
from memory    import memory
from agent     import agent
from config    import MACHINES, THRESHOLDS, AGENT_CHECK_INTERVAL

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #2e3250;
        margin-bottom: 0.5rem;
    }
    .status-normal   { color: #00c853; font-weight: 600; }
    .status-warning  { color: #ffab00; font-weight: 600; }
    .status-failure  { color: #ff5252; font-weight: 600; }
    .status-recovering { color: #40c4ff; font-weight: 600; }
    .thought-box {
        background: #1a1d2e;
        border-left: 3px solid #7c4dff;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-family: monospace;
        font-size: 13px;
    }
    .action-box {
        background: #1a2a1a;
        border-left: 3px solid #00c853;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-family: monospace;
        font-size: 13px;
    }
    .obs-box {
        background: #1a1a2a;
        border-left: 3px solid #448aff;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 4px 0;
        font-family: monospace;
        font-size: 13px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────
def _init_state():
    defaults = {
        "simulator_started": False,
        "agent_results":     {},
        "agent_running":     False,
        "auto_run":          False,
        "last_auto_run":     0.0,
        "selected_machine":  "M1",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Start simulator on first load ─────────────────
if not st.session_state["simulator_started"]:
    simulator.start()
    st.session_state["simulator_started"] = True
    time.sleep(2)   # let buffer warm up


# ── Colour helpers ────────────────────────────────
STATUS_COLOR = {
    "normal":     "#00c853",
    "warning":    "#ffab00",
    "failure":    "#ff5252",
    "recovering": "#40c4ff",
    "degrading":  "#ffab00",
    "critical":   "#ff5252",
}

def _status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#aaa")
    return f'<span style="color:{color};font-weight:600">{status.upper()}</span>'


# ── Build sensor chart ────────────────────────────
def _build_chart(machine_id: str) -> go.Figure:
    readings = simulator.get_readings(machine_id, n=20)
    if not readings:
        fig = go.Figure()
        fig.add_annotation(text="Warming up…", showarrow=False)
        return fig

    df = pd.DataFrame(readings)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        vertical_spacing=0.08,
    )

    # colour each point by status
    colors = df["status"].map({
        "normal":  "#00c853",
        "warning": "#ffab00",
        "failure": "#ff5252",
    }).fillna("#aaa")

    def _line(y_col, row, warn_t, crit_t, color_line):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[y_col],
            mode="lines+markers",
            line=dict(color=color_line, width=2),
            marker=dict(color=colors, size=6),
            name=y_col,
            showlegend=False,
        ), row=row, col=1)
        # threshold lines
        fig.add_hline(y=warn_t, line_dash="dot", line_color="#ffab00",
                      line_width=1, row=row, col=1)
        fig.add_hline(y=crit_t, line_dash="dash", line_color="#ff5252",
                      line_width=1, row=row, col=1)

    _line("temperature", 1,
          THRESHOLDS["temperature"]["warning"], THRESHOLDS["temperature"]["critical"],
          "#ef5350")
    _line("vibration",   2,
          THRESHOLDS["vibration"]["warning"],   THRESHOLDS["vibration"]["critical"],
          "#ffa726")
    _line("pressure",    3,
          THRESHOLDS["pressure"]["warning"],    THRESHOLDS["pressure"]["critical"],
          "#42a5f5")

    fig.update_layout(
        height=420,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#ccc", size=11),
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(gridcolor="#1e2130")
    return fig


# ── Run agent in background thread ───────────────
def _run_agent_async(machine_id: str):
    def _run():
        st.session_state["agent_running"] = True
        result = agent.run(machine_id)
        st.session_state["agent_results"][machine_id] = result
        st.session_state["agent_running"] = False
    threading.Thread(target=_run, daemon=True).start()


# ════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙ Maintenance AI")
    st.caption("Agentic Predictive Maintenance System")
    st.divider()

    # machine selector
    selected = st.selectbox(
        "Select machine to inspect",
        MACHINES,
        index=MACHINES.index(st.session_state["selected_machine"]),
    )
    st.session_state["selected_machine"] = selected

    st.divider()

    # manual agent trigger
    if st.button("▶  Run Agent on " + selected,
                 disabled=st.session_state["agent_running"]):
        _run_agent_async(selected)

    if st.button("▶▶  Run Agent on ALL machines",
                 disabled=st.session_state["agent_running"]):
        for mid in MACHINES:
            _run_agent_async(mid)

    st.divider()

    # auto-run toggle
    auto = st.toggle("Auto-run agent every 30s", value=st.session_state["auto_run"])
    st.session_state["auto_run"] = auto

    if st.session_state["agent_running"]:
        st.info("Agent is running…")

    st.divider()
    st.caption(f"Tick interval : 5 sec")
    st.caption(f"Buffer size   : 20 readings")
    st.caption(f"LLM model     : amazon.nova.lite")


# ════════════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════════════
st.title("⚙ Predictive Maintenance — Live Dashboard")

# ── Row 1: KPI cards for all machines ────────────
st.subheader("Machine Fleet Status")
cols = st.columns(5)
all_latest = simulator.get_all_latest()

for i, mid in enumerate(MACHINES):
    reading = all_latest.get(mid)
    with cols[i]:
        if reading:
            state  = simulator.get_state(mid)
            status = reading["status"]
            color  = STATUS_COLOR.get(status, "#aaa")
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:18px;font-weight:700">{mid}</div>
                <div style="color:{color};font-size:13px;margin:4px 0">{status.upper()}</div>
                <div style="font-size:12px;color:#aaa">🌡 {reading['temperature']}°C</div>
                <div style="font-size:12px;color:#aaa">〰 {reading['vibration']} mm/s</div>
                <div style="font-size:12px;color:#aaa">⬡ {reading['pressure']} bar</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:18px;font-weight:700">{mid}</div>
                <div style="color:#aaa;font-size:13px">Warming up…</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Row 2: Live sensor charts + agent panel ───────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader(f"Live Sensors — {selected}")
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(
        _build_chart(selected),
        use_container_width=True,
        key=f"chart_{selected}_{int(time.time())}",
    )

with right_col:
    st.subheader("Agent Thought Process")
    result = st.session_state["agent_results"].get(selected)

    if not result:
        st.info("Run the agent to see its reasoning here.")
    else:
        thought_container = st.container()
        with thought_container:
            for step in result.get("thought_steps", []):
                stype = step.get("type")
                if stype == "action":
                    st.markdown(
                        f'<div class="action-box">'
                        f'<b>ACTION</b> → <code>{step["tool"]}</code><br>'
                        f'Input: <code>{step["input"]}</code></div>',
                        unsafe_allow_html=True,
                    )
                elif stype == "observation":
                    preview = step["output"][:300].replace("\n", "<br>")
                    st.markdown(
                        f'<div class="obs-box"><b>OBSERVATION</b><br>{preview}</div>',
                        unsafe_allow_html=True,
                    )
                elif stype == "final":
                    st.markdown(
                        f'<div class="thought-box"><b>FINAL DECISION</b><br>'
                        f'{step["output"][:400]}</div>',
                        unsafe_allow_html=True,
                    )

st.divider()

# ── Row 3: Final report ───────────────────────────
st.subheader(f"Agent Report — {selected}")
result = st.session_state["agent_results"].get(selected)
if result and result.get("final_report"):
    st.markdown(
        f'<div style="background:#1e2130;border-radius:10px;padding:1rem 1.2rem;'
        f'border:1px solid #2e3250;font-size:14px;line-height:1.7">'
        f'{result["final_report"].replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("No report yet. Run the agent on this machine.")

st.divider()

# ── Row 4: Action log table ───────────────────────
st.subheader("Action History Log")
log = memory.get_recent_actions(n=30)
if log:
    df_log = pd.DataFrame(log)
    df_log = df_log[["timestamp", "machine_id", "action",
                      "severity", "reason", "job_id", "alert_id"]]
    df_log.columns = ["Time", "Machine", "Action", "Severity", "Reason", "Job ID", "Alert ID"]

    def _color_row(row):
        if row["Action"] == "alert_sent":
            return ["background-color:#2a1a1a"] * len(row)
        if row["Action"] == "maintenance_scheduled":
            return ["background-color:#1a2a1a"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_log.style.apply(_color_row, axis=1),
        use_container_width=True,
        height=250,
    )
else:
    st.info("No actions logged yet.")

# ── Row 5: Summary KPIs ───────────────────────────
st.subheader("Session Summary")
summary = memory.summary()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total actions",   summary["total_actions"])
k2.metric("Alerts fired",    summary["alerts_sent"])
k3.metric("Jobs scheduled",  summary["jobs_scheduled"])
k4.metric("Machines active", summary["machines_active"])

# ── Auto-refresh loop ─────────────────────────────
if st.session_state["auto_run"]:
    now = time.time()
    if now - st.session_state["last_auto_run"] > AGENT_CHECK_INTERVAL:
        st.session_state["last_auto_run"] = now
        _run_agent_async(selected)

# refresh every 5 seconds to show new sensor data
time.sleep(5)
st.rerun()
