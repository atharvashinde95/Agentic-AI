"""
streamlit_app.py
----------------
Streamlit dashboard for the LangChain Predictive Maintenance Agent.

Run with:
    streamlit run streamlit_app.py

What it shows:
  • Live sensor cards (Temperature, Vibration, Pressure)
  • Status badge (Normal / Warning / Failure)
  • Agent's full ReAct trace (Thought → Action → Observation chain)
  • Final Answer from the agent
  • Tool call timeline
  • Sensor trend analysis
  • Multi-panel live line chart with threshold lines
  • Scrollable event log
  • Start / Stop / Reset buttons
"""

import time
import json
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import memory_store as mem
from simulator import SensorSimulator
from agent import MaintenanceAgent


# ─────────────────────────────────────────────────────────────────────────── #
#  Page config
# ─────────────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Predictive Maintenance Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────── #
#  CSS — dark industrial terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────── #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, .stApp {
    background-color: #070d12;
    color: #c9d8e3;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Header ── */
.agent-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: 0.04em;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #0e2233;
    margin-bottom: 0.2rem;
}
.agent-subtitle {
    font-size: 0.78rem;
    color: #4a6880;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1.2rem;
}

/* ── Section labels ── */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #00d4ff;
    margin: 0.9rem 0 0.4rem 0;
}

/* ── Sensor cards ── */
.sensor-card {
    background: #0b1520;
    border: 1px solid #0e2233;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.sensor-card.warning  { border-color: #c9870a; }
.sensor-card.critical { border-color: #c0392b; }
.sensor-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4a6880;
    margin-bottom: 0.25rem;
}
.sensor-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.15rem;
}
.sensor-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a6880;
}
.sensor-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    margin-top: 0.35rem;
    padding: 0.1rem 0.5rem;
    border-radius: 3px;
    display: inline-block;
}
.badge-normal   { background:#001f14; color:#00d47e; }
.badge-warning  { background:#2b1c00; color:#e8a400; }
.badge-critical { background:#2b0000; color:#ff4c4c; }

/* ── Status panel ── */
.status-panel {
    background: #0b1520;
    border: 1px solid #0e2233;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.status-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.status-normal   { color: #00d47e; }
.status-warning  { color: #e8a400; }
.status-failure  { color: #ff4c4c; }

/* ── Pill badges ── */
.pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin: 0.1rem;
}
.pill-continue    { background:#001f14; color:#00d47e; border: 1px solid #004d25; }
.pill-alert       { background:#2b1c00; color:#e8a400; border: 1px solid #5a3c00; }
.pill-maintenance { background:#2b0000; color:#ff4c4c; border: 1px solid #5a0000; }
.pill-low         { background:#001f14; color:#00d47e; border: 1px solid #004d25; }
.pill-medium      { background:#2b1c00; color:#e8a400; border: 1px solid #5a3c00; }
.pill-high        { background:#2b0000; color:#ff4c4c; border: 1px solid #5a0000; }
.pill-tool        { background:#00142a; color:#00d4ff; border: 1px solid #003d5a; }

/* ── ReAct trace ── */
.react-box {
    background: #070d12;
    border: 1px solid #0e2233;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    max-height: 260px;
    overflow-y: auto;
    margin-bottom: 0.6rem;
}
.react-step {
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #0e2233;
}
.react-step:last-child { border-bottom: none; margin-bottom: 0; }
.react-tool   { color: #00d4ff; font-weight: 600; }
.react-input  { color: #c9d8e3; }
.react-obs    { color: #4a6880; }
.react-step-num { color: #1c4060; font-size: 0.6rem; }

/* ── Event log ── */
.log-box {
    background: #070d12;
    border: 1px solid #0e2233;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    max-height: 200px;
    overflow-y: auto;
}
.log-row { padding: 0.2rem 0; border-bottom: 1px solid #0b1520; }
.log-row:last-child { border-bottom: none; }
.log-ts      { color: #1c4060; }
.log-normal  { color: #00d47e; }
.log-warning { color: #e8a400; }
.log-failure { color: #ff4c4c; }

/* ── Trend table ── */
.trend-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0;
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #0e2233;
}
.trend-up   { color: #ff7b72; }
.trend-down { color: #00d47e; }
.trend-flat { color: #4a6880; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    border-radius: 4px;
    border: 1px solid #0e2233;
    background: #0b1520;
    color: #c9d8e3;
}
div[data-testid="stButton"] > button:hover {
    border-color: #00d4ff;
    color: #00d4ff;
}

/* ── Misc ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Session state init
# ─────────────────────────────────────────────────────────────────────────── #

def _init_state():
    if "running"    not in st.session_state: st.session_state.running    = False
    if "simulator"  not in st.session_state: st.session_state.simulator  = SensorSimulator()
    if "agent"      not in st.session_state: st.session_state.agent      = MaintenanceAgent()
    if "chart_data" not in st.session_state: st.session_state.chart_data = []
    if "result"     not in st.session_state: st.session_state.result     = None

_init_state()


# ─────────────────────────────────────────────────────────────────────────── #
#  Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def _status_css(status: str) -> str:
    return {"Normal": "status-normal", "Warning": "status-warning",
            "Failure": "status-failure"}.get(status, "status-normal")

def _sensor_card_css(level: str) -> str:
    return {"warning": "sensor-card warning", "critical": "sensor-card critical"
            }.get(level, "sensor-card")

def _badge_css(level: str) -> str:
    return {"warning": "badge-warning", "critical": "badge-critical"
            }.get(level, "badge-normal")

def _pill_action(action: str) -> str:
    return {"Continue": "pill pill-continue", "Alert": "pill pill-alert",
            "Maintenance": "pill pill-maintenance"}.get(action, "pill")

def _pill_risk(risk: str) -> str:
    return {"Low": "pill pill-low", "Medium": "pill pill-medium",
            "High": "pill pill-high"}.get(risk, "pill")

def _trend_css(trend: str) -> str:
    return {"rising": "trend-up", "falling": "trend-down"}.get(trend, "trend-flat")

def _trend_arrow(trend: str) -> str:
    return {"rising": "↑", "falling": "↓"}.get(trend, "→")

def _risk_icon(risk: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")


def build_chart(data: list) -> go.Figure:
    """Three-panel Plotly chart with threshold reference lines."""
    if not data:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#070d12", plot_bgcolor="#070d12",
                          height=380, margin=dict(l=5,r=5,t=5,b=5))
        return fig

    df = pd.DataFrame(data)
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        vertical_spacing=0.09,
    )

    SENSORS = [
        ("temperature", "#ff7b72", 1, 85,   100),
        ("vibration",   "#ffa657", 2,  6.0,   9.0),
        ("pressure",    "#58a6ff", 3,  7.0,   9.5),
    ]

    for col, color, row, warn, crit in SENSORS:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(x=df["tick"], y=df[col], mode="lines",
                       line=dict(color=color, width=1.8), showlegend=False),
            row=row, col=1,
        )
        fig.add_hline(y=warn, line_dash="dash", line_color="#e8a400",
                      line_width=1, opacity=0.5, row=row, col=1)
        fig.add_hline(y=crit, line_dash="dot",  line_color="#ff4c4c",
                      line_width=1, opacity=0.5, row=row, col=1)

    fig.update_layout(
        height=380,
        paper_bgcolor="#070d12",
        plot_bgcolor="#0b1520",
        font=dict(color="#4a6880", size=10, family="IBM Plex Mono"),
        margin=dict(l=8, r=8, t=28, b=8),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#0e2233",
                     tickfont=dict(size=9), row=3, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#0e2233")
    for i in range(1, 4):
        fig.update_annotations({"font": {"size": 10, "color": "#4a6880"}},
                                selector=dict(xref=f"x{'' if i==1 else i}"))
    return fig


def render_react_trace(tool_calls: list):
    """Render the ReAct Thought→Action→Observation chain as styled HTML."""
    if not tool_calls:
        return "<div class='react-box'><span style='color:#1c4060'>No steps recorded.</span></div>"

    html = "<div class='react-box'>"
    for i, step in enumerate(tool_calls, 1):
        tool  = step.get("tool", "?")
        inp   = step.get("tool_input", "")
        obs   = step.get("observation", "")

        # Truncate long observation
        obs_str = str(obs)[:280] + ("…" if len(str(obs)) > 280 else "")

        # Pretty-print JSON input if possible
        try:
            inp_pretty = json.dumps(json.loads(inp), indent=None) if inp else ""
        except Exception:
            inp_pretty = str(inp)[:120]

        html += f"""<div class='react-step'>
            <span class='react-step-num'>STEP {i}</span><br>
            <span class='react-tool'>⚙ {tool}</span><br>
            <span class='react-input'>&nbsp;&nbsp;▸ {inp_pretty[:160]}</span><br>
            <span class='react-obs'>&nbsp;&nbsp;◂ {obs_str}</span>
        </div>"""
    html += "</div>"
    return html


def render_log(entries: list):
    """Render last 20 log entries as styled HTML."""
    if not entries:
        return "<div class='log-box'><span style='color:#1c4060'>No events yet — start simulation.</span></div>"

    html = "<div class='log-box'>"
    for e in entries[:20]:
        css = {"Normal": "log-normal", "Warning": "log-warning",
               "Failure": "log-failure"}.get(e["status"], "log-normal")
        html += (
            f"<div class='log-row'>"
            f"<span class='log-ts'>[{e['time']}] #{e['tick']:>4}</span>  "
            f"<span class='{css}'>{e['status']:<8}</span>  "
            f"{e['action']:<12}  Risk:{e['risk']:<6}  {e['tool']}"
            f"</div>"
        )
    html += "</div>"
    return html


def run_one_cycle():
    """Invoke one agent cycle and update Streamlit session state."""
    reading = st.session_state.simulator.read()
    result  = st.session_state.agent.run_cycle(reading)
    st.session_state.result = result

    st.session_state.chart_data.append({
        "tick"        : reading["tick"],
        "temperature" : reading["temperature"],
        "vibration"   : reading["vibration"],
        "pressure"    : reading["pressure"],
    })
    if len(st.session_state.chart_data) > 80:
        st.session_state.chart_data.pop(0)


# ─────────────────────────────────────────────────────────────────────────── #
#  Layout
# ─────────────────────────────────────────────────────────────────────────── #

# ── Header ────────────────────────────────────────────────────────────────── #
st.markdown(
    "<div class='agent-header'>⚙ PREDICTIVE MAINTENANCE AGENT</div>"
    "<div class='agent-subtitle'>LangChain ReAct · Amazon Nova Lite · "
    "6 Tools · ConversationMemory · Real-time sensor fusion</div>",
    unsafe_allow_html=True,
)

# ── Control buttons ────────────────────────────────────────────────────────── #
b1, b2, b3, bx = st.columns([1, 1, 1, 6])
with b1:
    if st.button("▶ Start", use_container_width=True):
        st.session_state.running = True
with b2:
    if st.button("⏹ Stop",  use_container_width=True):
        st.session_state.running = False
with b3:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.running  = False
        st.session_state.simulator = SensorSimulator()
        st.session_state.agent     = MaintenanceAgent()
        st.session_state.chart_data = []
        st.session_state.result    = None
        mem.reset_store()

# ── Run cycle ──────────────────────────────────────────────────────────────── #
if st.session_state.running:
    run_one_cycle()

result  = st.session_state.result
reading = result["reading"] if result else {}

# ── Two-column layout ──────────────────────────────────────────────────────── #
left, right = st.columns([3, 2], gap="medium")

# ═══════════════════════════════════════════════════════════════════════════ #
#  LEFT column
# ═══════════════════════════════════════════════════════════════════════════ #
with left:

    # ── Sensor cards ──────────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Live Sensor Readings</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    def _card(col, label, value, unit, color, level):
        card_css  = _sensor_card_css(level)
        badge_css = _badge_css(level)
        col.markdown(
            f"<div class='{card_css}'>"
            f"<div class='sensor-label'>{label}</div>"
            f"<div class='sensor-value' style='color:{color}'>{value}</div>"
            f"<div class='sensor-unit'>{unit}</div>"
            f"<div class='sensor-badge {badge_css}'>{level.upper()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    sensor_status = result["sensor_status"] if result else {}
    _card(c1, "Temperature",
          f"{reading.get('temperature', 0):.1f}" if reading else "--",
          "°C", "#ff7b72",
          sensor_status.get("temperature", "normal"))
    _card(c2, "Vibration",
          f"{reading.get('vibration', 0):.2f}" if reading else "--",
          "mm/s", "#ffa657",
          sensor_status.get("vibration", "normal"))
    _card(c3, "Pressure",
          f"{reading.get('pressure', 0):.2f}" if reading else "--",
          "bar", "#58a6ff",
          sensor_status.get("pressure", "normal"))

    # ── Chart ──────────────────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Sensor History (last 80 ticks · dashed=warning · dotted=critical)</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        build_chart(st.session_state.chart_data),
        use_container_width=True,
        key="main_chart",
    )

    # ── ReAct trace ────────────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Agent ReAct Trace (Thought → Action → Observation)</div>",
                unsafe_allow_html=True)
    tool_calls = result["tool_calls"] if result else []
    st.markdown(render_react_trace(tool_calls), unsafe_allow_html=True)

    # ── Final answer ───────────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Agent Final Answer</div>", unsafe_allow_html=True)
    fa = result["final_answer"] if result else "Awaiting first cycle…"
    st.markdown(
        f"<div class='react-box' style='max-height:90px'>"
        f"<span style='color:#c9d8e3'>{fa[:400]}</span></div>",
        unsafe_allow_html=True,
    )

    # ── Event log ──────────────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Event Log</div>", unsafe_allow_html=True)
    st.markdown(render_log(mem.EVENT_LOG), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════ #
#  RIGHT column
# ═══════════════════════════════════════════════════════════════════════════ #
with right:

    # ── Main status panel ─────────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Machine Status</div>", unsafe_allow_html=True)

    if result:
        status_css = _status_css(result["status"])
        st.markdown(
            f"<div class='status-panel'>"
            f"<div class='status-main {status_css}'>"
            f"{result['status'].upper()}</div>"
            f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;color:#4a6880'>"
            f"Tick #{result['tick']} · Cycle #{result['cycle_count']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='status-panel'>"
            "<div class='status-main' style='color:#1c4060'>IDLE</div>"
            "<div style='font-size:0.7rem;color:#1c4060'>Press ▶ Start</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Decision / Risk / Confidence pills ───────────────────────────────── #
    if result:
        st.markdown("<div class='sec-label'>Decision Metrics</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown("<div style='font-size:0.6rem;color:#4a6880;font-family:IBM Plex Mono'>ACTION</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='{_pill_action(result['action'])}'>{result['action']}</span>", unsafe_allow_html=True)
        with d2:
            st.markdown("<div style='font-size:0.6rem;color:#4a6880;font-family:IBM Plex Mono'>RISK</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='{_pill_risk(result['risk'])}'>{_risk_icon(result['risk'])} {result['risk']}</span>", unsafe_allow_html=True)
        with d3:
            st.markdown("<div style='font-size:0.6rem;color:#4a6880;font-family:IBM Plex Mono'>CONFIDENCE</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='pill' style='background:#0b1520;border:1px solid #0e2233;color:#c9d8e3'>{result['confidence']}</span>", unsafe_allow_html=True)

    # ── Primary tool used ────────────────────────────────────────────────── #
    if result:
        tool_icon = {"send_alert": "⚠", "schedule_maintenance": "🔧",
                     "log_normal_cycle": "📝", "check_sensor_status": "📡",
                     "diagnose_condition": "🩺", "get_trend_analysis": "📊"
                     }.get(result["tool_used"], "⚙")
        st.markdown("<div class='sec-label'>Primary Action Tool</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='pill pill-tool'>{tool_icon} {result['tool_used']}</span>",
            unsafe_allow_html=True,
        )

    # ── Tool call sequence ───────────────────────────────────────────────── #
    if result and result["tool_calls"]:
        st.markdown("<div class='sec-label'>Tool Call Sequence</div>", unsafe_allow_html=True)
        seq_html = ""
        for i, tc in enumerate(result["tool_calls"]):
            arrow = " → " if i < len(result["tool_calls"]) - 1 else ""
            seq_html += (
                f"<span class='pill pill-tool' style='margin:1px;font-size:0.62rem'>"
                f"{tc['tool']}</span>{arrow}"
            )
        st.markdown(seq_html, unsafe_allow_html=True)

    # ── Diagnoses ────────────────────────────────────────────────────────── #
    if result and result["diagnoses"]:
        st.markdown("<div class='sec-label'>Fault Diagnoses</div>", unsafe_allow_html=True)
        for fault in result["diagnoses"]:
            icon = "🔴" if "Critical" in fault or "Severe" in fault else (
                   "🟡" if "No faults" not in fault else "🟢")
            st.markdown(
                f"<div style='font-family:IBM Plex Mono;font-size:0.72rem;"
                f"padding:0.2rem 0;color:#c9d8e3'>{icon} {fault}</div>",
                unsafe_allow_html=True,
            )

    # ── Trend analysis ───────────────────────────────────────────────────── #
    trends = mem.compute_trends()
    if trends:
        st.markdown("<div class='sec-label'>Sensor Trend Analysis</div>", unsafe_allow_html=True)
        trend_html = ""
        for sensor, info in trends.items():
            css   = _trend_css(info["trend"])
            arrow = _trend_arrow(info["trend"])
            trend_html += (
                f"<div class='trend-row'>"
                f"<span style='color:#4a6880'>{sensor.capitalize()}</span>"
                f"<span class='{css}'>{arrow} {info['trend'].upper()}</span>"
                f"<span style='color:#c9d8e3'>{info['average']}</span>"
                f"</div>"
            )
        st.markdown(trend_html, unsafe_allow_html=True)

    # ── LangChain agent info ─────────────────────────────────────────────── #
    st.markdown("<div class='sec-label'>Agent Configuration</div>", unsafe_allow_html=True)
    cfg_items = [
        ("Framework",  "LangChain ReAct"),
        ("LLM",        "Amazon Nova Lite"),
        ("Memory",     "ConversationBufferWindowMemory (k=6)"),
        ("Tools",      "6 @tool functions"),
        ("Executor",   "AgentExecutor (max 8 iterations)"),
        ("Cycles",     str(result["cycle_count"] if result else 0)),
    ]
    cfg_html = ""
    for k, v in cfg_items:
        cfg_html += (
            f"<div style='font-family:IBM Plex Mono;font-size:0.68rem;"
            f"display:flex;justify-content:space-between;"
            f"border-bottom:1px solid #0e2233;padding:0.18rem 0'>"
            f"<span style='color:#4a6880'>{k}</span>"
            f"<span style='color:#c9d8e3'>{v}</span></div>"
        )
    st.markdown(cfg_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────── #
#  Auto-refresh while running
# ─────────────────────────────────────────────────────────────────────────── #

if st.session_state.running:
    time.sleep(1.8)   # Pause between agent cycles (LLM call takes time)
    st.rerun()
