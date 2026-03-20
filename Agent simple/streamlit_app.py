"""
streamlit_app.py
----------------
Streamlit dashboard for the Autonomous Predictive Maintenance Agent.

Run with:
    streamlit run streamlit_app.py

Features:
  • Real-time sensor value display (Temperature, Vibration, Pressure)
  • Status indicators (Normal / Warning / Failure)
  • Agent decision & tool used
  • Line charts for all three sensors
  • Scrollable event log
  • Start / Stop buttons
  • Risk level & confidence score
"""

import time
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import SensorSimulator
from agent import MaintenanceAgent


# ──────────────────────────────────────────────────────────────────── #
#  Page config
# ──────────────────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="Predictive Maintenance Agent",
    page_icon="🔧",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────── #
#  Custom CSS — industrial dark theme
# ──────────────────────────────────────────────────────────────────── #

st.markdown("""
<style>
  /* Dark industrial background */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  
  /* Header */
  .main-header {
    font-family: 'Courier New', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    border-bottom: 2px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }
  
  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'Courier New', monospace;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
  }
  .metric-unit {
    font-size: 0.75rem;
    color: #8b949e;
  }
  
  /* Status badge */
  .status-normal   { color: #3fb950; font-weight: 700; font-size: 1.1rem; }
  .status-warning  { color: #d29922; font-weight: 700; font-size: 1.1rem; }
  .status-failure  { color: #f85149; font-weight: 700; font-size: 1.1rem; }
  
  /* Sensor color classes */
  .temp-color  { color: #ff7b72; }
  .vib-color   { color: #ffa657; }
  .pres-color  { color: #79c0ff; }
  
  /* Decision badge */
  .decision-continue    { background: #1c4428; color: #3fb950; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
  .decision-alert       { background: #3d2b00; color: #d29922; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
  .decision-maintenance { background: #3c1515; color: #f85149; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
  
  /* Log entries */
  .log-entry {
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
  }
  .log-normal      { color: #3fb950; }
  .log-warning     { color: #d29922; }
  .log-maintenance { color: #f85149; }
  
  /* Section titles */
  .section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #58a6ff;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
  }

  /* Hide Streamlit branding */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────── #
#  Session state initialisation
# ──────────────────────────────────────────────────────────────────── #

if "running" not in st.session_state:
    st.session_state.running = False

if "simulator" not in st.session_state:
    st.session_state.simulator = SensorSimulator()

if "agent" not in st.session_state:
    st.session_state.agent = MaintenanceAgent()

if "history" not in st.session_state:
    # history stores list of dicts for charting
    st.session_state.history = []

if "log_entries" not in st.session_state:
    st.session_state.log_entries = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ──────────────────────────────────────────────────────────────────── #
#  Helper functions
# ──────────────────────────────────────────────────────────────────── #

def status_color(status: str) -> str:
    return {"Normal": "#3fb950", "Warning": "#d29922", "Failure": "#f85149"}.get(status, "#8b949e")

def risk_emoji(risk: str) -> str:
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

def action_class(action: str) -> str:
    return {
        "Continue"   : "decision-continue",
        "Alert"      : "decision-alert",
        "Maintenance": "decision-maintenance",
    }.get(action, "")


def build_chart(history: list) -> go.Figure:
    """Build a multi-panel Plotly chart for all three sensors."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        vertical_spacing=0.08,
    )

    # Threshold lines
    thresholds = {
        "temperature": {"warning": 85, "critical": 100},
        "vibration"  : {"warning": 6,  "critical": 9},
        "pressure"   : {"warning": 7,  "critical": 9.5},
    }

    sensors = [
        ("temperature", "#ff7b72", 1),
        ("vibration",   "#ffa657", 2),
        ("pressure",    "#79c0ff", 3),
    ]

    for col, color, row in sensors:
        # Sensor line
        fig.add_trace(
            go.Scatter(
                x=df["tick"],
                y=df[col],
                mode="lines",
                name=col.capitalize(),
                line=dict(color=color, width=2),
                showlegend=False,
            ),
            row=row, col=1,
        )
        # Warning threshold
        fig.add_hline(
            y=thresholds[col]["warning"],
            line_dash="dash",
            line_color="#d29922",
            line_width=1,
            opacity=0.6,
            row=row, col=1,
        )
        # Critical threshold
        fig.add_hline(
            y=thresholds[col]["critical"],
            line_dash="dot",
            line_color="#f85149",
            line_width=1,
            opacity=0.6,
            row=row, col=1,
        )

    fig.update_layout(
        height=420,
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#21262d",
        tickfont=dict(size=9),
        title_text="Tick",
        row=3, col=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#21262d")

    return fig


def run_one_cycle():
    """Execute one agent cycle and update session state."""
    reading = st.session_state.simulator.read()
    result  = st.session_state.agent.run_cycle(reading)
    st.session_state.last_result = result

    # Append to charting history (keep last 60 points)
    st.session_state.history.append({
        "tick"       : reading["tick"],
        "temperature": reading["temperature"],
        "vibration"  : reading["vibration"],
        "pressure"   : reading["pressure"],
    })
    if len(st.session_state.history) > 60:
        st.session_state.history.pop(0)

    # Build log entry
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {
        "time"  : ts,
        "tick"  : reading["tick"],
        "status": result["status"],
        "action": result["action"],
        "risk"  : result["risk"],
        "tool"  : result["tool_result"]["tool"],
        "msg"   : result["tool_result"].get("message", "Normal cycle"),
    }
    st.session_state.log_entries.insert(0, entry)   # newest first
    if len(st.session_state.log_entries) > 50:
        st.session_state.log_entries.pop()


# ──────────────────────────────────────────────────────────────────── #
#  Layout — Header
# ──────────────────────────────────────────────────────────────────── #

st.markdown('<div class="main-header">🔧 Predictive Maintenance Agent</div>', unsafe_allow_html=True)
st.markdown(
    "<span style='font-size:0.85rem;color:#8b949e;'>Single autonomous agent · "
    "Rule-based decisions · LLM for edge cases only</span>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────── #
#  Layout — Control buttons
# ──────────────────────────────────────────────────────────────────── #

col_start, col_stop, col_reset, col_spacer = st.columns([1, 1, 1, 5])

with col_start:
    if st.button("▶ Start", type="primary", use_container_width=True):
        st.session_state.running = True

with col_stop:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False

with col_reset:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.simulator = SensorSimulator()
        st.session_state.agent = MaintenanceAgent()
        st.session_state.history = []
        st.session_state.log_entries = []
        st.session_state.last_result = None

# ──────────────────────────────────────────────────────────────────── #
#  Run cycle if active
# ──────────────────────────────────────────────────────────────────── #

if st.session_state.running:
    run_one_cycle()

result  = st.session_state.last_result
reading = result["reading"]   if result else {}

# ──────────────────────────────────────────────────────────────────── #
#  Layout — Main content
# ──────────────────────────────────────────────────────────────────── #

left_col, right_col = st.columns([3, 2])

with left_col:

    # ── Sensor cards ──────────────────────────────────────────────── #
    st.markdown('<div class="section-title">Live Sensor Readings</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    def sensor_card(container, label, value, unit, color, status="normal"):
        border_color = {"normal": "#30363d", "warning": "#d29922", "critical": "#f85149"}.get(status, "#30363d")
        container.markdown(
            f"""<div class="metric-card" style="border-color:{border_color};">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
                <div class="metric-unit">{unit}</div>
                <div style="font-size:0.7rem;color:#8b949e;margin-top:0.3rem;">{status.upper()}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    temp_val = f"{reading.get('temperature', 0):.1f}" if reading else "--"
    vib_val  = f"{reading.get('vibration',   0):.2f}" if reading else "--"
    pres_val = f"{reading.get('pressure',    0):.2f}" if reading else "--"

    t_status = result["sensor_status"].get("temperature", "normal") if result else "normal"
    v_status = result["sensor_status"].get("vibration",   "normal") if result else "normal"
    p_status = result["sensor_status"].get("pressure",    "normal") if result else "normal"

    sensor_card(c1, "Temperature", temp_val, "°C",   "#ff7b72", t_status)
    sensor_card(c2, "Vibration",   vib_val,  "mm/s", "#ffa657", v_status)
    sensor_card(c3, "Pressure",    pres_val, "bar",  "#79c0ff", p_status)

    # ── Chart ─────────────────────────────────────────────────────── #
    st.markdown('<div class="section-title">Sensor History</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(
        build_chart(st.session_state.history),
        use_container_width=True,
        key="sensor_chart",
    )

with right_col:

    # ── Agent status panel ────────────────────────────────────────── #
    st.markdown('<div class="section-title">Agent Status</div>', unsafe_allow_html=True)

    if result:
        status_label = result["status"]
        status_col   = status_color(status_label)

        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Overall Status</div>
                <div style="font-size:1.8rem;font-weight:700;color:{status_col};">{status_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Decision + Risk + Confidence in one row
        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"<div class='metric-label'>Decision</div>", unsafe_allow_html=True)
            action_cls = action_class(result["action"])
            st.markdown(f"<span class='{action_cls}'>{result['action']}</span>", unsafe_allow_html=True)
        with d2:
            st.markdown(f"<div class='metric-label'>Risk</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-weight:600'>{risk_emoji(result['risk'])} {result['risk']}</span>", unsafe_allow_html=True)
        with d3:
            st.markdown(f"<div class='metric-label'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-weight:600;color:#8b949e'>{result['confidence']}</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tool used
        tool_name = result["tool_result"]["tool"]
        tool_icon = {"alert_tool": "⚠️", "maintenance_tool": "🔧", "logging_tool": "📝"}.get(tool_name, "🔧")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Tool Invoked</div>
                <div style="font-size:1.1rem;font-weight:600;color:#58a6ff;">{tool_icon} {tool_name}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Diagnoses
        if result["diagnoses"]:
            st.markdown('<div class="section-title">Diagnoses</div>', unsafe_allow_html=True)
            for d in result["diagnoses"]:
                st.markdown(f"• {d}", unsafe_allow_html=True)

        # Trend summary
        trends = result.get("trend_summary", {})
        if trends:
            st.markdown('<div class="section-title">Sensor Trends</div>', unsafe_allow_html=True)
            for sensor, info in trends.items():
                arrow = {"rising": "↑", "falling": "↓", "stable": "→"}.get(info["trend"], "→")
                color = {"rising": "#f85149", "falling": "#3fb950", "stable": "#8b949e"}.get(info["trend"], "#8b949e")
                st.markdown(
                    f"<span style='font-size:0.85rem'>{sensor.capitalize()}: "
                    f"<span style='color:{color}'>{arrow} {info['trend'].upper()}</span> "
                    f"(avg {info['average']})</span>",
                    unsafe_allow_html=True,
                )

        # LLM explanation (shown only when used)
        if result.get("llm_used") and result.get("llm_explanation"):
            st.markdown('<div class="section-title">🤖 LLM Analysis</div>', unsafe_allow_html=True)
            st.info(result["llm_explanation"])
            st.caption(f"Total LLM calls this session: {result['llm_call_count']}")

    else:
        st.markdown(
            "<div style='color:#8b949e;text-align:center;padding:2rem 0;'>"
            "Press ▶ Start to begin simulation</div>",
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────── #
#  Layout — Event Log (full width)
# ──────────────────────────────────────────────────────────────────── #

st.markdown('<div class="section-title">Event Log</div>', unsafe_allow_html=True)

log_placeholder = st.empty()

if st.session_state.log_entries:
    log_html = ""
    for entry in st.session_state.log_entries[:20]:   # show last 20
        css_class = {
            "Normal"  : "log-normal",
            "Warning" : "log-warning",
            "Failure" : "log-maintenance",
        }.get(entry["status"], "")
        log_html += (
            f"<div class='log-entry'>"
            f"<span style='color:#8b949e'>[{entry['time']}]</span> "
            f"Tick #{entry['tick']:>4} | "
            f"<span class='{css_class}'>{entry['status']:<8}</span> | "
            f"{entry['action']:<12} | Risk: {entry['risk']:<6} | "
            f"{entry['tool']}"
            f"</div>"
        )
    log_placeholder.markdown(log_html, unsafe_allow_html=True)
else:
    log_placeholder.markdown(
        "<div style='color:#8b949e;font-size:0.85rem;'>No events yet.</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────── #
#  Auto-refresh while running
# ──────────────────────────────────────────────────────────────────── #

if st.session_state.running:
    time.sleep(1.5)
    st.rerun()
