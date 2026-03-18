# dashboard/app.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import threading
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.simulator import simulator
from agent.agent         import agent
from core.config         import MACHINES, THRESHOLDS, AGENT_CHECK_INTERVAL

# ══════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #2e3250;
    margin-bottom: 0.5rem;
    min-height: 130px;
}
.step-action {
    background:#1a2e1a; border-left:3px solid #00c853;
    padding:0.5rem 0.8rem; border-radius:0 6px 6px 0;
    margin:3px 0; font-family:monospace; font-size:12px;
}
.step-obs {
    background:#1a1a2e; border-left:3px solid #448aff;
    padding:0.5rem 0.8rem; border-radius:0 6px 6px 0;
    margin:3px 0; font-family:monospace; font-size:12px;
}
.step-final {
    background:#2a1a2e; border-left:3px solid #ce93d8;
    padding:0.5rem 0.8rem; border-radius:0 6px 6px 0;
    margin:3px 0; font-family:monospace; font-size:12px;
}
.report-box {
    background:#1e2130; border-radius:10px;
    padding:1rem 1.2rem; border:1px solid #2e3250;
    font-size:14px; line-height:1.75;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  Colour helpers
# ══════════════════════════════════════════════════════════
STATUS_COLOR = {
    "normal":     "#00c853",
    "warning":    "#ffab00",
    "failure":    "#ff5252",
    "degrading":  "#ffab00",
    "critical":   "#ff5252",
    "recovering": "#40c4ff",
}

# ══════════════════════════════════════════════════════════
#  Chart builders
# ══════════════════════════════════════════════════════════
def build_sensor_chart(machine_id: str) -> go.Figure:
    readings = simulator.get_readings(machine_id, n=20)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        vertical_spacing=0.10,
    )

    if not readings:
        fig.add_annotation(text="Warming up — please wait...", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _style_fig(fig)
        return fig

    df = pd.DataFrame(readings)
    point_colors = df["status"].map({
        "normal":  "#00c853",
        "warning": "#ffab00",
        "failure": "#ff5252",
    }).fillna("#aaa")

    def _add(col, row, line_color, warn, crit):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col],
            mode="lines+markers",
            line=dict(color=line_color, width=2),
            marker=dict(color=point_colors, size=6),
            showlegend=False,
        ), row=row, col=1)
        fig.add_hline(y=warn, line_dash="dot",  line_color="#ffab00", line_width=1, row=row, col=1)
        fig.add_hline(y=crit, line_dash="dash", line_color="#ff5252", line_width=1, row=row, col=1)

    _add("temperature", 1, "#ef5350",
         THRESHOLDS["temperature"]["warning"], THRESHOLDS["temperature"]["critical"])
    _add("vibration",   2, "#ffa726",
         THRESHOLDS["vibration"]["warning"],   THRESHOLDS["vibration"]["critical"])
    _add("pressure",    3, "#42a5f5",
         THRESHOLDS["pressure"]["warning"],    THRESHOLDS["pressure"]["critical"])

    _style_fig(fig)
    return fig


def build_fleet_bar(all_latest: dict) -> go.Figure:
    """
    Fleet overview — 3 subplots always visible side by side.
    Temperature / Vibration / Pressure each get their own panel.
    No toggling — no disappearing bars.
    """
    machines, temps, vibs, pres, colors = [], [], [], [], []
    for mid, r in all_latest.items():
        if r:
            machines.append(mid)
            temps.append(r["temperature"])
            vibs.append(r["vibration"])
            pres.append(r["pressure"])
            colors.append(STATUS_COLOR.get(r["status"], "#aaa"))

    if not machines:
        fig = go.Figure()
        fig.add_annotation(text="Warming up...", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        horizontal_spacing=0.08,
    )

    # Temperature — red shades
    fig.add_trace(go.Bar(
        x=machines, y=temps,
        marker_color=colors,
        text=[f"{v:.1f}" for v in temps],
        textposition="outside",
        textfont=dict(size=11, color="#ccc"),
        showlegend=False,
        name="Temp °C",
    ), row=1, col=1)

    # Vibration — orange shades
    fig.add_trace(go.Bar(
        x=machines, y=vibs,
        marker_color=colors,
        text=[f"{v:.3f}" for v in vibs],
        textposition="outside",
        textfont=dict(size=11, color="#ccc"),
        showlegend=False,
        name="Vib mm/s",
    ), row=1, col=2)

    # Pressure — blue shades
    fig.add_trace(go.Bar(
        x=machines, y=pres,
        marker_color=colors,
        text=[f"{v:.1f}" for v in pres],
        textposition="outside",
        textfont=dict(size=11, color="#ccc"),
        showlegend=False,
        name="Pres bar",
    ), row=1, col=3)

    # threshold reference lines
    fig.add_hline(y=72,   line_dash="dot",  line_color="#ffab00", line_width=1, row=1, col=1)
    fig.add_hline(y=90,   line_dash="dash", line_color="#ff5252", line_width=1, row=1, col=1)
    fig.add_hline(y=0.85, line_dash="dot",  line_color="#ffab00", line_width=1, row=1, col=2)
    fig.add_hline(y=1.80, line_dash="dash", line_color="#ff5252", line_width=1, row=1, col=2)
    fig.add_hline(y=36,   line_dash="dot",  line_color="#ffab00", line_width=1, row=1, col=3)
    fig.add_hline(y=46,   line_dash="dash", line_color="#ff5252", line_width=1, row=1, col=3)

    fig.update_layout(
        height=280,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#ccc", size=11),
        margin=dict(l=40, r=40, t=40, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#1e2130", showgrid=True)

    return fig


def _style_fig(fig):
    fig.update_layout(
        height=400,
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#ccc", size=11),
        margin=dict(l=50, r=20, t=40, b=20),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(gridcolor="#1e2130")


# ══════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════
def _init():
    defaults = {
        "sim_started":    False,
        "results":        {},
        "agent_running":  False,
        "auto_run":       False,
        "last_auto":      0.0,
        "selected":       "M1",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# start simulator once
if not st.session_state["sim_started"]:
    simulator.start()
    st.session_state["sim_started"] = True
    time.sleep(2)


# ══════════════════════════════════════════════════════════
#  Agent runner (background thread)
# ══════════════════════════════════════════════════════════
def _run_agent(machine_id: str):
    def _worker():
        st.session_state["agent_running"] = True
        result = agent.run(machine_id)
        st.session_state["results"][machine_id] = result
        st.session_state["agent_running"] = False
    threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Maintenance AI")
    st.caption("Agentic Predictive Maintenance System")
    st.divider()

    selected = st.selectbox(
        "Inspect machine",
        MACHINES,
        index=MACHINES.index(st.session_state["selected"]),
    )
    st.session_state["selected"] = selected
    st.divider()

    if st.button(f"▶  Run Agent — {selected}",
                 disabled=st.session_state["agent_running"],
                 use_container_width=True):
        _run_agent(selected)

    if st.button("▶▶  Run Agent — All Machines",
                 disabled=st.session_state["agent_running"],
                 use_container_width=True):
        for m in MACHINES:
            _run_agent(m)

    st.divider()
    st.session_state["auto_run"] = st.toggle(
        "Auto-run every 30 s", value=st.session_state["auto_run"]
    )

    if st.session_state["agent_running"]:
        st.info("🤖 Agent is thinking…")

    st.divider()
    st.caption("Tick interval  : 5 sec")
    st.caption("Buffer size    : 20 readings")
    st.caption(f"LLM            : amazon.nova.lite")


# ══════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════
st.title("⚙️ Predictive Maintenance — Live Dashboard")

# ── Fleet health cards ────────────────────────────────────
st.subheader("Fleet Status")
all_latest = simulator.get_all_latest()
cols = st.columns(5)

for i, mid in enumerate(MACHINES):
    r = all_latest.get(mid)
    with cols[i]:
        if r:
            c = STATUS_COLOR.get(r["status"], "#aaa")
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:20px;font-weight:700">{mid}</div>
                <div style="color:{c};font-size:13px;margin:4px 0 8px">{r['status'].upper()}</div>
                <div style="font-size:12px;color:#bbb">🌡 {r['temperature']} °C</div>
                <div style="font-size:12px;color:#bbb">〰 {r['vibration']} mm/s</div>
                <div style="font-size:12px;color:#bbb">⬡ {r['pressure']} bar</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:20px;font-weight:700">{mid}</div>
                <div style="color:#888;font-size:13px;margin-top:8px">Warming up…</div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── Fleet bar chart ───────────────────────────────────────
st.subheader("Fleet Sensor Overview")
st.plotly_chart(build_fleet_bar(all_latest), use_container_width=True)

st.divider()

# ── Live sensor chart + agent thought panel ───────────────
left, right = st.columns([3, 2])

with left:
    st.subheader(f"Live Sensors — {selected}")
    st.plotly_chart(
        build_sensor_chart(selected),
        use_container_width=True,
        key=f"chart_{selected}_{int(time.time()//5)}",
    )

with right:
    st.subheader("Agent Reasoning")
    result = st.session_state["results"].get(selected)
    if not result:
        st.info("Run the agent to see its step-by-step reasoning here.")
    else:
        for step in result.get("thought_steps", []):
            t = step.get("type")
            if t == "action":
                st.markdown(
                    f'<div class="step-action"><b>ACTION</b> → '
                    f'<code>{step["tool"]}</code><br>'
                    f'<span style="color:#aaa">Input: {step["input"][:120]}</span></div>',
                    unsafe_allow_html=True,
                )
            elif t == "observation":
                preview = step["output"][:300].replace("\n", "<br>")
                st.markdown(
                    f'<div class="step-obs"><b>OBSERVATION</b><br>{preview}</div>',
                    unsafe_allow_html=True,
                )
            elif t == "final":
                st.markdown(
                    f'<div class="step-final"><b>FINAL DECISION</b><br>'
                    f'{step["output"][:400]}</div>',
                    unsafe_allow_html=True,
                )

st.divider()

# ── Agent report ──────────────────────────────────────────
st.subheader(f"Maintenance Report — {selected}")
result = st.session_state["results"].get(selected)
if result and result.get("final_report"):
    st.markdown(
        f'<div class="report-box">'
        f'{result["final_report"].replace(chr(10), "<br>")}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("No report yet. Click 'Run Agent' to generate one.")

st.divider()

# ── Action log ────────────────────────────────────────────
st.subheader("Action History")
log = agent.memory.get_log(n=30)
if log:
    df_log = pd.DataFrame(log)[["timestamp", "machine_id", "action", "detail"]]
    df_log.columns = ["Time", "Machine", "Action", "Detail"]

    def _row_color(row):
        if row["Action"] == "alert_sent":
            return ["background-color:#2a1010"] * 4
        if row["Action"] == "maintenance_scheduled":
            return ["background-color:#102a10"] * 4
        return [""] * 4

    st.dataframe(
        df_log.style.apply(_row_color, axis=1),
        use_container_width=True,
        height=240,
    )
else:
    st.info("No actions logged yet.")

st.divider()

# ── Session summary KPIs ──────────────────────────────────
st.subheader("Session Summary")
s = agent.memory.summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total agent runs",  s["total"])
c2.metric("Alerts fired",      s["alerts"])
c3.metric("Jobs scheduled",    s["jobs"])
c4.metric("Machines monitored",s["machines"])

# ── Auto-run logic ────────────────────────────────────────
if st.session_state["auto_run"]:
    now = time.time()
    if now - st.session_state["last_auto"] > AGENT_CHECK_INTERVAL:
        st.session_state["last_auto"] = now
        _run_agent(selected)

# refresh every 5 seconds to show updated sensor data
time.sleep(5)
st.rerun()
