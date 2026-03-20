# dashboard/app.py
import sys, os, re, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.simulator import simulator
from agent.agent         import agent
from core.config         import MACHINES, THRESHOLDS

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

st.markdown("""
<style>
.card {
    background:#1e2130; border-radius:12px; padding:1rem;
    border:1px solid #2e3250; text-align:center; margin-bottom:6px;
}
.card-name   { font-size:22px; font-weight:700; }
.card-status { font-size:13px; font-weight:600; margin:4px 0 8px; }
.card-val    { font-size:12px; color:#aaa; line-height:1.9; }
.box-tool  { background:#0d2b0d; border-left:4px solid #00c853;
    padding:7px 12px; border-radius:0 8px 8px 0; margin:3px 0; font-size:13px; }
.box-obs   { background:#0d0d2b; border-left:4px solid #4488ff;
    padding:7px 12px; border-radius:0 8px 8px 0; margin:3px 0; font-size:13px; }
.box-final { background:#1a0d2b; border-left:4px solid #bb86fc;
    padding:7px 12px; border-radius:0 8px 8px 0; margin:3px 0; font-size:13px; }
.report {
    background:#1e2130; border-radius:10px; padding:1rem 1.2rem;
    border:1px solid #2e3250; font-size:14px; line-height:1.9;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  Colours
# ─────────────────────────────────────────────────────────
COLOR = {
    "normal":     "#00c853",
    "warning":    "#ffab00",
    "failure":    "#ff5252",
    "recovering": "#40c4ff",
    "degrading":  "#ffab00",
    "critical":   "#ff5252",
}

# ─────────────────────────────────────────────────────────
#  Thread-safe result store (never touch session_state from a thread)
# ─────────────────────────────────────────────────────────
_store: dict    = {}
_lock           = threading.Lock()
_running        = threading.Event()

def _save(mid, result):
    with _lock:
        _store[mid] = result

def _load(mid):
    with _lock:
        return _store.get(mid)

# ─────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state["started"] = False

if not st.session_state["started"]:
    simulator.start()
    st.session_state["started"] = True
    time.sleep(2)

# ─────────────────────────────────────────────────────────
#  Agent runner helpers
# ─────────────────────────────────────────────────────────
def _run_one(mid: str):
    """Run agent for a single machine in a background thread."""
    if _running.is_set():
        return
    def _work():
        _running.set()
        try:
            _save(mid, agent.run(mid))
        finally:
            _running.clear()
    threading.Thread(target=_work, daemon=True).start()


def _run_all():
    """Run agent for every machine sequentially in a background thread."""
    if _running.is_set():
        return
    def _work():
        _running.set()
        try:
            for mid in MACHINES:
                _save(mid, agent.run(mid))
        finally:
            _running.clear()
    threading.Thread(target=_work, daemon=True).start()

# ─────────────────────────────────────────────────────────
#  Chart helpers
# ─────────────────────────────────────────────────────────
def _style(fig, h=380):
    fig.update_layout(
        height=h, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(color="#ccc", size=11),
        margin=dict(l=50, r=20, t=36, b=20),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(gridcolor="#1e2130")


def sensor_chart(mid: str) -> go.Figure:
    rows = simulator.get_readings(mid, n=20)
    fig  = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
        vertical_spacing=0.10,
    )
    if not rows:
        fig.add_annotation(text="Warming up…", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        _style(fig); return fig

    df  = pd.DataFrame(rows)
    pts = df["status"].map(
        {"normal":"#00c853","warning":"#ffab00","failure":"#ff5252"}
    ).fillna("#aaa")

    def _add(col, row, color, warn, crit):
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col], mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=pts, size=6), showlegend=False,
        ), row=row, col=1)
        fig.add_hline(y=warn, line_dash="dot",  line_color="#ffab00",
                      line_width=1, row=row, col=1)
        fig.add_hline(y=crit, line_dash="dash", line_color="#ff5252",
                      line_width=1, row=row, col=1)

    _add("temperature", 1, "#ef5350",
         THRESHOLDS["temperature"]["warning"], THRESHOLDS["temperature"]["critical"])
    _add("vibration",   2, "#ffa726",
         THRESHOLDS["vibration"]["warning"],   THRESHOLDS["vibration"]["critical"])
    _add("pressure",    3, "#42a5f5",
         THRESHOLDS["pressure"]["warning"],    THRESHOLDS["pressure"]["critical"])
    _style(fig); return fig


def fleet_chart(latest: dict) -> go.Figure:
    mids, temps, vibs, pres, cols = [], [], [], [], []
    for mid, r in latest.items():
        if r:
            mids.append(mid)
            temps.append(r["temperature"])
            vibs.append(r["vibration"])
            pres.append(r["pressure"])
            cols.append(COLOR.get(r["status"], "#aaa"))

    if not mids:
        fig = go.Figure()
        fig.add_annotation(text="Warming up…", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.08,
        subplot_titles=("Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)"),
    )
    for ci, (data, lbl) in enumerate([(temps,"Temp"),(vibs,"Vib"),(pres,"Pres")], 1):
        fig.add_trace(go.Bar(
            x=mids, y=data, marker_color=cols, name=lbl,
            text=[f"{v:.2f}" for v in data],
            textposition="outside",
            textfont=dict(size=11, color="#ccc"),
            showlegend=False,
        ), row=1, col=ci)

    thresholds = [(1, 72, 90), (2, 0.85, 1.80), (3, 36, 46)]
    for ci, warn, crit in thresholds:
        fig.add_hline(y=warn, line_dash="dot",  line_color="#ffab00",
                      line_width=1, row=1, col=ci)
        fig.add_hline(y=crit, line_dash="dash", line_color="#ff5252",
                      line_width=1, row=1, col=ci)

    _style(fig, h=260)
    return fig


def _clean(text: str) -> str:
    """Strip <thinking>...</thinking> blocks from LLM output."""
    return re.sub(r"<thinking>.*?</thinking>", "", text,
                  flags=re.DOTALL).strip()


# ═════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ AI Maintenance")
    st.caption("Real-time predictive maintenance powered by an AI agent.")
    st.divider()

    busy    = _running.is_set()
    latest_ = simulator.get_all_latest()

    # figure out which machines actually need attention right now
    needs_attention = [
        mid for mid in MACHINES
        if latest_.get(mid) and latest_[mid]["status"] in ("warning", "failure")
    ]

    # ── LLM call counter ──────────────────────
    est_all   = len(MACHINES) * 4
    est_smart = len(needs_attention) * 4 if needs_attention else 0

    st.markdown("#### 🔋 LLM Call Estimator")
    col1, col2 = st.columns(2)
    col1.metric("Run All",   f"~{est_all} calls")
    col2.metric("Run Smart", f"~{est_smart} calls",
                delta=f"-{est_all - est_smart}" if est_smart < est_all else "same")
    st.divider()

    # ── Smart run (warning/failure only) ──────
    if needs_attention:
        if st.button("⚡  Run Smart — " + ", ".join(needs_attention),
                     disabled=busy, use_container_width=True, type="primary"):
            def _run_smart():
                _running.set()
                try:
                    for mid in needs_attention:
                        _save(mid, agent.run(mid))
                finally:
                    _running.clear()
            threading.Thread(target=_run_smart, daemon=True).start()
        st.caption(f"Only checks machines needing attention. "
                   f"Saves ~{est_all - est_smart} LLM calls.")
    else:
        st.success("✅ All machines healthy — no urgent runs needed.")

    st.divider()

    # ── Run ALL machines ──────────────────────
    if st.button("▶▶  Run Agent — ALL 5 Machines",
                 disabled=busy, use_container_width=True):
        _run_all()
    st.caption(f"Uses ~{est_all} LLM calls total.")

    st.divider()

    # ── Run individual machine ────────────────
    st.markdown("**Run one machine at a time:**")
    for mid in MACHINES:
        r      = latest_.get(mid)
        status = r["status"] if r else "unknown"
        emoji  = {"normal":"🟢","warning":"🟡","failure":"🔴",
                  "recovering":"🔵"}.get(status, "⚪")
        est    = 3 if status == "normal" else 4
        if st.button(f"{emoji}  {mid} — {status.upper()}  (~{est} calls)",
                     key=f"btn_{mid}", disabled=busy,
                     use_container_width=True):
            _run_one(mid)

    st.divider()

    if busy:
        st.warning("🤖 Agent is running… please wait.")
    else:
        st.info("💡 Use Run Smart to save API calls.")

    st.divider()
    st.markdown("**Legend:**")
    st.markdown("🟢 Normal  🟡 Warning  🔴 Failure  🔵 Recovering")
    st.markdown("Dotted line = warning threshold")
    st.markdown("Dashed line = critical threshold")


# ═════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═════════════════════════════════════════════════════════
st.title("⚙️ Predictive Maintenance — Live Dashboard")

# ── SECTION 1: Fleet status cards ────────────────────────
st.subheader("🏭  Fleet Status  —  All Machines")
st.caption("Updates every 5 seconds automatically.")

latest = simulator.get_all_latest()
fleet_cols = st.columns(5)
for i, mid in enumerate(MACHINES):
    r = latest.get(mid)
    with fleet_cols[i]:
        if r:
            c = COLOR.get(r["status"], "#aaa")
            has_result = _load(mid) is not None
            badge = "✅ Checked" if has_result else "⏳ Not checked"
            st.markdown(f"""
            <div class="card">
              <div class="card-name">{mid}</div>
              <div class="card-status" style="color:{c}">{r['status'].upper()}</div>
              <div class="card-val">
                🌡 {r['temperature']} °C<br>
                〰 {r['vibration']} mm/s<br>
                ⬡ {r['pressure']} bar<br>
                <span style="color:#666;font-size:11px">{badge}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
              <div class="card-name">{mid}</div>
              <div style="color:#555;font-size:12px;margin-top:12px">Starting…</div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── SECTION 2: Fleet sensor overview ─────────────────────
st.subheader("📊  Fleet Sensor Overview")
st.caption("All 5 machines compared. Dotted = warning.  Dashed = critical.")
st.plotly_chart(fleet_chart(latest), use_container_width=True)

st.divider()

# ── SECTION 3: Per-machine tabs ───────────────────────────
st.subheader("📈  Machine Detail — Live Sensors & Agent Report")
st.caption("Click a tab to view any machine. Run the agent first to see the report.")

tabs = st.tabs([f"  {m}  " for m in MACHINES])

for i, mid in enumerate(MACHINES):
    with tabs[i]:
        r_live = latest.get(mid)

        # live status pill
        if r_live:
            c = COLOR.get(r_live["status"], "#aaa")
            st.markdown(
                f'<span style="background:{c}22;color:{c};padding:4px 12px;'
                f'border-radius:20px;font-size:13px;font-weight:600;border:1px solid {c}">'
                f'{r_live["status"].upper()}</span>&nbsp;&nbsp;'
                f'<span style="color:#888;font-size:12px">'
                f'Temp: {r_live["temperature"]}°C  |  '
                f'Vib: {r_live["vibration"]} mm/s  |  '
                f'Pres: {r_live["pressure"]} bar</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

        # live sensor chart
        st.plotly_chart(
            sensor_chart(mid),
            use_container_width=True,
            key=f"sc_{mid}_{int(time.time()//5)}",
        )

        # agent report for this machine
        result = _load(mid)
        if not result:
            st.info(f"No agent report yet for {mid}. "
                    f"Click **▶  {mid}** in the sidebar or **Run All**.")
        else:
            report = _clean(result.get("final_report", ""))
            st.markdown(
                f'<div class="report">{report.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

            steps = result.get("thought_steps", [])
            if steps:
                with st.expander("🔍  See agent reasoning step by step"):
                    for step in steps:
                        t = step.get("type")
                        if t == "action":
                            st.markdown(
                                f'<div class="box-tool">'
                                f'<b>TOOL CALLED:</b> <code>{step["tool"]}</code><br>'
                                f'<span style="color:#aaa">Input: '
                                f'{str(step["input"])[:150]}</span></div>',
                                unsafe_allow_html=True)
                        elif t == "observation":
                            prev = step["output"][:400].replace("\n", "<br>")
                            st.markdown(
                                f'<div class="box-obs"><b>TOOL RESULT:</b>'
                                f'<br>{prev}</div>',
                                unsafe_allow_html=True)
                        elif t == "final":
                            cf = _clean(step.get("output",""))[:400]
                            st.markdown(
                                f'<div class="box-final"><b>DECISION:</b>'
                                f'<br>{cf}</div>',
                                unsafe_allow_html=True)

st.divider()

# ── SECTION 4: Action history ─────────────────────────────
st.subheader("📋  Action History — All Machines")
st.caption("Every action the agent has taken this session.")

log = agent.memory.get_log(n=30)
if log:
    df_log = pd.DataFrame(log)[["timestamp","machine_id","action","detail"]]
    df_log.columns = ["Time", "Machine", "Action", "Detail"]
    df_log["Detail"] = df_log["Detail"].str.replace(
        r"<thinking>.*?</thinking>", "", regex=True).str.strip()
    df_log["Detail"] = df_log["Detail"].str[:120]

    def _row_color(row):
        if row["Action"] == "alert_sent":
            return ["background-color:#2a1010"] * 4
        if row["Action"] == "maintenance_scheduled":
            return ["background-color:#102a10"] * 4
        return [""] * 4

    st.dataframe(
        df_log.style.apply(_row_color, axis=1),
        use_container_width=True, height=220,
    )

    s = agent.memory.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total agent runs",    s["total"])
    c2.metric("Alerts fired",        s["alerts"])
    c3.metric("Jobs scheduled",      s["jobs"])
    c4.metric("Machines checked",    s["machines"])
else:
    st.info("No actions yet. Run the agent to see results here.")

# ── auto-refresh every 5 seconds ──────────────────────────
time.sleep(5)
st.rerun()
