import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from tools import all_tools
from agent.agent_runner import build_agent, run_agent

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Industrial AI Agent",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Industrial Production Agent")
st.caption("Powered by Amazon Nova Lite · Capgemini Generative Engine")

# ── Session state ─────────────────────────────────────────────────────────────

if "agent_executor" not in st.session_state:
    with st.spinner("Initialising agent..."):
        st.session_state.agent_executor = build_agent(all_tools)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Layout: chat left, plant status right ────────────────────────────────────

chat_col, status_col = st.columns([2, 1])

# ── LEFT: Chat interface ──────────────────────────────────────────────────────

with chat_col:
    st.subheader("Production Feasibility Chat")
    st.markdown("""
    **Try asking:**
    - *Can we produce 5 batches of Product_A?*
    - *What is the maximum number of Product_B batches we can run?*
    - *What are the current process sensor readings?*
    - *Which products can we produce today?*
    """)
    st.divider()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about production feasibility...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning over plant data..."):
                response = run_agent(st.session_state.agent_executor, user_input)
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

# ── RIGHT: Live plant status (mirrors the SCADA image) ───────────────────────

with status_col:
    from opc_ua_simulator import opc_ua_simulator

    plant = opc_ua_simulator.get_full_plant_status()

    st.subheader("Live Plant Status")

    # ── Storage tanks ─────────────────────────────────────────────────────
    st.markdown("**Raw Material Tanks**")
    for tank_id, info in plant["tanks"].items():
        pct = info["fill_percent"]
        st.caption(f"{tank_id.replace('_', ' ').title()} · {info['material']}")
        st.progress(pct / 100, text=f"{pct}%  —  {info['level_litres']:,.3f} L")

    st.divider()

    # ── Mixer ─────────────────────────────────────────────────────────────
    st.markdown("**Mixer**")
    mixer = plant["mixer"]
    mixer_icon = "🟢" if mixer["status"] == "running" else "🔴"
    st.write(f"{mixer_icon} Status: `{mixer['status']}`")
    st.progress(mixer["fill_percent"] / 100, text=f"Fill: {mixer['fill_percent']}%")

    st.divider()

    # ── Pump ──────────────────────────────────────────────────────────────
    st.markdown("**Pump**")
    pump = plant["pump"]
    pump_icon = "🟢" if pump["status"] == "running" else "🔴"
    st.write(f"{pump_icon} Status: `{pump['status']}`")
    st.write(f"Speed: `{pump['speed_percent']}%`")

    st.divider()

    # ── Reactor ───────────────────────────────────────────────────────────
    st.markdown("**Reactor**")
    reactor = plant["reactor"]
    reactor_icon = "🟢" if reactor["status"] == "running" else "🟡" if reactor["status"] == "idle" else "🔴"
    st.write(f"{reactor_icon} Status: `{reactor['status']}`")
    st.progress(reactor["fill_percent"] / 100, text=f"Fill: {reactor['fill_percent']}%")

    st.divider()

    # ── Process sensors (digital readouts from image) ─────────────────────
    st.markdown("**Process Sensors**")
    sensors = plant["process_sensors"]

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.metric("PH",          f"{sensors['ph']['value']}")
        st.metric("Temp (°C)",   f"{sensors['temperature']['value']}")
    with s_col2:
        st.metric("Pressure (Kpa)", f"{sensors['pressure']['value']}")
        st.metric("Energy (kWh)",   f"{sensors['energy']['value']}")

