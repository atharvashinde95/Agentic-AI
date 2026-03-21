"""
tools.py
---------
Defines every tool the Maintenance Agent can call, using the
official LangChain @tool decorator.

Tool names here exactly match the tool names referenced in the
system prompt inside agent.py. This alignment is critical —
if the system prompt says "call immediate_shutdown" but the tool
is named "shutdown_machine", LangGraph won't route correctly.

Tool name → System prompt action label mapping:
  monitor_closely      → "Monitor closely"
  reduce_machine_load  → "Reduce machine load"
  schedule_maintenance → "Schedule maintenance"
  shift_to_backup      → "Shift production to backup machine"
  immediate_shutdown   → "Immediate shutdown required"
"""

from langchain_core.tools import tool


# ──────────────────────────────────────────────────────────────
# TOOL 1 — monitor_closely
# Triggered by: Low risk / healthy machine
# ──────────────────────────────────────────────────────────────
@tool
def monitor_closely() -> str:
    """
    Monitor the machine closely by logging all current sensor readings to the dashboard.
    Use this tool when the machine is healthy (Low risk, no anomalies detected),
    or as a final confirmation step after corrective actions have been applied.
    """
    return (
        "ACTION EXECUTED — Monitor Closely: "
        "All sensor readings have been logged to the dashboard. "
        "Machine is operating within safe parameters. "
        "Continuous monitoring enabled. No intervention required at this time."
    )


# ──────────────────────────────────────────────────────────────
# TOOL 2 — reduce_machine_load
# Triggered by: Medium or High risk
# ──────────────────────────────────────────────────────────────
@tool
def reduce_machine_load() -> str:
    """
    Reduce the machine's workload by 30% to lower thermal stress and vibration.
    Use this tool when temperature > 85°C, vibration > 7.0, or load > 85%.
    Helps prevent further degradation and buys time for maintenance.
    """
    return (
        "ACTION EXECUTED — Reduce Machine Load: "
        "Workload reduced by 30%. Thermal output decreasing. "
        "Vibration levels stabilising. Expected temperature drop of 8–12°C over next 10 minutes. "
        "Machine now operating at 70% capacity."
    )


# ──────────────────────────────────────────────────────────────
# TOOL 3 — schedule_maintenance
# Triggered by: High risk
# ──────────────────────────────────────────────────────────────
@tool
def schedule_maintenance() -> str:
    """
    Schedule an emergency maintenance visit and alert the technician team immediately.
    Use this tool when risk is High and the machine requires physical inspection and servicing.
    """
    return (
        "ACTION EXECUTED — Schedule Maintenance: "
        "Maintenance ticket #MT-2847 created. "
        "Emergency slot booked within 24 hours. "
        "Lead technician R. Sharma notified via SMS and email. "
        "Machine tagged for full mechanical inspection."
    )


# ──────────────────────────────────────────────────────────────
# TOOL 4 — shift_to_backup
# Triggered by: High risk
# ──────────────────────────────────────────────────────────────
@tool
def shift_to_backup() -> str:
    """
    Shift production workload from the primary machine to Backup Unit B.
    Use this tool for High risk situations to relieve the primary machine
    while maintaining overall production continuity.
    """
    return (
        "ACTION EXECUTED — Shift Production to Backup Machine: "
        "Production successfully rerouted to Backup Unit B. "
        "Backup unit activated and running at 94% efficiency. "
        "Primary machine load reduced to 15%. Production continuity maintained."
    )


# ──────────────────────────────────────────────────────────────
# TOOL 5 — immediate_shutdown
# Triggered by: EMERGENCY — temperature > 100°C OR vibration > 9.0
# Must always be called FIRST when emergency conditions exist
# ──────────────────────────────────────────────────────────────
@tool
def immediate_shutdown() -> str:
    """
    Execute an immediate emergency shutdown of the machine to prevent catastrophic failure or injury.
    Use this tool FIRST and IMMEDIATELY when temperature exceeds 100°C or vibration exceeds 9.0.
    This is the highest priority safety action — always call this before any other tool in emergencies.
    """
    return (
        "ACTION EXECUTED — Immediate Shutdown Required: "
        "EMERGENCY SHUTDOWN EXECUTED. All moving parts halted immediately. "
        "Safety interlock engaged. Power supply cut. "
        "Incident log #INC-9921 filed. Plant manager and safety team notified. "
        "Machine must not be restarted until cleared by a certified technician."
    )


# ──────────────────────────────────────────────────────────────
# TOOL LIST — passed to agent and LLM via create_react_agent
# ──────────────────────────────────────────────────────────────
ALL_TOOLS = [
    monitor_closely,
    reduce_machine_load,
    schedule_maintenance,
    shift_to_backup,
    immediate_shutdown,
]

