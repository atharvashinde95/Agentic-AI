"""
tools.py
--------
LangChain 1.0 tool definitions for the Predictive Maintenance Agent.

Each function is decorated with @tool from langchain.tools (stable in 1.0).
The create_agent() runtime discovers, selects, and invokes these tools
autonomously via tool-calling — the LLM decides which to call and when.

Tools:
  1. check_sensor_status    — threshold monitor (perception layer)
  2. diagnose_condition     — rule-based fault classifier
  3. send_alert             — issue a warning notification
  4. schedule_maintenance   — schedule a maintenance job
  5. log_normal_cycle       — record a healthy reading
  6. get_trend_analysis     — summarise sensor trends from memory
"""

import json
import datetime

import memory_store as mem
from langchain.tools import tool    # stable in LangChain 1.0


# ─────────────────────────────────────────────────────────────────────────── #
#  Thresholds — single source of truth shared by all tools
# ─────────────────────────────────────────────────────────────────────────── #

THRESHOLDS = {
    "temperature": {"warning": 85.0,  "critical": 100.0},
    "vibration"  : {"warning":  6.0,  "critical":   9.0},
    "pressure"   : {"warning":  7.0,  "critical":   9.5},
}


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 1 — Sensor Status Checker
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def check_sensor_status(sensor_json: str) -> str:
    """
    Evaluate raw sensor readings against known thresholds and return
    the severity level for each sensor (normal / warning / critical).

    Input  : JSON string with keys: temperature (float, °C),
             vibration (float, mm/s), pressure (float, bar), tick (int).
    Output : JSON string with sensor_status dict and a plain-text summary.

    Call this tool FIRST every cycle to understand the machine state.
    Example input: '{"temperature": 92.5, "vibration": 3.1, "pressure": 4.8, "tick": 5}'
    """
    try:
        data = json.loads(sensor_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON — check your input format."})

    status = {}
    for sensor, thr in THRESHOLDS.items():
        val = float(data.get(sensor, 0.0))
        if val >= thr["critical"]:
            status[sensor] = "critical"
        elif val >= thr["warning"]:
            status[sensor] = "warning"
        else:
            status[sensor] = "normal"

    critical = [s for s, v in status.items() if v == "critical"]
    warning  = [s for s, v in status.items() if v == "warning"]

    if critical:
        summary = f"CRITICAL condition on: {', '.join(critical)}"
    elif warning:
        summary = f"WARNING condition on: {', '.join(warning)}"
    else:
        summary = "All sensors within normal operating range."

    return json.dumps({
        "sensor_status": status,
        "summary"      : summary,
        "tick"         : data.get("tick", -1),
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 2 — Fault Diagnoser
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def diagnose_condition(status_json: str) -> str:
    """
    Apply rule-based fault logic to translate severity levels into named
    fault conditions and a recommended action.

    Input  : JSON string with a 'sensor_status' dict (output of
             check_sensor_status) and optionally a 'tick' int.
    Output : JSON string with fields: faults (list of strings),
             recommended_action (Continue / Alert / Maintenance),
             risk_level (Low / Medium / High).

    Call this tool AFTER check_sensor_status to know what to do next.
    Example input: '{"sensor_status": {"temperature": "warning", "vibration": "normal", "pressure": "normal"}}'
    """
    try:
        data   = json.loads(status_json)
        status = data.get("sensor_status", {})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON — provide sensor_status dict."})

    faults = []

    # Temperature rules
    if status.get("temperature") == "critical":
        faults.append("Severe Overheating — immediate shutdown risk")
    elif status.get("temperature") == "warning":
        faults.append("Overheating — cooling system inspection required")

    # Vibration rules
    if status.get("vibration") == "critical":
        faults.append("Severe Mechanical Imbalance / Bearing Failure")
    elif status.get("vibration") == "warning":
        faults.append("Mechanical Imbalance — inspect rotating components")

    # Pressure rules
    if status.get("pressure") == "critical":
        faults.append("Critical Pressure Spike — burst risk")
    elif status.get("pressure") == "warning":
        faults.append("Elevated Pressure — check relief valve")

    # Decision logic
    n_critical = sum(1 for v in status.values() if v == "critical")
    n_warning  = sum(1 for v in status.values() if v == "warning")

    if n_critical >= 1:
        action, risk = "Maintenance", "High"
    elif n_warning >= 2:
        action, risk = "Maintenance", "High"
    elif n_warning == 1:
        action, risk = "Alert", "Medium"
    else:
        action, risk = "Continue", "Low"
        faults = ["No faults detected — machine operating normally"]

    return json.dumps({
        "faults"              : faults,
        "recommended_action"  : action,
        "risk_level"          : risk,
        "critical_sensor_count": n_critical,
        "warning_sensor_count" : n_warning,
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 3 — Alert Sender
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def send_alert(alert_json: str) -> str:
    """
    Issue a maintenance alert when a sensor has entered the WARNING range.
    Records the event to the shared event log.

    Input  : JSON string with keys: sensor (str), value (float),
             risk (str), faults (list of str), tick (int).
    Output : JSON string confirming the alert, with a message and timestamp.

    Call this tool when recommended_action from diagnose_condition is "Alert".
    Example input: '{"sensor": "temperature", "value": 88.5, "risk": "Medium", "faults": ["Overheating"], "tick": 7}'
    """
    try:
        data = json.loads(alert_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    sensor = data.get("sensor",  "unknown")
    value  = data.get("value",   0.0)
    risk   = data.get("risk",    "Medium")
    faults = data.get("faults",  [])
    tick   = data.get("tick",    -1)

    threshold = THRESHOLDS.get(sensor, {}).get("warning", "N/A")
    ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message   = (
        f"⚠️  ALERT [{risk} Risk] | {sensor.upper()} = {value} "
        f"(warning threshold: {threshold}) | {ts}"
    )

    mem.log_event(
        status="Warning", action="Alert", risk=risk,
        tool="send_alert", message=message, tick=tick,
    )
    _write_log("ALERT", message)

    return json.dumps({
        "tool"     : "send_alert",
        "status"   : "raised",
        "sensor"   : sensor,
        "value"    : value,
        "threshold": threshold,
        "risk"     : risk,
        "faults"   : faults,
        "message"  : message,
        "timestamp": ts,
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 4 — Maintenance Scheduler
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def schedule_maintenance(maintenance_json: str) -> str:
    """
    Schedule a maintenance job when a critical or multi-warning condition
    is detected. Records the job to the shared event log.

    Input  : JSON string with keys: reason (str), urgency (str:
             Immediate or Scheduled), sensor (str), tick (int).
    Output : JSON string with a unique maintenance ID and confirmation.

    Call this tool when recommended_action from diagnose_condition is "Maintenance".
    Example input: '{"reason": "Severe Overheating", "urgency": "Immediate", "sensor": "temperature", "tick": 12}'
    """
    try:
        data = json.loads(maintenance_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    reason  = data.get("reason",  "Sensor anomaly detected")
    urgency = data.get("urgency", "Scheduled")
    sensor  = data.get("sensor",  "unknown")
    tick    = data.get("tick",    -1)

    ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mnt_id = f"MNT-{datetime.datetime.now().strftime('%H%M%S')}"
    message = (
        f"🔧 MAINTENANCE [{urgency}] | ID: {mnt_id} | "
        f"Reason: {reason} | Sensor: {sensor.upper()} | {ts}"
    )

    mem.log_event(
        status="Failure", action="Maintenance", risk="High",
        tool="schedule_maintenance", message=message, tick=tick,
    )
    _write_log("MAINTENANCE", message)

    return json.dumps({
        "tool"          : "schedule_maintenance",
        "maintenance_id": mnt_id,
        "reason"        : reason,
        "urgency"       : urgency,
        "sensor"        : sensor,
        "message"       : message,
        "timestamp"     : ts,
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 5 — Normal Cycle Logger
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def log_normal_cycle(cycle_json: str) -> str:
    """
    Record a healthy sensor cycle to the audit log when all readings
    are within normal operating limits and no action is required.

    Input  : JSON string with keys: temperature (float), vibration (float),
             pressure (float), tick (int).
    Output : JSON string confirming the log entry was written.

    Call this tool when recommended_action from diagnose_condition is "Continue".
    Example input: '{"temperature": 70.2, "vibration": 2.8, "pressure": 4.6, "tick": 3}'
    """
    try:
        data = json.loads(cycle_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input."})

    temp = data.get("temperature", 0.0)
    vib  = data.get("vibration",   0.0)
    pres = data.get("pressure",    0.0)
    tick = data.get("tick",        -1)
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detail = (
        f"T={temp}°C  V={vib}mm/s  P={pres}bar | "
        f"Status: Normal | tick={tick} | {ts}"
    )

    mem.log_event(
        status="Normal", action="Continue", risk="Low",
        tool="log_normal_cycle", message=detail, tick=tick,
    )
    _write_log("NORMAL", detail)

    return json.dumps({
        "tool"     : "log_normal_cycle",
        "status"   : "logged",
        "message"  : detail,
        "timestamp": ts,
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 6 — Trend Analyser
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def get_trend_analysis(dummy: str = "") -> str:
    """
    Retrieve trend direction (rising / stable / falling) and rolling
    averages for all three sensors from the last 10 readings in memory.

    Input  : Any string (ignored). Pass an empty string "".
    Output : JSON string with per-sensor trend direction and average value,
             or a message if there is insufficient history.

    Use this tool for borderline cases or when you want richer context
    about whether sensors are trending toward a dangerous threshold.
    """
    trends = mem.compute_trends()
    if not trends:
        return json.dumps({
            "message": "Not enough history yet (need at least 2 readings)."
        })
    return json.dumps({"trends": trends})


# ─────────────────────────────────────────────────────────────────────────── #
#  Internal file logger
# ─────────────────────────────────────────────────────────────────────────── #

def _write_log(event: str, details: str, log_file: str = "agent_log.txt"):
    """Append one line to the on-disk audit log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_file, "a") as f:
            f.write(f"[{ts}] [{event}] {details}\n")
    except Exception:
        pass   # non-fatal — in-memory log still works


# ─────────────────────────────────────────────────────────────────────────── #
#  Exported list — used by agent.py
# ─────────────────────────────────────────────────────────────────────────── #

ALL_TOOLS = [
    check_sensor_status,
    diagnose_condition,
    send_alert,
    schedule_maintenance,
    log_normal_cycle,
    get_trend_analysis,
]
