"""
tools.py
--------
LangChain tool definitions for the Predictive Maintenance Agent.

Each function is decorated with @tool so LangChain's ReAct agent
can discover, select, and invoke them autonomously based on the
agent's reasoning about the current sensor state.

Tools:
  1. check_sensor_status   — Threshold monitor (perception)
  2. diagnose_condition     — Rule-based fault classifier
  3. send_alert             — Issue a warning notification
  4. schedule_maintenance   — Book a maintenance job
  5. log_normal_cycle       — Record a normal (healthy) reading
  6. get_trend_analysis     — Summarise recent sensor trends from memory
"""

import json
import datetime
import memory_store as mem

from langchain.tools import tool


# ─────────────────────────────────────────────────────────────────────────── #
#  Sensor thresholds (single source of truth)
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
    the severity level (normal / warning / critical) for each sensor.

    Input:  JSON string with keys temperature (°C), vibration (mm/s),
            pressure (bar), tick (int).
    Output: JSON string mapping each sensor name to its severity level
            plus a brief human-readable summary.

    Use this tool FIRST in every cycle to understand the current machine state.
    """
    try:
        data = json.loads(sensor_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    status = {}
    for sensor, thresholds in THRESHOLDS.items():
        value = data.get(sensor, 0.0)
        if value >= thresholds["critical"]:
            status[sensor] = "critical"
        elif value >= thresholds["warning"]:
            status[sensor] = "warning"
        else:
            status[sensor] = "normal"

    # Build a human-readable summary line
    critical_sensors = [s for s, v in status.items() if v == "critical"]
    warning_sensors  = [s for s, v in status.items() if v == "warning"]

    if critical_sensors:
        summary = f"CRITICAL condition detected on: {', '.join(critical_sensors)}"
    elif warning_sensors:
        summary = f"WARNING condition detected on: {', '.join(warning_sensors)}"
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
    Apply rule-based fault logic to translate sensor severity levels into
    named fault conditions and a recommended action.

    Input:  JSON string with a 'sensor_status' dict (output from
            check_sensor_status) and a 'tick' int.
    Output: JSON string with 'faults' (list), 'recommended_action'
            (Continue / Alert / Maintenance), and 'risk_level'
            (Low / Medium / High).

    Use this tool AFTER check_sensor_status to determine what is wrong
    and what action the agent should take next.
    """
    try:
        data   = json.loads(status_json)
        status = data.get("sensor_status", {})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    faults = []

    # Temperature rules
    if status.get("temperature") == "critical":
        faults.append("Severe Overheating — immediate shutdown risk")
    elif status.get("temperature") == "warning":
        faults.append("Overheating — cooling system check required")

    # Vibration rules
    if status.get("vibration") == "critical":
        faults.append("Severe Mechanical Imbalance / Bearing Failure")
    elif status.get("vibration") == "warning":
        faults.append("Mechanical Imbalance — inspect rotating parts")

    # Pressure rules
    if status.get("pressure") == "critical":
        faults.append("Critical Pressure Spike — burst risk")
    elif status.get("pressure") == "warning":
        faults.append("Elevated Pressure — check relief valve")

    # Decision logic
    critical_count = sum(1 for v in status.values() if v == "critical")
    warning_count  = sum(1 for v in status.values() if v == "warning")

    if critical_count >= 1:
        action = "Maintenance"
        risk   = "High"
    elif warning_count >= 2:
        action = "Maintenance"
        risk   = "High"
    elif warning_count == 1:
        action = "Alert"
        risk   = "Medium"
    else:
        action = "Continue"
        risk   = "Low"
        faults = ["No faults detected — machine operating normally"]

    return json.dumps({
        "faults"              : faults,
        "recommended_action"  : action,
        "risk_level"          : risk,
        "critical_sensor_count": critical_count,
        "warning_sensor_count" : warning_count,
    })


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool 3 — Alert Sender
# ─────────────────────────────────────────────────────────────────────────── #

@tool
def send_alert(alert_json: str) -> str:
    """
    Issue a maintenance alert when a sensor has entered the WARNING range.
    Records the alert to the event log.

    Input:  JSON string with keys:
              sensor   — which sensor triggered (e.g. "temperature")
              value    — current sensor reading
              risk     — risk level string ("Medium" or "High")
              faults   — list of fault description strings
              tick     — current simulation tick
    Output: JSON string confirming the alert was raised, with a
            human-readable message and a timestamp.

    Use this tool when recommended_action is "Alert".
    """
    try:
        data = json.loads(alert_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    sensor  = data.get("sensor",  "unknown")
    value   = data.get("value",   0.0)
    risk    = data.get("risk",    "Medium")
    faults  = data.get("faults",  [])
    tick    = data.get("tick",    -1)

    threshold = THRESHOLDS.get(sensor, {}).get("warning", "N/A")
    ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
        f"⚠️ ALERT [{risk} Risk] | {sensor.upper()} = {value} "
        f"(warning threshold: {threshold}) | {ts}"
    )

    # Write to event log and file
    mem.log_event(
        status="Warning", action="Alert", risk=risk,
        tool="send_alert", message=message, tick=tick,
    )
    _write_log_file("ALERT", message)

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
    Schedule a maintenance job for the machine when a critical or multi-warning
    condition is detected.

    Input:  JSON string with keys:
              reason  — human-readable reason for maintenance
              urgency — "Immediate" or "Scheduled"
              sensor  — primary sensor that triggered this decision
              tick    — current simulation tick
    Output: JSON string with a unique maintenance ID, confirmation message,
            and timestamp.

    Use this tool when recommended_action is "Maintenance".
    """
    try:
        data = json.loads(maintenance_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    reason  = data.get("reason",  "Sensor anomaly detected")
    urgency = data.get("urgency", "Scheduled")
    sensor  = data.get("sensor",  "unknown")
    tick    = data.get("tick",    -1)

    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mnt_id = f"MNT-{datetime.datetime.now().strftime('%H%M%S')}"

    message = (
        f"🔧 MAINTENANCE [{urgency}] | ID: {mnt_id} | "
        f"Reason: {reason} | Sensor: {sensor.upper()} | {ts}"
    )

    mem.log_event(
        status="Failure", action="Maintenance", risk="High",
        tool="schedule_maintenance", message=message, tick=tick,
    )
    _write_log_file("MAINTENANCE", message)

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
    Record a healthy / normal sensor cycle to the audit log.
    No alert or maintenance action is needed.

    Input:  JSON string with keys temperature, vibration, pressure, tick.
    Output: JSON string confirming the log entry was written.

    Use this tool when recommended_action is "Continue".
    """
    try:
        data = json.loads(cycle_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})

    temp  = data.get("temperature", 0.0)
    vib   = data.get("vibration",   0.0)
    pres  = data.get("pressure",    0.0)
    tick  = data.get("tick",        -1)
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    detail = (
        f"T={temp}°C  V={vib}mm/s  P={pres}bar | "
        f"All Normal | tick={tick} | {ts}"
    )

    mem.log_event(
        status="Normal", action="Continue", risk="Low",
        tool="log_normal_cycle", message=detail, tick=tick,
    )
    _write_log_file("NORMAL", detail)

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
def get_trend_analysis(dummy_input: str = "") -> str:
    """
    Retrieve a trend summary (rising / stable / falling) and rolling
    averages for all three sensors from the agent's memory window
    (last 10 readings).

    Input:  Any string (ignored) — call this with an empty string "".
    Output: JSON string with per-sensor trend direction and average value.

    Use this tool when you need context about whether sensor values are
    trending toward danger, or to provide a richer explanation to the user.
    """
    trends = mem.compute_trends()
    if not trends:
        return json.dumps({"message": "Insufficient history for trend analysis (need ≥2 readings)"})

    return json.dumps({"trends": trends})


# ─────────────────────────────────────────────────────────────────────────── #
#  Internal helper
# ─────────────────────────────────────────────────────────────────────────── #

def _write_log_file(event: str, details: str, log_file: str = "agent_log.txt"):
    """Append one line to the persistent log file on disk."""
    ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{ts}] [{event}] {details}\n"
    try:
        with open(log_file, "a") as f:
            f.write(log_line)
    except Exception as e:
        pass  # Non-fatal — UI log is still updated via memory_store
