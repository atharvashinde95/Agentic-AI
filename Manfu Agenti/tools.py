# ─────────────────────────────────────────────
#  tools.py  —  5 LangChain tools for the agent
# ─────────────────────────────────────────────

import random
import string
from datetime import datetime, timedelta
from langchain.tools import tool

from simulator import simulator
from memory    import memory
from config    import THRESHOLDS, MACHINES


# ── Helper ───────────────────────────────────────
def _rand_id(prefix: str, n: int = 6) -> str:
    return prefix + "-" + "".join(random.choices(string.digits, k=n))


# ════════════════════════════════════════════════
#  Tool 1 — get_sensor_reading
# ════════════════════════════════════════════════
@tool
def get_sensor_reading(machine_id: str) -> str:
    """
    Fetches the last 10 real-time sensor readings for a machine.
    Always call this tool FIRST before any analysis.
    Input: machine_id  (e.g. 'M1', 'M2', 'M3', 'M4', 'M5')
    Returns: compact text summary of last 10 readings.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'. Valid: {MACHINES}"

    readings = simulator.get_readings(machine_id, n=10)

    if not readings:
        return f"No data yet for {machine_id}. Simulator may still be warming up."

    lines = [f"Sensor readings for {machine_id} (latest {len(readings)} ticks):"]
    for r in readings:
        lines.append(
            f"  [{r['timestamp']}]  "
            f"temp={r['temperature']}°C  "
            f"vib={r['vibration']}mm/s  "
            f"pres={r['pressure']}bar  "
            f"status={r['status']}"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════
#  Tool 2 — analyze_health
# ════════════════════════════════════════════════
@tool
def analyze_health(machine_id: str) -> str:
    """
    Computes a health score (0-100) and flags which sensors are anomalous.
    Call this SECOND after get_sensor_reading.
    Input: machine_id
    Returns: health score, per-sensor status flags, and overall assessment.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    readings = simulator.get_readings(machine_id, n=10)
    if len(readings) < 3:
        return f"Insufficient data for {machine_id}. Need at least 3 readings."

    # compute averages over last 10 readings
    avg_temp = sum(r["temperature"] for r in readings) / len(readings)
    avg_vib  = sum(r["vibration"]   for r in readings) / len(readings)
    avg_pres = sum(r["pressure"]    for r in readings) / len(readings)

    # score each sensor: 100 = perfect, 0 = extreme failure
    def sensor_score(value, warn_thresh, crit_thresh, baseline):
        if value <= warn_thresh:
            return 100
        if value >= crit_thresh:
            return max(0, int(100 - (value - crit_thresh) * 3))
        ratio = (value - warn_thresh) / (crit_thresh - warn_thresh)
        return int(100 - ratio * 60)

    t_score = sensor_score(avg_temp, THRESHOLDS["temperature"]["warning"],
                           THRESHOLDS["temperature"]["critical"], 65)
    v_score = sensor_score(avg_vib,  THRESHOLDS["vibration"]["warning"],
                           THRESHOLDS["vibration"]["critical"], 0.5)
    p_score = sensor_score(avg_pres, THRESHOLDS["pressure"]["warning"],
                           THRESHOLDS["pressure"]["critical"], 30)

    # overall health = weighted average (temp most important in industrial machines)
    health = int(t_score * 0.45 + v_score * 0.35 + p_score * 0.20)

    # flag each sensor
    def flag(value, warn, crit, name, unit):
        if value >= crit:
            return f"  {name}: {value}{unit}  ⚠ CRITICAL (threshold: {crit})"
        if value >= warn:
            return f"  {name}: {value}{unit}  ~ WARNING  (threshold: {warn})"
        return f"  {name}: {value}{unit}  ✓ normal"

    flags = [
        flag(round(avg_temp, 1), THRESHOLDS["temperature"]["warning"],
             THRESHOLDS["temperature"]["critical"], "Temperature", "°C"),
        flag(round(avg_vib,  3), THRESHOLDS["vibration"]["warning"],
             THRESHOLDS["vibration"]["critical"],   "Vibration",   "mm/s"),
        flag(round(avg_pres, 1), THRESHOLDS["pressure"]["warning"],
             THRESHOLDS["pressure"]["critical"],    "Pressure",    "bar"),
    ]

    if health >= 75:
        assessment = "HEALTHY — no immediate action needed"
    elif health >= 45:
        assessment = "DEGRADED — schedule preventive maintenance soon"
    else:
        assessment = "CRITICAL — immediate intervention required"

    return (
        f"Health analysis for {machine_id}:\n"
        f"  Health score : {health}/100\n"
        f"  Assessment   : {assessment}\n"
        f"Sensor averages (last {len(readings)} readings):\n"
        + "\n".join(flags)
    )


# ════════════════════════════════════════════════
#  Tool 3 — detect_trend
# ════════════════════════════════════════════════
@tool
def detect_trend(machine_id: str) -> str:
    """
    Detects whether sensor values are rising, stable, or falling over recent readings.
    Call this when health score is borderline (40-75) to check if things are worsening.
    Input: machine_id
    Returns: trend direction and rate for each sensor.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    readings = simulator.get_readings(machine_id, n=10)
    if len(readings) < 5:
        return f"Not enough data for trend analysis on {machine_id}. Need 5+ readings."

    def linear_slope(values: list[float]) -> float:
        """Simple least-squares slope."""
        n   = len(values)
        x   = list(range(n))
        x_m = sum(x) / n
        y_m = sum(values) / n
        num = sum((xi - x_m) * (yi - y_m) for xi, yi in zip(x, values))
        den = sum((xi - x_m) ** 2 for xi in x)
        return num / den if den != 0 else 0.0

    def trend_label(slope: float, unit: str, threshold: float) -> str:
        if abs(slope) < threshold:
            return f"stable  (slope: {slope:+.3f}{unit}/tick)"
        if slope > 0:
            return f"RISING  (slope: {slope:+.3f}{unit}/tick)"
        return f"falling (slope: {slope:+.3f}{unit}/tick)"

    temps  = [r["temperature"] for r in readings]
    vibs   = [r["vibration"]   for r in readings]
    pres   = [r["pressure"]    for r in readings]

    t_slope = linear_slope(temps)
    v_slope = linear_slope(vibs)
    p_slope = linear_slope(pres)

    # projected values in 5 ticks
    projected_temp = round(temps[-1] + t_slope * 5, 1)
    projected_vib  = round(vibs[-1]  + v_slope * 5, 3)

    concern = []
    if t_slope > 0.5:
        concern.append(f"temperature projected to reach {projected_temp}°C in 5 ticks")
    if v_slope > 0.05:
        concern.append(f"vibration projected to reach {projected_vib}mm/s in 5 ticks")

    concern_str = ("\nConcerns:\n  " + "\n  ".join(concern)) if concern else "\nNo immediate projection concerns."

    return (
        f"Trend analysis for {machine_id} (last {len(readings)} readings):\n"
        f"  Temperature : {trend_label(t_slope, '°C',   0.3)}\n"
        f"  Vibration   : {trend_label(v_slope, 'mm/s', 0.02)}\n"
        f"  Pressure    : {trend_label(p_slope, 'bar',  0.15)}"
        + concern_str
    )


# ════════════════════════════════════════════════
#  Tool 4 — schedule_maintenance
# ════════════════════════════════════════════════
@tool
def schedule_maintenance(machine_id: str, urgency: str, reason: str) -> str:
    """
    Books a preventive maintenance job for a machine.
    Call when health score is 40-74 (degraded) or trend is worsening.
    Input:
      machine_id : e.g. 'M1'
      urgency    : 'low' | 'medium' | 'high'
      reason     : brief explanation of why maintenance is needed
    Returns: job confirmation with ID and scheduled time.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    urgency = urgency.lower().strip()
    delay_hours = {"low": 48, "medium": 12, "high": 4}.get(urgency, 24)

    scheduled_time = (datetime.now() + timedelta(hours=delay_hours)).strftime(
        "%Y-%m-%d %H:%M"
    )
    job_id    = _rand_id("MNT")
    engineers = ["Rahul Sharma", "Priya Nair", "Amit Verma", "Sunita Patel"]
    engineer  = random.choice(engineers)

    # log to memory
    memory.log_action(
        machine_id=machine_id,
        action="maintenance_scheduled",
        severity=urgency,
        reason=reason,
        job_id=job_id,
    )
    memory.record_decision(machine_id, {
        "action":   "maintenance_scheduled",
        "job_id":   job_id,
        "urgency":  urgency,
    })

    # trigger recovery in simulator
    if urgency == "high":
        simulator.trigger_recovery(machine_id)

    return (
        f"Maintenance scheduled for {machine_id}:\n"
        f"  Job ID        : {job_id}\n"
        f"  Urgency       : {urgency.upper()}\n"
        f"  Scheduled at  : {scheduled_time}\n"
        f"  Assigned to   : {engineer}\n"
        f"  Reason        : {reason}\n"
        f"  Status        : CONFIRMED"
    )


# ════════════════════════════════════════════════
#  Tool 5 — send_alert
# ════════════════════════════════════════════════
@tool
def send_alert(machine_id: str, severity: str, reason: str) -> str:
    """
    Fires an immediate critical alert for a machine requiring urgent intervention.
    Call when health score < 40 OR any sensor is in critical zone.
    Input:
      machine_id : e.g. 'M1'
      severity   : 'warning' | 'critical' | 'emergency'
      reason     : detailed explanation of the failure condition
    Returns: alert confirmation with ID and notified contacts.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    # check memory — don't re-alert for same machine too soon
    if memory.was_recently_alerted(machine_id, within_n=2):
        return (
            f"Alert suppressed for {machine_id} — already alerted recently.\n"
            f"Check action log for previous alert details."
        )

    alert_id  = _rand_id("ALT")
    severity  = severity.lower().strip()

    contacts = {
        "warning":   ["floor-supervisor@factory.com"],
        "critical":  ["floor-supervisor@factory.com", "maintenance-lead@factory.com"],
        "emergency": ["floor-supervisor@factory.com", "maintenance-lead@factory.com",
                      "plant-manager@factory.com", "+91-9876543210 (on-call)"],
    }
    notified = contacts.get(severity, contacts["critical"])

    # log to memory
    memory.log_action(
        machine_id=machine_id,
        action="alert_sent",
        severity=severity,
        reason=reason,
        alert_id=alert_id,
    )
    memory.record_decision(machine_id, {
        "action":   "alert_sent",
        "alert_id": alert_id,
        "severity": severity,
    })

    # trigger recovery in simulator for emergency
    if severity == "emergency":
        simulator.trigger_recovery(machine_id)

    return (
        f"ALERT FIRED for {machine_id}:\n"
        f"  Alert ID   : {alert_id}\n"
        f"  Severity   : {severity.upper()}\n"
        f"  Reason     : {reason}\n"
        f"  Notified   : {', '.join(notified)}\n"
        f"  Fired at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Status     : DELIVERED"
    )


# ── Export all tools as a list for agent ─────────
ALL_TOOLS = [
    get_sensor_reading,
    analyze_health,
    detect_trend,
    schedule_maintenance,
    send_alert,
]
