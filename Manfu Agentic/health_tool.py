# tools/health_tool.py
from langchain.tools import tool
from simulator.simulator import simulator
from core.config import MACHINES, THRESHOLDS


@tool
def analyze_health(machine_id: str) -> str:
    """
    Computes a health score (0–100) for a machine and flags anomalous sensors.
    Call this SECOND after get_sensor_reading.
    Score >= 75  → healthy
    Score 40–74  → degraded, preventive maintenance needed
    Score  < 40  → critical, immediate action required
    Input : machine_id
    Output: health score, per-sensor flags, and overall assessment.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    readings = simulator.get_readings(machine_id, n=10)
    if len(readings) < 3:
        return f"Insufficient data for {machine_id}. Need at least 3 readings."

    avg_temp = sum(r["temperature"] for r in readings) / len(readings)
    avg_vib  = sum(r["vibration"]   for r in readings) / len(readings)
    avg_pres = sum(r["pressure"]    for r in readings) / len(readings)

    def score(val, warn, crit):
        if val <= warn:
            return 100
        if val >= crit:
            return max(0, int(100 - (val - crit) * 3))
        return int(100 - ((val - warn) / (crit - warn)) * 60)

    t_score = score(avg_temp, THRESHOLDS["temperature"]["warning"], THRESHOLDS["temperature"]["critical"])
    v_score = score(avg_vib,  THRESHOLDS["vibration"]["warning"],   THRESHOLDS["vibration"]["critical"])
    p_score = score(avg_pres, THRESHOLDS["pressure"]["warning"],    THRESHOLDS["pressure"]["critical"])

    # weighted: temperature most critical in industrial machines
    health = int(t_score * 0.45 + v_score * 0.35 + p_score * 0.20)

    def flag(val, warn, crit, name, unit):
        if val >= crit:
            return f"  {name}: {val}{unit}  ⚠ CRITICAL (limit: {crit})"
        if val >= warn:
            return f"  {name}: {val}{unit}  ~ WARNING  (limit: {warn})"
        return     f"  {name}: {val}{unit}  ✓ normal"

    flags = "\n".join([
        flag(round(avg_temp, 1), THRESHOLDS["temperature"]["warning"], THRESHOLDS["temperature"]["critical"], "Temperature", "°C"),
        flag(round(avg_vib,  3), THRESHOLDS["vibration"]["warning"],   THRESHOLDS["vibration"]["critical"],   "Vibration",   " mm/s"),
        flag(round(avg_pres, 1), THRESHOLDS["pressure"]["warning"],    THRESHOLDS["pressure"]["critical"],    "Pressure",    " bar"),
    ])

    if health >= 75:
        assessment = "HEALTHY — no action needed"
    elif health >= 40:
        assessment = "DEGRADED — schedule preventive maintenance soon"
    else:
        assessment = "CRITICAL — immediate intervention required"

    return (
        f"Health analysis for {machine_id}:\n"
        f"  Health score : {health}/100\n"
        f"  Assessment   : {assessment}\n"
        f"Sensor averages (last {len(readings)} readings):\n"
        f"{flags}"
    )
