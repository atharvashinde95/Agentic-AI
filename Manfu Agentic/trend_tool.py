# tools/trend_tool.py
from langchain.tools import tool
from simulator.simulator import simulator
from core.config import MACHINES


@tool
def detect_trend(machine_id: str) -> str:
    """
    Detects whether sensor values are rising, stable, or falling over recent readings.
    Call this when health score is borderline (40–74) to check if things are worsening.
    Input : machine_id
    Output: trend direction + rate per sensor + 5-tick projection.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    readings = simulator.get_readings(machine_id, n=10)
    if len(readings) < 5:
        return f"Not enough data for trend analysis on {machine_id}. Need at least 5 readings."

    def slope(values: list) -> float:
        n   = len(values)
        xm  = (n - 1) / 2
        ym  = sum(values) / n
        num = sum((i - xm) * (v - ym) for i, v in enumerate(values))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den else 0.0

    def label(s, unit, tol):
        if abs(s) < tol:
            return f"stable   (slope {s:+.3f}{unit}/tick)"
        return f"{'RISING' if s > 0 else 'falling'}  (slope {s:+.3f}{unit}/tick)"

    temps = [r["temperature"] for r in readings]
    vibs  = [r["vibration"]   for r in readings]
    pres  = [r["pressure"]    for r in readings]

    ts, vs, ps = slope(temps), slope(vibs), slope(pres)

    proj_temp = round(temps[-1] + ts * 5, 1)
    proj_vib  = round(vibs[-1]  + vs * 5, 3)

    concerns = []
    if ts > 0.5:
        concerns.append(f"temperature projected to reach {proj_temp}°C in 5 ticks")
    if vs > 0.05:
        concerns.append(f"vibration projected to reach {proj_vib} mm/s in 5 ticks")

    concern_str = ("\nConcerns:\n  " + "\n  ".join(concerns)) if concerns else "\nNo projection concerns."

    return (
        f"Trend analysis for {machine_id} (last {len(readings)} readings):\n"
        f"  Temperature : {label(ts, '°C',   0.30)}\n"
        f"  Vibration   : {label(vs, 'mm/s', 0.02)}\n"
        f"  Pressure    : {label(ps, 'bar',  0.15)}"
        + concern_str
    )
