# tools/sensor_tool.py
from langchain.tools import tool
from simulator.simulator import simulator
from core.config import MACHINES


@tool
def get_sensor_reading(machine_id: str) -> str:
    """
    Fetches the last 10 real-time sensor readings for a given machine.
    ALWAYS call this tool FIRST before any analysis.
    Input : machine_id — one of M1, M2, M3, M4, M5
    Output: compact text summary of the last 10 readings.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'. Valid machines: {MACHINES}"

    readings = simulator.get_readings(machine_id, n=10)

    if not readings:
        return f"No data yet for {machine_id}. Simulator is still warming up — wait a few seconds."

    lines = [f"Live sensor readings for {machine_id} (last {len(readings)} ticks):"]
    for r in readings:
        lines.append(
            f"  [{r['timestamp']}]  "
            f"temp={r['temperature']}°C  "
            f"vib={r['vibration']} mm/s  "
            f"pres={r['pressure']} bar  "
            f"status={r['status']}"
        )
    return "\n".join(lines)
