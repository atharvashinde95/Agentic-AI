"""
memory_store.py
---------------
A lightweight shared in-process store.

The LangChain tools need access to:
  1. Recent sensor readings  (for trend analysis inside tools)
  2. An append-only event log (so the UI can display it)

We keep these as module-level singletons so both the agent and the tools
can import and mutate them without passing objects around.
"""

from collections import deque
import datetime

# ── Sensor history (last 10 readings) ───────────────────────────────────── #
SENSOR_HISTORY: deque = deque(maxlen=10)

# ── Event log (append-only, newest first for UI) ─────────────────────────── #
EVENT_LOG: list = []

# ── Latest agent result (for Streamlit to pick up) ───────────────────────── #
LATEST_RESULT: dict = {}


def add_reading(reading: dict):
    """Push a new sensor reading onto the history buffer."""
    SENSOR_HISTORY.append(reading)


def get_history() -> list:
    return list(SENSOR_HISTORY)


def compute_trends() -> dict:
    """
    Return average + direction (rising / stable / falling) per sensor
    from the stored history window.
    """
    data = list(SENSOR_HISTORY)
    if len(data) < 2:
        return {}

    summary = {}
    for key in ["temperature", "vibration", "pressure"]:
        values = [r[key] for r in data]
        avg    = round(sum(values) / len(values), 2)
        if values[-1] > values[0] * 1.05:
            trend = "rising"
        elif values[-1] < values[0] * 0.95:
            trend = "falling"
        else:
            trend = "stable"
        summary[key] = {"average": avg, "trend": trend}

    return summary


def log_event(status: str, action: str, risk: str, tool: str, message: str, tick: int):
    """Prepend a new event to the global event log (newest first)."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    EVENT_LOG.insert(0, {
        "time"   : ts,
        "tick"   : tick,
        "status" : status,
        "action" : action,
        "risk"   : risk,
        "tool"   : tool,
        "message": message,
    })
    # Keep log from growing unboundedly
    if len(EVENT_LOG) > 200:
        EVENT_LOG.pop()


def reset_store():
    """Clear everything — called when the simulation is reset."""
    global EVENT_LOG, LATEST_RESULT
    SENSOR_HISTORY.clear()
    EVENT_LOG.clear()
    LATEST_RESULT = {}
