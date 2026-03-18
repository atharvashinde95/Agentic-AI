# tools/maintenance_tool.py
import random
import string
from datetime import datetime, timedelta
from langchain.tools import tool
from simulator.simulator import simulator
from core.config import MACHINES
from core.logger import agent_logger


def _job_id() -> str:
    return "MNT-" + "".join(random.choices(string.digits, k=6))

ENGINEERS = ["Rahul Sharma", "Priya Nair", "Amit Verma", "Sunita Patel", "Vikram Das"]


@tool
def schedule_maintenance(machine_id: str, urgency: str, reason: str) -> str:
    """
    Books a preventive maintenance job for a machine.
    Call when health score is 40–74 (degraded) or trend is worsening but not yet critical.
    Input:
      machine_id : M1 – M5
      urgency    : low | medium | high
      reason     : brief explanation of why maintenance is needed
    Output: job confirmation with ID, scheduled time, and assigned engineer.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    urgency    = urgency.lower().strip()
    delay_hrs  = {"low": 48, "medium": 12, "high": 4}.get(urgency, 24)
    scheduled  = (datetime.now() + timedelta(hours=delay_hrs)).strftime("%Y-%m-%d %H:%M")
    job_id     = _job_id()
    engineer   = random.choice(ENGINEERS)

    agent_logger.info("MAINTENANCE SCHEDULED | %s | job=%s | urgency=%s | reason=%s",
                      machine_id, job_id, urgency, reason)

    # high urgency — start recovery immediately
    if urgency == "high":
        simulator.trigger_recovery(machine_id)

    return (
        f"Maintenance scheduled for {machine_id}:\n"
        f"  Job ID       : {job_id}\n"
        f"  Urgency      : {urgency.upper()}\n"
        f"  Scheduled at : {scheduled}\n"
        f"  Engineer     : {engineer}\n"
        f"  Reason       : {reason}\n"
        f"  Status       : CONFIRMED"
    )
