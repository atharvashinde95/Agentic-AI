# tools/alert_tool.py
import random
import string
from datetime import datetime
from langchain.tools import tool
from simulator.simulator import simulator
from core.config import MACHINES
from core.logger import alert_logger


def _alert_id() -> str:
    return "ALT-" + "".join(random.choices(string.digits, k=6))

CONTACTS = {
    "warning":   ["floor-supervisor@factory.com"],
    "critical":  ["floor-supervisor@factory.com", "maintenance-lead@factory.com"],
    "emergency": ["floor-supervisor@factory.com", "maintenance-lead@factory.com",
                  "plant-manager@factory.com", "+91-9876543210 (on-call)"],
}


@tool
def send_alert(machine_id: str, severity: str, reason: str) -> str:
    """
    Fires an immediate alert for a machine requiring urgent intervention.
    Call when health score < 40 OR any sensor is in the critical zone.
    Input:
      machine_id : M1 – M5
      severity   : warning | critical | emergency
      reason     : detailed explanation of the failure condition
    Output: alert confirmation with ID and list of notified contacts.
    """
    if machine_id not in MACHINES:
        return f"ERROR: Unknown machine '{machine_id}'."

    severity   = severity.lower().strip()
    alert_id   = _alert_id()
    notified   = CONTACTS.get(severity, CONTACTS["critical"])
    fired_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert_logger.warning("ALERT FIRED | %s | id=%s | severity=%s | reason=%s",
                         machine_id, alert_id, severity, reason)

    # emergency — trigger recovery immediately
    if severity == "emergency":
        simulator.trigger_recovery(machine_id)

    return (
        f"ALERT FIRED for {machine_id}:\n"
        f"  Alert ID   : {alert_id}\n"
        f"  Severity   : {severity.upper()}\n"
        f"  Reason     : {reason}\n"
        f"  Notified   : {', '.join(notified)}\n"
        f"  Fired at   : {fired_at}\n"
        f"  Status     : DELIVERED"
    )
