"""
tools.py
--------
Agent tools — callable functions the agent can invoke.

Tools:
  1. alert_tool       — Send an alert for a warning condition
  2. maintenance_tool — Schedule a maintenance action
  3. logging_tool     — Log every cycle reading to a file
"""

import datetime
import os

# Path for the log file
LOG_FILE = "agent_log.txt"


# --------------------------------------------------------------------------- #
#  Tool 1: Alert
# --------------------------------------------------------------------------- #
def alert_tool(sensor: str, value: float, threshold: float, risk: str) -> dict:
    """
    Triggers an alert when a sensor reading exceeds a warning threshold.

    Args:
        sensor    : Name of the sensor (e.g. "temperature")
        value     : Measured value
        threshold : Warning threshold that was breached
        risk      : Risk level string — "Low", "Medium", or "High"

    Returns:
        A result dict with tool name, message, and timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = (
        f"⚠️  ALERT [{risk} Risk] | {sensor.upper()} = {value} "
        f"(threshold: {threshold}) | {timestamp}"
    )
    print(message)

    # Also write to log
    logging_tool(event="ALERT", details=message)

    return {
        "tool": "alert_tool",
        "sensor": sensor,
        "value": value,
        "threshold": threshold,
        "risk": risk,
        "message": message,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
#  Tool 2: Maintenance Scheduler
# --------------------------------------------------------------------------- #
def maintenance_tool(reason: str, urgency: str, sensor: str) -> dict:
    """
    Schedules a maintenance event for the machine.

    Args:
        reason  : Why maintenance is needed
        urgency : "Immediate" or "Scheduled"
        sensor  : Which sensor triggered the maintenance decision

    Returns:
        A result dict describing the scheduled maintenance.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    maintenance_id = f"MNT-{datetime.datetime.now().strftime('%H%M%S')}"

    message = (
        f"🔧 MAINTENANCE SCHEDULED [{urgency}] | ID: {maintenance_id} | "
        f"Reason: {reason} | Sensor: {sensor.upper()} | {timestamp}"
    )
    print(message)

    logging_tool(event="MAINTENANCE", details=message)

    return {
        "tool": "maintenance_tool",
        "maintenance_id": maintenance_id,
        "reason": reason,
        "urgency": urgency,
        "sensor": sensor,
        "message": message,
        "timestamp": timestamp,
    }


# --------------------------------------------------------------------------- #
#  Tool 3: Logger
# --------------------------------------------------------------------------- #
def logging_tool(event: str, details: str) -> dict:
    """
    Logs any event to a text file for auditing.

    Args:
        event   : Short event type label (e.g. "ALERT", "NORMAL", "MAINTENANCE")
        details : Full log message / details string

    Returns:
        A result dict confirming the log entry was written.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{event}] {details}\n"

    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_line)
    except Exception as e:
        print(f"Logging error: {e}")

    return {
        "tool": "logging_tool",
        "event": event,
        "timestamp": timestamp,
        "log_file": LOG_FILE,
    }
