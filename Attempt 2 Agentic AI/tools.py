"""
tools.py — Agent Action Tools
Each function represents one tool the agent can call.
The agent decides WHICH tool to use — it is never hardcoded.
"""

import logging
import os
from datetime import datetime

# Set up file logger (writes to maintenance_log.txt)
log_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(log_dir, "maintenance_log.txt")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────
# TOOL 1: Alert Tool
# ─────────────────────────────────────────────

def send_alert(sensor_data: dict, condition: str, confidence: float):
    """
    Send an urgent alert when anomalous conditions are detected.

    Args:
        sensor_data: Current sensor readings
        condition: Diagnosed machine condition
        confidence: Agent's confidence in diagnosis (0–1)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = (
        f"\n🚨 [ALERT] {timestamp}\n"
        f"   Condition  : {condition}\n"
        f"   Confidence : {confidence:.0%}\n"
        f"   Temp       : {sensor_data['temperature']}°C  |  "
        f"Vibration: {sensor_data['vibration']} mm/s  |  "
        f"Pressure: {sensor_data['pressure']} PSI\n"
    )
    print(message)
    logging.warning(f"ALERT | {condition} | Confidence: {confidence:.0%} | "
                    f"Temp={sensor_data['temperature']} Vib={sensor_data['vibration']} "
                    f"Pres={sensor_data['pressure']}")
    return "alert_sent"


# ─────────────────────────────────────────────
# TOOL 2: Maintenance Scheduling Tool
# ─────────────────────────────────────────────

def schedule_maintenance(sensor_data: dict, reason: str):
    """
    Simulate scheduling a maintenance task.

    Args:
        sensor_data: Current sensor readings that triggered this decision
        reason: Why maintenance is being scheduled
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Simulate a scheduled time (next business day, 8 AM)
    scheduled_time = "Tomorrow 08:00 AM"

    message = (
        f"\n🔧 [MAINTENANCE SCHEDULED] {timestamp}\n"
        f"   Reason     : {reason}\n"
        f"   Scheduled  : {scheduled_time}\n"
        f"   Temp       : {sensor_data['temperature']}°C  |  "
        f"Vibration: {sensor_data['vibration']} mm/s  |  "
        f"Pressure: {sensor_data['pressure']} PSI\n"
    )
    print(message)
    logging.critical(f"MAINTENANCE SCHEDULED | Reason: {reason} | "
                     f"Temp={sensor_data['temperature']} Vib={sensor_data['vibration']} "
                     f"Pres={sensor_data['pressure']}")
    return "maintenance_scheduled"


# ─────────────────────────────────────────────
# TOOL 3: Status Logging Tool
# ─────────────────────────────────────────────

def log_status(sensor_data: dict, condition: str, risk_level: str):
    """
    Log the current machine status during normal/low-risk operation.

    Args:
        sensor_data: Current sensor readings
        condition: Machine condition label
        risk_level: Low / Medium / High
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = {"Low": "✅", "Medium": "⚠️", "High": "❌"}.get(risk_level, "ℹ️")

    message = (
        f"{icon} [{timestamp}] Status: {condition} | Risk: {risk_level} | "
        f"Temp: {sensor_data['temperature']}°C | "
        f"Vib: {sensor_data['vibration']} mm/s | "
        f"Pres: {sensor_data['pressure']} PSI"
    )
    print(message)
    logging.info(f"STATUS | {condition} | Risk: {risk_level} | "
                 f"Temp={sensor_data['temperature']} Vib={sensor_data['vibration']} "
                 f"Pres={sensor_data['pressure']}")
    return "status_logged"
