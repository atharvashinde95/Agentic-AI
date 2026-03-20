"""
agent.py
--------
The single Autonomous Predictive Maintenance Agent.

Architecture:
  Main Agent
    ├── Monitoring Module  (perception — detect anomalies)
    ├── Diagnosis Module   (rule-based reasoning — classify condition)
    ├── Decision Module    (choose action: Continue / Alert / Maintenance)
    ├── Memory             (store last N readings, detect trends)
    └── Tools              (alert_tool, maintenance_tool, logging_tool)

⚡ LLM is called ONLY when signals are ambiguous / mixed.
"""

from collections import deque
from tools import alert_tool, maintenance_tool, logging_tool
from llm_client import call_llm, build_diagnosis_prompt


# ──────────────────────────────────────────────────────────────────────────── #
#  Thresholds
# ──────────────────────────────────────────────────────────────────────────── #

THRESHOLDS = {
    "temperature": {
        "warning":  85.0,   # °C
        "critical": 100.0,
    },
    "vibration": {
        "warning":  6.0,    # mm/s
        "critical": 9.0,
    },
    "pressure": {
        "warning":  7.0,    # bar
        "critical": 9.5,
    },
}

MEMORY_SIZE = 10   # Keep last N readings in memory


# ──────────────────────────────────────────────────────────────────────────── #
#  Memory Module
# ──────────────────────────────────────────────────────────────────────────── #

class AgentMemory:
    """Stores the last MEMORY_SIZE sensor readings and computes trends."""

    def __init__(self, size: int = MEMORY_SIZE):
        self.history = deque(maxlen=size)

    def add(self, reading: dict):
        self.history.append(reading)

    def get_all(self) -> list:
        return list(self.history)

    def trend_summary(self) -> dict:
        """
        Return the average and direction (rising / stable / falling)
        for each sensor over the stored history.
        """
        if len(self.history) < 2:
            return {}

        data = list(self.history)
        keys = ["temperature", "vibration", "pressure"]
        summary = {}

        for key in keys:
            values = [r[key] for r in data]
            avg = round(sum(values) / len(values), 2)
            trend = "stable"
            if values[-1] > values[0] * 1.05:
                trend = "rising"
            elif values[-1] < values[0] * 0.95:
                trend = "falling"
            summary[key] = {"average": avg, "trend": trend}

        return summary


# ──────────────────────────────────────────────────────────────────────────── #
#  Main Agent
# ──────────────────────────────────────────────────────────────────────────── #

class MaintenanceAgent:
    """
    Single autonomous agent that monitors, diagnoses, decides, and acts.
    """

    def __init__(self):
        self.memory = AgentMemory()
        self.llm_call_count = 0   # Track how many times we call LLM (should be rare)

    # ── 1. Monitoring Module ─────────────────────────────────────────────── #

    def monitor(self, reading: dict) -> dict:
        """
        Check each sensor against thresholds.
        Returns a dict with severity level for each sensor.
        """
        sensor_status = {}

        for sensor in ["temperature", "vibration", "pressure"]:
            value = reading[sensor]
            warn  = THRESHOLDS[sensor]["warning"]
            crit  = THRESHOLDS[sensor]["critical"]

            if value >= crit:
                sensor_status[sensor] = "critical"
            elif value >= warn:
                sensor_status[sensor] = "warning"
            else:
                sensor_status[sensor] = "normal"

        return sensor_status

    # ── 2. Diagnosis Module ──────────────────────────────────────────────── #

    def diagnose(self, reading: dict, sensor_status: dict) -> list:
        """
        Rule-based diagnosis: map sensor conditions to fault names.
        Returns a list of anomaly strings.
        """
        anomalies = []
        diagnoses = []

        # Temperature rules
        if sensor_status["temperature"] == "critical":
            anomalies.append("temperature_critical")
            diagnoses.append("Severe Overheating")
        elif sensor_status["temperature"] == "warning":
            anomalies.append("temperature_warning")
            diagnoses.append("Overheating")

        # Vibration rules
        if sensor_status["vibration"] == "critical":
            anomalies.append("vibration_critical")
            diagnoses.append("Severe Imbalance / Bearing Failure")
        elif sensor_status["vibration"] == "warning":
            anomalies.append("vibration_warning")
            diagnoses.append("Mechanical Imbalance")

        # Pressure rules
        if sensor_status["pressure"] == "critical":
            anomalies.append("pressure_critical")
            diagnoses.append("Severe Pressure Spike")
        elif sensor_status["pressure"] == "warning":
            anomalies.append("pressure_warning")
            diagnoses.append("High Pressure")

        # Trend-based secondary diagnosis
        trends = self.memory.trend_summary()
        if trends:
            for sensor_key in ["temperature", "vibration", "pressure"]:
                if (
                    trends.get(sensor_key, {}).get("trend") == "rising"
                    and sensor_status[sensor_key] == "normal"
                ):
                    diagnoses.append(f"Rising {sensor_key} trend (watch)")

        return anomalies, diagnoses

    # ── 3. Decision Module ───────────────────────────────────────────────── #

    def decide(self, sensor_status: dict, anomalies: list) -> tuple:
        """
        Rule-based decision engine.
        Returns (action, risk_level, primary_sensor, use_llm_flag).

        Actions: "Continue" | "Alert" | "Maintenance"
        Risk:    "Low" | "Medium" | "High"
        """
        critical_sensors = [s for s, v in sensor_status.items() if v == "critical"]
        warning_sensors  = [s for s, v in sensor_status.items() if v == "warning"]

        # Clear critical → immediate Maintenance
        if len(critical_sensors) >= 1:
            return (
                "Maintenance",
                "High",
                critical_sensors[0],
                False,
            )

        # Multiple warnings → escalate to Maintenance
        if len(warning_sensors) >= 2:
            return (
                "Maintenance",
                "High",
                warning_sensors[0],
                False,
            )

        # Single warning → Alert, but also check if LLM needed
        if len(warning_sensors) == 1:
            # LLM edge case: warning sensor + rising trend on another sensor
            trends = self.memory.trend_summary()
            rising_sensors = [
                k for k, v in trends.items()
                if v.get("trend") == "rising" and k not in warning_sensors
            ]
            use_llm = len(rising_sensors) > 0  # Mixed signal → ask LLM
            return (
                "Alert",
                "Medium",
                warning_sensors[0],
                use_llm,
            )

        # All normal
        return ("Continue", "Low", None, False)

    # ── 4. LLM Consultation ──────────────────────────────────────────────── #

    def consult_llm(self, reading: dict, anomalies: list) -> str:
        """
        Called ONLY for ambiguous edge cases.
        Returns an LLM-generated explanation string.
        """
        self.llm_call_count += 1
        prompt = build_diagnosis_prompt(
            readings=self.memory.get_all(),
            anomalies=anomalies,
            history_summary=self.memory.trend_summary(),
        )
        response = call_llm(prompt, max_tokens=200)
        return response

    # ── 5. Tool Dispatcher ───────────────────────────────────────────────── #

    def dispatch_tool(
        self,
        action: str,
        risk: str,
        primary_sensor: str,
        reading: dict,
        diagnoses: list,
    ) -> dict:
        """
        Calls the appropriate tool based on the agent's decision.
        """
        if action == "Alert":
            sensor_value = reading.get(primary_sensor, 0)
            threshold    = THRESHOLDS[primary_sensor]["warning"]
            return alert_tool(
                sensor=primary_sensor,
                value=sensor_value,
                threshold=threshold,
                risk=risk,
            )

        elif action == "Maintenance":
            reason = "; ".join(diagnoses) if diagnoses else "Critical sensor reading"
            urgency = "Immediate" if risk == "High" else "Scheduled"
            return maintenance_tool(
                reason=reason,
                urgency=urgency,
                sensor=primary_sensor,
            )

        else:
            # Continue — just log
            return logging_tool(
                event="NORMAL",
                details=(
                    f"T={reading['temperature']}°C  "
                    f"V={reading['vibration']}mm/s  "
                    f"P={reading['pressure']}bar | Status: Normal"
                ),
            )

    # ── Master Run Loop ──────────────────────────────────────────────────── #

    def run_cycle(self, reading: dict) -> dict:
        """
        Execute one full agent cycle for a single sensor reading.

        Returns a result dict containing everything the UI needs to display.
        """
        # Step 1 — Store reading in memory
        self.memory.add(reading)

        # Step 2 — Monitor (perception)
        sensor_status = self.monitor(reading)

        # Step 3 — Diagnose (rule-based reasoning)
        anomalies, diagnoses = self.diagnose(reading, sensor_status)

        # Step 4 — Decide (logic-based)
        action, risk, primary_sensor, use_llm = self.decide(sensor_status, anomalies)

        # Step 5 — Optionally consult LLM (edge cases only)
        llm_explanation = None
        if use_llm:
            llm_explanation = self.consult_llm(reading, anomalies)

        # Step 6 — Dispatch tool
        tool_result = self.dispatch_tool(
            action=action,
            risk=risk,
            primary_sensor=primary_sensor or "system",
            reading=reading,
            diagnoses=diagnoses,
        )

        # Determine overall status label for UI
        if action == "Maintenance":
            status = "Failure"
        elif action == "Alert":
            status = "Warning"
        else:
            status = "Normal"

        # Compute a simple confidence score
        confidence = self._compute_confidence(sensor_status, anomalies)

        return {
            "tick"           : reading["tick"],
            "reading"        : reading,
            "sensor_status"  : sensor_status,
            "anomalies"      : anomalies,
            "diagnoses"      : diagnoses,
            "action"         : action,
            "status"         : status,
            "risk"           : risk,
            "primary_sensor" : primary_sensor,
            "tool_result"    : tool_result,
            "llm_used"       : use_llm,
            "llm_explanation": llm_explanation,
            "confidence"     : confidence,
            "llm_call_count" : self.llm_call_count,
            "trend_summary"  : self.memory.trend_summary(),
        }

    def _compute_confidence(self, sensor_status: dict, anomalies: list) -> str:
        """Simple heuristic for confidence in the decision."""
        critical = sum(1 for v in sensor_status.values() if v == "critical")
        warning  = sum(1 for v in sensor_status.values() if v == "warning")
        if critical >= 1:
            return "High"
        if warning >= 2:
            return "High"
        if warning == 1:
            return "Medium"
        return "High"   # Normal state is confidently normal

    def reset(self):
        """Reset agent state (useful when restarting simulation)."""
        self.memory = AgentMemory()
        self.llm_call_count = 0
