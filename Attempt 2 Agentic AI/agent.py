"""
agent.py — Autonomous Predictive Maintenance Agent
Single agent with internal Monitoring, Diagnosis, Decision modules + Memory.
"""

from collections import deque
from tools import send_alert, schedule_maintenance, log_status


# ─────────────────────────────────────────────
# Threshold constants (tunable)
# ─────────────────────────────────────────────

THRESHOLDS = {
    # Temperature thresholds (°C)
    "temp_warning":  85.0,
    "temp_critical": 100.0,

    # Vibration thresholds (mm/s)
    "vib_warning":   0.85,
    "vib_critical":  1.20,

    # Pressure thresholds (PSI) — both low and high are bad
    "pres_low_warning":   88.0,
    "pres_low_critical":  80.0,
    "pres_high_warning":  115.0,
    "pres_high_critical": 125.0,
}

# How many past readings to keep in memory
MEMORY_SIZE = 8

# How many consecutive medium-risk cycles trigger maintenance
CONSECUTIVE_MEDIUM_LIMIT = 4


# ═════════════════════════════════════════════
# MAIN AGENT CLASS
# ═════════════════════════════════════════════

class MaintenanceAgent:
    """
    Single autonomous agent that:
    1. Monitors sensor data (Monitoring Module)
    2. Diagnoses machine condition (Diagnosis Module)
    3. Decides what action to take (Decision Module)
    4. Calls the appropriate tool
    5. Remembers past readings (Memory)
    """

    def __init__(self):
        # ── Memory: stores last N sensor readings ──
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Track how many consecutive medium-risk cycles happened
        self._consecutive_medium = 0

        # Track if maintenance was already scheduled (avoid spam)
        self._maintenance_active = False

    # ─────────────────────────────────────────
    # PUBLIC: Process one sensor reading cycle
    # ─────────────────────────────────────────

    def run_cycle(self, sensor_data: dict):
        """
        Main entry point for each sensor cycle.
        Runs all internal modules in sequence.
        """
        # Step 1 — Store reading in memory
        self._update_memory(sensor_data)

        # Step 2 — Monitoring Module: check thresholds
        anomalies = self._monitoring_module(sensor_data)

        # Step 3 — Diagnosis Module: interpret what's wrong
        condition, confidence = self._diagnosis_module(sensor_data, anomalies)

        # Step 4 — Compute risk level
        risk_level = self._compute_risk(anomalies, condition)

        # Step 5 — Decision Module: choose action
        decision, tool_used = self._decision_module(
            sensor_data, condition, risk_level, confidence
        )

        # Step 6 — Print cycle summary
        self._print_summary(sensor_data, condition, risk_level, confidence, decision, tool_used)

    # ─────────────────────────────────────────
    # MODULE A: Memory
    # ─────────────────────────────────────────

    def _update_memory(self, sensor_data: dict):
        """Store the latest sensor reading in memory."""
        self.memory.append(sensor_data)

    def _get_trend(self, key: str) -> str:
        """
        Detect if a sensor value is trending up, down, or stable
        by comparing the average of the first half vs second half of memory.

        Returns: "rising", "falling", or "stable"
        """
        if len(self.memory) < 4:
            return "stable"

        readings = [r[key] for r in self.memory]
        mid = len(readings) // 2
        avg_old = sum(readings[:mid]) / mid
        avg_new = sum(readings[mid:]) / (len(readings) - mid)

        delta_pct = (avg_new - avg_old) / (abs(avg_old) + 1e-9)

        if delta_pct > 0.03:    # More than 3% increase
            return "rising"
        elif delta_pct < -0.03: # More than 3% decrease
            return "falling"
        return "stable"

    # ─────────────────────────────────────────
    # MODULE B: Monitoring Module (Perception)
    # ─────────────────────────────────────────

    def _monitoring_module(self, sensor_data: dict) -> dict:
        """
        Compare current readings against thresholds.
        Returns a dictionary of anomalies detected.

        Anomaly severity: None / "warning" / "critical"
        """
        temp = sensor_data["temperature"]
        vib  = sensor_data["vibration"]
        pres = sensor_data["pressure"]

        anomalies = {
            "temperature": None,
            "vibration": None,
            "pressure": None,
        }

        # Temperature check
        if temp >= THRESHOLDS["temp_critical"]:
            anomalies["temperature"] = "critical"
        elif temp >= THRESHOLDS["temp_warning"]:
            anomalies["temperature"] = "warning"

        # Vibration check
        if vib >= THRESHOLDS["vib_critical"]:
            anomalies["vibration"] = "critical"
        elif vib >= THRESHOLDS["vib_warning"]:
            anomalies["vibration"] = "warning"

        # Pressure check (too low OR too high)
        if pres <= THRESHOLDS["pres_low_critical"] or pres >= THRESHOLDS["pres_high_critical"]:
            anomalies["pressure"] = "critical"
        elif pres <= THRESHOLDS["pres_low_warning"] or pres >= THRESHOLDS["pres_high_warning"]:
            anomalies["pressure"] = "warning"

        return anomalies

    # ─────────────────────────────────────────
    # MODULE C: Diagnosis Module (Reasoning)
    # ─────────────────────────────────────────

    def _diagnosis_module(self, sensor_data: dict, anomalies: dict) -> tuple[str, float]:
        """
        Interpret anomalies into a human-readable machine condition.
        Uses trend data from memory for better reasoning.

        Returns:
            condition: Description of what's wrong
            confidence: 0.0 – 1.0 confidence in diagnosis
        """
        temp_status = anomalies["temperature"]
        vib_status  = anomalies["vibration"]
        pres_status = anomalies["pressure"]

        temp_trend = self._get_trend("temperature")
        vib_trend  = self._get_trend("vibration")
        pres_trend = self._get_trend("pressure")

        # ── Multiple critical anomalies → severe failure risk ──
        critical_count = sum(1 for v in anomalies.values() if v == "critical")
        warning_count  = sum(1 for v in anomalies.values() if v == "warning")

        if critical_count >= 2:
            return "Severe Multi-System Failure Risk", 0.95

        # ── Single critical anomaly diagnoses ──
        if temp_status == "critical":
            if temp_trend == "rising":
                return "Critical Overheating (Rising)", 0.92
            return "Critical Overheating", 0.88

        if vib_status == "critical":
            if vib_trend == "rising":
                return "Severe Mechanical Failure (Worsening)", 0.91
            return "Severe Mechanical Vibration", 0.87

        if pres_status == "critical":
            if sensor_data["pressure"] < THRESHOLDS["pres_low_critical"]:
                return "Critical Pressure Drop (Possible Leak)", 0.89
            return "Critical Over-Pressure", 0.88

        # ── Warning-level diagnoses ──
        if temp_status == "warning" and vib_status == "warning":
            return "Overheating with Mechanical Stress", 0.78

        if temp_status == "warning":
            conf = 0.75 if temp_trend == "rising" else 0.65
            return "Elevated Temperature" + (" (Rising Trend)" if temp_trend == "rising" else ""), conf

        if vib_status == "warning":
            conf = 0.72 if vib_trend == "rising" else 0.62
            return "Elevated Vibration" + (" (Rising Trend)" if vib_trend == "rising" else ""), conf

        if pres_status == "warning":
            return "Abnormal Pressure", 0.68

        # ── All normal ──
        return "Normal Operation", 0.98

    # ─────────────────────────────────────────
    # Risk Level Computation
    # ─────────────────────────────────────────

    def _compute_risk(self, anomalies: dict, condition: str) -> str:
        """Map anomaly counts and condition to a risk level: Low / Medium / High."""
        critical_count = sum(1 for v in anomalies.values() if v == "critical")
        warning_count  = sum(1 for v in anomalies.values() if v == "warning")

        if critical_count >= 1 or "Severe" in condition:
            return "High"
        elif warning_count >= 1:
            return "Medium"
        return "Low"

    # ─────────────────────────────────────────
    # MODULE D: Decision Module (Action)
    # ─────────────────────────────────────────

    def _decision_module(
        self,
        sensor_data: dict,
        condition: str,
        risk_level: str,
        confidence: float,
    ) -> tuple[str, str]:
        """
        Autonomously decide which action/tool to use.
        Uses risk level, confidence, memory trends, and consecutive-cycle tracking.

        Returns:
            decision: Human-readable decision label
            tool_used: Name of the tool that was called
        """
        temp_trend = self._get_trend("temperature")
        vib_trend  = self._get_trend("vibration")

        # ── HIGH RISK: immediate action ──
        if risk_level == "High":
            self._consecutive_medium = 0

            if not self._maintenance_active:
                # Critical failure risk → schedule maintenance immediately
                self._maintenance_active = True
                schedule_maintenance(sensor_data, reason=condition)
                return "Schedule Maintenance (Critical)", "maintenance_tool"
            else:
                # Maintenance already scheduled → escalate with alert
                send_alert(sensor_data, condition, confidence)
                return "Send Alert (Maintenance Pending)", "alert_tool"

        # ── MEDIUM RISK: monitor trend; escalate if persistent ──
        elif risk_level == "Medium":
            self._consecutive_medium += 1
            self._maintenance_active = False

            if self._consecutive_medium >= CONSECUTIVE_MEDIUM_LIMIT:
                # Sustained medium risk → proactive maintenance
                self._consecutive_medium = 0
                schedule_maintenance(sensor_data, reason=f"Persistent issue: {condition}")
                return "Schedule Maintenance (Persistent Medium Risk)", "maintenance_tool"

            elif confidence >= 0.75 and (temp_trend == "rising" or vib_trend == "rising"):
                # Confident + rising trend → send alert
                send_alert(sensor_data, condition, confidence)
                return "Send Alert (Rising Trend)", "alert_tool"

            else:
                # Just log and keep watching
                log_status(sensor_data, condition, risk_level)
                return "Continue Monitoring (Medium)", "logging_tool"

        # ── LOW RISK: normal logging ──
        else:
            self._consecutive_medium = 0
            self._maintenance_active = False
            log_status(sensor_data, condition, risk_level)
            return "Continue Monitoring", "logging_tool"

    # ─────────────────────────────────────────
    # Print Cycle Summary
    # ─────────────────────────────────────────

    def _print_summary(
        self,
        sensor_data: dict,
        condition: str,
        risk_level: str,
        confidence: float,
        decision: str,
        tool_used: str,
    ):
        """Print the structured cycle summary to the terminal."""
        cycle = sensor_data["cycle"]
        risk_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        icon = risk_icons.get(risk_level, "⚪")

        print(
            f"\n{'─'*55}\n"
            f"  Cycle #{cycle:>3}  |  {icon} Risk: {risk_level:<6}  |  "
            f"Confidence: {confidence:.0%}\n"
            f"{'─'*55}\n"
            f"  Temp      : {sensor_data['temperature']:>7.2f} °C\n"
            f"  Vibration : {sensor_data['vibration']:>7.3f} mm/s\n"
            f"  Pressure  : {sensor_data['pressure']:>7.2f} PSI\n"
            f"  Trend     : Temp={self._get_trend('temperature'):<8} "
            f"Vib={self._get_trend('vibration')}\n"
            f"{'─'*55}\n"
            f"  Condition : {condition}\n"
            f"  Decision  : {decision}\n"
            f"  Tool Used : {tool_used}\n"
        )
