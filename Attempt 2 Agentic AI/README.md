# 🤖 Autonomous Predictive Maintenance Agent

A single autonomous agent that simulates real-time machine sensor data, detects anomalies, 
diagnoses machine conditions, and dynamically decides which action/tool to invoke.

---

## 📁 Project Structure

```
predictive_maintenance/
├── main.py        → Entry point — runs the agent loop
├── simulator.py   → Generates realistic sensor data (temp, vibration, pressure)
├── agent.py       → Single MaintenanceAgent with all internal modules
├── tools.py       → Three tools: alert, maintenance scheduler, status logger
├── utils.py       → Banner, config printing, helper functions
└── README.md      → This file
```

---

## 🏗️ Architecture

```
MaintenanceAgent
   ├── Memory          → deque of last 8 sensor readings; detects trends
   ├── Monitoring      → Compares readings to thresholds → anomaly severity
   ├── Diagnosis       → Interprets anomalies + trends → condition + confidence
   ├── Risk Scoring    → Low / Medium / High from anomaly count
   ├── Decision        → Chooses tool based on risk, confidence, trend, history
   └── Tools Called:
       ├── log_status()          → Normal / low risk
       ├── send_alert()          → Medium risk with rising trend
       └── schedule_maintenance()→ High risk OR persistent medium risk
```

---

## 🚀 How to Run

**Requirements:** Python 3.10+  (no third-party packages needed)

```bash
# Navigate to the project folder
cd predictive_maintenance

# Run the agent
python main.py
```

The agent will run for **30 cycles** (2 seconds apart) by default.

### Customization (edit main.py):
```python
CYCLE_INTERVAL_SECONDS = 2.0   # Change to 1.0 for faster cycles
MAX_CYCLES = 30                 # Set to None to run forever
```

---

## 📤 Output Example

```
───────────────────────────────────────────────────────
  Cycle #  6  |  🟡 Risk: Medium  |  Confidence: 72%
───────────────────────────────────────────────────────
  Temp      :   68.67 °C
  Vibration :   1.079 mm/s
  Pressure  :  103.14 PSI
  Trend     : Temp=stable   Vib=rising
───────────────────────────────────────────────────────
  Condition : Elevated Vibration (Rising Trend)
  Decision  : Continue Monitoring (Medium)
  Tool Used : logging_tool

🚨 [ALERT] 09:15:22
   Condition  : Critical Overheating (Rising)
   Confidence : 92%
   Temp       : 104.3°C  |  Vibration: 0.87 mm/s  |  Pressure: 99.5 PSI

🔧 [MAINTENANCE SCHEDULED] 09:15:30
   Reason     : Severe Multi-System Failure Risk
   Scheduled  : Tomorrow 08:00 AM
```

---

## 🧠 Agent Decision Logic

| Situation                             | Tool Triggered         |
|---------------------------------------|------------------------|
| All values normal                     | `log_status`           |
| Medium risk, 1–3 cycles               | `log_status`           |
| Medium risk + rising trend + confident| `send_alert`           |
| Medium risk for 4+ consecutive cycles | `schedule_maintenance` |
| Any critical anomaly                  | `schedule_maintenance` |
| Critical + maintenance already active | `send_alert`           |

---

## 📄 Log File

All events are automatically saved to `maintenance_log.txt` in the project folder.

---

## ⚙️ Threshold Tuning (agent.py)

```python
THRESHOLDS = {
    "temp_warning":  85.0,    # °C
    "temp_critical": 100.0,   # °C
    "vib_warning":   0.85,    # mm/s
    "vib_critical":  1.20,    # mm/s
    ...
}
```
