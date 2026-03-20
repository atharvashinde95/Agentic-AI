# 🔧 Autonomous Predictive Maintenance Agent

A single autonomous agent that simulates industrial machine sensors,
detects anomalies using rule-based logic, and takes actions via tools —
with minimal LLM calls reserved only for ambiguous edge cases.

---

## 📁 Project Structure

```
predictive_maintenance_agent/
│
├── main.py            ← Console entry point (no UI)
├── simulator.py       ← Sensor data simulator
├── agent.py           ← Main autonomous agent (all modules inside)
├── tools.py           ← Tool functions (alert, maintenance, logging)
├── llm_client.py      ← LLM wrapper for Amazon Nova Lite
├── streamlit_app.py   ← Streamlit dashboard UI
├── .env.example       ← Template for environment variables
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## ⚡ Quick Start

### 1. Clone / download the project

```bash
cd predictive_maintenance_agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API credentials

```bash
cp .env.example .env
```

Then open `.env` and fill in your values:

```
CAPGEMINI_API_KEY=your_api_key_here
CAPGEMINI_API_ENDPOINT=https://your-endpoint-url/v1/messages
```

> ⚠️ The agent works fully without LLM credentials — it falls back to
> rule-based decisions automatically if the API key is missing.

### 5. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open your browser at **http://localhost:8501**

### 6. (Optional) Run in console mode

```bash
python main.py
```

Press `Ctrl+C` to stop.

---

## 🏗️ Agent Architecture

```
Main Agent (agent.py)
   ├── Monitoring Module   → Checks thresholds per sensor
   ├── Diagnosis Module    → Maps readings to fault names (rule-based)
   ├── Decision Module     → Decides Continue / Alert / Maintenance
   ├── Memory              → Stores last 10 readings, computes trends
   └── Tools               → alert_tool, maintenance_tool, logging_tool
```

---

## 🧠 Decision Logic

| Condition                    | Action      | Risk   |
|------------------------------|-------------|--------|
| Any sensor critical          | Maintenance | High   |
| 2+ sensors in warning        | Maintenance | High   |
| 1 sensor in warning          | Alert       | Medium |
| Warning + rising trend (LLM) | Alert + LLM | Medium |
| All normal                   | Continue    | Low    |

---

## ⚠️ Sensor Thresholds

| Sensor      | Warning | Critical |
|-------------|---------|----------|
| Temperature | 85 °C   | 100 °C   |
| Vibration   | 6 mm/s  | 9 mm/s   |
| Pressure    | 7 bar   | 9.5 bar  |

---

## 🤖 LLM Usage Policy

The LLM is called **only** when:
- One sensor is in warning state **AND**
- Another sensor shows a rising trend (mixed signal edge case)

In all other cases, pure Python rule logic is used — keeping the system
fast, cheap, and reliable.

---

## 📊 Streamlit Dashboard Features

- Live sensor readings with colour-coded status cards
- Three-panel line chart with warning/critical threshold lines
- Agent decision, risk level, and confidence indicator
- Tool invocation display
- Trend analysis (rising / stable / falling per sensor)
- Scrollable real-time event log
- Start / Stop / Reset buttons

---

## 📝 Log File

All agent events are written to `agent_log.txt` in the project directory.
The file is appended on every run — useful for auditing.
