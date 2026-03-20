# ⚙️ Autonomous Predictive Maintenance Agent
### Pure LangChain · ReAct · Amazon Nova Lite · 6 Tools · ConversationMemory

A fully autonomous, tool-using industrial maintenance agent built on LangChain.
The LLM **autonomously decides** which tools to call, in what order, based on
its reasoning — this is a genuine agentic system, not a scripted pipeline.

---

## 📁 Project Structure

```
predictive_maintenance_agent/
│
├── simulator.py       ← Industrial sensor data generator
├── memory_store.py    ← Shared in-process state (history, event log)
├── tools.py           ← 6 LangChain @tool functions
├── llm_client.py      ← Custom BaseChatModel for Capgemini / Nova Lite
├── agent.py           ← LangChain ReAct AgentExecutor + ConversationMemory
├── streamlit_app.py   ← Real-time dashboard UI
├── main.py            ← Console entry point (verbose ReAct output)
├── .env.example       ← Credentials template
├── requirements.txt   ← Python dependencies
└── README.md
```

---

## 🏗️ Architecture

```
SensorSimulator
      │
      ▼ reading dict (tick, temp, vib, pressure)
MaintenanceAgent
  ├── CapgeminiNovaLite  (custom LangChain BaseChatModel)
  │     └── Calls Capgemini REST API → amazon.nova.lite-v1.0
  │
  ├── ReAct Prompt       (Thought → Action → Observation loop)
  │
  ├── AgentExecutor      (max 8 iterations, handles parsing errors)
  │     └── ConversationBufferWindowMemory (last 6 exchanges)
  │
  └── Tools (6 @tool functions):
        1. check_sensor_status    → threshold monitoring
        2. diagnose_condition     → rule-based fault classifier
        3. send_alert             → raise a warning notification
        4. schedule_maintenance   → book a maintenance job
        5. log_normal_cycle       → audit a healthy reading
        6. get_trend_analysis     → rolling sensor trend summary
```

---

## 🤖 What makes this a REAL agentic system

| Feature | Detail |
|---|---|
| **LangChain ReAct** | The LLM reasons, then picks a tool, reads the result, reasons again |
| **Tool autonomy** | The agent decides the tool call sequence — not hardcoded |
| **ConversationMemory** | The agent remembers the last 6 cycles of sensor readings |
| **Custom BaseChatModel** | Full LangChain interface — works with any LangChain chain/agent |
| **Intermediate steps** | Every Thought + Action + Observation is captured and shown in UI |
| **Graceful degradation** | Works without API credentials (returns stub, continues running) |

---

## 🛠️ Tools Detail

| Tool | When called by agent |
|---|---|
| `check_sensor_status` | First in every cycle — maps raw values to severity levels |
| `diagnose_condition` | After status check — maps severity to fault names + recommended action |
| `send_alert` | When `recommended_action == "Alert"` |
| `schedule_maintenance` | When `recommended_action == "Maintenance"` |
| `log_normal_cycle` | When `recommended_action == "Continue"` |
| `get_trend_analysis` | Optional — agent uses when it needs richer historical context |

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env — add CAPGEMINI_API_KEY and CAPGEMINI_API_ENDPOINT
```

### 3. Launch dashboard

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** → click **▶ Start**

### 4. Or run in console (verbose ReAct chain printed live)

```bash
python main.py
```

---

## 🔧 Sensor Thresholds

| Sensor | Warning | Critical |
|---|---|---|
| Temperature | 85 °C | 100 °C |
| Vibration | 6 mm/s | 9 mm/s |
| Pressure | 7 bar | 9.5 bar |

---

## 📊 Dashboard Features

- **Sensor cards** — live values with colour-coded severity borders
- **3-panel line chart** — temperature, vibration, pressure with threshold lines
- **ReAct trace panel** — every Step N: tool + input + observation
- **Final Answer** — the agent's own summary of the cycle
- **Tool call sequence** — pill badges showing which tools were invoked
- **Fault diagnoses** — extracted from `diagnose_condition` tool output
- **Trend analysis** — rising / stable / falling per sensor
- **Agent config** — LangChain framework details and cycle counter
- **Event log** — scrollable audit trail of all cycles

---

## 📝 Log File

All events (alerts, maintenance, normal cycles) are written to `agent_log.txt`.
The file is appended on every run — safe for long-running deployments.
