# Predictive Maintenance — Agentic AI System

Real-time industrial predictive maintenance powered by a LangChain ReAct agent,
live sensor simulation, and a Streamlit dashboard.

---

## Project Structure

```
predictive_maintenance/
│
├── core/
│   ├── config.py            ← all settings, constants, thresholds (loads .env)
│   └── logger.py            ← rotating log files (agent / alerts / errors)
│
├── simulator/
│   └── simulator.py         ← Machine state logic + SensorSimulator runner + buffer
│
├── agent/
│   └── agent.py             ← AgentMemory + system prompt + LangChain ReAct agent
│
├── tools/
│   ├── __init__.py          ← exports ALL_TOOLS list
│   ├── sensor_tool.py       ← get_sensor_reading
│   ├── health_tool.py       ← analyze_health
│   ├── trend_tool.py        ← detect_trend
│   ├── maintenance_tool.py  ← schedule_maintenance
│   └── alert_tool.py        ← send_alert
│
├── dashboard/
│   └── app.py               ← Streamlit dashboard + all charts
│
├── logs/                    ← auto-created at runtime
│
├── .env                     ← your LLM credentials (never commit)
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your credentials to .env
```
LLM_BASE_URL=https://your-capgemini-engine-url/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=amazon.nova.lite
```

### 3. Run
```bash
cd predictive_maintenance
streamlit run dashboard/app.py
```

---

## Agent Decision Rules

| Health Score | Action taken |
|---|---|
| >= 75 | No action — machine is healthy |
| 40 – 74 | detect_trend → schedule_maintenance |
| < 40 | send_alert immediately |

## Logs

logs/agent.log   — every agent decision
logs/alerts.log  — every alert fired
logs/errors.log  — any runtime errors
