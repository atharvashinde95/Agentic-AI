# Predictive Maintenance — Agentic AI System

Real-time industrial predictive maintenance powered by a LangChain ReAct agent,
live sensor simulation, and a Streamlit dashboard.

## Architecture

```
simulator.py   →  generates live sensor readings every 5 sec (5 machines)
memory.py      →  short-term + long-term agent memory
tools.py       →  5 LangChain tools the agent can call
agent.py       →  LangChain ReAct AgentExecutor (amazon.nova.lite)
app.py         →  Streamlit live dashboard
config.py      →  all settings and thresholds
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your LLM
Edit `config.py` and set:
```python
LLM_URL     = "https://your-capgemini-engine-url/v1"
LLM_API_KEY = "your-api-key-here"
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

## How it works

**Simulator** — 5 machines (M1–M5) each run a state machine:
- `normal` → `degrading` (2% chance per tick)
- `degrading` → `critical` (3% chance per tick)
- `critical` → `recovering` (after agent fires alert)
- `recovering` → `normal` (after 6 ticks)

Values use gaussian noise + gradual drift + sensor correlation for realism.

**Agent** — LangChain ReAct loop:
1. `get_sensor_reading` — always first
2. `analyze_health` — computes health score 0–100
3. `detect_trend` — called if score is borderline (40–74)
4. `schedule_maintenance` — if degraded
5. `send_alert` — if critical

**Dashboard** — Live charts, agent thought process, action log, KPI summary.

## Tools

| Tool | When called | Returns |
|------|-------------|---------|
| get_sensor_reading | Always first | Last 10 readings as text |
| analyze_health | Always second | Health score + sensor flags |
| detect_trend | Borderline score | Rising/stable/falling trend |
| schedule_maintenance | Score 40–74 | Job ID + scheduled time |
| send_alert | Score < 40 | Alert ID + notified contacts |
