# agent/agent.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
from collections import deque
from datetime import datetime

from langchain_openai        import ChatOpenAI
from langchain.agents        import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts  import ChatPromptTemplate, MessagesPlaceholder

from tools       import ALL_TOOLS
from core.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_AGENT_ITERATIONS, MACHINES
from core.logger import agent_logger, error_logger


# ══════════════════════════════════════════════════════════
#  AgentMemory
# ══════════════════════════════════════════════════════════
class AgentMemory:
    def __init__(self):
        self._lock        = threading.Lock()
        self._short:      dict[str, deque] = {}
        self._action_log: list[dict]       = []

    def record(self, machine_id: str, action: str, detail: str = ""):
        entry = {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": machine_id,
            "action":     action,
            "detail":     detail[:200],
        }
        with self._lock:
            if machine_id not in self._short:
                self._short[machine_id] = deque(maxlen=5)
            self._short[machine_id].append(entry)
            self._action_log.append(entry)

    def get_log(self, machine_id: str | None = None, n: int = 30) -> list[dict]:
        with self._lock:
            log = list(self._action_log)
        if machine_id:
            log = [e for e in log if e["machine_id"] == machine_id]
        return log[-n:]

    def summary(self) -> dict:
        with self._lock:
            log = list(self._action_log)
        return {
            "total":    len(log),
            "alerts":   sum(1 for e in log if e["action"] == "alert_sent"),
            "jobs":     sum(1 for e in log if e["action"] == "maintenance_scheduled"),
            "machines": len(set(e["machine_id"] for e in log)),
        }


# ══════════════════════════════════════════════════════════
#  System prompt
# ══════════════════════════════════════════════════════════
_SYSTEM = """You are an expert industrial maintenance AI agent for a manufacturing plant.
Your job is to monitor machine sensor data in real time, detect anomalies, and autonomously
take the correct maintenance action.

DECISION RULES — follow strictly:
1. ALWAYS call get_sensor_reading first to fetch live data.
2. ALWAYS call analyze_health second to get the health score.
3. If health score is 40-74 call detect_trend to check if worsening.
4. Then take ONE action:
   - Score >= 75                  → no action needed, machine is healthy
   - Score 40-74 or rising trend  → call schedule_maintenance
   - Score < 40 or critical sensor→ call send_alert
5. Never fire an alert for a single spike — look at the average across readings.
6. End with a clear report: what you found, what you did, and your recommendation.
"""


# ══════════════════════════════════════════════════════════
#  Thought logger
# ══════════════════════════════════════════════════════════
class _ThoughtLogger:
    def __init__(self):
        self.steps: list[dict] = []

    def log_action(self, tool: str, tool_input: str):
        self.steps.append({"type": "action", "tool": tool, "input": str(tool_input)})

    def log_observation(self, output: str):
        self.steps.append({"type": "observation", "output": str(output)[:600]})

    def log_final(self, output: str):
        self.steps.append({"type": "final", "output": output})

    def clear(self):
        self.steps = []


# ══════════════════════════════════════════════════════════
#  MaintenanceAgent
# ══════════════════════════════════════════════════════════
class MaintenanceAgent:
    def __init__(self):
        self.memory  = AgentMemory()
        self._lock   = threading.Lock()
        self._tlogger = _ThoughtLogger()

        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
            temperature=0.1,
            max_tokens=1024,
        )

        # openai_tools_agent is much more reliable than ReAct
        # for models served via OpenAI-compatible endpoints
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human",  "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        _agent = create_openai_tools_agent(
            llm=self._llm,
            tools=ALL_TOOLS,
            prompt=prompt,
        )

        self._executor = AgentExecutor(
            agent=_agent,
            tools=ALL_TOOLS,
            verbose=True,
            max_iterations=MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def run(self, machine_id: str) -> dict:
        if machine_id not in MACHINES:
            return {"error": f"Unknown machine: {machine_id}"}

        with self._lock:
            self._tlogger.clear()

        query = (
            f"Check machine {machine_id}. Fetch its live sensor readings, "
            f"assess its health, detect any trends if needed, and take the "
            f"appropriate maintenance action. Provide a full maintenance report."
        )

        try:
            result = self._executor.invoke({"input": query})
            report = result.get("output", "No output returned.")

            # extract thought steps from intermediate_steps
            for step in result.get("intermediate_steps", []):
                action, observation = step
                self._tlogger.log_action(
                    tool=action.tool,
                    tool_input=action.tool_input,
                )
                self._tlogger.log_observation(str(observation))

            self._tlogger.log_final(report)
            agent_logger.info("Agent run complete | %s | %s", machine_id, report[:120])
            self.memory.record(machine_id, "agent_run", report[:120])

        except Exception as exc:
            report = f"Agent error: {exc}"
            error_logger.error("Agent run failed | %s | %s", machine_id, exc)
            self._tlogger.log_final(report)

        return {
            "machine_id":    machine_id,
            "final_report":  report,
            "thought_steps": list(self._tlogger.steps),
        }

    def run_all(self) -> list[dict]:
        return [self.run(mid) for mid in MACHINES]


# ── Shared singleton ──────────────────────────────────────
agent = MaintenanceAgent()
