# agent/agent.py
import threading
from collections import deque
from datetime import datetime

from langchain.agents        import AgentExecutor, create_react_agent
from langchain.prompts       import PromptTemplate
from langchain_openai        import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from tools       import ALL_TOOLS
from core.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_AGENT_ITERATIONS, MACHINES
from core.logger import agent_logger, error_logger


# ══════════════════════════════════════════════════════════
#  AgentMemory  —  short-term decisions + long-term action log
# ══════════════════════════════════════════════════════════
class AgentMemory:
    """
    Short-term : last 5 decisions per machine — prevents duplicate alerts.
    Long-term  : full ordered action log for the dashboard history table.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self._short:     dict[str, deque] = {}
        self._action_log: list[dict]      = []

    def record(self, machine_id: str, action: str, detail: str = ""):
        entry = {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": machine_id,
            "action":     action,
            "detail":     detail,
        }
        with self._lock:
            if machine_id not in self._short:
                self._short[machine_id] = deque(maxlen=5)
            self._short[machine_id].append(entry)
            self._action_log.append(entry)

    def was_recently_alerted(self, machine_id: str, within: int = 2) -> bool:
        with self._lock:
            recent = list(self._short.get(machine_id, []))[-within:]
        return any(e["action"] in ("alert_sent", "maintenance_scheduled") for e in recent)

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
_SYSTEM_PROMPT = """You are an expert industrial maintenance AI agent for a manufacturing plant.
Your job is to monitor machine sensor data in real time, detect anomalies, and autonomously
take the correct maintenance action.

You have access to the following tools:
{tools}

DECISION RULES — follow these strictly:
1. ALWAYS start with get_sensor_reading to fetch live data.
2. ALWAYS call analyze_health second to get the health score.
3. If health score is 40–74 (degraded), call detect_trend to check if worsening.
4. Then take exactly ONE of these actions:
   - Score >= 75                   → no action, machine is healthy
   - Score 40–74 or rising trend   → call schedule_maintenance
   - Score < 40 or critical sensor → call send_alert
5. Never fire an alert for a single micro-spike — always look at the average.
6. End with a clear maintenance report explaining what you found and what you did.

Use this exact format:

Question: {{input}}
Thought: your reasoning about what to do next
Action: tool name — must be one of [{tool_names}]
Action Input: input to the tool
Observation: tool result
... repeat Thought / Action / Observation as needed ...
Thought: I now have enough information to write the final report.
Final Answer: your complete maintenance report
"""


# ══════════════════════════════════════════════════════════
#  Thought logger — captures ReAct steps for dashboard display
# ══════════════════════════════════════════════════════════
class _ThoughtLogger(BaseCallbackHandler):

    def __init__(self):
        self.steps: list[dict] = []

    def on_agent_action(self, action, **_):
        self.steps.append({
            "type":  "action",
            "tool":  action.tool,
            "input": str(action.tool_input),
        })

    def on_tool_end(self, output, **_):
        self.steps.append({
            "type":   "observation",
            "output": str(output)[:600],
        })

    def on_agent_finish(self, finish, **_):
        self.steps.append({
            "type":   "final",
            "output": finish.return_values.get("output", ""),
        })

    def clear(self):
        self.steps = []


# ══════════════════════════════════════════════════════════
#  MaintenanceAgent  —  the single agent
# ══════════════════════════════════════════════════════════
class MaintenanceAgent:
    """
    One LangChain ReAct AgentExecutor powered by amazon.nova.lite.
    Checks one machine per run — can be called concurrently from Streamlit.
    """

    def __init__(self):
        self.memory  = AgentMemory()
        self._logger = _ThoughtLogger()
        self._lock   = threading.Lock()

        llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
            temperature=0.1,
            max_tokens=1024,
        )

        prompt   = PromptTemplate.from_template(_SYSTEM_PROMPT + "\n{agent_scratchpad}")
        _agent   = create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

        self._executor = AgentExecutor(
            agent=_agent,
            tools=ALL_TOOLS,
            verbose=False,
            max_iterations=MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
            callbacks=[self._logger],
        )

    def run(self, machine_id: str) -> dict:
        """
        Run the full ReAct loop for one machine.
        Returns: { machine_id, final_report, thought_steps }
        """
        if machine_id not in MACHINES:
            return {"error": f"Unknown machine: {machine_id}"}

        with self._lock:
            self._logger.clear()

        query = (
            f"Check machine {machine_id}. Fetch its live sensor readings, "
            f"assess its health, detect any trends if needed, and take the "
            f"appropriate maintenance action. Provide a full maintenance report."
        )

        try:
            result = self._executor.invoke({"input": query})
            report = result.get("output", "No output returned.")
            agent_logger.info("Agent run complete | %s | %s", machine_id, report[:120])
            self.memory.record(machine_id, "agent_run", report[:120])
        except Exception as exc:
            report = f"Agent error: {exc}"
            error_logger.error("Agent run failed | %s | %s", machine_id, exc)

        return {
            "machine_id":    machine_id,
            "final_report":  report,
            "thought_steps": list(self._logger.steps),
        }

    def run_all(self) -> list[dict]:
        """Check all machines one by one."""
        return [self.run(mid) for mid in MACHINES]


# ── Shared singleton ──────────────────────────────────────
agent = MaintenanceAgent()
