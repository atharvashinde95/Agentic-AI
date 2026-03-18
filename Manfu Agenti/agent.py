# ─────────────────────────────────────────────
#  agent.py  —  LangChain ReAct agent
# ─────────────────────────────────────────────

from langchain.agents         import AgentExecutor, create_react_agent
from langchain.prompts        import PromptTemplate
from langchain_openai         import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from tools  import ALL_TOOLS
from memory import memory
from config import (
    LLM_URL, LLM_MODEL, LLM_API_KEY,
    MAX_AGENT_ITERATIONS, MACHINES
)


# ── Thought logger — captures ReAct reasoning ────
class ThoughtLogger(BaseCallbackHandler):
    """Collects every agent thought/action for display in Streamlit."""

    def __init__(self):
        self.steps: list[dict] = []

    def on_agent_action(self, action, **kwargs):
        self.steps.append({
            "type":  "action",
            "tool":  action.tool,
            "input": action.tool_input,
        })

    def on_tool_end(self, output, **kwargs):
        self.steps.append({
            "type":   "observation",
            "output": str(output)[:600],   # cap length for display
        })

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append({
            "type":   "final",
            "output": finish.return_values.get("output", ""),
        })

    def clear(self):
        self.steps = []


# ── System prompt — tells the LLM its role ───────
SYSTEM_PROMPT = """You are an expert industrial maintenance AI agent for a manufacturing plant.
Your job is to monitor machine sensor data, detect anomalies, and autonomously take maintenance actions.

You have access to these tools:
{tools}

STRICT RULES you must follow:
1. ALWAYS start by calling get_sensor_reading to get live data.
2. ALWAYS call analyze_health second to compute the health score.
3. If health score is between 40-74, call detect_trend to check if degrading.
4. Based on your analysis, decide ONE of:
   - Do nothing if machine is healthy (score >= 75)
   - Call schedule_maintenance if score 40-74 or trend is worsening
   - Call send_alert if score < 40 or any sensor is in critical zone
5. Never fire alerts for a single spike — look at the average across readings.
6. Always write a clear final report explaining what you found and what you did.

Use this exact format for your reasoning:

Question: {{input}}
Thought: (your reasoning about what to do next)
Action: (tool name — must be one of [{tool_names}])
Action Input: (input to the tool)
Observation: (tool result)
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to make a final decision.
Final Answer: (your complete maintenance report)
"""

# ── Build the LLM ─────────────────────────────────
def _build_llm():
    """
    Uses ChatOpenAI with a custom base_url to point at
    Capgemini Generative Engine (OpenAI-compatible endpoint).
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_URL,
        temperature=0.1,        # low temp = consistent, factual decisions
        max_tokens=1024,
    )


# ── Build the agent ───────────────────────────────
def _build_agent(llm):
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT + "\n{agent_scratchpad}")
    return create_react_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)


# ── Main entry point ──────────────────────────────
class MaintenanceAgent:
    """
    Single agent that checks one machine per run.
    Thread-safe — can be called from Streamlit callbacks.
    """

    def __init__(self):
        self.llm       = _build_llm()
        self.logger    = ThoughtLogger()
        agent          = _build_agent(self.llm)
        self.executor  = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=False,
            max_iterations=MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
            callbacks=[self.logger],
        )

    def run(self, machine_id: str) -> dict:
        """
        Run the full ReAct loop for one machine.
        Returns a result dict with final report + thought steps.
        """
        if machine_id not in MACHINES:
            return {"error": f"Unknown machine: {machine_id}"}

        self.logger.clear()

        query = (
            f"Check machine {machine_id}. "
            f"Analyze its current sensor readings, assess health, detect any trends, "
            f"and take the appropriate maintenance action if needed. "
            f"Provide a full maintenance report."
        )

        try:
            result = self.executor.invoke({"input": query})
            final  = result.get("output", "No output returned.")
        except Exception as e:
            final = f"Agent error: {str(e)}"

        # record to memory
        memory.record_decision(machine_id, {
            "action":  "agent_run_complete",
            "summary": final[:200],
        })

        return {
            "machine_id":   machine_id,
            "final_report": final,
            "thought_steps": list(self.logger.steps),
        }

    def run_all(self) -> list[dict]:
        """Check all machines sequentially."""
        return [self.run(mid) for mid in MACHINES]


# ── Singleton ─────────────────────────────────────
agent = MaintenanceAgent()
