"""
agent.py
--------
Autonomous Predictive Maintenance Agent — LangChain ReAct (modern, warning-free).

Memory approach (LangChain v0.3+)
──────────────────────────────────
ConversationBufferWindowMemory was deprecated in v0.3.1 and will be removed
in v1.0.  The modern idiomatic approach for a ReAct agent is:

  • Keep a plain Python deque of (human, ai) string pairs ourselves.
  • Format them as a "chat_history" string on every invoke() call.
  • Pass that string via the prompt variable {chat_history}.

This gives identical behaviour with zero deprecation warnings and zero
extra dependencies.

Architecture
────────────
  MaintenanceAgent
    ├── CapgeminiNovaLite   (custom BaseChatModel — llm_client.py)
    ├── 6 @tool functions   (tools.py)
    ├── ReAct PromptTemplate
    ├── AgentExecutor       (create_react_agent, max 8 iterations)
    └── Manual chat history window (last K=6 exchanges, plain deque)
"""

import json
import re
from collections import deque

import memory_store as mem
from llm_client import CapgeminiNovaLite
from tools import (
    check_sensor_status,
    diagnose_condition,
    send_alert,
    schedule_maintenance,
    log_normal_cycle,
    get_trend_analysis,
)

# ── Modern LangChain imports (no deprecated classes) ──────────────────────── #
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────────────────── #
#  History window size
# ─────────────────────────────────────────────────────────────────────────── #

HISTORY_WINDOW = 6   # Keep last N (human, ai) exchange pairs


# ─────────────────────────────────────────────────────────────────────────── #
#  ReAct prompt — includes {chat_history} for the manual memory window
# ─────────────────────────────────────────────────────────────────────────── #

REACT_PROMPT_TEMPLATE = """You are an Autonomous Predictive Maintenance Agent.
Your job: analyse live industrial sensor data, detect faults, and take the
correct maintenance action using your available tools.

Previous conversation context (most recent {window} cycles):
{chat_history}

TOOLS AVAILABLE:
{tools}

TOOL NAMES: {tool_names}

STRICT WORKFLOW — follow this exact sequence every single cycle:
  Step 1. Call check_sensor_status with the raw sensor JSON to get severity levels.
  Step 2. Call diagnose_condition with the output of step 1 to identify faults and get recommended_action.
  Step 3. Based on recommended_action:
          - "Maintenance" → call schedule_maintenance
          - "Alert"       → call send_alert
          - "Continue"    → call log_normal_cycle
  Step 4. (Optional) Call get_trend_analysis for richer context on borderline cases.
  Step 5. Output a Final Answer summarising the cycle outcome.

MANDATORY FORMAT — you MUST follow this exactly:
Thought: <your reasoning about the current sensor state>
Action: <one tool name exactly as listed in TOOL NAMES>
Action Input: <valid JSON string as the tool argument>
Observation: <tool result — filled in automatically, do not write this>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information to give the final answer.
Final Answer: {{"status": "<Normal|Warning|Failure>", "action": "<Continue|Alert|Maintenance>", "risk": "<Low|Medium|High>", "tool_used": "<action tool name>", "diagnoses": ["<fault 1>", "..."], "summary": "<one sentence>"}}

RULES:
- Always use double-quoted keys in JSON Action Inputs.
- Never skip steps 1 and 2.
- Always end with a Final Answer line in the JSON format shown above.
- For "Continue" the risk must always be "Low".
- Be concise in Thought steps — one or two sentences maximum.

{agent_scratchpad}"""


# ─────────────────────────────────────────────────────────────────────────── #
#  Agent
# ─────────────────────────────────────────────────────────────────────────── #

class MaintenanceAgent:
    """
    Single LangChain ReAct agent for autonomous predictive maintenance.

    Memory is managed as a plain rolling deque of string pairs —
    no deprecated LangChain memory classes used.

    Public API:
        result = agent.run_cycle(reading: dict)  → structured result dict
        agent.reset()                             → clear state
    """

    def __init__(self):
        # ── LLM ─────────────────────────────────────────────────── #
        self.llm = CapgeminiNovaLite(max_tokens=512, temperature=0.1)

        # ── Tools ────────────────────────────────────────────────── #
        self.tools = [
            check_sensor_status,
            diagnose_condition,
            send_alert,
            schedule_maintenance,
            log_normal_cycle,
            get_trend_analysis,
        ]

        # ── Prompt ───────────────────────────────────────────────── #
        self.prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

        # ── Manual rolling chat history (replaces deprecated memory) #
        # Each entry is a tuple: (human_msg: str, ai_msg: str)
        self._chat_history: deque = deque(maxlen=HISTORY_WINDOW)

        # ── AgentExecutor ────────────────────────────────────────── #
        react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,               # prints full ReAct chain to console
            handle_parsing_errors=True, # gracefully recover from bad LLM output
            max_iterations=8,
            return_intermediate_steps=True,
        )

        # ── Counters ─────────────────────────────────────────────── #
        self.cycle_count = 0

    # ── Build chat history string ──────────────────────────────────── #

    def _format_chat_history(self) -> str:
        """
        Render the rolling window of past cycles as a plain string.
        Example:
            Human: Analyse tick #4 — T=70°C …
            AI: {"status": "Normal", "action": "Continue", …}
        """
        if not self._chat_history:
            return "(no previous cycles yet)"
        lines = []
        for human, ai in self._chat_history:
            lines.append(f"Human: {human}")
            lines.append(f"AI: {ai}")
        return "\n".join(lines)

    def _update_history(self, human_msg: str, ai_response: str):
        """Append one (human, ai) pair to the rolling window."""
        self._chat_history.append((human_msg, ai_response))

    # ── Main public method ─────────────────────────────────────────── #

    def run_cycle(self, reading: dict) -> dict:
        """
        Execute one full autonomous agent cycle.

        Args:
            reading : dict from SensorSimulator.read()

        Returns:
            Structured result dict consumed by streamlit_app.py.
        """
        self.cycle_count += 1

        # Push reading into shared memory store (tools read from here)
        mem.add_reading(reading)

        # Build the human message for this cycle
        sensor_payload = json.dumps({
            "tick"       : reading["tick"],
            "temperature": reading["temperature"],
            "vibration"  : reading["vibration"],
            "pressure"   : reading["pressure"],
        })
        human_message = (
            f"Analyse sensor reading at tick #{reading['tick']} and take action.\n"
            f"Sensor data: {sensor_payload}"
        )

        try:
            raw = self.executor.invoke({
                "input"       : human_message,
                "chat_history": self._format_chat_history(),
                "window"      : HISTORY_WINDOW,
            })
        except Exception as e:
            return self._fallback_result(reading, str(e))

        final_answer = raw.get("output", "")
        intermediate = raw.get("intermediate_steps", [])

        # Update our manual history window with this exchange
        self._update_history(human_message[:120], final_answer[:200])

        result = self._parse_output(
            final_answer=final_answer,
            intermediate_steps=intermediate,
            reading=reading,
        )
        mem.LATEST_RESULT = result
        return result

    # ── Output parser ──────────────────────────────────────────────── #

    def _parse_output(
        self,
        final_answer: str,
        intermediate_steps: list,
        reading: dict,
    ) -> dict:
        """
        Parse the agent's Final Answer JSON and the intermediate steps
        into a clean result dict for the UI.
        """
        # Try to parse Final Answer as JSON
        parsed = {}
        json_match = re.search(r'\{[^{}]*\}', final_answer, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = {}

        action    = parsed.get("action",    "Continue")
        risk      = parsed.get("risk",      "Low")
        tool_used = parsed.get("tool_used", "log_normal_cycle")
        diagnoses = parsed.get("diagnoses", [])
        summary   = parsed.get("summary",   final_answer[:200])

        # Derive status from action (authoritative)
        status = {"Maintenance": "Failure", "Alert": "Warning"}.get(action, "Normal")

        # Unpack intermediate steps
        tool_calls    = []
        tool_results  = {}
        for step in intermediate_steps:
            if len(step) == 2:
                agent_action, observation = step
                t_name  = getattr(agent_action, "tool",       "unknown")
                t_input = getattr(agent_action, "tool_input", "")
                tool_calls.append({
                    "tool"       : t_name,
                    "tool_input" : t_input,
                    "observation": observation,
                })
                tool_results[t_name] = observation

        # Enrich from tool outputs when LLM's final answer was sparse
        sensor_status = self._extract_sensor_status(tool_results)
        if not diagnoses:
            diagnoses = self._extract_diagnoses(tool_results)
        if not tool_used or tool_used == "log_normal_cycle":
            tool_used = self._extract_primary_tool(tool_calls)

        confidence = "High" if json_match else "Medium"

        return {
            "tick"          : reading["tick"],
            "reading"       : reading,
            "sensor_status" : sensor_status,
            "status"        : status,
            "action"        : action,
            "risk"          : risk,
            "tool_used"     : tool_used,
            "tool_calls"    : tool_calls,
            "diagnoses"     : diagnoses if isinstance(diagnoses, list) else [str(diagnoses)],
            "summary"       : summary,
            "final_answer"  : final_answer,
            "confidence"    : confidence,
            "trend_summary" : mem.compute_trends(),
            "cycle_count"   : self.cycle_count,
        }

    # ── Extraction helpers ─────────────────────────────────────────── #

    def _extract_sensor_status(self, tool_results: dict) -> dict:
        raw = tool_results.get("check_sensor_status", "")
        if raw:
            try:
                p = json.loads(raw) if isinstance(raw, str) else raw
                return p.get("sensor_status", {})
            except Exception:
                pass
        return {}

    def _extract_diagnoses(self, tool_results: dict) -> list:
        raw = tool_results.get("diagnose_condition", "")
        if raw:
            try:
                p = json.loads(raw) if isinstance(raw, str) else raw
                return p.get("faults", [])
            except Exception:
                pass
        return []

    def _extract_primary_tool(self, tool_calls: list) -> str:
        action_tools = {"send_alert", "schedule_maintenance", "log_normal_cycle"}
        for step in reversed(tool_calls):
            if step["tool"] in action_tools:
                return step["tool"]
        return "log_normal_cycle"

    def _fallback_result(self, reading: dict, error_msg: str) -> dict:
        return {
            "tick"          : reading["tick"],
            "reading"       : reading,
            "sensor_status" : {},
            "status"        : "Normal",
            "action"        : "Continue",
            "risk"          : "Low",
            "tool_used"     : "log_normal_cycle",
            "tool_calls"    : [],
            "diagnoses"     : [f"Agent error: {error_msg}"],
            "summary"       : "Fallback to normal — agent encountered an error.",
            "final_answer"  : error_msg,
            "confidence"    : "Low",
            "trend_summary" : mem.compute_trends(),
            "cycle_count"   : self.cycle_count,
        }

    # ── Reset ──────────────────────────────────────────────────────── #

    def reset(self):
        """Clear all state — call when restarting the simulation."""
        self._chat_history.clear()
        self.cycle_count = 0
        mem.reset_store()
