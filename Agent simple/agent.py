"""
agent.py
--------
Autonomous Predictive Maintenance Agent — LangChain 0.3.x, zero warnings.

Import map (all verified stable on langchain 0.3.x)
----------------------------------------------------
  from langchain.agents       import AgentExecutor, create_react_agent
  from langchain_core.prompts import PromptTemplate
  (langchain_core.tools @tool is used in tools.py)

Memory
------
  No LangChain memory class is used.
  We maintain a plain Python deque of (human_str, ai_str) pairs and format
  them as a {chat_history} string injected into the prompt on every call.
  This is the recommended pattern after ConversationBufferWindowMemory was
  deprecated in LangChain 0.3.1.

ReAct loop
----------
  create_react_agent builds a Runnable that follows the classic
  Thought / Action / Action Input / Observation / Final Answer format.
  AgentExecutor drives the loop, captures intermediate_steps, and handles
  malformed LLM output gracefully (handle_parsing_errors=True).
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

# ── Stable langchain 0.3.x imports ───────────────────────────────────────── #
from langchain.agents       import AgentExecutor, create_react_agent   # 0.3.x
from langchain_core.prompts import PromptTemplate                       # core


# ─────────────────────────────────────────────────────────────────────────── #
#  Configuration
# ─────────────────────────────────────────────────────────────────────────── #

HISTORY_WINDOW = 6   # Number of (human, ai) pairs to keep in context
MAX_ITERATIONS = 8   # Hard cap on ReAct loop iterations per cycle


# ─────────────────────────────────────────────────────────────────────────── #
#  ReAct Prompt Template
#
#  LangChain PromptTemplate uses single braces for variables.
#  Literal braces that should appear in the output are doubled: {{ }}
#
#  Required template variables:
#    {tools}            – tool descriptions (injected by create_react_agent)
#    {tool_names}       – comma list of tool names (injected)
#    {chat_history}     – our manual history string
#    {window}           – history window size (for the header)
#    {input}            – the current human message
#    {agent_scratchpad} – running Thought/Action trace (injected by executor)
# ─────────────────────────────────────────────────────────────────────────── #

_PROMPT_TEXT = (
    "You are an Autonomous Predictive Maintenance Agent for industrial equipment.\n"
    "Analyse sensor data, detect faults, and take the correct action using tools.\n\n"
    "Recent cycle history (last {window} cycles):\n"
    "{chat_history}\n\n"
    "Available tools:\n"
    "{tools}\n\n"
    "Tool names: {tool_names}\n\n"
    "Mandatory workflow per cycle\n"
    "----------------------------\n"
    "1. Call check_sensor_status  with the raw sensor JSON.\n"
    "2. Call diagnose_condition   with the output of step 1.\n"
    "3. Based on recommended_action in the diagnosis:\n"
    '   - "Maintenance" -> call schedule_maintenance\n'
    '   - "Alert"       -> call send_alert\n'
    '   - "Continue"    -> call log_normal_cycle\n'
    "4. Optionally call get_trend_analysis for borderline cases.\n"
    "5. Produce a Final Answer.\n\n"
    "You MUST use this exact format for every step:\n\n"
    "Thought: <one or two sentences of reasoning>\n"
    "Action: <exact tool name from tool names list>\n"
    "Action Input: <valid JSON string>\n"
    "Observation: <tool result — do NOT write this yourself>\n"
    "... (repeat Thought/Action/Action Input/Observation until done)\n"
    "Thought: I now have all the information I need.\n"
    'Final Answer: {{"status": "<Normal|Warning|Failure>", '
    '"action": "<Continue|Alert|Maintenance>", '
    '"risk": "<Low|Medium|High>", '
    '"tool_used": "<last action tool name>", '
    '"diagnoses": ["<fault>"], '
    '"summary": "<one sentence>"}}\n\n'
    "Rules\n"
    "-----\n"
    "- Always quote JSON keys with double quotes.\n"
    "- Never skip steps 1 and 2.\n"
    "- Always end with a Final Answer on its own line.\n"
    '- "Continue" means risk must be "Low".\n'
    "- Keep Thought steps to one or two sentences.\n\n"
    "Input: {input}\n\n"
    "{agent_scratchpad}"
)

REACT_PROMPT = PromptTemplate.from_template(_PROMPT_TEXT)


# ─────────────────────────────────────────────────────────────────────────── #
#  Agent class
# ─────────────────────────────────────────────────────────────────────────── #

class MaintenanceAgent:
    """
    Single LangChain ReAct agent for autonomous predictive maintenance.

    Uses langchain 0.3.x stable APIs only:
      - create_react_agent (langchain.agents)
      - AgentExecutor      (langchain.agents)
      - PromptTemplate     (langchain_core.prompts)
      - BaseChatModel      (langchain_core — via llm_client.py)
      - @tool              (langchain_core — via tools.py)

    Memory is a plain Python deque — no deprecated LangChain memory class.
    """

    def __init__(self):

        # ── LLM ─────────────────────────────────────────────────── #
        self.llm = CapgeminiNovaLite(max_tokens=512, temperature=0.1)

        # ── Tools (all decorated with @tool from langchain_core) ─── #
        self.tools = [
            check_sensor_status,
            diagnose_condition,
            send_alert,
            schedule_maintenance,
            log_normal_cycle,
            get_trend_analysis,
        ]

        # ── ReAct agent runnable ─────────────────────────────────── #
        react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=REACT_PROMPT,
        )

        # ── Executor ─────────────────────────────────────────────── #
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            verbose=True,               # Full ReAct chain printed to console
            handle_parsing_errors=True, # Recover from malformed LLM output
            max_iterations=MAX_ITERATIONS,
            return_intermediate_steps=True,
        )

        # ── Manual rolling chat history (plain deque) ─────────────── #
        # Each entry: (human_msg_str, ai_final_answer_str)
        self._chat_history: deque = deque(maxlen=HISTORY_WINDOW)

        # ── Counters ─────────────────────────────────────────────── #
        self.cycle_count = 0

    # ── History helpers ────────────────────────────────────────────── #

    def _format_chat_history(self) -> str:
        """Render the deque as a readable string for the prompt."""
        if not self._chat_history:
            return "(no previous cycles recorded yet)"
        lines = []
        for human, ai in self._chat_history:
            lines.append(f"Human: {human}")
            lines.append(f"Agent: {ai}")
        return "\n".join(lines)

    def _save_to_history(self, human_msg: str, ai_answer: str):
        """Append one exchange to the rolling window."""
        self._chat_history.append((
            human_msg[:150],    # keep short to save tokens
            ai_answer[:200],
        ))

    # ── Main public method ─────────────────────────────────────────── #

    def run_cycle(self, reading: dict) -> dict:
        """
        Execute one full autonomous agent cycle.

        Args:
            reading : dict produced by SensorSimulator.read()

        Returns:
            Structured result dict ready for streamlit_app.py.
        """
        self.cycle_count += 1
        mem.add_reading(reading)

        # Human message for this cycle
        sensor_payload = json.dumps({
            "tick"        : reading["tick"],
            "temperature" : reading["temperature"],
            "vibration"   : reading["vibration"],
            "pressure"    : reading["pressure"],
        })
        human_message = (
            f"Analyse sensor reading at tick #{reading['tick']}. "
            f"Sensor data: {sensor_payload}"
        )

        try:
            raw = self.executor.invoke({
                "input"        : human_message,
                "chat_history" : self._format_chat_history(),
                "window"       : HISTORY_WINDOW,
            })
        except Exception as exc:
            return self._fallback_result(reading, str(exc))

        final_answer = raw.get("output", "")
        intermediate = raw.get("intermediate_steps", [])

        # Save to history window
        self._save_to_history(human_message, final_answer)

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
        """Parse Final Answer JSON + intermediate steps into a UI-ready dict."""

        # Attempt to parse the Final Answer as JSON
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

        # Status is derived from action (single source of truth)
        status = {"Maintenance": "Failure", "Alert": "Warning"}.get(action, "Normal")

        # Unpack intermediate steps
        tool_calls   = []
        tool_results = {}
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

        # Enrich from intermediate steps when Final Answer was sparse
        sensor_status = self._extract_sensor_status(tool_results)
        if not diagnoses:
            diagnoses = self._extract_diagnoses(tool_results)
        if not tool_used or tool_used == "log_normal_cycle":
            tool_used = self._extract_primary_tool(tool_calls)

        confidence = "High" if json_match else "Medium"

        return {
            "tick"         : reading["tick"],
            "reading"      : reading,
            "sensor_status": sensor_status,
            "status"       : status,
            "action"       : action,
            "risk"         : risk,
            "tool_used"    : tool_used,
            "tool_calls"   : tool_calls,
            "diagnoses"    : diagnoses if isinstance(diagnoses, list) else [str(diagnoses)],
            "summary"      : summary,
            "final_answer" : final_answer,
            "confidence"   : confidence,
            "trend_summary": mem.compute_trends(),
            "cycle_count"  : self.cycle_count,
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
            "tick"         : reading["tick"],
            "reading"      : reading,
            "sensor_status": {},
            "status"       : "Normal",
            "action"       : "Continue",
            "risk"         : "Low",
            "tool_used"    : "log_normal_cycle",
            "tool_calls"   : [],
            "diagnoses"    : [f"Agent error: {error_msg}"],
            "summary"      : "Agent encountered an error — defaulted to normal.",
            "final_answer" : error_msg,
            "confidence"   : "Low",
            "trend_summary": mem.compute_trends(),
            "cycle_count"  : self.cycle_count,
        }

    # ── Reset ──────────────────────────────────────────────────────── #

    def reset(self):
        """Clear all state when restarting the simulation."""
        self._chat_history.clear()
        self.cycle_count = 0
        mem.reset_store()
