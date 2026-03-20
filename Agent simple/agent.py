"""
agent.py
--------
Autonomous Predictive Maintenance Agent — LangChain 1.0 native API.

LangChain 1.0 changes that affect this file
────────────────────────────────────────────
  REMOVED (do NOT import):
    ❌  from langchain.agents import AgentExecutor
    ❌  from langchain.agents import create_react_agent
    ❌  from langchain.memory import ConversationBufferWindowMemory

  NEW in 1.0 (what we use):
    ✅  from langchain.agents import create_agent
    ✅  create_agent(model, tools, prompt) → compiled LangGraph graph

  The compiled graph is invoked with:
    agent.invoke({"messages": [HumanMessage(content=...)]})

Memory
──────
  LangChain 1.0 agents keep message history automatically in the graph
  state.  We additionally maintain a compact rolling summary (plain deque)
  that we prepend to the system prompt for long-term context across
  Streamlit's repeated script executions (which re-create the agent
  object each time the user presses Start).

  Because Streamlit stores the agent in session_state, the graph's own
  message history persists for the lifetime of the browser session.

Architecture
────────────
  MaintenanceAgent
    ├── CapgeminiNovaLite     (custom BaseChatModel — langchain_core)
    ├── 6 @tool functions     (tools.py — from langchain.tools import tool)
    ├── System prompt string  (injected into every invoke call)
    ├── create_agent graph    (langchain 1.0 — built on langgraph runtime)
    └── Manual history deque  (rolling plain-text summary, 6 cycles)
"""

import json
import re
from collections import deque

import memory_store as mem
from llm_client import CapgeminiNovaLite
from tools import ALL_TOOLS

# ── LangChain 1.0 imports ────────────────────────────────────────────────── #
from langchain.agents import create_agent        # NEW in langchain 1.0  ✅
from langchain_core.messages import HumanMessage, SystemMessage


# ─────────────────────────────────────────────────────────────────────────── #
#  System prompt
# ─────────────────────────────────────────────────────────────────────────── #

_SYSTEM_PROMPT_BASE = """\
You are an Autonomous Predictive Maintenance Agent for industrial equipment.
Your job is to analyse live machine sensor data and take the correct
maintenance action using the tools available to you.

MANDATORY WORKFLOW — follow this sequence every single cycle:
1. Call check_sensor_status   with the raw sensor JSON to get severity levels.
2. Call diagnose_condition    with the output of step 1 to get fault names
   and a recommended_action (Continue / Alert / Maintenance).
3. Based on recommended_action:
   - "Maintenance" → call schedule_maintenance
   - "Alert"       → call send_alert
   - "Continue"    → call log_normal_cycle
4. Optionally call get_trend_analysis for borderline or mixed-signal cases.
5. Return a final JSON summary.

FINAL ANSWER FORMAT (always end with this exact JSON):
{
  "status": "<Normal|Warning|Failure>",
  "action": "<Continue|Alert|Maintenance>",
  "risk": "<Low|Medium|High>",
  "tool_used": "<name of the action tool called in step 3>",
  "diagnoses": ["<fault description>"],
  "summary": "<one sentence>"
}

RULES:
- Always call steps 1 and 2 before deciding action.
- Use double-quoted keys in all JSON strings.
- "Continue" action must have risk "Low".
- Be concise. Limit reasoning to 1-2 sentences per step.
"""


# ─────────────────────────────────────────────────────────────────────────── #
#  Agent
# ─────────────────────────────────────────────────────────────────────────── #

HISTORY_WINDOW = 6   # Rolling summary of past N cycles


class MaintenanceAgent:
    """
    Single LangChain 1.0 autonomous agent for predictive maintenance.

    Uses:
      create_agent(model, tools, prompt)  ← langchain 1.0 native API
      BaseChatModel subclass              ← langchain_core (all versions)
      @tool decorator                     ← langchain.tools (stable in 1.0)
      Plain deque                         ← memory (no deprecated LC class)
    """

    def __init__(self):
        # ── LLM ─────────────────────────────────────────────────── #
        self.llm = CapgeminiNovaLite(max_tokens=512, temperature=0.1)

        # ── Tools ────────────────────────────────────────────────── #
        self.tools = ALL_TOOLS

        # ── Build LangChain 1.0 agent ────────────────────────────── #
        # create_agent returns a compiled LangGraph graph.
        # It accepts: model (BaseChatModel), tools (list), prompt (str).
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            prompt=_SYSTEM_PROMPT_BASE,
        )

        # ── Rolling plain-text history (cross-session context) ────── #
        self._history: deque = deque(maxlen=HISTORY_WINDOW)

        # ── Counters ─────────────────────────────────────────────── #
        self.cycle_count = 0

    # ── History helpers ────────────────────────────────────────────── #

    def _history_context(self) -> str:
        if not self._history:
            return ""
        lines = ["\nRecent cycle context:"]
        for h in self._history:
            lines.append(f"  Tick #{h['tick']}: {h['status']} | {h['action']} | {h['summary'][:80]}")
        return "\n".join(lines)

    def _save_history(self, result: dict):
        self._history.append({
            "tick"   : result.get("tick", -1),
            "status" : result.get("status", "Normal"),
            "action" : result.get("action", "Continue"),
            "summary": result.get("summary", "")[:80],
        })

    # ── Main public method ─────────────────────────────────────────── #

    def run_cycle(self, reading: dict) -> dict:
        """
        Execute one full autonomous agent cycle.

        Args:
            reading : dict from SensorSimulator.read()

        Returns:
            Structured result dict for streamlit_app.py.
        """
        self.cycle_count += 1
        mem.add_reading(reading)

        # Build the user message for this cycle
        sensor_json = json.dumps({
            "tick"        : reading["tick"],
            "temperature" : reading["temperature"],
            "vibration"   : reading["vibration"],
            "pressure"    : reading["pressure"],
        })
        user_msg = (
            f"Analyse sensor reading at tick #{reading['tick']}. "
            f"Sensor data: {sensor_json}"
            f"{self._history_context()}"
        )

        try:
            # LangChain 1.0 create_agent graph is invoked with a messages list
            raw = self.agent.invoke({
                "messages": [HumanMessage(content=user_msg)]
            })
        except Exception as exc:
            return self._fallback_result(reading, str(exc))

        # Extract the final AI message from the returned state
        messages = raw.get("messages", [])
        final_text = ""
        tool_calls_log = []

        for msg in messages:
            content = getattr(msg, "content", "")
            msg_type = type(msg).__name__

            if msg_type == "AIMessage" and content:
                final_text = str(content)   # last AIMessage = final answer

            # Capture tool call steps for the UI trace
            tool_calls_info = getattr(msg, "tool_calls", None)
            if tool_calls_info:
                for tc in tool_calls_info:
                    tool_calls_log.append({
                        "tool"       : tc.get("name", "unknown"),
                        "tool_input" : json.dumps(tc.get("args", {})),
                        "observation": "",   # filled in below
                    })

            # ToolMessages carry the tool response
            if msg_type == "ToolMessage":
                tool_call_id = getattr(msg, "tool_call_id", "")
                # Match observation back to the tool call
                for tc in reversed(tool_calls_log):
                    if tc["observation"] == "":
                        tc["observation"] = str(content)[:300]
                        break

        result = self._parse_final(
            final_text=final_text,
            tool_calls_log=tool_calls_log,
            reading=reading,
        )
        self._save_history(result)
        mem.LATEST_RESULT = result
        return result

    # ── Output parser ──────────────────────────────────────────────── #

    def _parse_final(
        self,
        final_text: str,
        tool_calls_log: list,
        reading: dict,
    ) -> dict:
        """Parse the agent's final text + tool trace into a UI-ready dict."""

        # Try to extract JSON from the final answer
        parsed = {}
        json_match = re.search(r'\{[^{}]*\}', final_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        action    = parsed.get("action",    "Continue")
        risk      = parsed.get("risk",      "Low")
        tool_used = parsed.get("tool_used", "log_normal_cycle")
        diagnoses = parsed.get("diagnoses", [])
        summary   = parsed.get("summary",   final_text[:200] if final_text else "Cycle complete.")

        # Status derived from action
        status = {"Maintenance": "Failure", "Alert": "Warning"}.get(action, "Normal")

        # Enrich from tool call log when final answer was sparse
        tool_results = {tc["tool"]: tc["observation"] for tc in tool_calls_log}

        if not diagnoses:
            diagnoses = self._extract_diagnoses(tool_results)
        sensor_status = self._extract_sensor_status(tool_results)
        if not tool_used or tool_used == "log_normal_cycle":
            tool_used = self._extract_primary_tool(tool_calls_log)

        # Also update event log for normal cycles (agent may skip log_normal_cycle)
        if status == "Normal" and not any(
            tc["tool"] == "log_normal_cycle" for tc in tool_calls_log
        ):
            mem.log_event(
                status="Normal", action="Continue", risk="Low",
                tool="log_normal_cycle",
                message=f"T={reading['temperature']}°C V={reading['vibration']}mm/s P={reading['pressure']}bar",
                tick=reading["tick"],
            )

        return {
            "tick"         : reading["tick"],
            "reading"      : reading,
            "sensor_status": sensor_status,
            "status"       : status,
            "action"       : action,
            "risk"         : risk,
            "tool_used"    : tool_used,
            "tool_calls"   : tool_calls_log,
            "diagnoses"    : diagnoses if isinstance(diagnoses, list) else [str(diagnoses)],
            "summary"      : summary,
            "final_answer" : final_text,
            "confidence"   : "High" if json_match else "Medium",
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
        for tc in reversed(tool_calls):
            if tc["tool"] in action_tools:
                return tc["tool"]
        return "log_normal_cycle"

    def _fallback_result(self, reading: dict, error_msg: str) -> dict:
        """Safe fallback when the agent graph raises an exception."""
        mem.log_event(
            status="Normal", action="Continue", risk="Low",
            tool="fallback", message=f"Agent error: {error_msg}", tick=reading["tick"],
        )
        return {
            "tick"         : reading["tick"],
            "reading"      : reading,
            "sensor_status": {},
            "status"       : "Normal",
            "action"       : "Continue",
            "risk"         : "Low",
            "tool_used"    : "fallback",
            "tool_calls"   : [],
            "diagnoses"    : [f"Agent error: {error_msg[:120]}"],
            "summary"      : "Agent error — defaulted to normal.",
            "final_answer" : error_msg,
            "confidence"   : "Low",
            "trend_summary": mem.compute_trends(),
            "cycle_count"  : self.cycle_count,
        }

    # ── Reset ──────────────────────────────────────────────────────── #

    def reset(self):
        """Clear all state. Rebuild the agent graph."""
        self._history.clear()
        self.cycle_count = 0
        mem.reset_store()
        # Rebuild agent to get a fresh message history in the graph state
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            prompt=_SYSTEM_PROMPT_BASE,
        )
