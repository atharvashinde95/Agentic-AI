"""
agent.py
--------
The Autonomous Predictive Maintenance Agent — built entirely on LangChain.

Architecture
────────────
  MaintenanceAgent
    ├── LLM              : CapgeminiNovaLite (custom BaseChatModel)
    ├── Tools            : 6 @tool functions from tools.py
    ├── ReAct Prompt     : System prompt that shapes agent reasoning
    ├── AgentExecutor    : LangChain's Thought→Action→Observation loop
    └── ConversationBufferWindowMemory : Keeps last K exchanges in context

How it works per cycle
──────────────────────
  1. Sensor reading is serialised to JSON and handed to the agent.
  2. The AgentExecutor runs the ReAct loop:
       Thought   → LLM reasons about what to do
       Action    → LLM picks a tool from the 6 available
       Action Input → LLM produces the JSON argument for that tool
       Observation → Tool executes and returns its result
       ... (loop until the LLM outputs "Final Answer:")
  3. The final answer and all intermediate steps are parsed and
     returned as a structured result dict for the Streamlit UI.
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

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


# ─────────────────────────────────────────────────────────────────────────── #
#  ReAct system prompt
# ─────────────────────────────────────────────────────────────────────────── #

SYSTEM_PROMPT = """You are an Autonomous Predictive Maintenance Agent.
Your job: analyse live industrial sensor data, detect faults, and take the
correct maintenance action using your available tools.

TOOLS AVAILABLE:
{tools}

TOOL NAMES: {tool_names}

STRICT WORKFLOW — follow this every single cycle:
  Step 1. Call check_sensor_status with the raw sensor JSON to get severity levels.
  Step 2. Call diagnose_condition with the output of step 1 to get fault names and recommended_action.
  Step 3. Based on recommended_action:
          - "Maintenance" → call schedule_maintenance
          - "Alert"       → call send_alert
          - "Continue"    → call log_normal_cycle
  Step 4. Optionally call get_trend_analysis if you want richer context.
  Step 5. Output a Final Answer summarising: status, action taken, risk level, key findings.

FORMAT — you MUST use this exact format:
Thought: <your reasoning>
Action: <tool name exactly as listed>
Action Input: <valid JSON string>
Observation: <tool result — filled in automatically>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to give the final answer.
Final Answer: <JSON with keys: status, action, risk, tool_used, diagnoses, summary>

RULES:
- Always use valid JSON strings as Action Inputs.
- Never skip check_sensor_status or diagnose_condition.
- Always end with a Final Answer in JSON format.
- Be concise in Thought steps.
- For "Continue" action the Final Answer risk must be "Low".

{agent_scratchpad}
"""


# ─────────────────────────────────────────────────────────────────────────── #
#  Agent class
# ─────────────────────────────────────────────────────────────────────────── #

class MaintenanceAgent:
    """
    Single LangChain-powered autonomous agent.

    Public API:
      agent.run_cycle(reading: dict) → result dict
      agent.reset()
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
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

        # ── Memory (keeps last 6 human/AI exchanges) ─────────────── #
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=6,
            return_messages=True,
        )

        # ── Build ReAct agent + executor ─────────────────────────── #
        react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
        self.executor = AgentExecutor(
            agent=react_agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,               # prints full ReAct chain to console
            handle_parsing_errors=True, # gracefully handle malformed LLM output
            max_iterations=8,           # prevent runaway loops
            return_intermediate_steps=True,
        )

        # ── Internal counters ────────────────────────────────────── #
        self.cycle_count = 0

    # ── Public: run one full agent cycle ──────────────────────────── #

    def run_cycle(self, reading: dict) -> dict:
        """
        Execute one autonomous agent cycle.

        Args:
            reading: dict from SensorSimulator.read()

        Returns:
            Structured result dict ready for Streamlit UI.
        """
        self.cycle_count += 1

        # Push reading into shared memory store for tools to access
        mem.add_reading(reading)

        # Serialise reading for the agent's input
        sensor_input = json.dumps({
            "tick"       : reading["tick"],
            "temperature": reading["temperature"],
            "vibration"  : reading["vibration"],
            "pressure"   : reading["pressure"],
        })

        human_message = (
            f"Analyse the following sensor reading and take appropriate action.\n"
            f"Sensor data: {sensor_input}"
        )

        try:
            raw = self.executor.invoke({"input": human_message})
        except Exception as e:
            # If the executor fails entirely, return a safe fallback
            return self._fallback_result(reading, str(e))

        # ── Parse agent output ────────────────────────────────────── #
        final_answer = raw.get("output", "")
        intermediate = raw.get("intermediate_steps", [])

        result = self._parse_final_answer(
            final_answer=final_answer,
            intermediate_steps=intermediate,
            reading=reading,
        )

        # Write to shared store so Streamlit can read it
        mem.LATEST_RESULT = result
        return result

    # ── Internal helpers ──────────────────────────────────────────── #

    def _parse_final_answer(
        self,
        final_answer: str,
        intermediate_steps: list,
        reading: dict,
    ) -> dict:
        """
        Parse the agent's Final Answer (which should be JSON) into a
        structured result dict. Falls back to reasonable defaults if
        the LLM did not produce valid JSON.
        """
        # Try to parse the final answer as JSON
        parsed_answer = {}
        json_match = re.search(r'\{.*\}', final_answer, re.DOTALL)
        if json_match:
            try:
                parsed_answer = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed_answer = {}

        # Extract fields with sensible defaults
        status   = parsed_answer.get("status",    "Normal")
        action   = parsed_answer.get("action",    "Continue")
        risk     = parsed_answer.get("risk",      "Low")
        tool_used = parsed_answer.get("tool_used", "log_normal_cycle")
        diagnoses = parsed_answer.get("diagnoses", [])
        summary   = parsed_answer.get("summary",   final_answer[:200])

        # Map action → status label
        if action == "Maintenance":
            status = "Failure"
        elif action == "Alert":
            status = "Warning"
        else:
            status = "Normal"

        # Collect tool calls from intermediate steps
        tool_calls = []
        tool_results = {}
        for step in intermediate_steps:
            if len(step) == 2:
                agent_action, observation = step
                tool_name = getattr(agent_action, "tool", "unknown")
                tool_input = getattr(agent_action, "tool_input", "")
                tool_calls.append({
                    "tool"       : tool_name,
                    "tool_input" : tool_input,
                    "observation": observation,
                })
                tool_results[tool_name] = observation

        # Try to pull richer info from intermediate steps
        sensor_status = self._extract_sensor_status(tool_results)
        if not diagnoses:
            diagnoses = self._extract_diagnoses(tool_results)
        if not tool_used or tool_used == "log_normal_cycle":
            tool_used = self._extract_primary_tool(tool_calls)

        # Confidence based on how certain the agent's answer looked
        confidence = "High" if json_match else "Medium"

        return {
            "tick"            : reading["tick"],
            "reading"         : reading,
            "sensor_status"   : sensor_status,
            "status"          : status,
            "action"          : action,
            "risk"            : risk,
            "tool_used"       : tool_used,
            "tool_calls"      : tool_calls,
            "diagnoses"       : diagnoses if isinstance(diagnoses, list) else [str(diagnoses)],
            "summary"         : summary,
            "final_answer"    : final_answer,
            "confidence"      : confidence,
            "trend_summary"   : mem.compute_trends(),
            "cycle_count"     : self.cycle_count,
        }

    def _extract_sensor_status(self, tool_results: dict) -> dict:
        """Pull sensor_status dict out of check_sensor_status tool result."""
        raw = tool_results.get("check_sensor_status", "")
        if raw:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                return parsed.get("sensor_status", {})
            except Exception:
                pass
        return {"temperature": "unknown", "vibration": "unknown", "pressure": "unknown"}

    def _extract_diagnoses(self, tool_results: dict) -> list:
        """Pull faults list out of diagnose_condition tool result."""
        raw = tool_results.get("diagnose_condition", "")
        if raw:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                return parsed.get("faults", [])
            except Exception:
                pass
        return []

    def _extract_primary_tool(self, tool_calls: list) -> str:
        """
        Identify the last action tool used
        (the one that took the maintenance/alert/log action).
        """
        action_tools = {
            "send_alert", "schedule_maintenance", "log_normal_cycle"
        }
        for step in reversed(tool_calls):
            if step["tool"] in action_tools:
                return step["tool"]
        return "log_normal_cycle"

    def _fallback_result(self, reading: dict, error_msg: str) -> dict:
        """Return a safe result dict when the executor fails entirely."""
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

    def reset(self):
        """Clear memory and counters — call when restarting simulation."""
        self.memory.clear()
        self.cycle_count = 0
        mem.reset_store()
