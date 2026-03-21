"""
agent.py
---------
MaintenanceAgent — built with LangChain + LangGraph.

Architecture:
  - LLM     : ChatOpenAI pointed at your Capgemini endpoint (OpenAI-compatible)
  - Tools   : LangChain @tool decorated functions from tools.py
  - Loop    : LangGraph create_react_agent — full ReAct cycle
                Reason → Act → Observe → Reason → ... → Done

The system prompt is the one provided, which instructs the LLM to:
  - Detect anomalies against defined thresholds
  - Assign health (Healthy / Warning / Critical) and risk (Low / Medium / High)
  - Autonomously select and call the right tools
  - Return a strict JSON final assessment
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS


# ─────────────────────────────────────────────────────────────
# CONFIGURATION — Set your Capgemini API credentials here
# ─────────────────────────────────────────────────────────────
API_KEY = "YOUR_CAPGEMINI_API_KEY"       # ← Replace with your key
API_URL = "YOUR_CAPGEMINI_ENDPOINT_URL"  # ← Replace with your endpoint
MODEL   = "amazon.nova.lite-v1:0"


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an Autonomous Predictive Maintenance and Decision-Making AI Agent used in industrial environments.
Your role is NOT just to analyze data, but to:
- Detect anomalies
- Predict failure risks
- Make decisions
- Recommend and trigger operational actions
You must behave like an intelligent agent that can take proactive decisions.

-------------------------------------
INPUT DATA:
You will receive machine sensor data in the following format:
{
  "machine_id": "<id>",
  "temperature": <value in °C>,
  "vibration": <value>,
  "load": <percentage>,
  "runtime_hours": <value>
}

-------------------------------------
YOUR TASKS:
1. Analyze the machine condition based on the input data.
2. Identify if there are any anomalies:
   - High temperature (>85°C)
   - High vibration (>7)
   - High load (>85%)
   - Long runtime (>100 hours)
3. Predict failure risk:
   - "Low" → Normal operation
   - "Medium" → Warning state
   - "High" → Likely failure soon
4. Determine machine health:
   - "Healthy"
   - "Warning"
   - "Critical"
5. Decide which tools to call autonomously based on your assessment (VERY IMPORTANT):
   Available tools and when to call them:
   - monitor_closely         → Low risk / no anomalies / healthy machine
   - reduce_machine_load     → Medium or High risk / high temperature, vibration, or load
   - schedule_maintenance    → High risk / machine needs physical inspection
   - shift_to_backup         → High risk / primary machine must be relieved
   - immediate_shutdown      → EMERGENCY: temperature > 100°C OR vibration > 9 — call FIRST
6. After calling all necessary tools, provide your final assessment as strict JSON.

-------------------------------------
DECISION RULES:
  LOW risk    (0 anomalies)   → call: monitor_closely only
  MEDIUM risk (1–2 anomalies) → call: reduce_machine_load, then monitor_closely
  HIGH risk   (3+ anomalies)  → call: reduce_machine_load, schedule_maintenance, shift_to_backup
  EMERGENCY   (temp > 100 OR vibration > 9) → call: immediate_shutdown FIRST, then others

-------------------------------------
IMPORTANT RULES:
- NEVER call the same tool twice
- Always call immediate_shutdown first if emergency conditions exist
- If multiple anomalies exist → increase risk level
- If temperature AND vibration are both high → mark as "Critical"
- If risk is High → must include at least 2 strong actions
- Prioritize safety over performance
- Be deterministic and logical

-------------------------------------
FINAL OUTPUT FORMAT (strict JSON only, after all tools are called):
{
  "machine_id": "<from input>",
  "health": "Healthy | Warning | Critical",
  "risk": "Low | Medium | High",
  "anomalies": ["list of detected anomaly strings"],
  "actions": ["list of action labels taken"],
  "reason": "Short explanation of why these decisions were made"
}
"""


# ─────────────────────────────────────────────────────────────
# BUILD AGENT
# ─────────────────────────────────────────────────────────────
def build_agent():
    """
    Initialise the LLM and build the LangGraph ReAct agent.

    ChatOpenAI is used because:
      - It supports any OpenAI-compatible endpoint via base_url
      - It supports .bind_tools() which LangGraph requires
      - Capgemini's Nova Lite endpoint is OpenAI-format compatible

    create_react_agent returns a compiled StateGraph that automatically
    handles the full Reason → Act → Observe loop.
    """
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=API_URL,
        temperature=0,       # Deterministic — same input = same output always
        max_tokens=1024,
    )

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ─────────────────────────────────────────────────────────────
# PARSE FINAL JSON FROM AGENT RESPONSE
# ─────────────────────────────────────────────────────────────
def extract_final_json(text: str) -> dict:
    """
    Extract and parse the strict JSON block from the agent's final message.
    Handles cases where the LLM wraps JSON in markdown code fences.

    Returns the parsed dict, or an empty dict if parsing fails.
    """
    if not text:
        return {}

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Find the outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────────────────────────
# RUN AGENT
# ─────────────────────────────────────────────────────────────
def run_agent(
    machine_id: str,
    temperature: float,
    vibration: float,
    load: float,
    runtime_hours: float,
) -> dict:
    """
    Run the MaintenanceAgent for the given sensor readings.

    The user message passes sensor data as JSON (matching the format
    defined in the system prompt) so the LLM receives exactly what
    it was instructed to expect.

    Returns:
        {
            "steps": [
                {
                    "step_number": int,
                    "type":        "tool_call" | "observation" | "final",
                    "content":     str,
                    "tool_name":   str | None,
                }
            ],
            "final_json":    dict,   ← parsed strict JSON from agent
            "final_summary": str,    ← raw final message text
            "error":         str | None,
        }
    """

    # Build user message as JSON — matches system prompt's expected input format
    sensor_payload = {
        "machine_id":     machine_id,
        "temperature":    temperature,
        "vibration":      vibration,
        "load":           load,
        "runtime_hours":  runtime_hours,
    }

    user_message = (
        f"Analyze this machine and take all necessary maintenance actions.\n\n"
        f"Sensor data:\n{json.dumps(sensor_payload, indent=2)}"
    )

    # Build agent
    try:
        agent = build_agent()
    except Exception as e:
        return {
            "steps": [], "final_json": {}, "final_summary": "",
            "error": f"Failed to build agent: {str(e)}",
        }

    # Invoke agent
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]}
        )
    except Exception as e:
        return {
            "steps": [], "final_json": {}, "final_summary": "",
            "error": f"Agent invocation failed: {str(e)}",
        }

    # ── PARSE MESSAGE TRACE ──────────────────────────────────
    steps         = []
    final_summary = ""
    final_json    = {}
    step_num      = 0

    messages = result.get("messages", [])

    for msg in messages:

        if isinstance(msg, HumanMessage):
            continue  # Skip input message

        if isinstance(msg, SystemMessage):
            continue  # Skip system prompt echo

        step_num += 1

        # AIMessage — agent reasoning + optional tool calls
        if isinstance(msg, AIMessage):
            content    = msg.content or ""
            tool_calls = getattr(msg, "tool_calls", []) or []

            if tool_calls:
                for tc in tool_calls:
                    tool_name = (
                        tc.get("name", "unknown")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "unknown")
                    )
                    steps.append({
                        "step_number": step_num,
                        "type":        "tool_call",
                        "content":     content if content else f"Decided to call: {tool_name}",
                        "tool_name":   tool_name,
                    })
            else:
                # Final message — no more tool calls
                steps.append({
                    "step_number": step_num,
                    "type":        "final",
                    "content":     content,
                    "tool_name":   None,
                })
                final_summary = content
                final_json    = extract_final_json(content)

        # ToolMessage — observation from a tool execution
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or "tool"
            steps.append({
                "step_number": step_num,
                "type":        "observation",
                "content":     msg.content,
                "tool_name":   tool_name,
            })

    return {
        "steps":         steps,
        "final_json":    final_json,
        "final_summary": final_summary,
        "error":         None,
    }
