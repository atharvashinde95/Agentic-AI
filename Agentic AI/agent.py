"""
agent.py
---------
Builds the LangChain + LangGraph tool-calling agent.

Uses langgraph.prebuilt.create_react_agent — the modern
replacement for the removed AgentExecutor in LangChain 1.x.

Flow:
  User query
      ↓
  LLM reads query + tool description
      ↓
  LLM extracts the expression  e.g. "128 + 374"
      ↓
  calculate("128 + 374") runs in Python
      ↓
  LLM formats the final answer
"""

import os
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.math_tool import calculate
from utils.llm_client import get_llm


SYSTEM_PROMPT = """You are a precise AI Math Assistant powered by Capgemini's Generative Engine.

You have ONE tool: calculate(expression)

Rules:
- Always use the calculate tool for any math. Never compute in your head.
- Extract only the numeric expression from the user's question.
  Example: "What is 5 plus 3?" → calculate("5 + 3")
- For multi-step word problems, build the full expression first.
  Example: "I have 10 apples, buy 5 more, give away 3" → calculate("10 + 5 - 3")
- After getting the result, explain it in a friendly way.
- If the question is not math-related, politely say so.
"""


def build_agent():
    """
    Create and return a langgraph ReAct agent.
    Call once, reuse across queries.
    """
    llm   = get_llm()
    agent = create_react_agent(
        model        = llm,
        tools        = [calculate],
        prompt       = SYSTEM_PROMPT,
    )
    return agent


def run_query(agent, question: str) -> dict:
    """
    Run one question through the agent.
    Returns answer text + list of tool call steps.
    """
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    answer   = messages[-1].content if messages else "No answer."

    # Extract tool call steps for display
    steps = []
    for msg in messages:
        # Tool call request (AIMessage with tool_calls)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "type":       "call",
                    "tool":       tc["name"],
                    "expression": tc["args"].get("expression", ""),
                })
        # Tool result (ToolMessage)
        if hasattr(msg, "name") and msg.name == "calculate":
            steps.append({
                "type":   "result",
                "output": msg.content,
            })

    return {"answer": answer, "steps": steps}
