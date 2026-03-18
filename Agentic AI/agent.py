"""
agent.py  —  LangGraph ReAct Agent
------------------------------------
Fix: create_react_agent from langgraph.prebuilt is deprecated in LangGraph v1.0.
     Replaced with create_agent from langchain.agents (new stable API).

Change summary:
  OLD: from langgraph.prebuilt import create_react_agent
       create_react_agent(model=llm, tools=[...], prompt="...")

  NEW: from langchain.agents import create_agent
       create_agent(model=llm, tools=[...], system_prompt="...")
"""

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent          # ✅ new stable import
from tools.math_tool import calculate
from utils.llm_client import get_llm


SYSTEM_PROMPT = """You are a precise AI Math Assistant powered by Capgemini's Generative Engine.

You have ONE tool: calculate(expression)

Rules:
- ALWAYS use the calculate tool for any math. Never compute in your head.
- Extract only the numeric expression from the user's question.
  Example: "What is 5 plus 3?" -> calculate("5 + 3")
- For word problems, build the full expression first.
  Example: "I have 10 apples, buy 5 more, give 3 away" -> calculate("10 + 5 - 3")
- After getting the result, explain it clearly.
- If not a math question, politely say so.
"""


def build_agent():
    """
    Build and return the agent.
    Uses create_agent (langchain.agents) — no deprecation warnings.
    Note: parameter is system_prompt= not prompt=
    """
    agent = create_agent(
        model         = get_llm(),
        tools         = [calculate],
        system_prompt = SYSTEM_PROMPT,   # ✅ renamed from prompt= to system_prompt=
    )
    return agent


def run_query(agent, question: str) -> dict:
    """
    Run one question through the agent.
    Returns answer + list of tool call steps for the UI trace.
    """
    result   = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    answer   = messages[-1].content if messages else "No answer."

    steps = []
    tool_was_called = False

    for msg in messages:
        # AIMessage — LLM decided to call a tool
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n  TOOL CALLED   : {tc['name']}")
                print(f"  EXPRESSION    : {tc['args'].get('expression', '')}")
                steps.append({
                    "type":       "call",
                    "tool":       tc["name"],
                    "expression": tc["args"].get("expression", ""),
                })
                tool_was_called = True

        # ToolMessage — Python returned the result
        if hasattr(msg, "name") and msg.name == "calculate":
            print(f"  PYTHON RESULT : {msg.content}")
            steps.append({"type": "result", "output": msg.content})

    if tool_was_called:
        print(f"  STATUS        : Tool was called — Python did the math")
    else:
        print(f"  STATUS        : WARNING — No tool called, LLM answered itself!")

    print(f"  FINAL ANSWER  : {answer}\n")
    return {"answer": answer, "steps": steps}
