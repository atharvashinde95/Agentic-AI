"""
agent.py  —  LangGraph ReAct Agent
"""
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
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
    agent = create_react_agent(
        model=get_llm(),
        tools=[calculate],
        prompt=SYSTEM_PROMPT,
    )
    return agent

def run_query(agent, question: str) -> dict:
    result   = agent.invoke({"messages": [HumanMessage(content=question)]})
    messages = result.get("messages", [])
    answer   = messages[-1].content if messages else "No answer."

    steps = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "type":       "call",
                    "tool":       tc["name"],
                    "expression": tc["args"].get("expression", ""),
                })
        if hasattr(msg, "name") and msg.name == "calculate":
            steps.append({"type": "result", "output": msg.content})

    return {"answer": answer, "steps": steps}
