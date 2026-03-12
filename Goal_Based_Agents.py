Goal-Based Agents
What is a Goal-Based Agent?

A Goal-Based Agent is an AI system that makes decisions based on a specific goal it wants to achieve. Unlike earlier agents that simply react to inputs, a Goal-Based Agent thinks ahead and plans actions to reach a desired outcome. 🎯

Previous agents like Simple Reflex Agents only reacted to the current situation. But a Goal-Based Agent is different because it:

Has a clear goal to achieve

Plans a sequence of steps to reach that goal

Evaluates different possible actions

Chooses the action that moves it closer to the goal

Modern implementations often use a reasoning engine like GPT-4 or LLaMA to decide what steps to take instead of relying only on fixed rules.

How It Works

A Goal-Based Agent works by continuously reasoning, acting, and evaluating results until the goal is achieved.

Process:

The agent receives a goal.

It reasons about possible actions that move it closer to the goal.

It uses tools or actions to perform the task.

It observes the result.

It reasons again and repeats until the goal is completed.

Architecture of a Goal-Based Agent
[Goal]
   ↓
[Reasoning Engine]
(What action moves me closer?)
   ↓
[Select / Call a Tool]
   ↓
[Observe Result]
   ↓
[Reason Again]
   ↓
Repeat until Goal Achieved

This loop of Reason → Act → Observe → Repeat is often called the ReAct loop (Reason + Act).

Example

Imagine an AI assistant with the goal:

Goal: Book the cheapest flight to Delhi

The agent might:

Search flight websites

Compare prices

Check available dates

Select the best option

Complete the booking

Each step moves the agent closer to achieving the goal.

Why Goal-Based Agents Are Powerful

Goal-Based Agents are more advanced because they can:

Plan multiple steps ahead

Adapt actions based on results

Use tools to complete tasks

Solve complex problems

This makes them useful in applications like:

AI assistants

automated research agents

task automation systems

intelligent robotics 

"""
Goal-Based Agent — Research Assistant
======================================
Architecture: Goal → Plan → Tool Call → Observe → Replan → ... → Done

The LLM (Claude) acts as the reasoning engine.
It decides WHICH tool to call and WHEN based on the goal.

Tools available:
  - search(query)            : simulate searching for information
  - summarize(text)          : condense a block of text
  - write_report(title, sections) : compile the final report
  - finish(answer)           : signal the goal is complete

Run:
  pip install anthropic
  export ANTHROPIC_API_KEY=your_key_here
  python goal_based_agent.py
"""

import json
import anthropic

# ─────────────────────────────────────────────────────
# 1. TOOL DEFINITIONS  (what the LLM can choose from)
# ─────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "search",
        "description": (
            "Search for information about a topic. "
            "Returns a short fact-based passage on the query."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'Python history and creator'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "summarize",
        "description": "Summarize a long block of text into 2-3 concise sentences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "write_report",
        "description": "Compile all gathered information into a final structured report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The report title"
                },
                "sections": {
                    "type": "array",
                    "description": "List of report sections",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["heading", "content"]
                    }
                }
            },
            "required": ["title", "sections"]
        }
    },
    {
        "name": "finish",
        "description": "Call this when the goal is fully achieved and the report is ready.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary of what was accomplished"
                }
            },
            "required": ["summary"]
        }
    }
]


# ─────────────────────────────────────────────────────
# 2. TOOL EXECUTOR  (what actually happens when called)
#    In a real agent these would be real API calls.
#    Here we simulate results for learning purposes.
# ─────────────────────────────────────────────────────
SIMULATED_KNOWLEDGE = {
    "python history":
        "Python was created by Guido van Rossum and first released in 1991. "
        "It was designed for readability and simplicity, inspired by ABC language. "
        "Python 2.0 launched in 2000, and Python 3.0 in 2008, which broke backward compatibility.",

    "python features":
        "Python supports multiple programming paradigms: procedural, object-oriented, and functional. "
        "Key features include dynamic typing, garbage collection, and a vast standard library. "
        "It uses indentation for code blocks, making it highly readable.",

    "python popularity":
        "Python is consistently ranked as the most popular programming language (TIOBE index, 2024). "
        "It dominates data science, machine learning, and web development. "
        "Major users include Google, NASA, Instagram, and Netflix.",

    "python use cases":
        "Python is widely used in AI/ML (TensorFlow, PyTorch), web development (Django, FastAPI), "
        "data analysis (pandas, NumPy), automation, scripting, and scientific computing. "
        "It is also the primary language for Raspberry Pi projects.",
}

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result as a string."""

    if tool_name == "search":
        query = tool_input["query"].lower()
        # Find the best matching simulated result
        for key, value in SIMULATED_KNOWLEDGE.items():
            if any(word in query for word in key.split()):
                return value
        return f"No specific results found for '{tool_input['query']}'. Try a more specific query."

    elif tool_name == "summarize":
        text = tool_input["text"]
        # Truncate to first 2 sentences as a simple simulation
        sentences = text.split(". ")
        return ". ".join(sentences[:2]) + "."

    elif tool_name == "write_report":
        title = tool_input["title"]
        sections = tool_input["sections"]
        lines = [f"\n{'='*55}", f"  {title}", f"{'='*55}"]
        for section in sections:
            lines.append(f"\n## {section['heading']}")
            lines.append(section["content"])
        lines.append(f"\n{'='*55}")
        report = "\n".join(lines)
        return report  # Return so the LLM can confirm it

    elif tool_name == "finish":
        return f"✅ Goal achieved: {tool_input['summary']}"

    return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────────────
def make_state(goal: str) -> dict:
    return {
        "goal":      goal,
        "messages":  [],   # full conversation with LLM
        "steps":     [],   # log of (tool, input, result)
        "report":    None, # final report when done
        "done":      False,
        "turn":      0,
    }


# ─────────────────────────────────────────────────────
# 4. GOAL-BASED AGENT LOOP
# ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Goal-Based Research Agent.

Your job is to achieve the user's research goal by using the available tools.
Think step by step:
1. Decide what information you need
2. Use 'search' to gather facts (search multiple aspects)
3. Use 'summarize' if needed to condense long results
4. Once you have enough info, use 'write_report' to compile everything
5. Finally call 'finish' to signal completion

Be thorough — use search at least 3 times to cover different aspects of the topic.
Always end by calling write_report then finish."""


def run_goal_agent(goal: str, max_turns: int = 10) -> dict:
    """
    Main agent loop.
    Keeps calling the LLM until it calls 'finish' or hits max_turns.
    """
    client = anthropic.Anthropic()
    state  = make_state(goal)

    # Seed the conversation with the goal
    state["messages"].append({
        "role": "user",
        "content": f"Research goal: {goal}"
    })

    print(f"\n{'='*55}")
    print(f"  Goal-Based Agent Starting")
    print(f"  Goal: {goal}")
    print(f"{'='*55}")

    for turn in range(max_turns):
        state["turn"] = turn + 1
        print(f"\n── Turn {turn + 1} ──────────────────────────────")

        # ── Step A: Ask the LLM what to do next ───────
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=state["messages"]
        )

        # ── Step B: Add LLM response to history ───────
        state["messages"].append({
            "role": "assistant",
            "content": response.content
        })

        # ── Step C: Process each content block ────────
        tool_results = []

        for block in response.content:

            if block.type == "text" and block.text:
                print(f"  💭 LLM thinks: {block.text[:120]}...")

            elif block.type == "tool_use":
                tool_name  = block.name
                tool_input = block.input
                tool_id    = block.id

                print(f"  🔧 Calling tool : {tool_name}")
                print(f"     Input        : {json.dumps(tool_input)[:80]}...")

                # Execute the tool
                result = execute_tool(tool_name, tool_input)

                print(f"     Result       : {str(result)[:80]}...")

                # Log the step
                state["steps"].append({
                    "turn":   turn + 1,
                    "tool":   tool_name,
                    "input":  tool_input,
                    "result": result
                })

                # Collect tool result for the next message
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tool_id,
                    "content":     result
                })

                # Store report when written
                if tool_name == "write_report":
                    state["report"] = result

                # Check if done
                if tool_name == "finish":
                    state["done"] = True

        # ── Step D: Send tool results back to LLM ─────
        if tool_results:
            state["messages"].append({
                "role": "user",
                "content": tool_results
            })

        # ── Step E: Check stop conditions ─────────────
        if state["done"]:
            print(f"\n✅ Goal achieved in {turn + 1} turns!")
            break

        if response.stop_reason == "end_turn" and not tool_results:
            print("\n⚠️  LLM stopped without calling finish.")
            break

    return state


# ─────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────
def main():
    goal = input("Enter your research goal (or press Enter for default):\n> ").strip()
    if not goal:
        goal = "Research the Python programming language: its history, features, and use cases"

    state = run_goal_agent(goal)

    # Print the final report
    if state["report"]:
        print(state["report"])

    # Print agent summary
    print(f"\n── Agent Summary ───────────────────────")
    print(f"  Total turns : {state['turn']}")
    print(f"  Tools used  : {len(state['steps'])}")
    for step in state["steps"]:
        tool_label = step['tool']
        inp = list(step['input'].values())[0]
        print(f"    Turn {step['turn']}: {tool_label}({str(inp)[:45]}...)")


if __name__ == "__main__":
    main()
