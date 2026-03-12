Model-Based Agents
What is a Model-Based Agent?

A Model-Based Agent is an AI agent that improves upon the Simple Reflex Agent by adding an internal state (memory). This means the agent can remember past interactions and use that information to make better decisions.

Instead of reacting only to the current input, the agent:

Remembers past interactions

Builds a model of the environment

Uses both current input and history to decide actions

In simple terms:

Input + Memory → Decision → Action

This allows the agent to handle more complex situations and ongoing interactions.

How It Works

A Model-Based Agent follows these steps:

Receive Input – The agent observes the current situation.

Check Internal State – It refers to stored information from past interactions.

Match Conditions – It evaluates rules or decision logic.

Perform Action – The agent takes the appropriate action.

Update Internal State – It stores the new information for future decisions.

Architecture of a Model-Based Agent
[Input] + [Internal State / Memory]
               ↓
        [Condition Checker]
               ↓
            [Action]
               ↓
      [Update Internal State]

The key addition compared to simple agents is the internal state, which continuously updates as the agent interacts with the environment.

Example

Imagine a customer support chatbot.

If a user asks about a refund, a simple system might always give the same reply.
But a model-based agent remembers previous messages.

Example conversation:

User: "I want a refund."
Agent: "Please provide your order ID."
User: "Here it is."

The agent remembers the previous step and continues the process instead of restarting the conversation.

Reflex Agent vs Model-Based Agent
Feature	Simple Reflex Agent	Model-Based Agent
Memory	❌ No memory	✅ Stores past information
Context Awareness	❌ Treats every input separately	✅ Understands previous interactions
Multi-step Tasks	❌ Cannot handle	✅ Can manage multi-step conversations
Example	Responds to “refund” with fixed reply	Remembers that the user already asked about a refund
Why Model-Based Agents Are Useful

Model-Based Agents are important because many real-world environments are partially observable. This means the agent cannot see the entire situation at once, so it must rely on memory and internal models to make good decisions.

They are commonly used in:

Conversational AI systems

Autonomous robots

Navigation systems

Intelligent assistants 


"""
Model-Based Reflex Agent — Customer Support Bot
================================================
Architecture: Perceive → Update State → Match Rule → Act
KEY UPGRADE: Agent now maintains internal state (memory).

Internal State tracks:
  - Full conversation history
  - Topics already discussed
  - User's name (if shared)
  - Number of turns
  - Whether escalation is needed
"""

import re
from datetime import datetime


# ─────────────────────────────────────────────────────
# 1. INTERNAL STATE  (the "model of the world")
# ─────────────────────────────────────────────────────
def make_state() -> dict:
    """Create a fresh internal state for a new session."""
    return {
        "history":        [],       # list of (role, message) tuples
        "topics_seen":    set(),    # tags of topics already discussed
        "user_name":      None,     # extracted if user shares it
        "turn":           0,        # how many exchanges happened
        "awaiting":       None,     # what we're waiting for (e.g. "order_id")
        "escalate":       False,    # should we escalate to human?
        "session_start":  datetime.now().strftime("%H:%M:%S"),
    }


# ─────────────────────────────────────────────────────
# 2. RULES  (now context-aware using state)
# ─────────────────────────────────────────────────────
RULES = [
    # ── Greeting ──────────────────────────────────────
    {
        "tag": "GREET",
        "condition": lambda msg, state: bool(
            re.search(r"\b(hello|hi|hey|good morning|good evening)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            f"👋 Hello {state['user_name']}! How can I help you today?"
            if state["user_name"]
            else "👋 Hello! I'm your support agent. What's your name?"
        ),
        "next_awaiting": "name",
    },

    # ── User provides their name ───────────────────────
    {
        "tag": "NAME",
        "condition": lambda msg, state: (
            state["awaiting"] == "name"
            and len(msg.split()) <= 3
            and not re.search(r"\b(refund|order|cancel|ship)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            f"Nice to meet you, {msg.strip().title()}! How can I help you today?"
        ),
        "next_awaiting": None,
        "update": lambda msg, state: state.update({"user_name": msg.strip().title()}),
    },

    # ── Refund ─────────────────────────────────────────
    {
        "tag": "REFUND",
        "condition": lambda msg, state: bool(
            re.search(r"\b(refund|money back|return)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            "You've already started a refund request this session. "
            "Would you like me to escalate this to a senior agent?"
            if "REFUND" in state["topics_seen"]
            else "💸 I can help with your refund! What is your order number?"
        ),
        "next_awaiting": "order_id",
    },

    # ── User provides order ID ─────────────────────────
    {
        "tag": "ORDER_ID",
        "condition": lambda msg, state: (
            state["awaiting"] == "order_id"
            and bool(re.search(r"\b([A-Z]{0,3}\d{4,})\b", msg, re.I))
        ),
        "action": lambda msg, state: (
            f"✅ Got it! I've logged order "
            f"{re.search(r'[A-Z0-9]+', msg.upper()).group()} for a refund. "
            f"You'll receive a confirmation email within 24 hours, "
            f"{state['user_name'] or 'there'}!"
        ),
        "next_awaiting": None,
    },

    # ── Shipping ───────────────────────────────────────
    {
        "tag": "SHIPPING",
        "condition": lambda msg, state: bool(
            re.search(r"\b(ship|delivery|where.*order|tracking|arrived|late)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            "You asked about shipping earlier too — still no update? "
            "Let me escalate this for you."
            if "SHIPPING" in state["topics_seen"]
            else "📦 Orders typically arrive in 5-7 days. "
                 "Can you share your order ID for live tracking?"
        ),
        "next_awaiting": "order_id",
    },

    # ── Auth ───────────────────────────────────────────
    {
        "tag": "AUTH",
        "condition": lambda msg, state: bool(
            re.search(r"\b(password|login|can't log|locked out|access)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            "🔐 Still having login trouble? I'm escalating this to our "
            "technical team right away."
            if "AUTH" in state["topics_seen"]
            else "🔐 Let's fix your login! Try 'Forgot Password' on the login page. "
                 "Want me to send a reset link to your email?"
        ),
        "next_awaiting": None,
    },

    # ── Cancellation ───────────────────────────────────
    {
        "tag": "CANCEL",
        "condition": lambda msg, state: bool(
            re.search(r"\b(cancel|cancellation|stop|unsubscribe)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            f"I understand, {state['user_name'] or 'there'}. "
            "I've flagged your cancellation request for a senior agent to process."
            if "CANCEL" in state["topics_seen"]
            else "❌ I can process your cancellation. "
                 "Before I do — would a 20% discount change your mind?"
        ),
        "next_awaiting": None,
    },

    # ── Thanks ─────────────────────────────────────────
    {
        "tag": "THANKS",
        "condition": lambda msg, state: bool(
            re.search(r"\b(thank|thanks|appreciate|great|awesome)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            f"😊 Anytime{', ' + state['user_name'] if state['user_name'] else ''}! "
            "Is there anything else I can help you with?"
        ),
        "next_awaiting": None,
    },

    # ── Escalate / Angry ───────────────────────────────
    {
        "tag": "ESCALATE",
        "condition": lambda msg, state: bool(
            re.search(r"\b(human|agent|manager|escalate|angry|frustrated|useless)\b", msg, re.I)
        ),
        "action": lambda msg, state: (
            "☎️ Understood! Connecting you to a human agent now. "
            f"Average wait time is ~3 minutes. "
            f"(Session summary: {len(state['history'])} messages, "
            f"topics: {', '.join(state['topics_seen']) or 'none'})"
        ),
        "next_awaiting": None,
        "update": lambda msg, state: state.update({"escalate": True}),
    },
]

DEFAULT_ACTION = lambda msg, state: (
    f"🤷 I'm not sure how to help with that"
    f"{', ' + state['user_name'] if state['user_name'] else ''}. "
    "Type 'agent' to speak with a human."
)


# ─────────────────────────────────────────────────────
# 3. AGENT PIPELINE
# ─────────────────────────────────────────────────────
def perceive(user_input: str) -> str:
    """Step 1: Perceive — normalize input."""
    return user_input.strip()


def update_state(percept: str, state: dict) -> None:
    """Step 2: Update state BEFORE matching rules."""
    state["turn"] += 1
    state["history"].append(("user", percept))


def match_rule(percept: str, state: dict) -> dict:
    """Step 3: Match — find first rule whose condition is True."""
    for rule in RULES:
        if rule["condition"](percept, state):
            return rule
    return {"tag": "DEFAULT", "action": DEFAULT_ACTION, "next_awaiting": None}


def act(percept: str, rule: dict, state: dict) -> str:
    """Step 4: Act — generate response."""
    # Run any side-effect updates (e.g. save name)
    if "update" in rule:
        rule["update"](percept, state)

    # Update topics seen and awaiting
    state["topics_seen"].add(rule["tag"])
    state["awaiting"] = rule.get("next_awaiting", None)

    response = rule["action"](percept, state)
    state["history"].append(("agent", response))
    return response


def run_agent(user_input: str, state: dict) -> str:
    """Full pipeline for one turn."""
    percept = perceive(user_input)
    update_state(percept, state)
    rule     = match_rule(percept, state)

    print(f"  [rule: {rule['tag']} | turn: {state['turn']} | "
          f"seen: {state['topics_seen']} | awaiting: {state['awaiting']}]")

    return act(percept, rule, state)


# ─────────────────────────────────────────────────────
# 4. MAIN LOOP
# ─────────────────────────────────────────────────────
def main():
    state = make_state()

    print("=" * 55)
    print("   Model-Based Agent — Customer Support")
    print("   (Agent remembers context across turns)")
    print("   Type 'state' to inspect memory | 'quit' to exit")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Agent: 👋 Goodbye! Have a great day.")
            break

        # Debug: print the internal state
        if user_input.lower() == "state":
            print("\n── Internal State ──────────────────")
            for k, v in state.items():
                if k != "history":
                    print(f"  {k:15} : {v}")
            print(f"  {'history':15} : {len(state['history'])} messages")
            print("────────────────────────────────────")
            continue

        response = run_agent(user_input, state)
        print(f"Agent: {response}")

        if state.get("escalate"):
            print("\n[Session ended — escalated to human agent]")
            break


if __name__ == "__main__":
    main()
