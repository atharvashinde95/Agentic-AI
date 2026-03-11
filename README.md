# Agentic-AI
Types of Agents : 

Simple Reflex Agents
What is a Simple Reflex Agent?

A Simple Reflex Agent is the most basic type of AI agent. It makes decisions using a simple rule:

IF this condition → THEN perform this action

The agent:

Does not store memory

Does not plan

Does not consider past experiences

It simply follows the process:

Perceive input → Match a rule → Take action

Example

A common example is a thermostat that controls temperature.

Rules:

IF temperature < 68°F → Turn ON the heater

IF temperature > 74°F → Turn ON the AC

The system only checks the current temperature and reacts immediately.

Explanation:

Input / Perception – The agent receives information from the environment.

Condition Checker – The agent checks predefined IF–THEN rules.

Action – It performs the action that matches the rule.

Key Characteristics

Simple and fast

Deterministic decision-making

Uses predefined rules

However, it has limitations:

Cannot learn from experience

Cannot remember past events

Cannot handle situations that do not match existing rules



"""
==============================================
  SIMPLE REFLEX AGENT — Customer Support Bot
==============================================
Architecture:
  [Perception] → [Rule Matcher] → [Action]

No memory. No planning. Just IF → THEN rules.
"""

import re
RULES = [
    {
        "tag": "GREET",
        "condition": lambda msg: bool(re.search(r"\b(hello|hi|hey|good morning|howdy)\b", msg, re.I)),
        "response": "👋 Hello! Welcome to support. How can I help you today?"
    },
    {
        "tag": "REFUND",
        "condition": lambda msg: bool(re.search(r"\b(refund|money back|return)\b", msg, re.I)),
        "response": "💸 I can help with your refund! Please share your order number and we'll process it within 3–5 business days."
    },
    {
        "tag": "LOGIN",
        "condition": lambda msg: bool(re.search(r"\b(password|login|can'?t log|cannot log|access|locked out)\b", msg, re.I)),
        "response": "🔐 Let's fix your login! Click 'Forgot Password' on the login page, or I can send a reset link to your email."
    },
    {
        "tag": "SHIPPING",
        "condition": lambda msg: bool(re.search(r"\b(ship|delivery|where.*order|tracking|arrived|late)\b", msg, re.I)),
        "response": "📦 I'll check your shipment! Orders typically arrive in 5–7 days. Share your order ID for real-time tracking."
    },
    {
        "tag": "CANCEL",
        "condition": lambda msg: bool(re.search(r"\b(cancel|cancellation|stop|unsubscribe)\b", msg, re.I)),
        "response": "❌ I can process your cancellation. Before I do — would a pause or discount help? If not, I'll cancel right away."
    },
    {
        "tag": "THANKS",
        "condition": lambda msg: bool(re.search(r"\b(thank|thanks|appreciate|great|awesome)\b", msg, re.I)),
        "response": "😊 You're welcome! Is there anything else I can help you with?"
    },
]

# Fallback if no rule matches
DEFAULT_RESPONSE = {
    "tag": "DEFAULT",
    "response": "🤷 I'm not sure how to handle that. Let me connect you with a human agent."
}


# ─────────────────────────────────────────────
#  AGENT CORE
# ─────────────────────────────────────────────

class SimpleReflexAgent:
    """
    A Simple Reflex Agent.
    Scans rules top-to-bottom and fires the FIRST matching rule.
    Has NO memory of past interactions.
    """

    def __init__(self, rules, default):
        self.rules = rules
        self.default = default

    def perceive(self, user_input: str) -> str:
        """Step 1 — Perceive: clean and return the input."""
        return user_input.strip()

    def match_rule(self, percept: str) -> dict:
        """Step 2 — Match: scan rules, return first match."""
        for rule in self.rules:
            if rule["condition"](percept):
                return rule
        return self.default

    def act(self, rule: dict) -> str:
        """Step 3 — Act: return the response for matched rule."""
        return rule["response"]

    def run(self, user_input: str) -> str:
        """Full pipeline: perceive → match → act."""
        percept = self.perceive(user_input)
        rule    = self.match_rule(percept)
        response = self.act(rule)

        # Show internals for learning purposes
        print(f"  [Matched Rule] → {rule['tag']}")
        return response


# ─────────────────────────────────────────────
#  MAIN — Interactive Chat Loop
# ─────────────────────────────────────────────

def main():
    agent = SimpleReflexAgent(rules=RULES, default=DEFAULT_RESPONSE)

    print("=" * 50)
    print("   Simple Reflex Agent — Customer Support")
    print("=" * 50)
    print("Type a message and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: 👋 Goodbye!")
            break

        response = agent.run(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
