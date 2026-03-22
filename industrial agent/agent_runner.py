import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "amazon.nova-lite-v1:0"
TEMPERATURE     = 0
MAX_ITERATIONS  = 10
BASE_URL        = os.getenv("CAPGEMINI_BASE_URL")
API_KEY         = os.getenv("CAPGEMINI_API_KEY", "placeholder")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an industrial production assistant for a batch manufacturing plant.

Your job is to determine if a production run is feasible by analysing:
- Raw material availability (tank levels in litres)
- Machine operational status (running or stopped)
- Product recipe requirements (materials and machines needed per batch)

You have access to three tools:
1. get_material_availability  — checks current tank levels
2. get_machine_states         — checks which machines are running
3. get_product_details        — retrieves recipe for a specific product

Always reason step by step before calling a tool. After gathering all necessary 
information, provide a clear GO or NO-GO decision with a specific explanation 
of why — including which resource is the bottleneck if production cannot proceed.

If materials are sufficient but a machine is down, say so explicitly.
If a machine is running but materials are low, calculate exactly how many batches 
are possible before materials run out.

Available products: Product_A, Product_B, Product_C

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: reason about what you need to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your clear GO or NO-GO decision with full explanation

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(tools: list) -> AgentExecutor:
    """
    Initialises the LLM, builds the ReAct agent, and wraps it in an executor.
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        base_url=BASE_URL,
        api_key=API_KEY,
        # If Capgemini requires a custom header (e.g. x-api-key), add:
        # default_headers={"x-api-key": API_KEY}
    )

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
    )


# ── Run function (called by app.py) ───────────────────────────────────────────

def run_agent(agent_executor: AgentExecutor, user_query: str) -> str:
    """
    Runs the ReAct loop for a given user query.
    Returns the final answer as a string.
    """
    try:
        result = agent_executor.invoke({"input": user_query})
        return result.get("output", "No response generated.")
    except Exception as e:
        return f"Agent error: {str(e)}"
