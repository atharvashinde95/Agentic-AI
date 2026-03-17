"""
utils/llm_client.py
--------------------
Configures the LangChain ChatOpenAI client
pointing at Capgemini's Generative Engine.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model    = os.getenv("MODEL_NAME", "gpt-4o"),
        api_key  = os.getenv("CAPGEMINI_API_KEY", "your-key-here"),
        base_url = os.getenv("CAPGEMINI_BASE_URL", "https://api.openai.com/v1"),
        temperature = 0,        # deterministic — important for math
    )
