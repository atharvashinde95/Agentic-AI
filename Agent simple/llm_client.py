"""
llm_client.py
-------------
LangChain-compatible LLM wrapper for the Capgemini Generative Engine API
(model: amazon.nova.lite-v1.0).

Implements LangChain's BaseChatModel interface so it can be dropped
directly into any LangChain chain or agent executor.

⚡ LLM Usage Rule:
   This LLM is used by the ReAct agent for ALL reasoning steps
   (Thought → Action selection → Final Answer).
   The agent itself decides when to call which tool.
"""

import os
import json
import requests
from typing import Any, List, Optional, Iterator

from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration

load_dotenv()

API_KEY      = os.getenv("CAPGEMINI_API_KEY", "")
API_ENDPOINT = os.getenv("CAPGEMINI_API_ENDPOINT", "")
MODEL_ID     = "amazon.nova.lite-v1.0"


class CapgeminiNovaLite(BaseChatModel):
    """
    Custom LangChain ChatModel that calls the Capgemini Generative Engine
    (Amazon Nova Lite) REST API.

    Compatible with LangChain's agent executors, chains, and tools.
    """

    model_name: str = MODEL_ID
    max_tokens: int = 512
    temperature: float = 0.2   # Low temperature = deterministic / reliable agent

    @property
    def _llm_type(self) -> str:
        return "capgemini-nova-lite"

    def _convert_messages_to_payload(self, messages: List[BaseMessage]) -> dict:
        """
        Convert LangChain message objects to the API's expected format.
        Handles System, Human, and AI messages.
        """
        formatted = []
        system_text = None

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_text = msg.content
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback — treat as user message
                formatted.append({"role": "user", "content": str(msg.content)})

        payload = {
            "model"     : self.model_name,
            "max_tokens": self.max_tokens,
            "messages"  : formatted,
        }
        if system_text:
            payload["system"] = system_text

        return payload

    def _parse_response(self, data: dict) -> str:
        """
        Extract text from the API response.
        Handles both Anthropic-style (content list) and OpenAI-style (choices) formats.
        """
        # Anthropic / Bedrock style
        if "content" in data:
            content = data["content"]
            if isinstance(content, list) and content:
                return content[0].get("text", str(content[0]))
            return str(content)

        # OpenAI-compatible style
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        # Raw fallback
        return json.dumps(data)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core method called by LangChain to generate a response.
        """
        if not API_KEY or not API_ENDPOINT:
            # Graceful degradation — return a stub response so the system
            # can still run in demo mode without credentials
            stub = (
                "I cannot connect to the LLM API (missing credentials). "
                "Please set CAPGEMINI_API_KEY and CAPGEMINI_API_ENDPOINT in .env"
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=stub))])

        payload = self._convert_messages_to_payload(messages)

        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "x-api-key"    : API_KEY,
        }

        try:
            response = requests.post(
                API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            text = self._parse_response(data)

        except requests.exceptions.Timeout:
            text = "Action: log_normal_cycle\nAction Input: {}"
        except requests.exceptions.RequestException as e:
            text = f"LLM request failed: {str(e)}"
        except Exception as e:
            text = f"Unexpected LLM error: {str(e)}"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    # LangChain requires this property
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "max_tokens": self.max_tokens}
