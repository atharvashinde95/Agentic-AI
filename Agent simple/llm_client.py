"""
llm_client.py
-------------
LangChain-compatible LLM wrapper for the Capgemini Generative Engine API
(model: amazon.nova.lite-v1.0).

Implements langchain_core.language_models.chat_models.BaseChatModel so it
slots directly into create_react_agent / AgentExecutor on langchain 0.3.x.

Only langchain_core is imported here — that package is stable across all
0.3.x versions and has no deprecations.
"""

import os
import json
import requests
from typing import Any, List, Optional

from dotenv import load_dotenv

# langchain_core is stable on 0.3.x — safe to import
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration

load_dotenv()

API_KEY      = os.getenv("CAPGEMINI_API_KEY",      "")
API_ENDPOINT = os.getenv("CAPGEMINI_API_ENDPOINT", "")
MODEL_ID     = "amazon.nova.lite-v1.0"


class CapgeminiNovaLite(BaseChatModel):
    """
    Custom LangChain ChatModel wrapping the Capgemini REST API.

    Works with langchain 0.3.x AgentExecutor + create_react_agent.
    No third-party LangChain integration packages required.
    """

    model_name : str   = MODEL_ID
    max_tokens : int   = 512
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "capgemini-nova-lite"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "max_tokens": self.max_tokens}

    # ── Message conversion ─────────────────────────────────────────────── #

    def _build_payload(self, messages: List[BaseMessage]) -> dict:
        """
        Convert LangChain messages → Capgemini / Anthropic REST payload.
        System messages are extracted and sent as a top-level "system" key.
        """
        system_content = None
        formatted      = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            else:
                # Fallback for any other message type
                formatted.append({"role": "user", "content": str(msg.content)})

        payload: dict = {
            "model"      : self.model_name,
            "max_tokens" : self.max_tokens,
            "messages"   : formatted,
        }
        if system_content:
            payload["system"] = system_content

        return payload

    def _extract_text(self, data: dict) -> str:
        """
        Pull the text response out of the API JSON.
        Handles both Anthropic-style (content[]) and OpenAI-style (choices[]).
        """
        if "content" in data:
            content = data["content"]
            if isinstance(content, list) and content:
                return content[0].get("text", str(content[0]))
            return str(content)

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return json.dumps(data)

    # ── Core generate ──────────────────────────────────────────────────── #

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Called by LangChain on every LLM invocation."""

        # ── Graceful no-credentials fallback ──────────────────────── #
        if not API_KEY or not API_ENDPOINT:
            stub = (
                "I cannot reach the LLM (missing credentials in .env).\n"
                "Thought: I will log this as a normal cycle.\n"
                "Action: log_normal_cycle\n"
                'Action Input: {"temperature": 0, "vibration": 0, "pressure": 0, "tick": 0}\n'
                "Observation: logged\n"
                'Final Answer: {"status": "Normal", "action": "Continue", '
                '"risk": "Low", "tool_used": "log_normal_cycle", '
                '"diagnoses": ["LLM unavailable"], "summary": "Running without LLM credentials."}'
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=stub))]
            )

        # ── Live API call ──────────────────────────────────────────── #
        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "x-api-key"    : API_KEY,
        }

        try:
            resp = requests.post(
                API_ENDPOINT,
                headers=headers,
                json=self._build_payload(messages),
                timeout=20,
            )
            resp.raise_for_status()
            text = self._extract_text(resp.json())

        except requests.exceptions.Timeout:
            text = (
                "Thought: The LLM timed out. I will log this cycle as normal.\n"
                "Action: log_normal_cycle\n"
                'Action Input: {"temperature": 0, "vibration": 0, "pressure": 0, "tick": -1}\n'
                "Observation: logged\n"
                'Final Answer: {"status": "Normal", "action": "Continue", "risk": "Low", '
                '"tool_used": "log_normal_cycle", "diagnoses": ["LLM timeout"], '
                '"summary": "LLM timed out — defaulted to normal."}'
            )
        except Exception as e:
            text = (
                f"Thought: API error: {e}. Logging as normal.\n"
                "Action: log_normal_cycle\n"
                'Action Input: {"temperature": 0, "vibration": 0, "pressure": 0, "tick": -1}\n'
                "Observation: logged\n"
                'Final Answer: {"status": "Normal", "action": "Continue", "risk": "Low", '
                '"tool_used": "log_normal_cycle", "diagnoses": ["API error"], '
                '"summary": "API error — defaulted to normal."}'
            )

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )
