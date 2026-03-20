"""
llm_client.py
-------------
LangChain 1.0-compatible LLM wrapper for the Capgemini Generative Engine API
(model: amazon.nova.lite-v1.0).

Implements langchain_core.language_models.chat_models.BaseChatModel so it
works with langchain 1.0's create_agent() which requires a BaseChatModel.

Only langchain_core is imported here — that package is stable across all
LangChain versions (0.3.x and 1.0+).
"""

import os
import json
import requests
from typing import Any, List, Optional

from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration

load_dotenv()

API_KEY      = os.getenv("CAPGEMINI_API_KEY",      "")
API_ENDPOINT = os.getenv("CAPGEMINI_API_ENDPOINT", "")
MODEL_ID     = "amazon.nova.lite-v1.0"


class CapgeminiNovaLite(BaseChatModel):
    """
    Custom LangChain 1.0 BaseChatModel for the Capgemini REST API.

    Accepted by create_agent() in langchain 1.0 because it is a BaseChatModel.
    No third-party integration package required.
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

    # ── Message → payload ─────────────────────────────────────────────── #

    def _build_payload(self, messages: List[BaseMessage]) -> dict:
        """
        Convert LangChain message objects to the REST API payload format.
        System messages are extracted and sent as a top-level 'system' key.
        ToolMessages (tool call results) are mapped to 'user' role with a
        clear prefix so the model understands their context.
        """
        system_content = None
        formatted      = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user",      "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, ToolMessage):
                # Tool results come back as ToolMessage in langchain 1.0
                formatted.append({
                    "role"   : "user",
                    "content": f"[Tool result for {msg.tool_call_id}]: {msg.content}",
                })
            else:
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
        """Pull text from API response — handles Anthropic and OpenAI formats."""
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

        if not API_KEY or not API_ENDPOINT:
            # Graceful no-credentials mode — agent still runs in fallback
            stub = (
                "I don't have LLM credentials. I'll use the available sensor data "
                "and call log_normal_cycle as a safe default action."
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=stub))]
            )

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
            text = "API timeout — proceeding with safe default."
        except Exception as e:
            text = f"API error ({e}) — proceeding with safe default."

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    # ── bind_tools support (required by create_agent in LangChain 1.0) ── #

    def bind_tools(self, tools, **kwargs):
        """
        LangChain 1.0's create_agent() calls bind_tools() on the model.
        We return a simple wrapper that appends tool schemas to every
        system message so the LLM knows what tools are available.
        """
        return _ToolBoundModel(base_model=self, tools=tools)


# ─────────────────────────────────────────────────────────────────────────── #
#  Tool-bound wrapper — returned by bind_tools()
# ─────────────────────────────────────────────────────────────────────────── #

class _ToolBoundModel(BaseChatModel):
    """
    Thin wrapper around CapgeminiNovaLite that injects tool descriptions
    into the system prompt so the model can reason about tool selection.

    This satisfies the interface expected by create_agent() in LangChain 1.0
    without requiring a native tool-calling API on the Capgemini endpoint.
    """

    base_model: Any
    tools: Any

    @property
    def _llm_type(self) -> str:
        return "capgemini-nova-lite-tool-bound"

    @property
    def _identifying_params(self) -> dict:
        return self.base_model._identifying_params

    def _build_tool_description(self) -> str:
        lines = ["You have access to these tools — call them by name:\n"]
        for t in self.tools:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "No description available.")
            lines.append(f"  Tool: {name}\n  Description: {desc}\n")
        lines.append(
            "\nTo use a tool respond with:\n"
            "Action: <tool_name>\n"
            "Action Input: <json_string>\n"
            "When done, respond with:\n"
            "Final Answer: <your summary json>"
        )
        return "\n".join(lines)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Inject tool descriptions into the system message, then call base model."""
        tool_desc    = self._build_tool_description()
        augmented    = []
        has_system   = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                augmented.append(SystemMessage(
                    content=f"{msg.content}\n\n{tool_desc}"
                ))
                has_system = True
            else:
                augmented.append(msg)

        if not has_system:
            augmented.insert(0, SystemMessage(content=tool_desc))

        return self.base_model._generate(augmented, stop=stop, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return _ToolBoundModel(base_model=self.base_model, tools=tools)
