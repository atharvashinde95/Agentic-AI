"""
llm_client.py
-------------
Thin wrapper around the Capgemini Generative Engine API
(Amazon Nova Lite: amazon.nova.lite-v1.0).

⚡ USAGE RULE: Call this ONLY for edge cases / explanations.
   Most decisions are handled by rule-based logic in agent.py.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

API_KEY      = os.getenv("CAPGEMINI_API_KEY", "")
API_ENDPOINT = os.getenv("CAPGEMINI_API_ENDPOINT", "")
MODEL_ID     = "amazon.nova.lite-v1.0"


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """
    Send a prompt to the Amazon Nova Lite model and return the text response.

    Args:
        prompt     : The full prompt string to send to the LLM.
        max_tokens : Maximum tokens for the response (keep low to save cost).

    Returns:
        The LLM's response as a plain string, or an error message.
    """
    if not API_KEY or not API_ENDPOINT:
        return (
            "[LLM Unavailable] API key or endpoint not set in .env. "
            "Using rule-based fallback."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "x-api-key": API_KEY,
    }

    payload = {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        # Parse the response — handle both OpenAI-style and Bedrock-style formats
        if "content" in data:
            # Bedrock / Anthropic style
            content = data["content"]
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", str(content[0]))
            return str(content)

        if "choices" in data:
            # OpenAI-compatible style
            return data["choices"][0]["message"]["content"]

        # Fallback: return raw JSON
        return json.dumps(data)

    except requests.exceptions.Timeout:
        return "[LLM Error] Request timed out. Using rule-based decision."
    except requests.exceptions.RequestException as e:
        return f"[LLM Error] {str(e)}"
    except Exception as e:
        return f"[LLM Error] Unexpected error: {str(e)}"


def build_diagnosis_prompt(readings: list, anomalies: list, history_summary: dict) -> str:
    """
    Build a concise prompt for the LLM when the agent encounters
    ambiguous or mixed sensor signals.

    Args:
        readings        : List of recent sensor dicts (last 5–10)
        anomalies       : List of anomaly strings detected this cycle
        history_summary : Dict with trend info from memory module

    Returns:
        A formatted prompt string.
    """
    latest = readings[-1] if readings else {}
    prompt = (
        "You are a predictive maintenance expert analyzing industrial sensor data.\n\n"
        f"Latest readings:\n"
        f"  Temperature : {latest.get('temperature', 'N/A')} °C\n"
        f"  Vibration   : {latest.get('vibration', 'N/A')} mm/s\n"
        f"  Pressure    : {latest.get('pressure', 'N/A')} bar\n\n"
        f"Detected anomalies: {', '.join(anomalies) if anomalies else 'None'}\n\n"
        f"Trend summary: {history_summary}\n\n"
        "Given the mixed signals, provide:\n"
        "1. A brief diagnosis (1 sentence)\n"
        "2. Recommended action: Continue / Alert / Maintenance\n"
        "3. Confidence: Low / Medium / High\n\n"
        "Keep your response under 80 words."
    )
    return prompt
