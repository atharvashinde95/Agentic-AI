"""
config.py — Centralised configuration for the Blog Writing Agent.
All secrets are read from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Capgemini / AWS-Nova LLM ──────────────────────────────────────────────────
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://your-capgemini-llm-endpoint/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "dummy-key")
LLM_MODEL: str = os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# ── Capgemini / AWS-Nova Image Generation ─────────────────────────────────────
IMG_BASE_URL: str = os.getenv("IMG_BASE_URL", "https://your-capgemini-imggen-endpoint")
IMG_API_KEY: str = os.getenv("IMG_API_KEY", "dummy-key")
IMG_MODEL: str = os.getenv("IMG_MODEL", "amazon.nova-canvas-v1:0")
IMG_WIDTH: int = int(os.getenv("IMG_WIDTH", "1024"))
IMG_HEIGHT: int = int(os.getenv("IMG_HEIGHT", "1024"))

# ── Output settings ───────────────────────────────────────────────────────────
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Blog defaults (overridable via UI) ────────────────────────────────────────
DEFAULT_TONE: str = "informative and engaging"
DEFAULT_LENGTH: str = "medium"
DEFAULT_NUM_IMAGES: int = 3
DEFAULT_LANGUAGE: str = "English"

LENGTH_WORD_MAP: dict = {
    "short":  "600-800 words",
    "medium": "1000-1400 words",
    "long":   "1800-2400 words",
}
