import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://your-capgemini-llm-endpoint/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "your-llm-api-key")
LLM_MODEL    = os.getenv("LLM_MODEL",    "amazon.nova-lite-v1:0")

# Image Generation
IMG_BASE_URL = os.getenv("IMG_BASE_URL", "https://your-capgemini-image-endpoint")
IMG_API_KEY  = os.getenv("IMG_API_KEY",  "your-img-api-key")
IMG_MODEL    = os.getenv("IMG_MODEL",    "amazon.nova-canvas-v1:0")

# Blog defaults
LENGTH_MAP = {
    "short":  "600-800 words",
    "medium": "1000-1400 words",
    "long":   "1800-2400 words",
}
