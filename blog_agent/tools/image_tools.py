"""
tools/image_tools.py — LangChain @tool definition for image generation.

Wraps the Capgemini Nova-Canvas HTTP call so agents can call it
as a standard LangChain tool.
"""

import logging
from langchain_core.tools import tool
from utils.image_client import generate_image as _generate_image

logger = logging.getLogger(__name__)


@tool
def generate_blog_image(prompt: str, section_name: str) -> str:
    """
    Generate an image for a blog section using the Capgemini Nova-Canvas API.

    Sends the prompt to the image generation endpoint, saves the result
    as a PNG file in the outputs/ directory, and returns the file path.

    Returns the saved image file path, or an empty string on failure.
    """
    logger.info("[Tool:generate_blog_image] section=%s  prompt=%.80s...", section_name, prompt)
    path = _generate_image(prompt=prompt, section_name=section_name)
    if path:
        logger.info("[Tool:generate_blog_image] saved → %s", path)
    else:
        logger.warning("[Tool:generate_blog_image] failed for section: %s", section_name)
    return path
