"""
agents/image_prompt.py — Image Prompt Agent

Calls: generate_image_prompt (tool) — once per image section
Produces: image_prompt filled in each BlogSection
"""

from __future__ import annotations
import logging

from state import BlogState
from tools.llm_tools import generate_image_prompt

logger = logging.getLogger(__name__)


def run_image_prompt_agent(state: BlogState) -> BlogState:
    """LangGraph node: Image Prompt Agent."""
    image_sections = [s for s in state.outline.sections if s.needs_image]
    logger.info("▶ [ImagePrompt] %d sections need prompts", len(image_sections))
    state.logs.append(f"🎨  Image Prompt Agent: Crafting {len(image_sections)} prompts...")
    state.current_step = "image_prompting"

    if state.error:
        return state

    for section in image_sections:
        try:
            # ── Call the @tool ────────────────────────────────────────────
            prompt_text = generate_image_prompt.invoke({
                "topic":           state.topic,
                "section_heading": section.heading,
                "content_preview": (section.content or section.image_prompt or "")[:300],
            })

            section.image_prompt = prompt_text
            logger.info("  ✔ prompt for [%s]: %.80s...", section.heading, prompt_text)

        except Exception as exc:
            logger.exception("[ImagePrompt] section=%s  err=%s", section.heading, exc)
            section.image_prompt = (
                f"Professional photograph related to {section.heading}, "
                "sharp focus, natural lighting, 4K"
            )
            state.logs.append(f"⚠️  Fallback prompt used for: {section.heading}")

    state.logs.append("✅  Image Prompt Agent: All prompts ready.")
    return state
