"""
agents/planner.py — Planner Agent

Calls: generate_outline (tool)
Produces: structured BlogOutline stored in state
"""

from __future__ import annotations
import json
import logging

from state import BlogState, BlogOutline, BlogSection
from tools.llm_tools import generate_outline

logger = logging.getLogger(__name__)


def run_planner(state: BlogState) -> BlogState:
    """LangGraph node: Planner Agent."""
    logger.info("▶ [Planner] topic=%s", state.topic)
    state.logs.append("🗂️  Planner Agent: Generating blog outline...")
    state.current_step = "planning"

    try:
        # ── Call the @tool ────────────────────────────────────────────────
        raw_json = generate_outline.invoke({
            "topic":               state.topic,
            "keywords":            state.keywords or "none",
            "tone":                state.tone,
            "length":              state.length,
            "num_images":          state.num_images,
            "language":            state.language,
            "custom_instructions": state.custom_instructions or "none",
        })

        data = json.loads(raw_json)

        # ── Parse sections ────────────────────────────────────────────────
        sections = []
        image_count = 0
        for s in data.get("sections", []):
            needs_img = s.get("needs_image", False) and image_count < state.num_images
            if needs_img:
                image_count += 1
            sections.append(BlogSection(
                heading=s["heading"],
                needs_image=needs_img,
                image_prompt=s.get("image_hint", ""),
            ))

        state.outline = BlogOutline(
            title=data.get("title", state.topic),
            meta_description=data.get("meta_description", ""),
            slug=data.get("slug", state.topic.lower().replace(" ", "-")),
            keywords=data.get("keywords", []),
            sections=sections,
        )

        logger.info("✔ [Planner] title=%s  sections=%d  images=%d",
                    state.outline.title, len(sections), image_count)
        state.logs.append(
            f"✅  Planner: Outline ready — {len(sections)} sections, {image_count} image slots."
        )

    except Exception as exc:
        logger.exception("[Planner] failed: %s", exc)
        state.error = f"Planner failed: {exc}"
        state.logs.append(f"❌  Planner error: {exc}")

    return state
