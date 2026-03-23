"""
agents/writer.py — Content Writer Agent

Calls: write_blog_content (tool)
Produces: raw_content stored in state
"""

from __future__ import annotations
import logging
import re

from state import BlogState
from tools.llm_tools import write_blog_content
import config

logger = logging.getLogger(__name__)


def run_writer(state: BlogState) -> BlogState:
    """LangGraph node: Content Writer Agent."""
    logger.info("▶ [Writer] title=%s", state.outline.title)
    state.logs.append("✍️  Writer Agent: Drafting full blog content...")
    state.current_step = "writing"

    if state.error:
        return state

    # Build outline text for the tool
    outline_text = "\n".join(
        f"{i}. {s.heading}"
        for i, s in enumerate(state.outline.sections, 1)
    )

    try:
        # ── Call the @tool ────────────────────────────────────────────────
        content = write_blog_content.invoke({
            "title":               state.outline.title,
            "outline_sections":    outline_text,
            "keywords":            ", ".join(state.outline.keywords),
            "tone":                state.tone,
            "length":              state.length,
            "language":            state.language,
            "custom_instructions": state.custom_instructions or "none",
        })

        state.raw_content = content
        _map_content_to_sections(state)

        words = len(content.split())
        logger.info("✔ [Writer] words≈%d", words)
        state.logs.append(f"✅  Writer: Draft complete — approx. {words} words.")

    except Exception as exc:
        logger.exception("[Writer] failed: %s", exc)
        state.error = f"Writer failed: {exc}"
        state.logs.append(f"❌  Writer error: {exc}")

    return state


def _map_content_to_sections(state: BlogState) -> None:
    """Split raw_content by ## headings and assign to matching sections."""
    chunks = re.split(r"(?=^## )", state.raw_content, flags=re.MULTILINE)
    heading_map: dict[str, str] = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        first_line = chunk.split("\n")[0].lstrip("#").strip()
        body = "\n".join(chunk.split("\n")[1:]).strip()
        heading_map[first_line.lower()] = body
    for section in state.outline.sections:
        body = heading_map.get(section.heading.lower())
        if body:
            section.content = body
