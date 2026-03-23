"""Writer Agent — writes full blog content."""

import logging
from tools.tools import write_blog_content
import config

logger = logging.getLogger(__name__)


def run_writer(state: dict) -> dict:
    if state.get("error"):
        return state

    logger.info("[Writer] Writing content...")
    state["logs"].append("✍️  Writer: Drafting blog content...")

    outline  = state["outline"]
    sections = "\n".join(f"- {s['heading']}" for s in outline.get("sections", []))
    word_count = config.LENGTH_MAP.get(state["length"], "1000-1400 words")

    try:
        content = write_blog_content.invoke({
            "title":    outline.get("title", state["topic"]),
            "sections": sections,
            "keywords": ", ".join(outline.get("keywords", [])),
            "tone":     state["tone"],
            "length":   word_count,
            "language": state["language"],
        })

        state["raw_content"] = content
        words = len(content.split())
        state["logs"].append(f"✅  Writer: Done — ~{words} words.")
        logger.info("[Writer] Done — ~%d words", words)

    except Exception as e:
        logger.error("[Writer] Error: %s", e)
        state["error"] = str(e)
        state["logs"].append(f"❌  Writer error: {e}")

    return state
