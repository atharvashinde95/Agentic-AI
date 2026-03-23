"""
agents/seo_optimizer.py — SEO Optimizer Agent

Calls: seo_optimize_content (tool)
Produces: seo_content stored in state
"""

from __future__ import annotations
import logging

from state import BlogState
from tools.llm_tools import seo_optimize_content

logger = logging.getLogger(__name__)


def run_seo_optimizer(state: BlogState) -> BlogState:
    """LangGraph node: SEO Optimizer Agent."""
    logger.info("▶ [SEO] title=%s", state.outline.title)
    state.logs.append("🔍  SEO Agent: Optimising content...")
    state.current_step = "seo_optimization"

    if state.error:
        return state

    try:
        # ── Call the @tool ────────────────────────────────────────────────
        optimised = seo_optimize_content.invoke({
            "draft":    state.raw_content,
            "keywords": ", ".join(state.outline.keywords),
            "tone":     state.tone,
            "language": state.language,
        })

        state.seo_content = optimised
        logger.info("✔ [SEO] optimisation complete.")
        state.logs.append("✅  SEO Agent: Content optimised.")

    except Exception as exc:
        logger.exception("[SEO] failed: %s", exc)
        state.seo_content = state.raw_content          # non-fatal fallback
        state.logs.append(f"⚠️  SEO fallback (raw content used): {exc}")

    return state
