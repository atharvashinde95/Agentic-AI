"""SEO Agent — optimises the blog draft."""

import logging
from tools.tools import seo_optimize

logger = logging.getLogger(__name__)


def run_seo_agent(state: dict) -> dict:
    if state.get("error"):
        return state

    logger.info("[SEO] Optimising content...")
    state["logs"].append("🔍  SEO Agent: Optimising content...")

    try:
        optimised = seo_optimize.invoke({
            "draft":    state["raw_content"],
            "keywords": ", ".join(state["outline"].get("keywords", [])),
            "tone":     state["tone"],
        })

        state["seo_content"] = optimised
        state["logs"].append("✅  SEO Agent: Done.")
        logger.info("[SEO] Done")

    except Exception as e:
        logger.error("[SEO] Error: %s", e)
        state["seo_content"] = state["raw_content"]   # fallback
        state["logs"].append(f"⚠️  SEO fallback (raw content used): {e}")

    return state
