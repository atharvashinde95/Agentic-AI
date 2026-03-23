"""Planner Agent — generates blog outline."""

import json
import logging
from tools.tools import generate_outline

logger = logging.getLogger(__name__)


def run_planner(state: dict) -> dict:
    logger.info("[Planner] Starting for topic: %s", state["topic"])
    state["logs"].append("🗂️  Planner: Generating outline...")

    try:
        raw = generate_outline.invoke({
            "topic":      state["topic"],
            "keywords":   state["keywords"],
            "tone":       state["tone"],
            "length":     state["length"],
            "num_images": state["num_images"],
            "language":   state["language"],
        })

        outline = json.loads(raw)

        # Cap needs_image to num_images
        count = 0
        for s in outline.get("sections", []):
            if s.get("needs_image") and count < state["num_images"]:
                count += 1
            else:
                s["needs_image"] = False

        state["outline"] = outline
        state["logs"].append(f"✅  Planner: Outline ready — {len(outline['sections'])} sections.")
        logger.info("[Planner] Done — %d sections", len(outline["sections"]))

    except Exception as e:
        logger.error("[Planner] Error: %s", e)
        state["error"] = str(e)
        state["logs"].append(f"❌  Planner error: {e}")

    return state
