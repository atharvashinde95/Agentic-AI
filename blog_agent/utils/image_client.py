"""
utils/image_client.py — HTTP client for the Capgemini Nova-Canvas image
generation endpoint.

Nova-Canvas accepts a POST to the model's invocation URL and returns
the generated image as a base64-encoded string inside a JSON body.
We decode it and save it to disk, returning the file path.
"""

from __future__ import annotations
import base64
import json
import logging
import os
import re
import time
import uuid

import httpx

import config

logger = logging.getLogger(__name__)


def _sanitise_prompt(prompt: str) -> str:
    """Keep prompt within 512 chars and strip unsupported characters."""
    prompt = re.sub(r"[^\x00-\x7F]+", " ", prompt)
    return prompt[:512].strip()


def generate_image(
    prompt: str,
    section_name: str = "section",
    output_dir: str | None = None,
) -> str:
    """
    Call the Capgemini Nova-Canvas endpoint to generate one image.

    Returns
    -------
    str
        Absolute path to the saved PNG file, or "" on failure.
    """
    out_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    clean_prompt = _sanitise_prompt(prompt)
    filename = f"{section_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(out_dir, filename)

    # ── Build request body (Nova-Canvas native format) ────────────────────
    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": clean_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "width": config.IMG_WIDTH,
            "height": config.IMG_HEIGHT,
            "quality": "standard",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {config.IMG_API_KEY}",
    }

    logger.info("Image generation request | section=%s  prompt=%.80s...", section_name, clean_prompt)

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                config.IMG_BASE_URL,
                headers=headers,
                content=json.dumps(payload),
            )
            response.raise_for_status()
            body = response.json()

        # Nova-Canvas returns: {"images": ["<base64>", ...], ...}
        images = body.get("images") or body.get("artifacts") or []
        if not images:
            logger.error("No images in response: %s", body)
            return ""

        img_bytes = base64.b64decode(images[0])
        with open(filepath, "wb") as fh:
            fh.write(img_bytes)

        logger.info("Image saved → %s", filepath)
        return filepath

    except httpx.HTTPStatusError as exc:
        logger.error("Image API HTTP error %s: %s", exc.response.status_code, exc.response.text[:300])
        return ""
    except Exception as exc:
        logger.exception("Image generation failed: %s", exc)
        return ""
