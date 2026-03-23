"""Image Agent — generates image prompts then images, builds final output."""

import logging
import os
import re

from tools.tools import generate_image_prompt, generate_image

logger = logging.getLogger(__name__)


def run_image_agent(state: dict) -> dict:
    if state.get("error"):
        return state

    outline  = state["outline"]
    sections = outline.get("sections", [])
    img_sections = [s for s in sections if s.get("needs_image")]

    state["image_prompts"] = []
    state["image_paths"]   = []

    if not img_sections:
        state["logs"].append("ℹ️  Image Agent: No images requested.")
    else:
        state["logs"].append(f"🎨  Image Agent: Creating {len(img_sections)} image(s)...")
        logger.info("[ImageAgent] %d images to generate", len(img_sections))

        content = state.get("seo_content") or state.get("raw_content", "")

        for section in img_sections:
            heading = section["heading"]

            # Step 1 — generate prompt
            try:
                preview = content[:300]
                img_prompt = generate_image_prompt.invoke({
                    "topic":           state["topic"],
                    "heading":         heading,
                    "content_preview": preview,
                })
                state["image_prompts"].append({"heading": heading, "prompt": img_prompt})
                logger.info("[ImageAgent] Prompt ready for: %s", heading)
            except Exception as e:
                img_prompt = f"Professional photo related to {heading}, sharp focus, natural lighting"
                state["image_prompts"].append({"heading": heading, "prompt": img_prompt})
                state["logs"].append(f"⚠️  Prompt fallback for: {heading}")

            # Step 2 — generate image
            state["logs"].append(f"  🔄 Generating image: {heading}")
            try:
                path = generate_image.invoke({"prompt": img_prompt, "section_name": heading})
                if path:
                    state["image_paths"].append({"heading": heading, "path": path})
                    state["logs"].append(f"  ✅ Image ready: {heading}")
                    logger.info("[ImageAgent] Image saved: %s", path)
                else:
                    state["logs"].append(f"  ⚠️  Image failed: {heading}")
            except Exception as e:
                logger.error("[ImageAgent] Image error for %s: %s", heading, e)
                state["logs"].append(f"  ⚠️  Image error {heading}: {e}")

    # Build final Markdown and HTML in memory
    state["final_markdown"] = _build_markdown(state)
    state["final_html"]     = _build_html(state)
    state["logs"].append("✅  Blog complete!")
    logger.info("[ImageAgent] Final output built in memory")

    return state


def _build_markdown(state: dict) -> str:
    outline  = state["outline"]
    content  = state.get("seo_content") or state.get("raw_content", "")
    img_map  = {i["heading"]: i["path"] for i in state.get("image_paths", [])}

    md = f"# {outline.get('title', state['topic'])}\n\n"
    if outline.get("meta"):
        md += f"> {outline['meta']}\n\n"

    # Inject images after matching headings
    for section in outline.get("sections", []):
        heading = section["heading"]
        if heading in img_map:
            img_path = os.path.abspath(img_map[heading])
            img_tag  = f"\n\n![{heading}]({img_path})\n"
            escaped  = re.escape(heading)
            content  = re.sub(
                rf"(^##\s+{escaped}\s*$)", rf"\1{img_tag}",
                content, count=1, flags=re.MULTILINE
            )

    md += content
    return md


def _build_html(state: dict) -> str:
    outline = state["outline"]
    md      = state.get("final_markdown", "")
    title   = outline.get("title", state["topic"])
    meta    = outline.get("meta", "")
    kw      = ", ".join(outline.get("keywords", []))

    # Simple Markdown → HTML
    body = md
    body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", body, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",  body, flags=re.MULTILINE)
    body = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   body, flags=re.MULTILINE)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         body)
    body = re.sub(r"^> (.+)$",      r"<blockquote>\1</blockquote>", body, flags=re.MULTILINE)
    body = re.sub(
        r"!\[(.+?)\]\((.+?)\)",
        r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', body
    )

    paragraphs = []
    for block in re.split(r"\n{2,}", body):
        block = block.strip()
        if not block:
            continue
        if block.startswith("<"):
            paragraphs.append(block)
        else:
            paragraphs.append(f"<p>{block.replace(chr(10), ' ')}</p>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{meta}">
  <meta name="keywords" content="{kw}">
  <title>{title}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 860px; margin: 40px auto;
            padding: 0 20px; color: #1a1a1a; line-height: 1.8; }}
    h1   {{ font-size: 2.2rem; }}
    h2   {{ font-size: 1.5rem; margin-top: 2em; color: #222; }}
    h3   {{ font-size: 1.2rem; color: #333; }}
    blockquote {{ border-left: 4px solid #ccc; margin: 1em 0; padding: .5em 1em; color: #555; }}
    figure     {{ margin: 2em 0; text-align: center; }}
    figure img {{ max-width: 100%; border-radius: 8px; }}
    figcaption {{ font-size: .85rem; color: #666; margin-top: .4em; }}
  </style>
</head>
<body>
{"".join(paragraphs)}
</body>
</html>"""
