"""
agents/image_generator.py — Image Generator Agent

Calls: generate_blog_image (tool) — once per image section
Produces: image_path filled in each BlogSection
Also builds final_markdown and final_html directly in state (no file saving).
"""

from __future__ import annotations
import logging
import os
import re

from state import BlogState
from tools.image_tools import generate_blog_image

logger = logging.getLogger(__name__)


def _inject_images(state: BlogState) -> str:
    """Insert image Markdown references after each matching ## heading."""
    content = state.seo_content or state.raw_content
    for section in state.outline.sections:
        if not section.needs_image or not section.image_path:
            continue
        abs_path = os.path.abspath(section.image_path)
        img_md = f"\n\n![{section.heading}]({abs_path})\n"
        escaped = re.escape(section.heading)
        content = re.sub(
            rf"(^##\s+{escaped}\s*$)",
            rf"\1{img_md}",
            content, count=1, flags=re.MULTILINE,
        )
    return content


def _to_html(title: str, meta: str, slug: str, keywords: list[str], md: str) -> str:
    """Minimal Markdown → self-contained HTML string."""
    body = md
    body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", body, flags=re.MULTILINE)
    body = re.sub(r"^## (.+)$",  r"<h2>\1</h2>",  body, flags=re.MULTILINE)
    body = re.sub(r"^# (.+)$",   r"<h1>\1</h1>",   body, flags=re.MULTILINE)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         body)
    body = re.sub(
        r"!\[(.+?)\]\((.+?)\)",
        r'<figure><img src="\2" alt="\1"/><figcaption>\1</figcaption></figure>',
        body,
    )
    paragraphs = []
    for block in re.split(r"\n{2,}", body):
        block = block.strip()
        if not block:
            continue
        paragraphs.append(block if block.startswith("<") else f"<p>{block.replace(chr(10), ' ')}</p>")
    kw = ", ".join(keywords)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <meta name="description" content="{meta}"/>
  <meta name="keywords" content="{kw}"/>
  <title>{title}</title>
  <style>
    body{{font-family:Georgia,serif;max-width:860px;margin:40px auto;padding:0 20px;color:#1a1a1a;line-height:1.75}}
    h1{{font-size:2.2rem;margin-bottom:.3em}}
    h2{{font-size:1.5rem;margin-top:2em;color:#222}}
    h3{{font-size:1.2rem;color:#333}}
    p{{margin:.8em 0}}
    figure{{margin:2em 0;text-align:center}}
    figure img{{max-width:100%;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.12)}}
    figcaption{{font-size:.85rem;color:#666;margin-top:.4em}}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {chr(10).join(paragraphs)}
</body>
</html>"""


def run_image_generator(state: BlogState) -> BlogState:
    """LangGraph node: Image Generator Agent — generates images then assembles final output."""
    image_sections = [s for s in state.outline.sections if s.needs_image]
    logger.info("▶ [ImageGen] generating %d images", len(image_sections))
    state.logs.append(f"🖼️  Image Generator: Creating {len(image_sections)} images...")
    state.current_step = "image_generation"

    if state.error:
        return state

    success = 0
    for section in image_sections:
        if not section.image_prompt:
            continue
        state.logs.append(f"  🔄 Generating: {section.heading}")
        try:
            path = generate_blog_image.invoke({
                "prompt":       section.image_prompt,
                "section_name": section.heading,
            })
            if path:
                section.image_path = path
                success += 1
                state.logs.append(f"  ✅ Image ready: {section.heading}")
            else:
                state.logs.append(f"  ⚠️ No image returned for: {section.heading}")
        except Exception as exc:
            logger.exception("[ImageGen] section=%s err=%s", section.heading, exc)
            state.logs.append(f"  ⚠️ Failed for {section.heading}: {exc}")

    state.logs.append(f"✅  Image Generator: {success}/{len(image_sections)} images done.")

    # ── Build final outputs in memory (no file saving) ────────────────────
    md = f"# {state.outline.title}\n\n"
    if state.outline.meta_description:
        md += f"> {state.outline.meta_description}\n\n"
    md += _inject_images(state)
    state.final_markdown = md

    state.final_html = _to_html(
        title=state.outline.title,
        meta=state.outline.meta_description,
        slug=state.outline.slug,
        keywords=state.outline.keywords,
        md=md,
    )

    state.is_complete = True
    logger.info("✔ [ImageGen] final_markdown and final_html ready in state.")
    state.logs.append("✅  Blog complete — content ready in memory.")

    return state
