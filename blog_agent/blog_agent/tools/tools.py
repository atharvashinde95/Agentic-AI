"""
tools/tools.py
All @tool functions used by the agents.
"""

import json
import re
import base64
import logging

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

logger = logging.getLogger(__name__)


# ── shared LLM instance ───────────────────────────────────────────────────────
def get_llm(temperature=0.7):
    return ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        temperature=temperature,
        max_tokens=4096,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Generate outline
# ─────────────────────────────────────────────────────────────────────────────
@tool
def generate_outline(topic: str, keywords: str, tone: str,
                     length: str, num_images: int, language: str) -> str:
    """Generate a structured JSON blog outline from the given topic and settings."""

    system = """You are a content strategist. Return ONLY valid JSON, no extra text.
Schema:
{
  "title": "Blog title",
  "slug": "url-slug",
  "meta": "150 char meta description",
  "keywords": ["kw1","kw2","kw3"],
  "sections": [
    {"heading": "Introduction", "needs_image": true},
    {"heading": "Section heading", "needs_image": false}
  ]
}
Rules:
- Always include Introduction and Conclusion sections
- 4-6 body sections depending on length
- Mark needs_image=true for at most {num_images} sections
"""

    human = """Topic: {topic}
Keywords: {keywords}
Tone: {tone}
Length: {length}
Language: {language}
Images: {num_images}"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain  = prompt | get_llm(temperature=0.5) | StrOutputParser()

    result = chain.invoke({
        "topic": topic, "keywords": keywords, "tone": tone,
        "length": length, "language": language, "num_images": num_images
    })

    # Strip markdown fences if model wraps response
    result = re.sub(r"^```(?:json)?|```$", "", result.strip(), flags=re.MULTILINE).strip()
    logger.info("[Tool] generate_outline done")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Write blog content
# ─────────────────────────────────────────────────────────────────────────────
@tool
def write_blog_content(title: str, sections: str, keywords: str,
                       tone: str, length: str, language: str) -> str:
    """Write full Markdown blog content based on the outline sections provided."""

    system = """You are an expert blog writer.
Write engaging, human-like blog content in Markdown.
- Use ## for each section heading (exactly as given)
- Tone: {tone}
- Target length: {length}
- Language: {language}
- Weave keywords naturally: {keywords}
- Do NOT include the blog title in output
- Output ONLY the Markdown content"""

    human = """Title: {title}
Sections to write:
{sections}

Write the full blog now."""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain  = prompt | get_llm(temperature=0.75) | StrOutputParser()

    result = chain.invoke({
        "title": title, "sections": sections, "keywords": keywords,
        "tone": tone, "length": length, "language": language
    })

    logger.info("[Tool] write_blog_content done")
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — SEO optimise
# ─────────────────────────────────────────────────────────────────────────────
@tool
def seo_optimize(draft: str, keywords: str, tone: str) -> str:
    """Improve the blog draft for SEO — keyword density, readability, and flow."""

    system = """You are an SEO editor.
Improve the blog for:
- Natural keyword integration (no stuffing)
- Shorter paragraphs and clearer sentences
- Stronger headings
- Smooth transitions

Rules:
- Keep all ## headings exactly as they are
- Preserve tone: {tone}
- Return ONLY the improved Markdown, no commentary"""

    human = """Keywords: {keywords}

--- DRAFT ---
{draft}
--- END ---"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain  = prompt | get_llm(temperature=0.4) | StrOutputParser()

    result = chain.invoke({"draft": draft, "keywords": keywords, "tone": tone})
    logger.info("[Tool] seo_optimize done")
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Generate image prompt
# ─────────────────────────────────────────────────────────────────────────────
@tool
def generate_image_prompt(topic: str, heading: str, content_preview: str) -> str:
    """Generate a Nova-Canvas image generation prompt for a blog section."""

    system = """You are an AI art director.
Write ONE image generation prompt under 400 characters.
- Describe the scene vividly
- Include style: professional photography, sharp focus, natural lighting
- No text, logos, or words in the image
- Return ONLY the prompt, nothing else"""

    human = """Blog topic: {topic}
Section: {heading}
Summary: {content_preview}"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain  = prompt | get_llm(temperature=0.6) | StrOutputParser()

    result = chain.invoke({"topic": topic, "heading": heading, "content_preview": content_preview})
    logger.info("[Tool] generate_image_prompt done for: %s", heading)
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Generate image
# ─────────────────────────────────────────────────────────────────────────────
@tool
def generate_image(prompt: str, section_name: str) -> str:
    """Call the Capgemini Nova-Canvas API, save PNG, return file path or empty string."""

    import os, uuid
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt[:512]},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "width": 1024,
            "height": 1024,
            "quality": "standard",
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.IMG_API_KEY}",
    }

    try:
        response = httpx.post(config.IMG_BASE_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        body   = response.json()
        images = body.get("images") or body.get("artifacts") or []
        if not images:
            logger.warning("[Tool] generate_image: no images in response")
            return ""

        fname    = f"{section_name.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.png"
        filepath = os.path.join(out_dir, fname)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(images[0]))

        logger.info("[Tool] generate_image saved: %s", filepath)
        return filepath

    except Exception as e:
        logger.error("[Tool] generate_image failed: %s", e)
        return ""
