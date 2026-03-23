"""
tools/llm_tools.py — LangChain @tool definitions for all LLM-based operations.

Each tool wraps one specific LLM capability used by the agents.
Agents import and call these tools directly — keeping agent logic thin.
"""

import json
import logging
import re

from langchain_core.tools import tool

from utils.llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Generate blog outline
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_outline(
    topic: str,
    keywords: str,
    tone: str,
    length: str,
    num_images: int,
    language: str,
    custom_instructions: str,
) -> str:
    """
    Generate a structured JSON blog outline from user inputs.

    Returns a JSON string containing: title, meta_description, slug,
    keywords list, and sections (each with heading, needs_image, image_hint).
    """
    word_count = config.LENGTH_WORD_MAP.get(length, "1000-1400 words")

    system = f"""You are a senior content strategist and SEO expert.
Create a detailed blog outline. Respond ONLY with valid JSON — no markdown fences, no extra text.

JSON schema:
{{
  "title": "Engaging blog post title",
  "meta_description": "150-160 char SEO meta description",
  "slug": "url-friendly-slug",
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "sections": [
    {{
      "heading": "Section heading",
      "needs_image": true,
      "image_hint": "brief visual concept"
    }}
  ]
}}

Rules:
- Include Introduction and Conclusion as sections
- 4-7 body sections based on length
- needs_image=true for at most {num_images} sections total
- Language: {language}"""

    human = f"""Topic: {topic}
Keywords: {keywords or 'none'}
Tone: {tone}
Length: {length} ({word_count})
Images: {num_images}
Custom instructions: {custom_instructions or 'none'}

Generate the outline now."""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | get_llm(temperature=0.5) | StrOutputParser()

    logger.info("[Tool:generate_outline] topic=%s", topic)
    raw = chain.invoke({})
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Write blog content
# ─────────────────────────────────────────────────────────────────────────────

@tool
def write_blog_content(
    title: str,
    outline_sections: str,
    keywords: str,
    tone: str,
    length: str,
    language: str,
    custom_instructions: str,
) -> str:
    """
    Write full blog content in Markdown based on a structured outline.

    outline_sections should be a newline-separated list like:
      1. Introduction
      2. What is AI?
      ...

    Returns the complete blog body in Markdown (no title line).
    """
    word_count = config.LENGTH_WORD_MAP.get(length, "1000-1400 words")

    system = f"""You are an expert blog writer. Write engaging, human-like blog content.

Rules:
- Tone: {tone}
- Target length: {word_count}
- Language: {language}
- Use the exact headings from the outline, prefixed with ##
- Write detailed, valuable paragraphs under each heading
- Do NOT include the blog title in your output
- Output ONLY Markdown content — no preamble"""

    human = f"""Blog Title: {title}
Target Keywords: {keywords}
Custom Instructions: {custom_instructions or 'none'}

Outline:
{outline_sections}

Write the full blog post now."""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | get_llm(temperature=0.75) | StrOutputParser()

    logger.info("[Tool:write_blog_content] title=%s", title)
    return chain.invoke({}).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — SEO optimise content
# ─────────────────────────────────────────────────────────────────────────────

@tool
def seo_optimize_content(
    draft: str,
    keywords: str,
    tone: str,
    language: str,
) -> str:
    """
    Review and improve a blog draft for SEO best practices.

    Improves keyword density, readability, headings, and flow.
    Keeps all ## headings intact. Returns improved Markdown content.
    """
    system = f"""You are an SEO expert and content editor. Improve the blog draft for:
1. Keyword integration — naturally weave keywords, no stuffing
2. Readability — short paragraphs, clear sentences
3. Heading optimisation — include keywords where natural
4. Smooth transitions between sections
5. Engagement — hooks, power words

Rules:
- Keep ALL ## headings exactly as they are
- Do NOT add new sections
- Preserve tone: {tone}
- Output ONLY the improved Markdown — no commentary"""

    human = f"""Keywords: {keywords}
Language: {language}

--- DRAFT ---
{draft}
--- END ---

Return the SEO-optimised version now."""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | get_llm(temperature=0.4) | StrOutputParser()

    logger.info("[Tool:seo_optimize_content] keywords=%s", keywords)
    return chain.invoke({}).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Generate image prompt
# ─────────────────────────────────────────────────────────────────────────────

@tool
def generate_image_prompt(
    topic: str,
    section_heading: str,
    content_preview: str,
) -> str:
    """
    Generate a detailed image generation prompt for a blog section.

    Returns a single image prompt string under 400 characters,
    suitable for Nova-Canvas or similar image generation models.
    """
    system = """You are a professional AI art director.
Write ONE detailed image generation prompt for a blog section.

Requirements:
- Describe the scene/subject clearly in 1-2 sentences
- Specify photographic style: lighting, camera angle, mood
- Include style keywords: e.g. professional photography, sharp focus, 4K
- Keep under 400 characters
- Do NOT include text, logos, or words in the description
- Output ONLY the prompt — nothing else"""

    human = f"""Blog topic: {topic}
Section heading: {section_heading}
Section summary: {content_preview}

Write the image prompt:"""

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | get_llm(temperature=0.6) | StrOutputParser()

    logger.info("[Tool:generate_image_prompt] section=%s", section_heading)
    return chain.invoke({}).strip()
