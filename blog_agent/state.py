"""
state.py — LangGraph shared state schema (typed with Pydantic).
Every agent reads from and writes to this single state object.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class BlogSection(BaseModel):
    heading: str
    content: str = ""
    image_prompt: str = ""
    image_path: str = ""
    needs_image: bool = False


class BlogOutline(BaseModel):
    title: str = ""
    meta_description: str = ""
    slug: str = ""
    sections: list[BlogSection] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class BlogState(BaseModel):
    # ── User inputs ──────────────────────────────────────────────────────────
    topic: str = ""
    keywords: str = ""
    tone: str = "informative and engaging"
    length: str = "medium"
    num_images: int = 3
    language: str = "English"
    custom_instructions: str = ""

    # ── Agent outputs (filled progressively) ────────────────────────────────
    outline: BlogOutline = Field(default_factory=BlogOutline)
    raw_content: str = ""
    seo_content: str = ""
    final_markdown: str = ""
    final_html: str = ""

    # ── Execution metadata ───────────────────────────────────────────────────
    logs: list[str] = Field(default_factory=list)
    current_step: str = "idle"
    error: Optional[str] = None
    is_complete: bool = False

    class Config:
        arbitrary_types_allowed = True
