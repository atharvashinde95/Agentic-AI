"""
graph.py — LangGraph pipeline definition for the Blog Writing Agent.

Graph flow:
  planner → writer → seo_optimizer → image_prompt → image_generator → END
                                              ↓ (no images)
                                         image_generator (builds output) → END
"""

from __future__ import annotations
import logging

from langgraph.graph import StateGraph, END

from state import BlogState
from agents.planner import run_planner
from agents.writer import run_writer
from agents.seo_optimizer import run_seo_optimizer
from agents.image_prompt import run_image_prompt_agent
from agents.image_generator import run_image_generator

logger = logging.getLogger(__name__)


def should_generate_images(state: BlogState) -> str:
    if state.error:
        return "end"
    has_image_sections = any(s.needs_image for s in state.outline.sections)
    if has_image_sections and state.num_images > 0:
        return "image_prompt"
    return "image_generator"


def _wrap(fn):
    """Wrap a BlogState function as a LangGraph dict-based node."""
    def node(state_dict: dict) -> dict:
        blog_state = BlogState(**state_dict)
        result = fn(blog_state)
        return result.model_dump()
    return node


def build_graph() -> StateGraph:
    graph = StateGraph(dict)

    graph.add_node("planner",         _wrap(run_planner))
    graph.add_node("writer",          _wrap(run_writer))
    graph.add_node("seo_optimizer",   _wrap(run_seo_optimizer))
    graph.add_node("image_prompt",    _wrap(run_image_prompt_agent))
    graph.add_node("image_generator", _wrap(run_image_generator))

    graph.set_entry_point("planner")

    graph.add_edge("planner",     "writer")
    graph.add_edge("writer",      "seo_optimizer")

    graph.add_conditional_edges(
        "seo_optimizer",
        lambda s: should_generate_images(BlogState(**s)),
        {
            "image_prompt":    "image_prompt",
            "image_generator": "image_generator",
            "end":             END,
        },
    )

    graph.add_edge("image_prompt",    "image_generator")
    graph.add_edge("image_generator", END)

    return graph.compile()


BLOG_GRAPH = build_graph()


def run_pipeline(user_inputs: dict) -> BlogState:
    """Run the full pipeline and return a BlogState."""
    initial = BlogState(**user_inputs).model_dump()
    final_dict = BLOG_GRAPH.invoke(initial)
    return BlogState(**final_dict)
