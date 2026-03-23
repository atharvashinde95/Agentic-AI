"""
graph.py — LangGraph pipeline.
Nodes: planner → writer → seo_agent → image_agent → END
"""

import logging
from langgraph.graph import StateGraph, END

from state import BlogState
from agents.planner    import run_planner
from agents.writer     import run_writer
from agents.seo_agent  import run_seo_agent
from agents.image_agent import run_image_agent

logger = logging.getLogger(__name__)


def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("planner",    run_planner)
    graph.add_node("writer",     run_writer)
    graph.add_node("seo_agent",  run_seo_agent)
    graph.add_node("image_agent", run_image_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "writer")
    graph.add_edge("writer",     "seo_agent")
    graph.add_edge("seo_agent",  "image_agent")
    graph.add_edge("image_agent", END)

    return graph.compile()


GRAPH = build_graph()


def run_pipeline(inputs: dict) -> dict:
    """Run the full blog pipeline. Returns the final state dict."""
    initial_state = BlogState(
        topic=inputs.get("topic", ""),
        keywords=inputs.get("keywords", ""),
        tone=inputs.get("tone", "informative"),
        length=inputs.get("length", "medium"),
        num_images=inputs.get("num_images", 3),
        language=inputs.get("language", "English"),
        outline={},
        raw_content="",
        seo_content="",
        image_prompts=[],
        image_paths=[],
        final_markdown="",
        final_html="",
        logs=[],
        error="",
    )
    result = GRAPH.invoke(initial_state)
    return result
