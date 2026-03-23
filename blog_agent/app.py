"""
app.py — Streamlit UI for the AI Blog Writing Agent.

Tabs:
  1. ✍️  Generate  — inputs + run pipeline
  2. 📄  Blog Post  — rendered Markdown output
  3. 🌐  HTML Preview — iframe of HTML output
  4. 🖼️  Images     — gallery of generated images
  5. ⚙️  Settings   — API config override
"""

from __future__ import annotations
import logging
import os
import sys
import threading
import time

import streamlit as st

# ── Logging: file→terminal, NOT Streamlit ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("app")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 70%, #533483 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 2rem;
  }
  .agent-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.9rem;
    color: #374151;
  }
  .agent-card.active {
    border-color: #6366f1;
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    font-weight: 600;
    color: #4338ca;
  }
  .agent-card.done   { border-color: #22c55e; background: linear-gradient(135deg, #f0fdf4, #dcfce7); color: #15803d; }
  .agent-card.error  { border-color: #ef4444; background: linear-gradient(135deg, #fef2f2, #fee2e2); color: #b91c1c; }

  .step-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #cbd5e1; flex-shrink: 0;
  }
  .step-dot.active { background: #6366f1; box-shadow: 0 0 0 3px #c7d2fe; }
  .step-dot.done   { background: #22c55e; }
  .step-dot.error  { background: #ef4444; }

  .log-box {
    background: #0f172a;
    color: #94a3b8;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    border-radius: 10px;
    padding: 14px 16px;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.6;
  }
  .log-line { margin: 0; padding: 1px 0; }
  .log-line.success { color: #4ade80; }
  .log-line.error   { color: #f87171; }
  .log-line.warn    { color: #fbbf24; }
  .log-line.info    { color: #94a3b8; }

  .metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
  }
  .metric-val { font-size: 2rem; font-weight: 700; color: #1e293b; }
  .metric-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }

  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  div[data-testid="stTabs"] button {
    font-size: 0.92rem;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "blog_state": None,
        "running": False,
        "logs": [],
        "current_step": "idle",
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ── Pipeline runner (runs in same thread, yields via session state) ────────
AGENT_STEPS = [
    ("planner",         "🗂️",  "Planner Agent"),
    ("writing",         "✍️",  "Writer Agent"),
    ("seo_optimization","🔍",  "SEO Optimizer"),
    ("image_prompting", "🎨",  "Image Prompt Agent"),
    ("image_generation","🖼️",  "Image Generator"),
    ("assembling",      "📦",  "Assembler Agent"),
]


def _step_status(step_key: str, current_step: str, is_complete: bool, has_error: bool) -> str:
    order = [s[0] for s in AGENT_STEPS]
    if has_error and current_step == step_key:
        return "error"
    if is_complete:
        return "done"
    try:
        cur_idx = order.index(current_step)
        step_idx = order.index(step_key)
    except ValueError:
        return "idle"
    if step_idx < cur_idx:
        return "done"
    if step_idx == cur_idx:
        return "active"
    return "idle"


def _render_log_box(logs: list[str]):
    lines_html = ""
    for line in logs[-60:]:
        if "✅" in line or "✔" in line:
            css = "success"
        elif "❌" in line:
            css = "error"
        elif "⚠️" in line:
            css = "warn"
        else:
            css = "info"
        escaped = line.replace("<", "&lt;").replace(">", "&gt;")
        lines_html += f'<p class="log-line {css}">{escaped}</p>'
    st.markdown(f'<div class="log-box">{lines_html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">✍️ AI Blog Writing Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by LangGraph · Amazon Nova LLM · Nova Canvas Image Gen</div>', unsafe_allow_html=True)

tab_gen, tab_blog, tab_html, tab_images, tab_settings = st.tabs([
    "🚀 Generate",
    "📄 Blog Post",
    "🌐 HTML Preview",
    "🖼️ Images",
    "⚙️ Settings",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    # ── Left: Inputs ──────────────────────────────────────────────────────
    with left_col:
        st.markdown("### 📝 Blog Configuration")

        topic = st.text_input(
            "Blog Topic *",
            placeholder="e.g. The Future of Renewable Energy in India",
            help="Main subject of your blog post",
        )

        keywords = st.text_input(
            "Target Keywords",
            placeholder="e.g. solar energy, wind power, green technology",
            help="Comma-separated SEO keywords",
        )

        col_tone, col_len = st.columns(2)
        with col_tone:
            tone = st.selectbox("Tone", [
                "informative and engaging",
                "professional and authoritative",
                "casual and conversational",
                "inspirational and motivational",
                "technical and detailed",
                "simple and beginner-friendly",
            ])
        with col_len:
            length = st.selectbox("Length", ["short", "medium", "long"],
                                  index=1, help="short=~700w  medium=~1200w  long=~2000w")

        col_img, col_lang = st.columns(2)
        with col_img:
            num_images = st.slider("Number of Images", 0, 6, 3)
        with col_lang:
            language = st.selectbox("Language", ["English", "Hindi", "French", "Spanish", "German", "Arabic"])

        custom_instructions = st.text_area(
            "Custom Instructions (optional)",
            placeholder="e.g. Focus on Indian market, include 2024 statistics, avoid technical jargon...",
            height=90,
        )

        st.markdown("---")
        generate_clicked = st.button("🚀 Generate Blog Post", disabled=st.session_state.running)

    # ── Right: Agent Status + Logs ────────────────────────────────────────
    with right_col:
        st.markdown("### 🤖 Agent Pipeline")

        state = st.session_state.blog_state
        current_step = st.session_state.current_step
        is_complete   = state.is_complete if state else False
        has_error     = bool(st.session_state.error)

        for step_key, icon, label in AGENT_STEPS:
            status = _step_status(step_key, current_step, is_complete, has_error)
            dot_cls = f"step-dot {status}"
            card_cls = f"agent-card {status}"
            spinner = "⟳ " if status == "active" else ""
            st.markdown(
                f'<div class="{card_cls}"><div class="{dot_cls}"></div>'
                f'{icon} {spinner}{label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("##### 📋 Live Logs")
        logs = st.session_state.logs or ["Waiting for generation to start..."]
        _render_log_box(logs)

        # Metrics when complete
        if is_complete and state:
            st.markdown("---")
            st.markdown("##### 📊 Output Stats")
            m1, m2, m3 = st.columns(3)
            word_count = len((state.final_markdown or "").split())
            img_count  = sum(1 for s in state.outline.sections if s.image_path)
            sec_count  = len(state.outline.sections)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{word_count}</div><div class="metric-lbl">Words</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{sec_count}</div><div class="metric-lbl">Sections</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-val">{img_count}</div><div class="metric-lbl">Images</div></div>', unsafe_allow_html=True)


    # ── Pipeline execution ─────────────────────────────────────────────────
    if generate_clicked:
        if not topic.strip():
            st.error("⚠️ Please enter a blog topic.")
            st.stop()

        # Reset state
        st.session_state.blog_state  = None
        st.session_state.running     = True
        st.session_state.logs        = ["🚀 Starting Blog Writing Agent..."]
        st.session_state.current_step = "planning"
        st.session_state.error       = None

        logger.info("=" * 60)
        logger.info("NEW BLOG GENERATION REQUEST")
        logger.info("  Topic     : %s", topic)
        logger.info("  Keywords  : %s", keywords)
        logger.info("  Tone      : %s", tone)
        logger.info("  Length    : %s", length)
        logger.info("  Images    : %d", num_images)
        logger.info("  Language  : %s", language)
        logger.info("=" * 60)

        # Import here to pick up any .env changes from Settings tab
        from graph import run_pipeline

        user_inputs = {
            "topic":               topic.strip(),
            "keywords":            keywords.strip(),
            "tone":                tone,
            "length":              length,
            "num_images":          num_images,
            "language":            language,
            "custom_instructions": custom_instructions.strip(),
        }

        with st.spinner("Running agent pipeline..."):
            try:
                result = run_pipeline(user_inputs)
                st.session_state.blog_state  = result
                st.session_state.logs        = result.logs
                st.session_state.current_step = result.current_step
                st.session_state.error       = result.error

                if result.error:
                    logger.error("Pipeline completed with error: %s", result.error)
                    st.error(f"❌ Pipeline error: {result.error}")
                else:
                    logger.info("Pipeline completed successfully.")
                    st.success("✅ Blog post generated successfully! Switch to the tabs above to view it.")
                    st.balloons()

            except Exception as exc:
                logger.exception("Unhandled pipeline exception: %s", exc)
                st.session_state.error = str(exc)
                st.session_state.logs.append(f"❌ Fatal error: {exc}")
                st.error(f"❌ Fatal error: {exc}")

        st.session_state.running = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Blog Post (Markdown rendered)
# ══════════════════════════════════════════════════════════════════════════════
with tab_blog:
    state = st.session_state.blog_state
    if not state or not state.final_markdown:
        st.info("📭 No blog generated yet. Go to the **Generate** tab to create one.")
    else:
        # Metadata bar
        col_title, col_slug = st.columns([2, 1])
        with col_title:
            st.markdown(f"**Title:** {state.outline.title}")
        with col_slug:
            st.markdown(f"**Slug:** `{state.outline.slug}`")

        if state.outline.meta_description:
            st.caption(f"**Meta:** {state.outline.meta_description}")

        if state.outline.keywords:
            st.markdown(" ".join([f"`{kw}`" for kw in state.outline.keywords]))

        st.markdown("---")
        st.markdown(state.seo_content or state.raw_content)

        st.markdown("---")
        st.download_button(
            "⬇️ Download Markdown",
            data=state.final_markdown,
            file_name=f"{state.outline.slug}.md",
            mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HTML Preview
# ══════════════════════════════════════════════════════════════════════════════
with tab_html:
    state = st.session_state.blog_state
    if not state or not state.final_html:
        st.info("📭 No blog generated yet.")
    else:
        st.components.v1.html(state.final_html, height=750, scrolling=True)
        st.download_button(
            "⬇️ Download HTML",
            data=state.final_html,
            file_name=f"{state.outline.slug}.html",
            mime="text/html",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Images Gallery
# ══════════════════════════════════════════════════════════════════════════════
with tab_images:
    state = st.session_state.blog_state
    if not state:
        st.info("📭 No blog generated yet.")
    else:
        image_sections = [s for s in state.outline.sections if s.needs_image]
        if not image_sections:
            st.info("No images were requested for this blog.")
        else:
            generated = [s for s in image_sections if s.image_path and os.path.exists(s.image_path)]
            failed    = [s for s in image_sections if not s.image_path or not os.path.exists(s.image_path)]

            if generated:
                st.markdown(f"### 🖼️ Generated Images ({len(generated)})")
                cols = st.columns(min(len(generated), 3))
                for i, section in enumerate(generated):
                    with cols[i % 3]:
                        st.image(section.image_path, caption=section.heading, use_container_width=True)
                        with open(section.image_path, "rb") as f:
                            st.download_button(
                                f"⬇️ Download",
                                data=f.read(),
                                file_name=os.path.basename(section.image_path),
                                mime="image/png",
                                key=f"dl_img_{i}",
                            )
            if failed:
                st.warning(f"⚠️ {len(failed)} image(s) could not be generated: {', '.join(s.heading for s in failed)}")

            # Show prompts used
            with st.expander("🔍 View Image Generation Prompts"):
                for s in image_sections:
                    st.markdown(f"**{s.heading}**")
                    st.code(s.image_prompt, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Settings
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("### ⚙️ API & Model Configuration")
    st.info("Changes here update the `.env` file and take effect on the next generation run.")

    with st.form("settings_form"):
        st.markdown("#### 🧠 LLM Settings")
        llm_url   = st.text_input("LLM Base URL",   value=os.getenv("LLM_BASE_URL", ""), placeholder="https://...")
        llm_key   = st.text_input("LLM API Key",    value=os.getenv("LLM_API_KEY", ""),  type="password")
        llm_model = st.text_input("LLM Model Name", value=os.getenv("LLM_MODEL", "amazon.nova-lite-v1:0"))
        llm_temp  = st.slider("Temperature", 0.0, 1.0, float(os.getenv("LLM_TEMPERATURE", "0.7")), 0.05)

        st.markdown("#### 🖼️ Image Generation Settings")
        img_url   = st.text_input("Image API Base URL", value=os.getenv("IMG_BASE_URL", ""), placeholder="https://...")
        img_key   = st.text_input("Image API Key",      value=os.getenv("IMG_API_KEY", ""),  type="password")
        img_model = st.text_input("Image Model Name",   value=os.getenv("IMG_MODEL", "amazon.nova-canvas-v1:0"))
        img_w     = st.number_input("Image Width",  value=int(os.getenv("IMG_WIDTH", "1024")),  step=64)
        img_h     = st.number_input("Image Height", value=int(os.getenv("IMG_HEIGHT", "1024")), step=64)

        st.markdown("#### 📁 Output")
        out_dir   = st.text_input("Output Directory", value=os.getenv("OUTPUT_DIR", "outputs"))

        submitted = st.form_submit_button("💾 Save Settings")

    if submitted:
        env_content = f"""# Auto-generated by Blog Agent Settings tab
LLM_BASE_URL={llm_url}
LLM_API_KEY={llm_key}
LLM_MODEL={llm_model}
LLM_TEMPERATURE={llm_temp}

IMG_BASE_URL={img_url}
IMG_API_KEY={img_key}
IMG_MODEL={img_model}
IMG_WIDTH={int(img_w)}
IMG_HEIGHT={int(img_h)}

OUTPUT_DIR={out_dir}
"""
        with open(".env", "w") as f:
            f.write(env_content)

        # Reload dotenv
        from dotenv import load_dotenv
        load_dotenv(override=True)

        st.success("✅ Settings saved to `.env`. They will be used on the next run.")
        logger.info("Settings updated and saved to .env")
