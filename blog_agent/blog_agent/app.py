"""
app.py — Streamlit UI for the AI Blog Writing Agent.

Tabs: Generate | Blog Post | HTML Preview | Images | Settings
Logs stream to the terminal in real time.
"""

import logging
import sys
import os

import streamlit as st

# All logs go to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

st.set_page_config(page_title="AI Blog Agent", page_icon="✍️", layout="wide")

# ── simple CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-size: 1rem; width: 100%;
}
.log-box {
    background: #0f172a; color: #94a3b8;
    font-family: monospace; font-size: 0.8rem;
    border-radius: 8px; padding: 12px;
    max-height: 260px; overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# ── session state defaults ────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result  = None
if "running" not in st.session_state:
    st.session_state.running = False

st.title("✍️ AI Blog Writing Agent")
st.caption("Powered by LangGraph · Amazon Nova LLM · Nova Canvas")

tab_gen, tab_blog, tab_html, tab_images, tab_settings = st.tabs([
    "🚀 Generate", "📄 Blog Post", "🌐 HTML Preview", "🖼️ Images", "⚙️ Settings"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate
# ══════════════════════════════════════════════════════════════════════════════
with tab_gen:
    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.subheader("Blog Settings")
        topic    = st.text_input("Topic *", placeholder="e.g. Future of Renewable Energy")
        keywords = st.text_input("Keywords", placeholder="e.g. solar, wind, green energy")

        c1, c2 = st.columns(2)
        with c1:
            tone = st.selectbox("Tone", [
                "informative and engaging",
                "professional and authoritative",
                "casual and conversational",
                "inspirational",
                "technical and detailed",
                "beginner-friendly",
            ])
        with c2:
            length = st.selectbox("Length", ["short", "medium", "long"], index=1)

        c3, c4 = st.columns(2)
        with c3:
            num_images = st.slider("Images", 0, 6, 3)
        with c4:
            language = st.selectbox("Language", ["English", "Hindi", "French", "Spanish"])

        run_btn = st.button("🚀 Generate Blog Post", disabled=st.session_state.running)

    with col2:
        st.subheader("Pipeline Status")

        result = st.session_state.result
        agents = [
            ("🗂️", "Planner"),
            ("✍️", "Writer"),
            ("🔍", "SEO Agent"),
            ("🖼️", "Image Agent"),
        ]

        logs  = result["logs"] if result else []
        done  = bool(result and result.get("final_markdown"))

        for icon, name in agents:
            # determine if this agent ran based on log keywords
            ran = any(name.lower().split()[0] in l.lower() for l in logs)
            color = "#22c55e" if ran else "#e2e8f0"
            st.markdown(
                f'<div style="border:1px solid {color};border-radius:8px;'
                f'padding:8px 14px;margin-bottom:8px;font-size:.9rem;">'
                f'{icon} {name}</div>',
                unsafe_allow_html=True
            )

        if logs:
            log_html = "".join(
                f'<div style="color:{"#4ade80" if "✅" in l else "#f87171" if "❌" in l else "#94a3b8"}">{l}</div>'
                for l in logs[-40:]
            )
            st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

        if done:
            words = len(result["final_markdown"].split())
            imgs  = len(result.get("image_paths", []))
            st.markdown(f"**Words:** {words} &nbsp;|&nbsp; **Images:** {imgs}")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_btn:
        if not topic.strip():
            st.error("Please enter a topic.")
            st.stop()

        st.session_state.result  = None
        st.session_state.running = True

        from graph import run_pipeline

        with st.spinner("Running agent pipeline..."):
            result = run_pipeline({
                "topic": topic, "keywords": keywords, "tone": tone,
                "length": length, "num_images": num_images, "language": language,
            })

        st.session_state.result  = result
        st.session_state.running = False

        if result.get("error"):
            st.error(f"Error: {result['error']}")
        else:
            st.success("✅ Blog generated! Switch to the tabs above to view it.")
            st.balloons()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Blog Post
# ══════════════════════════════════════════════════════════════════════════════
with tab_blog:
    result = st.session_state.result
    if not result or not result.get("final_markdown"):
        st.info("Generate a blog first.")
    else:
        outline = result.get("outline", {})
        st.markdown(f"**Title:** {outline.get('title','')}")
        st.caption(f"Slug: `{outline.get('slug','')}` | Meta: {outline.get('meta','')}")
        kws = outline.get("keywords", [])
        if kws:
            st.markdown(" ".join(f"`{k}`" for k in kws))
        st.divider()
        st.markdown(result["seo_content"] or result["raw_content"])
        st.divider()
        st.download_button(
            "⬇️ Download Markdown",
            data=result["final_markdown"],
            file_name=f"{outline.get('slug','blog')}.md",
            mime="text/markdown",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HTML Preview
# ══════════════════════════════════════════════════════════════════════════════
with tab_html:
    result = st.session_state.result
    if not result or not result.get("final_html"):
        st.info("Generate a blog first.")
    else:
        st.components.v1.html(result["final_html"], height=720, scrolling=True)
        st.download_button(
            "⬇️ Download HTML",
            data=result["final_html"],
            file_name=f"{result['outline'].get('slug','blog')}.html",
            mime="text/html",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Images
# ══════════════════════════════════════════════════════════════════════════════
with tab_images:
    result = st.session_state.result
    if not result:
        st.info("Generate a blog first.")
    else:
        image_paths = result.get("image_paths", [])
        if not image_paths:
            st.info("No images were generated.")
        else:
            cols = st.columns(min(len(image_paths), 3))
            for i, img in enumerate(image_paths):
                if os.path.exists(img["path"]):
                    with cols[i % 3]:
                        st.image(img["path"], caption=img["heading"], use_container_width=True)
                        with open(img["path"], "rb") as f:
                            st.download_button(
                                "⬇️ Download",
                                data=f.read(),
                                file_name=os.path.basename(img["path"]),
                                mime="image/png",
                                key=f"img_{i}",
                            )

        with st.expander("View image prompts used"):
            for p in result.get("image_prompts", []):
                st.markdown(f"**{p['heading']}**")
                st.code(p["prompt"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Settings
# ══════════════════════════════════════════════════════════════════════════════
with tab_settings:
    st.subheader("API Configuration")
    st.info("Changes are saved to `.env` and used on the next run.")

    with st.form("settings"):
        st.markdown("**LLM**")
        llm_url   = st.text_input("LLM Base URL",   value=os.getenv("LLM_BASE_URL",""))
        llm_key   = st.text_input("LLM API Key",    value=os.getenv("LLM_API_KEY",""), type="password")
        llm_model = st.text_input("LLM Model",      value=os.getenv("LLM_MODEL","amazon.nova-lite-v1:0"))

        st.markdown("**Image Generation**")
        img_url   = st.text_input("Image Base URL", value=os.getenv("IMG_BASE_URL",""))
        img_key   = st.text_input("Image API Key",  value=os.getenv("IMG_API_KEY",""), type="password")
        img_model = st.text_input("Image Model",    value=os.getenv("IMG_MODEL","amazon.nova-canvas-v1:0"))

        if st.form_submit_button("💾 Save"):
            env = f"""LLM_BASE_URL={llm_url}
LLM_API_KEY={llm_key}
LLM_MODEL={llm_model}
IMG_BASE_URL={img_url}
IMG_API_KEY={img_key}
IMG_MODEL={img_model}
"""
            with open(".env", "w") as f:
                f.write(env)
            st.success("Saved to .env — restart the app for changes to take effect.")
